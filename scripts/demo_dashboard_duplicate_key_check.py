"""Live check: the dashboard refuses duplicate JSON keys before sending, and still submits a valid payload.

    python scripts/demo_dashboard_duplicate_key_check.py [API_URL] [ARTIFACT_DIR]

Drives the real dashboard (AppTest) against the LIVE local API with the real aggregate artifacts. The network call is wrapped by a
counter that still performs the real request, so "nothing was sent" is observed, not assumed.
"""
import json
import os
import sys
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

import fraudrisk.dashboard_data as dd

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
art = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/real").resolve()
os.environ["FRAUDRISK_ARTIFACT_DIR"] = str(art)
os.environ["FRAUDRISK_API_URL"] = url
app = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

sent = []
real_call = dd.call_score_api
dd.call_score_api = lambda base, feats, timeout=10.0: (sent.append(feats), real_call(base, feats, timeout))[1]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")


def with_dup(key):
    body = json.dumps(dd.example_features())
    val = json.dumps(dd.example_features()[key])
    return body.replace(f'"{key}": {val}', f'"{key}": 999.0, "{key}": {val}', 1)


at = AppTest.from_file(app, default_timeout=120).run()
for key in ("V1", "V28", "Amount"):
    text = with_dup(key)
    check(f"json.loads would silently drop 999.0 for {key}", json.loads(text)[key] == dd.example_features()[key])
    at.text_area(key="payload_text").set_value(text).run()
    n = len(sent)
    at.button(key="score_button").click().run()
    shown = at.json[0].value
    shown = json.loads(shown) if isinstance(shown, str) else shown
    check(f"duplicate '{key}': nothing sent to the API", len(sent) == n, f"(requests so far: {len(sent)})")
    check(f"duplicate '{key}': clear error shown", any("Nothing was scored" in e.value and f"duplicate key '{key}'" in e.value for e in at.error))
    check(f"duplicate '{key}': no model_score displayed", "model_score" not in shown and shown.get("submitted") is False)
at.text_area(key="payload_text").set_value(json.dumps(dd.example_features())).run()
at.button(key="score_button").click().run()
shown = at.json[0].value
shown = json.loads(shown) if isinstance(shown, str) else shown
check("valid payload: exactly one request sent", len(sent) == 1)
check("valid payload: real API response shown", "model_score" in shown and shown["threshold"] == httpx.get(url + "/model").json()["threshold"],
      f"(review_flag {shown.get('review_flag')})")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
