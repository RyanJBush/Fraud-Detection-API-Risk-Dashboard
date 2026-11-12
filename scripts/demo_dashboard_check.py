"""Drive the Streamlit dashboard (AppTest) against the REAL aggregate artifacts and the LIVE local API.

    python scripts/demo_dashboard_check.py [API_URL] [ARTIFACT_DIR]

Checks: it renders; the example request really reaches the API and the page shows the API's response; the validation
slider moves only the validation section; the frozen threshold, the "frozen" metrics, every test table and every artifact
file are identical before and after; the API still answers identically. Prints aggregates only.
"""
import json
import os
import sys
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from fraudrisk.dashboard_data import call_score_api, example_features
from fraudrisk.provenance import sha256_file

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
art = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/real").resolve()
os.environ["FRAUDRISK_ARTIFACT_DIR"] = str(art)
os.environ["FRAUDRISK_API_URL"] = url
app = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
files = ["frozen_config.json", "test_metrics.json", "model_pipeline.joblib", "threshold_selection_validation.json",
         "threshold_candidates_model.csv.gz"]
before = {f: sha256_file(art / f) for f in files}
api_before = call_score_api(url, example_features())[1]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")


at = AppTest.from_file(app, default_timeout=120).run()
check("dashboard renders without exceptions or errors", not at.exception and not at.error)
check("6 sections", len(at.header) == 6, str([h.value[:2] for h in at.header]))
at.button(key="score_button").click().run()
shown = at.json[0].value
shown = json.loads(shown) if isinstance(shown, str) else shown
check("page shows the actual API response", shown == api_before and "review_flag" in shown, f"(model_score {shown['model_score']!r})")
metrics0 = [(m.label, m.value) for m in at.metric]
tables0 = [t.value.to_csv() for t in at.table]
print("frozen section:", metrics0)
n = at.slider(key="val_rank").max
moved = False
for target in (0, int(n // 3), int(n)):
    at.slider(key="val_rank").set_value(int(target)).run()
    tables = [t.value.to_csv() for t in at.table]
    check(f"slider -> rank {target:,}: frozen metrics unchanged", [(m.label, m.value) for m in at.metric] == metrics0)
    check(f"slider -> rank {target:,}: frozen validation view + all 6 test tables unchanged", tables[1:] == tables0[1:])
    moved |= tables[0] != tables0[0]
check("explored validation table did change with the slider", moved)
check("API response still identical after slider moves", call_score_api(url, example_features())[1] == api_before)
check("artifact files unchanged (SHA-256)", {f: sha256_file(art / f) for f in files} == before)
check("API threshold == frozen threshold",
      httpx.get(url + "/model").json()["threshold"] == float.fromhex(
          json.loads((art / "frozen_config.json").read_text())["thresholds"]["model"]["threshold_hex"]))
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
