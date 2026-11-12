"""Send a battery of malformed /score requests to the LIVE local API and print the status for each.

    python scripts/demo_malformed_requests.py [API_URL]

Exit 0 only if every case got the documented status (400 or 422) and nothing returned 5xx.
"""
import json
import sys

import httpx

from fraudrisk import constants as C

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
good = {k: 0.0 for k in C.FEATURE_COLUMNS}
good["Amount"] = 25.0


def body_with(name, literal):
    return "{" + '"features": {' + ", ".join(f'"{k}": {literal if k == name else json.dumps(v)}' for k, v in good.items()) + "}}"


cases = [
    ("valid request", json.dumps({"features": good}), 200),
    ("string number '0.0'", body_with("V1", '"0.0"'), 422),
    ("boolean true", body_with("V2", "true"), 422),
    ("boolean false (Amount)", body_with("Amount", "false"), 422),
    ("null", body_with("V3", "null"), 422),
    ("array value", body_with("V4", "[1]"), 422),
    ("1e999 (overflows to inf)", body_with("V5", "1e999"), 422),
    ("integer 10**400", body_with("V6", "1" + "0" * 400), 422),
    ("integer with 5000 digits", body_with("V7", "1" + "0" * 5000), 400),
    ("NaN literal", body_with("V8", "NaN"), 400),
    ("Infinity literal", body_with("V9", "Infinity"), 400),
    ("V28 = 1e308 (finite; overflows while scoring)", body_with("V28", "1e308"), 422),
    ("V28 = -1e308 (finite; overflows while scoring)", body_with("V28", "-1e308"), 422),
    ("V28 = 1.7976931348623157e308 (float max)", body_with("V28", "1.7976931348623157e308"), 422),
    ("V1 = 1e308 (finite; score saturates, valid)", body_with("V1", "1e308"), 200),
    ("duplicate key V1 (999.0 then 0.0)", '{"features": {"V1": 999.0, "V1": 0.0}}', 400),
    ("negative Amount", body_with("Amount", "-1"), 422),
    ("missing feature V10", json.dumps({"features": {k: v for k, v in good.items() if k != "V10"}}), 422),
    ("extra feature field", json.dumps({"features": {**good, "extra": 1}}), 422),
    ("Class in features", json.dumps({"features": {**good, "Class": 0}}), 422),
    ("Time in features", json.dumps({"features": {**good, "Time": 5}}), 422),
    ("Class at envelope level", json.dumps({"features": good, "Class": 0}), 422),
    ("missing envelope key", json.dumps({"feature": good}), 422),
    ("features is a list", json.dumps({"features": [1, 2, 3]}), 422),
    ("body is a list", json.dumps([good]), 422),
    ("empty body", "", 400),
    ("invalid JSON", "{", 400),
    ("duplicate feature key", '{"features": {"V1": 1, "V1": 2}}', 400),
    ("body over 16 KiB", '{"features": {"pad": "' + "x" * 20000 + '"}}', 400),
    ("deeply nested", "[" * 100000, 400),
]
bad = 0
with httpx.Client(base_url=url, timeout=20) as c:
    print(f"{'case':32s} {'expected':>8s} {'got':>5s}  error code")
    for name, body, want in cases:
        r = c.post("/score", content=body.encode(), headers={"content-type": "application/json"})
        code = r.json().get("error", {}).get("code", "-") if r.status_code != 200 else "-"
        ok = r.status_code == want and r.status_code < 500
        bad += not ok
        print(f"{name:32s} {want:8d} {r.status_code:5d}  {code}{'' if ok else '   <-- UNEXPECTED'}")
print("ALL AS DOCUMENTED, no 5xx" if not bad else f"{bad} UNEXPECTED")
sys.exit(1 if bad else 0)
