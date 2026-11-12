"""Compare the LIVE local API with the saved offline test scores on real test rows (local run, aggregate output only).

    python scripts/demo_api_agreement.py [API_URL] [ARTIFACT_DIR] [DATA.arff]

Posts every labeled-fraud test row, the 150 test rows whose saved score is closest to the frozen threshold, and 1,000
seeded-random test rows. For each: the API model_score must equal the saved offline score up to float noise
(rtol 1e-13: a one-row call and the saved batch call may differ in the last bits), the API threshold must equal the
frozen threshold exactly, and review_flag must equal the offline `score >= threshold`. Prints counts and maxima only;
no row-level values are printed or written.

SCOPE: a zero count here is a statement about the rows posted (listed in the output), NOT about every row. A one-row API score
can differ from the saved batch score in the last bits, and the exact `>=` rule is kept, so a row whose batch score equals the
threshold exactly can flag differently (confirmed at the validation boundary; see
scripts/single_row_vs_batch_scan.py for the full scan).
"""
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from fraudrisk import constants as C
from fraudrisk.dataio import load_dataset
from fraudrisk.provenance import read_json

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
art = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/real")
data = sys.argv[3] if len(sys.argv) > 3 else "data/raw/creditcard.arff"

frozen = read_json(art / "frozen_config.json")
thr = float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
saved = pd.read_csv(art / "test_scores.csv.gz")
loaded = load_dataset(data)
index_of = {rid: i for i, rid in enumerate(loaded.row_ids)}

fraud = saved[saved["Class"] == 1]
near = saved.iloc[np.argsort(np.abs(saved["model_score"].to_numpy() - thr))[:150]]
rand = saved.sample(1000, random_state=20261111)
chosen = pd.concat([fraud, near, rand]).drop_duplicates("row_id")

info = httpx.get(url + "/model", timeout=10).json()
assert info["threshold"] == thr, "API threshold differs from frozen_config.json"
n = bad_score = bad_flag = bad_thr = 0
max_rel = 0.0
flagged = 0
with httpx.Client(base_url=url, timeout=10) as client:
    for _, r in chosen.iterrows():
        feats = {k: float(loaded.frame.at[index_of[r["row_id"]], k]) for k in C.FEATURE_COLUMNS}
        resp = client.post("/score", json={"features": feats})
        assert resp.status_code == 200, resp.text[:200]
        b = resp.json()
        n += 1
        rel = abs(b["model_score"] - r["model_score"]) / max(abs(r["model_score"]), 1e-300)
        max_rel = max(max_rel, rel)
        bad_score += rel > 1e-13
        bad_thr += b["threshold"] != thr
        offline_flag = bool(r["model_score"] >= thr)
        bad_flag += b["review_flag"] != offline_flag
        flagged += b["review_flag"]
print(f"API {url} model {info['model_version']} threshold_version {info['threshold_version']}")
print(f"rows posted: {n} (all {len(fraud)} labeled-fraud test rows, 150 nearest the threshold, 1000 random; deduplicated)")
print(f"max relative difference API vs saved offline score: {max_rel:.3e}")
print(f"score mismatches (>1e-13 relative): {bad_score} | threshold mismatches: {bad_thr} | review_flag mismatches: {bad_flag}")
print(f"flagged by the API among those rows: {flagged}")
sys.exit(0 if (bad_score == bad_thr == bad_flag == 0) else 1)
