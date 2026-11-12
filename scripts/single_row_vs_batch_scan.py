"""Quantify batch-versus-single-row scoring differences on every validation and test row (local, aggregate output).

    python scripts/single_row_vs_batch_scan.py [ARTIFACT_DIR] [DATA.arff]

For each row the saved batch score (validation_scores / test_scores) is compared with the score the API path computes for that
one row (`features_to_frame` + `model_scores`, exactly what POST /score runs). Reports counts, the maximum relative difference and
every row whose `score >= frozen threshold` decision differs (row ids and labels only, no feature values). Nothing is changed or
saved except the printed report; the exact >= rule is applied to both scores, with no rounding or tolerance.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudrisk import constants as C
from fraudrisk.dataio import load_dataset
from fraudrisk.modeling import features_to_frame, model_scores
from fraudrisk.provenance import read_json

art = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/real")
data = sys.argv[2] if len(sys.argv) > 2 else "data/raw/creditcard.arff"
frozen = read_json(art / "frozen_config.json")
thr = float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
pipe = joblib.load(art / frozen["model_file"])  # our own artifact; hash is checked by the API/evaluate paths
loaded = load_dataset(data)
index_of = {rid: i for i, rid in enumerate(loaded.row_ids)}
print(f"frozen threshold {thr!r}")

diffs = []
for part, fname in (("validation", "validation_scores.csv.gz"), ("test", "test_scores.csv.gz")):
    saved = pd.read_csv(art / fname)
    n = len(saved)
    t0 = time.time()
    batch = saved["model_score"].to_numpy()
    single = np.empty(n)
    cols = {k: loaded.frame[k].to_numpy(dtype="float64") for k in C.FEATURE_COLUMNS}
    idx = np.array([index_of[r] for r in saved["row_id"]])
    for j in range(n):
        feats = {k: float(cols[k][idx[j]]) for k in C.FEATURE_COLUMNS}
        single[j] = model_scores(pipe, features_to_frame(feats))[0]
    rel = np.abs(single - batch) / np.maximum(np.abs(batch), 1e-300)
    flag_b, flag_s = batch >= thr, single >= thr
    bad = np.flatnonzero(flag_b != flag_s)
    print(f"{part}: rows={n} | rows with any score difference={int((single != batch).sum())} | max relative difference={rel.max():.3e}"
          f" | flag differences={len(bad)} | exactly-on-threshold batch scores={int((batch == thr).sum())} | {time.time() - t0:.0f}s")
    for j in bad:
        print(f"   flag differs: row_id={saved['row_id'].iloc[j]} Class={int(saved['Class'].iloc[j])} "
              f"batch={batch[j]!r} (flag {bool(flag_b[j])}) single={single[j]!r} (flag {bool(flag_s[j])})")
    diffs.append((part, len(bad)))
print("SUMMARY", {p: c for p, c in diffs})
