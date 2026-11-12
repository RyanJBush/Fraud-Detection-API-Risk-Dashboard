"""Independent re-computation of the split and the leakage audit (different code path, no fraudrisk imports).

    python -I scripts/independent_split_check.py DATA.arff ARTIFACT_DIR

Re-derives partition sizes, label counts, timestamp boundaries, and the cross-partition repeated-feature-vector
counts using plain pandas groupby on the raw float columns (no hashing), then compares them with the values
saved by `python -m fraudrisk split`. Exit 0 only if everything matches.
"""
import gzip
import json
import sys

import numpy as np
import pandas as pd

data, art = sys.argv[1], sys.argv[2]
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
with open(data, encoding="utf-8") as f:
    skip = next(i for i, ln in enumerate(f) if ln.strip().lower() == "@data") + 1
df = pd.read_csv(data, skiprows=skip, header=None, names=cols, quotechar="'", dtype={"Class": str})
df["Class"] = df["Class"].astype(int)
df["row_number"] = np.arange(len(df))
n = len(df)

s = df.sort_values(["Time", "row_number"], kind="mergesort").reset_index(drop=True)
b1n, b2n = int(np.floor(0.60 * n)), int(np.floor(0.80 * n))
b1 = int(s.index[s["Time"] == s.loc[b1n, "Time"]].min())
b2 = int(s.index[s["Time"] == s.loc[b2n, "Time"]].min())
s["split"] = np.where(s.index < b1, "train", np.where(s.index < b2, "validation", "test"))

ok = True
def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} {name}: recomputed={got} saved={want}")

man = json.load(open(f"{art}/split_manifest.json"))
check("nominal boundaries", (b1n, b2n),
      (man["rule"]["nominal_boundaries"]["train_end_exclusive_sorted_position"],
       man["rule"]["nominal_boundaries"]["validation_end_exclusive_sorted_position"]))
check("adjusted boundaries", (b1, b2),
      (man["rule"]["adjusted_boundaries"]["train_end_exclusive_sorted_position"],
       man["rule"]["adjusted_boundaries"]["validation_end_exclusive_sorted_position"]))
for name in ("train", "validation", "test"):
    p = s[s["split"] == name]
    got = (len(p), int(p["Class"].sum()), float(p["Time"].min()), float(p["Time"].max()))
    q = man["partitions"][name]
    check(f"{name} rows/fraud/time_min/time_max", got, (q["n_rows"], q["n_fraud"], q["time_min"], q["time_max"]))
t = {k: set(s.loc[s["split"] == k, "Time"]) for k in ("train", "validation", "test")}
check("no Time value shared between partitions",
      (len(t["train"] & t["validation"]), len(t["validation"] & t["test"]), len(t["train"] & t["test"])), (0, 0, 0))

mem = pd.read_csv(f"{art}/split_membership.csv.gz").sort_values("row_number")
check("saved membership equals independent assignment",
      bool((mem["split"].to_numpy() == s.sort_values("row_number")["split"].to_numpy()).all()), True)

feat = [f"V{i}" for i in range(1, 29)] + ["Amount"]
g = s.groupby(feat)["split"].nunique()
crossing = g[g > 1]
aud = json.load(open(f"{art}/leakage_audit.json"))["audit"]
check("cross-partition repeated fingerprints", int(len(crossing)), aud["cross_partition_repeated_fingerprints"])
key = s.set_index(feat).index
in_cross = key.isin(crossing.index)
check("rows in crossing groups", int(in_cross.sum()), aud["rows_in_crossing_fingerprint_groups"])
tr_keys = set(s.loc[s["split"] == "train", feat].itertuples(index=False, name=None))
va_keys = set(s.loc[s["split"] == "validation", feat].itertuples(index=False, name=None))
tst = s[s["split"] == "test"]
tk = list(tst[feat].itertuples(index=False, name=None))
check("test rows with fingerprint in train", int(sum(k in tr_keys for k in tk)), aud["test_rows_with_fingerprint_in_train"]["rows"])
check("test rows with fingerprint in train or validation", int(sum((k in tr_keys) or (k in va_keys) for k in tk)),
      aud["test_rows_with_fingerprint_in_train_or_validation"]["rows"])
vk = list(s[s["split"] == "validation"][feat].itertuples(index=False, name=None))
check("validation rows with fingerprint in train", int(sum(k in tr_keys for k in vk)),
      aud["validation_rows_with_fingerprint_in_train"]["rows"])
check("fraud rows in crossing groups", int(s.loc[in_cross, "Class"].sum()), aud["crossing_group_descriptors"]["fraud_rows_in_crossing_groups"])
print("ALL MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
