"""Independent re-computation of the secondary subset, the model fit, the validation threshold and every test metric.

    python -I scripts/independent_results_check.py DATA.arff ARTIFACT_DIR

Separate code path: it does not import `fraudrisk`. It parses the ARFF itself, re-derives the split and the
secondary subset with pandas, refits StandardScaler + LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
class_weight=None) on the training rows only, re-selects the validation-F1 threshold from sklearn's ROC counts
with exact Fraction arithmetic, and recomputes the test confusion counts and metrics from the saved test scores.
Exit 0 only if everything matches what the pipeline saved.
"""
import gzip
import hashlib
import json
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler

data, art = sys.argv[1], sys.argv[2]
ok = True


def check(name, got, want, tol=None):
    global ok
    if tol is None:
        good = got == want
    else:
        good = abs(got - want) <= tol
    ok &= bool(good)
    print(f"{'OK  ' if good else 'FAIL'} {name}: independent={got!r} saved={want!r}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
feat = [f"V{i}" for i in range(1, 29)] + ["Amount"]
with open(data, encoding="utf-8") as f:
    skip = next(i for i, ln in enumerate(f) if ln.strip().lower() == "@data") + 1
df = pd.read_csv(data, skiprows=skip, header=None, names=cols, quotechar="'", dtype={"Class": str})
df["Class"] = df["Class"].astype(int)
df["row_number"] = np.arange(len(df))
n = len(df)
prefix = sha(data)[:12]
df["row_id"] = [f"{prefix}-{i:07d}" for i in range(n)]

# --- split (independent)
s = df.sort_values(["Time", "row_number"], kind="mergesort").reset_index(drop=True)
b1 = int(s.index[s["Time"] == s.loc[(60 * n) // 100, "Time"]].min())
b2 = int(s.index[s["Time"] == s.loc[(80 * n) // 100, "Time"]].min())
s["split"] = np.where(s.index < b1, "train", np.where(s.index < b2, "validation", "test"))

# --- secondary subset (independent, plain tuples of floats; no hashing)
keyset = set()
for part in ("train", "validation"):
    keyset |= set(s.loc[s["split"] == part, feat].itertuples(index=False, name=None))
tst = s[s["split"] == "test"].copy()
tst["unseen"] = [k not in keyset for k in tst[feat].itertuples(index=False, name=None)]
ids_ind = sorted(tst.loc[tst["unseen"], "row_id"])
ids_saved = sorted(pd.read_csv(f"{art}/secondary_subset_row_ids.csv.gz")["row_id"])
check("secondary subset size", len(ids_ind), len(ids_saved))
check("secondary subset row ids identical", ids_ind == ids_saved, True)
check("secondary positives", int(tst.loc[tst["unseen"], "Class"].sum()), 75)
check("excluded test rows / positives", (int((~tst["unseen"]).sum()), int(tst.loc[~tst["unseen"], "Class"].sum())), (1722, 0))
check("test rows total", len(tst), 56962)

# --- refit on training rows only and compare scores bit for bit
tr = s[s["split"] == "train"]
va = s[s["split"] == "validation"]
scaler = StandardScaler().fit(tr[feat])
clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None).fit(scaler.transform(tr[feat]), tr["Class"])
check("independent refit n_iter < max_iter", int(clf.n_iter_[0]) < 2000, True)
p_val = clf.predict_proba(scaler.transform(va[feat]))[:, 1]
p_tst = clf.predict_proba(scaler.transform(tst[feat]))[:, 1]
vs = pd.read_csv(f"{art}/validation_scores.csv.gz").set_index("row_id").loc[va["row_id"]]
ts = pd.read_csv(f"{art}/test_scores.csv.gz").set_index("row_id").loc[tst["row_id"]]
check("validation scores max abs diff vs refit", float(np.max(np.abs(vs["model_score"].to_numpy() - p_val))), 0.0, tol=1e-12)
check("test scores max abs diff vs refit", float(np.max(np.abs(ts["model_score"].to_numpy() - p_tst))), 0.0, tol=1e-12)
check("validation scores bit-identical to refit", bool(np.array_equal(vs["model_score"].to_numpy(), p_val)), True)
check("test scores bit-identical to refit", bool(np.array_equal(ts["model_score"].to_numpy(), p_tst)), True)

# --- validation-F1 threshold via sklearn ROC counts + exact fractions (highest threshold wins ties)
yv = va["Class"].to_numpy()
fpr, tpr, thr = roc_curve(yv, p_val, drop_intermediate=False)
P, N = int(yv.sum()), int((1 - yv).sum())
best = None
for k in range(1, len(thr)):  # thr[0] is the synthetic +inf point
    tp, fp = int(round(tpr[k] * P)), int(round(fpr[k] * N))
    f1 = Fraction(2 * tp, 2 * tp + fp + (P - tp))
    if best is None or f1 > best[0]:  # thr is descending: strict > keeps the highest threshold on ties
        best = (f1, float(thr[k]), tp, fp)
frozen = json.load(open(f"{art}/frozen_config.json"))
check("model threshold re-selected on validation", best[1], frozen["thresholds"]["model"]["threshold"])
check("validation F1 at that threshold", float(best[0]), json.load(open(f"{art}/threshold_selection_validation.json"))["model"]["validation_f1"], tol=1e-15)
prev = float(tr["Class"].mean())
check("baseline score = train prevalence", prev, frozen["thresholds"]["baseline"]["baseline_prevalence"])
check("baseline threshold = its only validation score", prev, frozen["thresholds"]["baseline"]["threshold"])

# --- test metrics recomputed from saved scores and frozen thresholds
t = json.load(open(f"{art}/test_metrics.json"))["populations"]
thr_m, thr_b = frozen["thresholds"]["model"]["threshold"], frozen["thresholds"]["baseline"]["threshold"]
ts2 = ts.assign(row_id=ts.index)
masks = {"primary": np.ones(len(ts), bool), "secondary_previously_unseen_feature_vector_subset": ts["in_secondary_subset"].to_numpy()}
for pop, m in masks.items():
    y = ts["Class"].to_numpy()[m]
    for name, col, th in (("model", "model_score", thr_m), ("constant_score_baseline", "baseline_score", thr_b)):
        sc = ts[col].to_numpy()[m]
        fl = sc >= th
        tp, fp = int((fl & (y == 1)).sum()), int((fl & (y == 0)).sum())
        fn, tn = int((~fl & (y == 1)).sum()), int((~fl & (y == 0)).sum())
        got = t[pop][name]
        check(f"{pop}/{name} TP FP FN TN", (tp, fp, fn, tn), (got["tp"], got["fp"], got["fn"], got["tn"]))
        check(f"{pop}/{name} rows, fraud", (len(y), int(y.sum())), (got["n_rows"], got["n_fraud"]))
        check(f"{pop}/{name} precision", tp / (tp + fp) if tp + fp else None, got["precision"], tol=1e-15 if tp + fp else None)
        check(f"{pop}/{name} recall", tp / int(y.sum()), got["recall"], tol=1e-15)
        check(f"{pop}/{name} F1", 2 * tp / (2 * tp + fp + fn), got["f1"], tol=1e-15)
        check(f"{pop}/{name} fraction flagged", int(fl.sum()) / len(y), got["fraction_flagged"], tol=1e-15)
        check(f"{pop}/{name} average precision", float(average_precision_score(y, sc)), got["average_precision"], tol=1e-12)
exc = ~masks["secondary_previously_unseen_feature_vector_subset"]
for name, col, th in (("model", "model_score", thr_m), ("constant_score_baseline", "baseline_score", thr_b)):
    y = ts["Class"].to_numpy()[exc]
    fl = ts[col].to_numpy()[exc] >= th
    got = t["excluded_test_rows"][name]
    check(f"excluded/{name} TP FP FN TN", (int((fl & (y == 1)).sum()), int((fl & (y == 0)).sum()), int((~fl & (y == 1)).sum()), int((~fl & (y == 0)).sum())),
          (got["tp"], got["fp"], got["fn"], got["tn"]))

# --- artifact hashes recorded in the frozen config
check("data sha256", sha(data), frozen["data_sha256"])
check("model sha256", sha(f"{art}/{frozen['model_file']}"), frozen["model_sha256"])
check("membership sha256", sha(f"{art}/split_membership.csv.gz"), frozen["split_membership_sha256"])
check("secondary ids sha256", sha(f"{art}/secondary_subset_row_ids.csv.gz"), frozen["secondary_subset"]["ids_sha256"])
check("frozen_at <= evaluated_at", frozen["frozen_at_utc"] <= json.load(open(f"{art}/test_metrics.json"))["evaluated_at_utc"], True)
print("ALL MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
