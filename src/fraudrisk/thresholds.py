"""Threshold selection (validation only) and metric arithmetic.

Decision rule everywhere: review_flag = model_score >= threshold   (full float64 precision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import constants as C
from .errors import EvaluationPrerequisiteError


def review_flags(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype="float64") >= float(threshold)


def _check_inputs(y, scores):
    y = np.asarray(y)
    s = np.asarray(scores, dtype="float64")
    if y.shape != s.shape or y.ndim != 1:
        raise EvaluationPrerequisiteError("labels and scores must be 1-D arrays of equal length")
    if len(y) == 0:
        raise EvaluationPrerequisiteError("empty partition")
    if not np.all(np.isfinite(s)):
        raise EvaluationPrerequisiteError("scores must be finite")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise EvaluationPrerequisiteError("labels must be 0/1")
    return y.astype("int64"), s


def confusion_counts(y: np.ndarray, flags: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flags).astype(bool)
    return {
        "tp": int((y & f).sum()),
        "fp": int((~y & f).sum()),
        "tn": int((~y & ~f).sum()),
        "fn": int((y & ~f).sum()),
    }


def select_max_f1_threshold(y_val, scores_val) -> tuple[dict, pd.DataFrame]:
    """Evaluate every distinct validation score as a threshold (inclusive >=); maximise F1.

    F1 = 2TP / (2TP + FP + FN), compared EXACTLY with integer cross-multiplication.
    Scanning from the highest threshold down and replacing the incumbent only on a strictly
    larger F1 means ties resolve to the highest threshold, independent of input row order.
    """
    y, s = _check_inputs(y_val, scores_val)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise EvaluationPrerequisiteError("validation partition has no positive labels; cannot select a threshold")
    n_neg = len(y) - n_pos
    uniq, inv = np.unique(s, return_inverse=True)  # ascending distinct scores
    k = len(uniq)
    pos_at = np.bincount(inv, weights=y, minlength=k).astype("int64")
    all_at = np.bincount(inv, minlength=k).astype("int64")
    neg_at = all_at - pos_at
    tp = np.cumsum(pos_at[::-1])[::-1]  # rows with score >= uniq[j]
    fp = np.cumsum(neg_at[::-1])[::-1]
    fn = n_pos - tp
    tn = n_neg - fp
    num = 2 * tp
    den = 2 * tp + fp + fn

    best = None  # (j, num, den)
    for j in range(k - 1, -1, -1):  # highest threshold first
        nj, dj = int(num[j]), int(den[j])
        if best is None or nj * best[2] > best[1] * dj:  # strictly greater F1
            best = (j, nj, dj)
    j, nj, dj = best
    thr = float(uniq[j])
    table = pd.DataFrame(
        {
            "threshold": uniq,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1_numerator": num, "f1_denominator": den,
            "f1": num / den,
        }
    ).iloc[::-1].reset_index(drop=True)  # highest threshold first
    result = {
        "threshold": thr,
        "threshold_hex": thr.hex(),
        "decision_rule": C.THRESHOLD_RULE,
        "objective": C.THRESHOLD_OBJECTIVE,
        "validation_f1": nj / dj,
        "validation_f1_numerator": nj,
        "validation_f1_denominator": dj,
        "validation_tp": int(tp[j]), "validation_fp": int(fp[j]),
        "validation_fn": int(fn[j]), "validation_tn": int(tn[j]),
        "n_candidate_thresholds": int(k),
        "n_validation_rows": int(len(y)),
        "n_validation_positive": n_pos,
        "n_candidates_tied_at_best_f1": int(sum(1 for i in range(k) if int(num[i]) * dj == nj * int(den[i]))),
    }
    return result, table


def evaluate_at_threshold(y, scores, threshold: float) -> dict:
    """All reported metrics at a frozen threshold. Precision is undefined (null) if nothing is flagged."""
    y, s = _check_inputs(y, scores)
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise EvaluationPrerequisiteError("partition has no positive labels; metrics are not defined")
    flags = review_flags(s, threshold)
    cm = confusion_counts(y, flags)
    n_flagged = cm["tp"] + cm["fp"]
    if n_flagged:
        precision = cm["tp"] / n_flagged
        precision_status = "defined"
    else:
        precision = None
        precision_status = "undefined_no_rows_flagged"
    recall = cm["tp"] / n_pos
    f1 = (2 * cm["tp"]) / (2 * cm["tp"] + cm["fp"] + cm["fn"])  # denominator > 0 because n_pos > 0
    return {
        "n_rows": n,
        "n_fraud": n_pos,
        "n_non_fraud": n - n_pos,
        "fraud_prevalence": n_pos / n,
        "threshold": float(threshold),
        "threshold_hex": float(threshold).hex(),
        "decision_rule": C.THRESHOLD_RULE,
        **cm,
        "n_flagged": n_flagged,
        "fraction_flagged": n_flagged / n,
        "precision": precision,
        "precision_defined": precision is not None,
        "precision_status": precision_status,
        "recall": recall,
        "f1": f1,
        "f1_definition": "2TP / (2TP + FP + FN)",
        "average_precision": float(average_precision_score(y, s)),
        "average_precision_method": "sklearn.metrics.average_precision_score (not trapezoidal PR-AUC)",
    }


def confusion_summary(y, scores, threshold: float) -> dict:
    """Counts and rates at a frozen threshold WITHOUT requiring positives (used for excluded rows).

    Ratios that are undefined (empty population, no positives, nothing flagged) are null with a status.
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype="float64")
    n = int(len(y))
    n_pos = int(y.sum()) if n else 0
    flags = review_flags(s, threshold) if n else np.zeros(0, dtype=bool)
    cm = confusion_counts(y, flags) if n else {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    n_flagged = cm["tp"] + cm["fp"]
    return {
        "n_rows": n, "n_fraud": n_pos, "n_non_fraud": n - n_pos,
        "fraud_prevalence": (n_pos / n) if n else None,
        "threshold": float(threshold), "threshold_hex": float(threshold).hex(),
        **cm, "n_flagged": n_flagged,
        "fraction_flagged": (n_flagged / n) if n else None,
        "precision": (cm["tp"] / n_flagged) if n_flagged else None,
        "precision_status": "defined" if n_flagged else "undefined_no_rows_flagged",
        "recall": (cm["tp"] / n_pos) if n_pos else None,
        "recall_status": "defined" if n_pos else "undefined_no_positive_rows",
        "note": "counts only; average precision and F1 are not reported for a population that may have no positives",
    }
