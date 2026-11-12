"""Pre-fit data validation: schema, finiteness, ranges, labels, duplicates, label conflicts."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import constants as C
from .errors import ConflictingLabelsError, DataValidationError


def fingerprints(frame: pd.DataFrame, columns) -> np.ndarray:
    """SHA-256 of the exact float64 bytes of the chosen columns, one hex string per row.

    -0.0 is normalised to +0.0 (so equal values hash equally). Inputs must be finite.
    """
    arr = np.ascontiguousarray(frame.loc[:, list(columns)].to_numpy(dtype="float64") + 0.0)
    mv = memoryview(arr).cast("B")
    width = arr.shape[1] * arr.itemsize
    return np.array(
        [hashlib.sha256(mv[i * width : (i + 1) * width]).hexdigest() for i in range(arr.shape[0])],
        dtype=object,
    )


def _dup_stats(fp: np.ndarray) -> dict:
    counts = pd.Series(fp).value_counts()
    multi = counts[counts > 1]
    return {
        "unique_values": int(len(counts)),
        "extra_copies": int((counts - 1).sum()),  # rows beyond the first of each repeated group
        "groups_with_repeats": int(len(multi)),
        "rows_in_repeated_groups": int(multi.sum()),
    }


def inspect_frame(frame: pd.DataFrame, expect_published: bool = False) -> dict:
    """Return a structured report. Does not raise on data problems; see `errors` / `status`."""
    errors: list[str] = []
    cols = list(frame.columns)
    expected = list(C.SOURCE_COLUMNS)
    missing = [c for c in expected if c not in cols]
    unexpected = [c for c in cols if c not in expected]
    report: dict = {
        "n_rows": int(len(frame)),
        "expected_columns": expected,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "column_order_matches_source": cols == expected,
    }
    if missing:
        errors.append(f"missing columns: {missing}")
    if unexpected:
        errors.append(f"unexpected columns: {unexpected}")
    if missing:
        report.update(errors=errors, status="failed", conflicting_label_groups=None)
        return report

    numeric_cols = [c for c in expected if c != C.LABEL_COLUMN]
    num = frame[numeric_cols]
    na = num.isna().sum()
    raw_labels = frame[C.LABEL_COLUMN].astype(str)
    label_missing_mask = frame[C.LABEL_COLUMN].isna() | (raw_labels == "")
    report["missing_values"] = {
        "per_column": {c: int(v) for c, v in na.items() if v},
        "label_missing": int(label_missing_mask.sum()),
        "total": int(na.sum()) + int(label_missing_mask.sum()),
    }
    if report["missing_values"]["total"]:
        errors.append(f"missing values present: {report['missing_values']['total']}")

    arr = num.to_numpy(dtype="float64")
    report["nonfinite_values"] = int((~np.isfinite(arr) & ~np.isnan(arr)).sum())
    if report["nonfinite_values"]:
        errors.append(f"non-finite (inf) numeric values: {report['nonfinite_values']}")

    with np.errstate(invalid="ignore"):
        neg_amount = int((frame[C.AMOUNT_COLUMN] < 0).sum())
        neg_time = int((frame[C.TIME_COLUMN] < 0).sum())
    report["negative_amount_rows"] = neg_amount
    report["negative_time_rows"] = neg_time
    if neg_amount:
        errors.append(f"negative Amount rows: {neg_amount}")
    if neg_time:
        errors.append(f"negative Time rows: {neg_time}")

    observed = sorted(set(raw_labels[~label_missing_mask]))
    report["class_values_observed"] = observed
    bad_labels = [v for v in observed if v not in ("0", "1")]
    report["invalid_label_values"] = bad_labels
    if bad_labels:
        errors.append(f"labels outside {{0,1}}: {bad_labels}")
    report["class_counts"] = {v: int((raw_labels == v).sum()) for v in ("0", "1")}
    both = report["class_counts"]["0"] > 0 and report["class_counts"]["1"] > 0
    report["both_classes_present"] = bool(both)
    if not both:
        errors.append("Class must contain both 0 and 1")

    # Time must survive loading as real elapsed-time values (not a 0..N-1 row index, not constant).
    t = frame[C.TIME_COLUMN].to_numpy(dtype="float64")
    tfinite = t[np.isfinite(t)]
    report["time_check"] = {
        "present": True,
        "min": float(tfinite.min()) if tfinite.size else None,
        "max": float(tfinite.max()) if tfinite.size else None,
        "distinct_values": int(len(np.unique(tfinite))),
        "is_non_decreasing_in_file_order": bool(np.all(np.diff(t) >= 0)) if t.size > 1 else True,
        "equals_row_index": bool(np.array_equal(t, np.arange(len(t), dtype="float64"))),
    }
    if report["time_check"]["distinct_values"] < 2:
        errors.append("Time has fewer than 2 distinct values; elapsed-time values did not survive loading")
    if report["time_check"]["equals_row_index"]:
        errors.append("Time equals the row index; it looks replaced by a row identifier")

    if not errors:
        feat_fp = fingerprints(frame, C.FEATURE_COLUMNS)
        time_cols = [C.TIME_COLUMN, *C.FEATURE_COLUMNS]
        time_feat_fp = fingerprints(frame, time_cols)
        labels = raw_labels.astype(int).to_numpy()
        full_fp = fingerprints(
            frame.assign(**{C.LABEL_COLUMN: labels.astype("float64")}), [*time_cols, C.LABEL_COLUMN]
        )
        report["duplicates"] = {
            "identical_records_all_columns_incl_time_and_class": _dup_stats(full_fp),
            "identical_time_plus_features": _dup_stats(time_feat_fp),
            "identical_scoring_features_excluding_time": _dup_stats(feat_fp),
        }
        g = pd.DataFrame({"fp": feat_fp, "y": labels}).groupby("fp")["y"].nunique()
        conflicting = g[g > 1]
        report["conflicting_label_groups"] = int(len(conflicting))
        report["conflicting_label_rows"] = int(pd.Series(feat_fp).isin(conflicting.index).sum())
        report["conflict_definition"] = (
            "same V1-V28+Amount vector (exact float64 bytes, Time excluded) with more than one Class value"
        )
        if len(conflicting):
            errors.append(f"conflicting labels: {len(conflicting)} feature vectors with both classes")
    else:
        report["conflicting_label_groups"] = None

    if expect_published and not errors:
        n = len(frame)
        n1 = report["class_counts"]["1"]
        n0 = report["class_counts"]["0"]
        pub = C.PUBLISHED_EXPECTATIONS
        prev = 100.0 * n1 / n
        checks = {
            "n_rows": {"expected": pub["n_rows"], "observed": n, "match": n == pub["n_rows"]},
            "n_fraud": {"expected": pub["n_fraud"], "observed": n1, "match": n1 == pub["n_fraud"]},
            "n_non_fraud": {"expected": pub["n_non_fraud"], "observed": n0, "match": n0 == pub["n_non_fraud"]},
            "fraud_prevalence_percent": {
                "published_approx": pub["fraud_prevalence_percent_approx"],
                "observed": prev,
                "match": abs(prev - pub["fraud_prevalence_percent_approx"]) < 5e-5,
            },
            "time_within_two_days_seconds": {
                "limit": 172800,
                "observed_max": report["time_check"]["max"],
                "match": report["time_check"]["max"] is not None and report["time_check"]["max"] <= 172800,
            },
        }
        report["published_expectations"] = checks
        if not all(v["match"] for v in checks.values()):
            errors.append(
                "observed data does not match published expectations: "
                + ", ".join(k for k, v in checks.items() if not v["match"])
            )

    report["errors"] = errors
    report["status"] = "ok" if not errors else "failed"
    return report


def validate_frame(frame: pd.DataFrame, expect_published: bool = False) -> dict:
    """Inspect and raise the right error type; returns the report when clean."""
    report = inspect_frame(frame, expect_published=expect_published)
    if report["status"] != "ok":
        err = ConflictingLabelsError if report.get("conflicting_label_groups") else DataValidationError
        exc = err(
            ("STOP FOR REVIEW: " if err is ConflictingLabelsError else "") + "; ".join(report["errors"])
        )
        exc.report = report  # type: ignore[attr-defined]
        raise exc
    return report


def labels_as_int(frame: pd.DataFrame) -> np.ndarray:
    raw = frame[C.LABEL_COLUMN].astype(str)
    if not set(raw.unique()) <= {"0", "1"}:
        raise DataValidationError("labels outside {0,1}")
    return raw.astype(int).to_numpy(dtype="int64")
