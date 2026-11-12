"""Chronological partitioning and the cross-partition fingerprint audit.

Rule (fixed before fitting):
  * sort by Time, ties broken by original row number
  * nominal boundaries at floor(0.60 N) and floor(0.80 N) (integer arithmetic)
  * move each boundary to the start of its timestamp group
  * equal timestamps never cross partitions
  * fail (no random fallback) if a partition is empty or unusable
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .errors import PartitionUnusableError
from .validation import fingerprints


@dataclass(frozen=True)
class SplitPlan:
    n_rows: int
    order: np.ndarray                    # original row positions in (Time, row_number) order
    nominal_boundaries: tuple[int, int]  # floor(0.60N), floor(0.80N) in sorted positions
    boundaries: tuple[int, int]          # after moving to timestamp-group starts
    assignment: np.ndarray               # split name per ORIGINAL row position


def _group_start(sorted_time: np.ndarray, b: int) -> int:
    """Index of the first element with the same Time as sorted_time[b] (b itself if b is at a group start)."""
    if b <= 0:
        return 0
    if b >= len(sorted_time):
        return len(sorted_time)
    return int(np.searchsorted(sorted_time, sorted_time[b], side="left"))


def plan_chronological_split(time: np.ndarray, row_numbers: np.ndarray | None = None) -> SplitPlan:
    time = np.asarray(time, dtype="float64")
    n = len(time)
    if n == 0:
        raise PartitionUnusableError("cannot split an empty dataset")
    if not np.all(np.isfinite(time)):
        raise PartitionUnusableError("Time contains non-finite values")
    rn = np.arange(n) if row_numbers is None else np.asarray(row_numbers)
    order = np.lexsort((rn, time))  # primary key Time, secondary key original row number
    sorted_time = time[order]
    nominal = ((C.TRAIN_PERCENT * n) // 100, (C.VALIDATION_END_PERCENT * n) // 100)
    b1 = _group_start(sorted_time, nominal[0])
    b2 = _group_start(sorted_time, nominal[1])
    names = np.empty(n, dtype=object)
    names[order[:b1]] = "train"
    names[order[b1:b2]] = "validation"
    names[order[b2:]] = "test"
    return SplitPlan(n_rows=n, order=order, nominal_boundaries=nominal, boundaries=(b1, b2), assignment=names)


def summarize_split(plan: SplitPlan, time: np.ndarray, y: np.ndarray) -> dict:
    """Counts, prevalence, timestamp ranges and usability for each partition (observed, not rebalanced)."""
    time = np.asarray(time, dtype="float64")
    y = np.asarray(y)
    parts = {}
    reasons: list[str] = []
    for name in C.SPLIT_NAMES:
        m = plan.assignment == name
        n = int(m.sum())
        n1 = int(y[m].sum()) if n else 0
        parts[name] = {
            "n_rows": n,
            "n_fraud": n1,
            "n_non_fraud": n - n1,
            "fraud_prevalence": (n1 / n) if n else None,
            "time_min": float(time[m].min()) if n else None,
            "time_max": float(time[m].max()) if n else None,
        }
        if n == 0:
            reasons.append(f"{name} partition is empty")
        else:
            if n1 == 0:
                reasons.append(f"{name} partition has no positive labels")
            if n - n1 == 0:
                reasons.append(f"{name} partition has no negative labels")
    strict = all(
        parts[a]["time_max"] is not None and parts[b]["time_min"] is not None
        and parts[a]["time_max"] < parts[b]["time_min"]
        for a, b in (("train", "validation"), ("validation", "test"))
    )
    if not strict:
        reasons.append("timestamp groups cross partitions or a partition is empty (Time ranges not strictly ordered)")
    total = sum(p["n_rows"] for p in parts.values())
    if total != plan.n_rows:
        reasons.append(f"rows not preserved: {total} != {plan.n_rows}")
    return {
        "n_rows": plan.n_rows,
        "rule": {
            "sort": "Time ascending, ties by original row number",
            "nominal_boundaries": {
                "train_end_exclusive_sorted_position": plan.nominal_boundaries[0],
                "validation_end_exclusive_sorted_position": plan.nominal_boundaries[1],
                "formula": "floor(0.60*N), floor(0.80*N) via integer arithmetic (60*N)//100, (80*N)//100",
            },
            "adjusted_boundaries": {
                "train_end_exclusive_sorted_position": plan.boundaries[0],
                "validation_end_exclusive_sorted_position": plan.boundaries[1],
                "rows_moved_earlier": [
                    plan.nominal_boundaries[0] - plan.boundaries[0],
                    plan.nominal_boundaries[1] - plan.boundaries[1],
                ],
                "adjustment": "each boundary moved to the first row of its Time group",
            },
        },
        "partitions": parts,
        "time_ranges_strictly_ordered": bool(strict),
        "rows_preserved": total == plan.n_rows,
        "usable": not reasons,
        "unusable_reasons": reasons,
        "randomization": "none; no random fallback exists in this code path",
    }


def assert_usable(summary: dict) -> None:
    if not summary["usable"]:
        raise PartitionUnusableError("; ".join(summary["unusable_reasons"]))


def cross_partition_audit(
    frame: pd.DataFrame, assignment: np.ndarray, y: np.ndarray, row_ids: list[str]
) -> tuple[dict, pd.DataFrame]:
    """Audit repeated V1-V28+Amount fingerprints (Time excluded) within and across partitions.

    Identical feature vectors do NOT prove identical transactions; this audit only
    counts them. No rows are dropped or re-assigned here.
    """
    fp = fingerprints(frame, C.FEATURE_COLUMNS)
    df = pd.DataFrame({"fp": fp, "split": assignment, "y": np.asarray(y), "row_id": row_ids,
                       "Time": frame[C.TIME_COLUMN].to_numpy()})
    nsplits = df.groupby("fp")["split"].nunique()
    sizes = df.groupby("fp")["split"].size()
    crossing_fps = set(nsplits[nsplits > 1].index)
    df["crosses"] = df["fp"].isin(crossing_fps)

    per_split = {}
    for name in C.SPLIT_NAMES:
        d = df[df["split"] == name]
        c = d["fp"].value_counts()
        rep = c[c > 1]
        per_split[name] = {
            "rows": int(len(d)),
            "repeated_fingerprint_groups_within_partition": int(len(rep)),
            "extra_copies_within_partition": int((c - 1).sum()),
            "rows_whose_fingerprint_also_in_another_partition": int(d["crosses"].sum()),
            "fraud_rows_whose_fingerprint_also_in_another_partition": int(d.loc[d["crosses"], "y"].sum()),
        }

    sets = {n: set(df.loc[df["split"] == n, "fp"]) for n in C.SPLIT_NAMES}
    tr, va, te = sets["train"], sets["validation"], sets["test"]

    def rows_in(name: str, other: set) -> dict:
        d = df[(df["split"] == name) & (df["fp"].isin(other))]
        return {"rows": int(len(d)), "fraud_rows": int(d["y"].sum())}

    pair_counts = {
        "train_and_validation": len(tr & va),
        "train_and_test": len(tr & te),
        "validation_and_test": len(va & te),
        "all_three": len(tr & va & te),
    }
    # Label agreement within crossing groups (overall conflicts were already ruled out by data-check).
    lab = df[df["crosses"]].groupby("fp")["y"].nunique()
    full_fp = fingerprints(frame, [C.TIME_COLUMN, *C.FEATURE_COLUMNS])
    full_nsplits = pd.DataFrame({"fp": full_fp, "split": assignment}).groupby("fp")["split"].nunique()
    # Descriptive only (no treatment is applied): how far apart in Time are copies that cross a boundary,
    # and what Amounts do crossing groups carry.
    cross = df[df["crosses"]]
    order_idx = {n: i for i, n in enumerate(C.SPLIT_NAMES)}
    gaps = []
    for _, g in cross.groupby("fp"):
        g = g.assign(o=g["split"].map(order_idx)).sort_values(["Time"], kind="stable")
        # smallest Time difference between two copies that sit in different partitions
        t, o = g["Time"].to_numpy(), g["o"].to_numpy()
        best = None
        for i in range(len(g) - 1):
            if o[i] != o[i + 1]:
                d = float(t[i + 1] - t[i])
                best = d if best is None else min(best, d)
        if best is not None:
            gaps.append(best)
    amt = frame.loc[df["crosses"].to_numpy(), C.AMOUNT_COLUMN].to_numpy(dtype="float64")
    descriptors = {
        "n_crossing_groups_with_adjacent_cross_partition_copies": len(gaps),
        "min_time_gap_seconds_between_cross_partition_copies": {
            "min": float(np.min(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "max": float(np.max(gaps)) if gaps else None,
        },
        "amount_in_crossing_rows": {
            "share_zero_amount": float(np.mean(amt == 0.0)) if amt.size else None,
            "median": float(np.median(amt)) if amt.size else None,
            "max": float(np.max(amt)) if amt.size else None,
        },
        "fraud_rows_in_crossing_groups": int(cross["y"].sum()),
    }
    audit = {
        "crossing_group_descriptors": descriptors,
        "fingerprint_definition": "SHA-256 of float64 bytes of V1..V28 + Amount in saved order; Time excluded; -0.0 normalised",
        "caveat": "Identical feature vectors do not prove identical transactions. No entity identifiers exist, "
                  "so entity separation across partitions cannot be established.",
        "cross_partition_repeated_fingerprints": int(len(crossing_fps)),
        "rows_in_crossing_fingerprint_groups": int(df["crosses"].sum()),
        "crossing_fingerprints_by_partition_pair": pair_counts,
        "validation_rows_with_fingerprint_in_train": rows_in("validation", tr),
        "test_rows_with_fingerprint_in_train": rows_in("test", tr),
        "test_rows_with_fingerprint_in_validation": rows_in("test", va),
        "test_rows_with_fingerprint_in_train_or_validation": rows_in("test", tr | va),
        "crossing_groups_with_conflicting_labels": int((lab > 1).sum()),
        "time_plus_features_fingerprints_crossing_partitions": int((full_nsplits > 1).sum()),
        "per_partition": per_split,
        "largest_crossing_group_size": int(sizes[list(crossing_fps)].max()) if crossing_fps else 0,
        "review_required": bool(len(crossing_fps) > 0),
        "treatment_applied": "none (no rows dropped, no re-split, no reweighting)",
    }
    rows = df[df["crosses"]].copy()
    rows["fingerprint_prefix"] = rows["fp"].str[:16]
    rows["group_size"] = rows["fp"].map(sizes)
    rows = rows.rename(columns={"y": "Class"})[
        ["row_id", "split", "Class", "Time", "group_size", "fingerprint_prefix"]
    ].sort_values(["fingerprint_prefix", "Time", "row_id"], kind="stable")
    return audit, rows.reset_index(drop=True)


def unseen_test_mask(frame: pd.DataFrame, assignment: np.ndarray) -> np.ndarray:
    """True for test rows whose exact V1..V28 + Amount vector appears in NO training or validation row.

    Uses input features only: no labels, no scores, no rounding (exact float64 bytes, -0.0 normalised, Time
    excluded). A test row matching only another TEST row stays in the subset.
    """
    fp = fingerprints(frame, C.FEATURE_COLUMNS)
    assignment = np.asarray(assignment)
    seen = set(fp[assignment != "test"])
    return np.array([(a == "test") and (f not in seen) for a, f in zip(assignment, fp)], dtype=bool)
