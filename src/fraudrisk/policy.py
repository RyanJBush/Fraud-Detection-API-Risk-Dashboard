"""Explicit overlap policy tied to the saved leakage audit.

Overlap detection stays active. A policy JSON (reviewed by the independent reviewer) records the policy text
and the exact findings it was written against. The pipeline proceeds past a detected overlap only if a policy
is supplied AND the freshly recomputed findings equal the policy's `expected` block key for key. Any
difference (data checksum, split checksum, counts) stops the run with exit code 8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import OverlapPolicyMismatchError
from .provenance import read_json, sha256_file


def load_policy(path: str | Path) -> tuple[dict, str]:
    p = Path(path)
    if not p.is_file():
        raise OverlapPolicyMismatchError(f"overlap policy file not found: {p}")
    pol = read_json(p)
    for k in ("policy_id", "decision", "expected"):
        if k not in pol:
            raise OverlapPolicyMismatchError(f"overlap policy is missing key {k!r}")
    return pol, sha256_file(p)


def observed_findings(*, data_sha256: str, membership_sha256: str, summary: dict, audit: dict,
                      unseen_mask: np.ndarray, y: np.ndarray, assignment: np.ndarray) -> dict:
    """The quantities a policy is allowed to pin. All derived from freshly computed data."""
    test = assignment == "test"
    sec = unseen_mask
    exc = test & ~unseen_mask
    parts = summary["partitions"]
    return {
        "data_sha256": data_sha256,
        "membership_sha256": membership_sha256,
        "n_rows": summary["n_rows"],
        "train_rows": parts["train"]["n_rows"],
        "validation_rows": parts["validation"]["n_rows"],
        "test_rows": parts["test"]["n_rows"],
        "train_fraud": parts["train"]["n_fraud"],
        "validation_fraud": parts["validation"]["n_fraud"],
        "test_fraud": parts["test"]["n_fraud"],
        "cross_partition_repeated_fingerprints": audit["cross_partition_repeated_fingerprints"],
        "rows_in_crossing_fingerprint_groups": audit["rows_in_crossing_fingerprint_groups"],
        "validation_rows_with_fingerprint_in_train": audit["validation_rows_with_fingerprint_in_train"]["rows"],
        "test_rows_with_fingerprint_in_train": audit["test_rows_with_fingerprint_in_train"]["rows"],
        "test_rows_with_fingerprint_in_train_or_validation":
            audit["test_rows_with_fingerprint_in_train_or_validation"]["rows"],
        "fraud_rows_in_crossing_groups": audit["crossing_group_descriptors"]["fraud_rows_in_crossing_groups"],
        "crossing_groups_with_conflicting_labels": audit["crossing_groups_with_conflicting_labels"],
        "secondary_subset_rows": int(sec.sum()),
        "secondary_subset_fraud": int(np.asarray(y)[sec].sum()),
        "excluded_test_rows": int(exc.sum()),
        "excluded_test_fraud": int(np.asarray(y)[exc].sum()),
    }


def compare(policy: dict, observed: dict) -> tuple[bool, list[dict]]:
    exp = policy["expected"]
    rows = []
    for k in sorted(set(exp) | set(observed)):
        rows.append({
            "key": k,
            "expected": exp.get(k, "<absent from policy>"),
            "observed": observed.get(k, "<not computed>"),
            "match": k in exp and k in observed and exp[k] == observed[k],
        })
    return all(r["match"] for r in rows), rows


def enforce(policy: dict, observed: dict) -> list[dict]:
    ok, rows = compare(policy, observed)
    if not ok:
        bad = [f"{r['key']}: expected {r['expected']!r}, observed {r['observed']!r}" for r in rows if not r["match"]]
        raise OverlapPolicyMismatchError(
            "STOP: findings differ from the reviewed report (policy "
            f"{policy['policy_id']!r}): " + "; ".join(bad)
        )
    return rows
