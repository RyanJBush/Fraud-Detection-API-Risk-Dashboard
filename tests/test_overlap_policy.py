"""Secondary-subset selection, overlap policy enforcement, and population reconciliation (SYNTHETIC data)."""
import json

import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C
from fraudrisk.cli import main
from fraudrisk.errors import OverlapPolicyMismatchError
from fraudrisk.policy import compare, enforce
from fraudrisk.split import plan_chronological_split, unseen_test_mask

from conftest import FIXTURE, make_frame


# ---------------------------------------------------------------- unseen_test_mask (unit)
def _frame_with_assignment(n=20):
    df = make_frame(n)
    plan = plan_chronological_split(df["Time"].to_numpy())
    return df, plan.assignment


def _copy_features(df, dst, src):
    df.loc[dst, list(C.FEATURE_COLUMNS)] = df.loc[src, list(C.FEATURE_COLUMNS)].to_numpy()


def test_unseen_mask_hand_calculated():
    df, a = _frame_with_assignment(20)  # train 0-11, validation 12-15, test 16-19
    assert list(a[[0, 11, 12, 15, 16, 19]]) == ["train", "train", "validation", "validation", "test", "test"]
    _copy_features(df, 17, 2)    # test row 17 == a train row     -> excluded
    _copy_features(df, 18, 13)   # test row 18 == a validation row -> excluded
    _copy_features(df, 19, 16)   # test row 19 == another TEST row only -> stays in the subset
    m = unseen_test_mask(df, a)
    assert m.tolist() == [False] * 16 + [True, False, False, True]
    assert not m[a != "test"].any()


def test_unseen_mask_uses_features_only_not_labels_time_or_scores():
    df, a = _frame_with_assignment(20)
    _copy_features(df, 17, 2)
    base = unseen_test_mask(df, a)
    flipped = df.assign(Class=1 - df["Class"], Time=df["Time"] + 123.0)  # labels and Time changed
    assert np.array_equal(base, unseen_test_mask(flipped, a))


def test_unseen_mask_exact_no_rounding_and_negative_zero():
    df, a = _frame_with_assignment(20)
    _copy_features(df, 17, 2)
    df.loc[17, "V3"] = df.loc[17, "V3"] + 1e-12  # not byte-equal any more -> NOT a match
    assert unseen_test_mask(df, a)[17]
    _copy_features(df, 18, 3)
    df.loc[3, "V1"], df.loc[18, "V1"] = 0.0, -0.0  # equal values -> a match
    assert not unseen_test_mask(df, a)[18]


# ---------------------------------------------------------------- policy comparison (unit)
POLICY = {"policy_id": "p", "decision": {}, "expected": {"a": 1, "b": "x"}}


def test_policy_compare_requires_every_key_exactly():
    assert compare(POLICY, {"a": 1, "b": "x"})[0]
    assert not compare(POLICY, {"a": 2, "b": "x"})[0]
    assert not compare(POLICY, {"a": 1})[0]                    # a pinned quantity was not computed
    assert not compare(POLICY, {"a": 1, "b": "x", "c": 3})[0]  # a computed quantity the policy never reviewed
    with pytest.raises(OverlapPolicyMismatchError) as ei:
        enforce(POLICY, {"a": 2, "b": "x"})
    assert ei.value.exit_code == C.EXIT_POLICY_MISMATCH and "expected 1, observed 2" in str(ei.value)


# ---------------------------------------------------------------- end to end with overlap
def run(cmd, data, out, *extra):
    return main([cmd, "--data", str(data), "--data-kind", "synthetic", "--out", str(out), *extra])


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _overlap_fixture(tmp_path):
    """Fixture where three test rows repeat train/validation vectors; they are made Class 0 (non-fraud)."""
    df = pd.read_csv(FIXTURE)
    plan = plan_chronological_split(df["Time"].to_numpy())
    a = plan.assignment
    neg_train = [i for i in range(len(df)) if a[i] == "train" and df.loc[i, "Class"] == 0][:2]
    neg_val = [i for i in range(len(df)) if a[i] == "validation" and df.loc[i, "Class"] == 0][:1]
    test_neg = [i for i in range(len(df)) if a[i] == "test" and df.loc[i, "Class"] == 0][:3]
    for dst, src in zip(test_neg, neg_train + neg_val):
        df.loc[dst, list(C.FEATURE_COLUMNS)] = df.loc[src, list(C.FEATURE_COLUMNS)].to_numpy()
    p = tmp_path / "overlap.csv"
    df.to_csv(p, index=False)
    return p


def _write_policy(tmp_path, out, mutate=None, name="policy.json"):
    """A policy pinned to this synthetic run's own findings (plumbing test; real policy is in policy/)."""
    audit = J(out / "leakage_audit.json")
    split = J(out / "split_manifest.json")
    sec = J(out / "secondary_subset_manifest.json")
    parts = split["partitions"]
    a = audit["audit"]
    exp = {
        "data_sha256": split["source_sha256"], "membership_sha256": audit["membership_sha256"],
        "n_rows": split["n_rows"], "train_rows": parts["train"]["n_rows"],
        "validation_rows": parts["validation"]["n_rows"], "test_rows": parts["test"]["n_rows"],
        "train_fraud": parts["train"]["n_fraud"], "validation_fraud": parts["validation"]["n_fraud"],
        "test_fraud": parts["test"]["n_fraud"],
        "cross_partition_repeated_fingerprints": a["cross_partition_repeated_fingerprints"],
        "rows_in_crossing_fingerprint_groups": a["rows_in_crossing_fingerprint_groups"],
        "validation_rows_with_fingerprint_in_train": a["validation_rows_with_fingerprint_in_train"]["rows"],
        "test_rows_with_fingerprint_in_train": a["test_rows_with_fingerprint_in_train"]["rows"],
        "test_rows_with_fingerprint_in_train_or_validation":
            a["test_rows_with_fingerprint_in_train_or_validation"]["rows"],
        "fraud_rows_in_crossing_groups": a["crossing_group_descriptors"]["fraud_rows_in_crossing_groups"],
        "crossing_groups_with_conflicting_labels": a["crossing_groups_with_conflicting_labels"],
        "secondary_subset_rows": sec["n_rows"], "secondary_subset_fraud": sec["n_fraud_reported_only"],
        "excluded_test_rows": sec["n_excluded_test_rows"],
        "excluded_test_fraud": parts["test"]["n_fraud"] - sec["n_fraud_reported_only"],
    }
    if mutate:
        mutate(exp)
    path = tmp_path / name
    path.write_text(json.dumps({"policy_id": "synthetic-test-policy", "decision": {"x": "test"}, "expected": exp}))
    return path


def test_full_pipeline_with_overlap_and_policy_reconciles(tmp_path):
    p = _overlap_fixture(tmp_path)
    out = tmp_path / "o"
    assert run("split", p, out) == C.EXIT_CROSS_PARTITION_OVERLAP     # detection active
    assert not (out / "model_pipeline.joblib").exists()
    pol = _write_policy(tmp_path, out)
    assert run("split", p, out, "--overlap-policy", str(pol)) == 0       # explicit policy tied to the audit
    ids_file = out / "secondary_subset_row_ids.csv.gz"
    ids_hash_before = ids_file.read_bytes()
    assert run("train", p, out, "--overlap-policy", str(pol)) == 0
    assert run("evaluate", p, out, "--overlap-policy", str(pol)) == 0
    assert ids_file.read_bytes() == ids_hash_before                      # subset was frozen before fitting

    t = J(out / "test_metrics.json")
    pops = t["populations"]
    prim, sec = pops["primary"], pops["secondary_previously_unseen_feature_vector_subset"]
    exc = pops["excluded_test_rows"]
    assert sec["label"] == "previously unseen feature-vector subset"
    assert exc["n_rows"] == 3 and prim["n_rows"] == sec["n_rows"] + 3
    for name in ("model", "constant_score_baseline"):
        for k in ("n_rows", "n_fraud", "tp", "fp", "tn", "fn", "n_flagged"):
            assert prim[name][k] == sec[name][k] + exc[name][k]
        assert exc[name]["n_fraud"] == 0 and exc[name]["tp"] == 0 and exc[name]["fn"] == 0
        assert exc[name]["recall"] is None and exc[name]["recall_status"] == "undefined_no_positive_rows"
        assert prim[name]["recall"] == sec[name]["recall"]               # identical: all positives retained
        assert prim[name]["threshold"] == sec[name]["threshold"]         # same frozen threshold
    assert sec["model"]["fraud_prevalence"] > prim["model"]["fraud_prevalence"]
    assert t["reconciliation"]["all_reconcile"] and t["reconciliation"]["all_positive_rows_retained_in_secondary"]
    text = json.dumps(t).lower()
    assert "entity-independent" not in sec["label"] and "leakage-free" not in sec["label"]
    assert "pure measure of leakage" in text
    assert J(out / "frozen_config.json")["same_model_and_thresholds_for_all_test_populations"] is True
    scores = pd.read_csv(out / "test_scores.csv.gz")
    assert scores["in_secondary_subset"].sum() == sec["n_rows"]


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(secondary_subset_rows=e["secondary_subset_rows"] + 1),
    lambda e: e.update(data_sha256="0" * 64),
    lambda e: e.update(cross_partition_repeated_fingerprints=e["cross_partition_repeated_fingerprints"] + 1),
    lambda e: e.pop("test_fraud"),
])
def test_findings_that_differ_from_the_reviewed_report_stop_every_stage(tmp_path, mutate):
    p = _overlap_fixture(tmp_path)
    out = tmp_path / "o"
    run("split", p, out)
    bad = _write_policy(tmp_path, out, mutate, name="bad_policy.json")
    assert run("split", p, out, "--overlap-policy", str(bad)) == C.EXIT_POLICY_MISMATCH
    good = _write_policy(tmp_path, out)
    assert run("split", p, out, "--overlap-policy", str(good)) == 0
    assert run("train", p, out, "--overlap-policy", str(bad)) == C.EXIT_POLICY_MISMATCH
    assert not (out / "model_pipeline.joblib").exists()
    assert run("train", p, out, "--overlap-policy", str(good)) == 0
    assert run("evaluate", p, out, "--overlap-policy", str(bad)) == C.EXIT_POLICY_MISMATCH
    assert not (out / "test_metrics.json").exists()


def test_data_changed_after_audit_is_detected(tmp_path):
    p = _overlap_fixture(tmp_path)
    out = tmp_path / "o"
    run("split", p, out)
    pol = _write_policy(tmp_path, out)
    run("split", p, out, "--overlap-policy", str(pol))
    df = pd.read_csv(p)
    df.loc[1499, "V5"] += 0.5  # change one feature value of the data file
    df.to_csv(p, index=False)
    assert run("train", p, out, "--overlap-policy", str(pol)) != 0


def test_no_overlap_means_secondary_equals_primary_and_no_policy_needed(tmp_path):
    out = tmp_path / "o"
    for c in ("data-check", "split", "train", "evaluate"):
        assert run(c, FIXTURE, out) == 0
    pops = J(out / "test_metrics.json")["populations"]
    assert pops["excluded_test_rows"]["n_rows"] == 0
    sec = pops["secondary_previously_unseen_feature_vector_subset"]
    assert pops["primary"]["model"] == sec["model"]
    assert pops["primary"]["constant_score_baseline"] == sec["constant_score_baseline"]
