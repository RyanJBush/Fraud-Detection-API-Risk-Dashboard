"""End-to-end CLI tests on the SYNTHETIC fixture only. No real data, no network."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C
from fraudrisk.cli import main

from conftest import FIXTURE


def run(cmd, data, out, *extra):
    return main([cmd, "--data", str(data), "--data-kind", "synthetic", "--out", str(out), *extra])


@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_run")
    assert run("data-check", FIXTURE, out) == 0
    assert run("split", FIXTURE, out) == 0
    assert run("train", FIXTURE, out) == 0
    assert run("evaluate", FIXTURE, out) == 0
    return out


def J(p):
    return json.loads((p).read_text(encoding="utf-8"))


def test_every_artifact_is_labelled_synthetic(pipeline_out):
    for name in ["data_quality_report.json", "split_manifest.json", "leakage_audit.json", "model_config.json",
                 "threshold_selection_validation.json", "frozen_config.json", "test_metrics.json",
                 "error_analysis.json"]:
        d = J(pipeline_out / name)
        assert d["data_kind"] == "synthetic" and d["notice"].startswith("SYNTHETIC FIXTURE"), name


def test_split_manifest_and_membership(pipeline_out):
    m = J(pipeline_out / "split_manifest.json")
    assert m["usable"] and m["rows_preserved"] and m["time_ranges_strictly_ordered"]
    mem = pd.read_csv(pipeline_out / "split_membership.csv.gz")
    assert len(mem) == 1500 and mem["row_id"].is_unique
    assert sum(p["n_rows"] for p in m["partitions"].values()) == 1500
    assert mem.groupby("split")["Time"].agg(["min", "max"]).loc["train", "max"] < \
        mem.groupby("split")["Time"].min()["validation"]


def test_frozen_before_test_and_threshold_rule(pipeline_out):
    f = J(pipeline_out / "frozen_config.json")
    t = J(pipeline_out / "test_metrics.json")
    assert f["frozen_before_test_scoring"] and f["refit_on_train_plus_validation"] is False
    assert f["frozen_at_utc"] <= t["evaluated_at_utc"]
    assert t["frozen_config_sha256"]
    assert f["decision_rule"] == "review_flag = model_score >= threshold"
    assert f["feature_order"] == list(C.FEATURE_COLUMNS)


def test_baseline_and_model_evaluated_on_identical_rows_with_consistent_arithmetic(pipeline_out):
    t = J(pipeline_out / "test_metrics.json")
    scores = pd.read_csv(pipeline_out / "test_scores.csv.gz")
    prim = t["populations"]["primary"]
    assert prim["n_rows"] == len(scores)
    for key in ("model", "constant_score_baseline"):
        m = prim[key]
        assert m["n_rows"] == len(scores)
        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["n_rows"]
        assert m["tp"] + m["fn"] == m["n_fraud"]
        assert m["fraction_flagged"] == (m["tp"] + m["fp"]) / m["n_rows"]
    b = prim["constant_score_baseline"]
    assert b["fraction_flagged"] == 1.0  # constant score at its own selected threshold flags every row
    assert b["recall"] == 1.0
    # recompute the model's confusion counts from the saved scores and frozen threshold
    thr = float.fromhex(J(pipeline_out / "frozen_config.json")["thresholds"]["model"]["threshold_hex"])
    flags = scores["model_score"].to_numpy() >= thr
    y = scores["Class"].to_numpy().astype(bool)
    assert int((flags & y).sum()) == prim["model"]["tp"] and int((flags & ~y).sum()) == prim["model"]["fp"]


def test_validation_threshold_is_in_validation_scores_and_maximises_f1(pipeline_out):
    sel = J(pipeline_out / "threshold_selection_validation.json")["model"]
    val = pd.read_csv(pipeline_out / "validation_scores.csv.gz")
    assert sel["threshold"] in set(val["model_score"].tolist())
    table = pd.read_csv(pipeline_out / "threshold_candidates_model.csv.gz")
    assert table["f1"].max() == pytest.approx(sel["validation_f1"])
    assert sel["n_candidate_thresholds"] == val["model_score"].nunique() == len(table)


def test_error_analysis_reports_earliest_examples_by_time(pipeline_out):
    e = J(pipeline_out / "error_analysis.json")["primary"]["model"]
    scores = pd.read_csv(pipeline_out / "test_scores.csv.gz")
    mem = pd.read_csv(pipeline_out / "split_membership.csv.gz").set_index("row_id")
    thr = float.fromhex(J(pipeline_out / "frozen_config.json")["thresholds"]["model"]["threshold_hex"])
    fp = scores[(scores["model_score"] >= thr) & (scores["Class"] == 0)]
    if len(fp):
        first = fp.assign(Time=mem.loc[fp["row_id"], "Time"].to_numpy()).sort_values(["Time", "row_id"]).iloc[0]
        assert e["earliest_false_positive"]["row_id"] == first["row_id"]
        assert set(e["earliest_false_positive"]["feature_values"]) == set(C.FEATURE_COLUMNS)
    else:
        assert e["earliest_false_positive"] is None


def test_model_config_records_versions_features_and_convergence(pipeline_out):
    mc = J(pipeline_out / "model_config.json")
    assert mc["feature_order"] == list(C.FEATURE_COLUMNS)
    assert mc["model_config"]["C"] == 1.0 and mc["model_config"]["max_iter"] == 2000
    assert mc["training_provenance"]["converged"] is True
    assert mc["training_provenance"]["test_rows_excluded_from_fitting_and_threshold_selection"] is True
    assert "test_rows_present_in_memory_during_fit" not in mc["training_provenance"]
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(mc["dependency_versions"])


def test_pipeline_is_deterministic_across_runs(pipeline_out, tmp_path):
    for cmd in ("data-check", "split", "train", "evaluate"):
        assert run(cmd, FIXTURE, tmp_path) == 0
    for name in ("validation_scores.csv.gz", "test_scores.csv.gz", "split_membership.csv.gz"):
        a = gzip.open(pipeline_out / name).read()
        b = gzip.open(tmp_path / name).read()
        assert a == b, name
    assert J(pipeline_out / "frozen_config.json")["thresholds"] == J(tmp_path / "frozen_config.json")["thresholds"]


def test_evaluate_refuses_tampered_model(tmp_path):
    for cmd in ("data-check", "split", "train"):
        assert run(cmd, FIXTURE, tmp_path) == 0
    with open(tmp_path / "model_pipeline.joblib", "ab") as f:
        f.write(b"x")
    assert run("evaluate", FIXTURE, tmp_path) == C.EXIT_EVALUATION_PREREQUISITE


def test_conflicting_labels_stop_data_check(tmp_path):
    df = pd.read_csv(FIXTURE)
    df.loc[10, C.FEATURE_COLUMNS] = df.loc[3, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[10, "Class"] = 1 - df.loc[3, "Class"]
    p = tmp_path / "conflict.csv"
    df.to_csv(p, index=False)
    assert run("data-check", p, tmp_path / "o") == C.EXIT_CONFLICTING_LABELS
    assert run("split", p, tmp_path / "o") == C.EXIT_CONFLICTING_LABELS


def test_cross_partition_overlap_stops_before_fitting_without_policy(tmp_path):
    df = pd.read_csv(FIXTURE)
    df.loc[1490, list(C.FEATURE_COLUMNS)] = df.loc[5, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[1490, "Class"] = df.loc[5, "Class"]
    p = tmp_path / "overlap.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "o"
    assert run("split", p, out) == C.EXIT_CROSS_PARTITION_OVERLAP
    audit = J(out / "leakage_audit.json")["audit"]
    assert audit["cross_partition_repeated_fingerprints"] >= 1 and audit["review_required"]
    assert (out / "secondary_subset_row_ids.csv.gz").exists()  # frozen at split time, before any fit
    assert run("train", p, out) == C.EXIT_CROSS_PARTITION_OVERLAP
    assert not (out / "model_pipeline.joblib").exists()  # nothing was fitted


def test_real_kind_requires_manifest(tmp_path):
    assert main(["data-check", "--data", str(FIXTURE), "--data-kind", "real", "--out", str(tmp_path)]) == \
        C.EXIT_VALIDATION_FAILED
