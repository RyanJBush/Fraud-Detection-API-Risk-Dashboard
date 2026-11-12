"""Batch evaluation versus single-row serving.

A one-row predict_proba and a many-row call can differ in the last bits (BLAS summation order). The exact `>=` rule is
kept (no rounding, no epsilon, no threshold change), so a score sitting exactly on the threshold may flag differently
when served singly. These tests pin that behaviour: the API must equal OFFLINE SINGLE-ROW scoring exactly, and the
confirmed real boundary case must stay documented and reproducible (skipped when the private real data is absent).
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudrisk import constants as C
from fraudrisk.api import create_app, load_bundle
from fraudrisk.modeling import features_to_frame, model_scores
from fraudrisk.provenance import read_json
from fraudrisk.thresholds import review_flags

from conftest import FIXTURE

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
REAL_ART = ROOT / "artifacts" / "real"
REAL_DATA = ROOT / "data" / "raw" / "creditcard.arff"


def test_api_equals_offline_single_row_scoring_exactly_on_every_synthetic_test_row(synthetic_run):
    bundle = load_bundle(synthetic_run)
    raw = pd.read_csv(FIXTURE)
    mem = pd.read_csv(synthetic_run / "split_membership.csv.gz")
    rows = raw.iloc[mem.loc[mem["split"] == "test", "row_number"].to_numpy()]
    with TestClient(create_app(synthetic_run)) as c:
        for _, r in rows.iterrows():
            feats = {k: float(r[k]) for k in C.FEATURE_COLUMNS}
            body = c.post("/score", json={"features": feats}).json()
            offline = float(model_scores(bundle.pipeline, features_to_frame(feats))[0])
            assert body["model_score"] == offline                          # bit for bit: same code path
            assert body["review_flag"] is bool(offline >= bundle.threshold)


def test_single_row_and_batch_scores_are_close_but_flags_follow_each_exact_score(synthetic_run):
    bundle = load_bundle(synthetic_run)
    raw = pd.read_csv(FIXTURE)
    X = raw[list(C.FEATURE_COLUMNS)]
    batch = model_scores(bundle.pipeline, X)
    single = np.array([model_scores(bundle.pipeline, features_to_frame({k: float(X.iloc[i][k]) for k in C.FEATURE_COLUMNS}))[0]
                       for i in range(len(X))])
    assert np.allclose(single, batch, rtol=1e-13, atol=0.0)
    # at a threshold equal to a batch score the two scorings are each judged by the exact rule, never by a tolerance
    for j in (3, 400, 900):
        t = float(batch[j])
        assert bool(review_flags(batch[[j]], t)[0]) is True
        assert bool(review_flags(single[[j]], t)[0]) is bool(single[j] >= t)


@pytest.mark.skipif(not (REAL_ART.is_dir() and REAL_DATA.is_file()), reason="needs the private real artifacts and raw data (local only)")
def test_confirmed_real_boundary_case_is_reproduced_and_api_matches_offline_single_row():
    from fraudrisk.dataio import load_dataset

    frozen = read_json(REAL_ART / "frozen_config.json")
    thr = float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
    saved = pd.read_csv(REAL_ART / "validation_scores.csv.gz").set_index("row_id")
    loaded = load_dataset(REAL_DATA)
    ids = {r: i for i, r in enumerate(loaded.row_ids)}
    bundle = load_bundle(REAL_ART)

    def feats(rid):
        return {k: float(loaded.frame.at[ids[rid], k]) for k in C.FEATURE_COLUMNS}

    # Locate the private validation boundary dynamically; no real row identifier or per-row score is published.
    boundary = saved[saved["model_score"] == thr]
    assert len(boundary) == 1
    boundary_row = str(boundary.index[0])
    batch_score = float(boundary.iloc[0]["model_score"])
    single = float(model_scores(bundle.pipeline, features_to_frame(feats(boundary_row)))[0])
    assert batch_score == thr               # the threshold was selected AS this batch score
    assert single != batch_score
    assert abs(single - batch_score) < 2e-16 and abs(single - batch_score) / batch_score < 2e-15
    assert batch_score >= thr and not (single >= thr)               # flagged in the batch, not flagged singly
    with TestClient(create_app(REAL_ART)) as c:
        body = c.post("/score", json={"features": feats(boundary_row)}).json()
        assert body["model_score"] == single and body["threshold"] == thr and body["review_flag"] is False
        # elsewhere the API still equals offline single-row scoring exactly, and tracks the saved batch scores closely
        sample = saved.sample(25, random_state=5)
        for rid, r in sample.iterrows():
            b = c.post("/score", json={"features": feats(rid)}).json()
            off = float(model_scores(bundle.pipeline, features_to_frame(feats(rid)))[0])
            assert b["model_score"] == off
            assert abs(b["model_score"] - r["model_score"]) <= 1e-13 * max(1.0, abs(r["model_score"]))
