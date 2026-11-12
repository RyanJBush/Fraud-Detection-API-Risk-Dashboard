"""Numerical failure DURING scoring (finite inputs whose standardised or linear score overflows).

Checking the raw inputs for finiteness is not enough: 1e308 is finite, but (1e308 - mean) / scale or the dot product
with the coefficients can overflow to inf/NaN. Those cases must become a controlled ScoringNumericalError (HTTP 422),
while unrelated faults must still surface as faults, and no global warning/err state may be changed.
"""
import copy
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudrisk import constants as C
from fraudrisk.api import create_app
from fraudrisk.errors import DataValidationError, ScoringNumericalError
from fraudrisk.modeling import features_to_frame, model_scores

FMAX = float(np.finfo("float64").max)
EXTREMES = (1e150, 1e300, 1e308, FMAX)


@pytest.fixture(scope="module")
def pipe(synthetic_run):
    return joblib.load(synthetic_run / "model_pipeline.joblib")  # our own artifact from this test session


def base_features():
    f = {k: 0.0 for k in C.FEATURE_COLUMNS}
    f["Amount"] = 25.0
    return f


def score_or_controlled(pipe, feats):
    """Return the score, or None if (and only if) the numerical-overflow error was raised."""
    try:
        s = model_scores(pipe, features_to_frame(feats))
    except ScoringNumericalError:
        return None
    assert s.shape == (1,) and np.isfinite(s).all() and ((0.0 <= s) & (s <= 1.0)).all(), s
    return float(s[0])


# ------------------------------------------------------------------------- extreme finite values, every feature
@pytest.mark.filterwarnings("error")  # any RuntimeWarning leaking out of scoring fails the test
@pytest.mark.parametrize("feature", list(C.FEATURE_COLUMNS))
def test_extreme_single_feature_never_leaks_an_exception_or_a_warning(pipe, feature):
    for v in EXTREMES + tuple(-e for e in EXTREMES):
        if feature == C.AMOUNT_COLUMN and v < 0:
            continue
        f = base_features()
        f[feature] = v
        score_or_controlled(pipe, f)


@pytest.mark.filterwarnings("error")
def test_random_combinations_of_extremes(pipe):
    rng = np.random.default_rng(7)
    pool = np.array([0.0, 1.0, -1.0, 1e150, -1e150, 1e300, -1e300, 1e308, -1e308, FMAX, -FMAX])
    outcomes = {"scored": 0, "controlled": 0}
    for _ in range(400):
        vals = rng.choice(pool, size=len(C.FEATURE_COLUMNS))
        f = {k: float(v) for k, v in zip(C.FEATURE_COLUMNS, vals)}
        f["Amount"] = abs(f["Amount"])
        r = score_or_controlled(pipe, f)
        outcomes["scored" if r is not None else "controlled"] += 1
    assert outcomes["scored"] > 0 and outcomes["controlled"] > 0, outcomes  # the test exercises both outcomes


def test_overflow_in_the_linear_score_with_finite_standardised_features(pipe):
    """Every standardised feature is finite, but the sum of coef * z overflows: the second stage must catch it."""
    clf, sc = pipe.named_steps["classifier"], pipe.named_steps["scaler"]
    coef = clf.coef_[0]
    x = np.sign(coef) * 1e308 * np.minimum(sc.scale_, 1.0)  # each |z| <= 1e308, each term pushes the same way
    x[-1] = abs(x[-1])  # Amount must be non-negative
    with np.errstate(all="ignore"):
        z = (x - sc.mean_) / sc.scale_
        decision = z @ coef
    assert np.isfinite(z).all(), "construction must keep standardisation finite"
    assert not np.isfinite(decision), "construction must overflow only in the linear score"
    f = {k: float(v) for k, v in zip(C.FEATURE_COLUMNS, x)}
    with pytest.raises(ScoringNumericalError, match="too extreme to score"):
        model_scores(pipe, features_to_frame(f))


def test_overflow_in_standardisation_with_a_tiny_variance_feature(pipe):
    """Real V28 has a small scale; emulate that on a copy: (1e306 - mean) / 1e-3 overflows in the scaler itself."""
    p2 = copy.deepcopy(pipe)
    j = list(C.FEATURE_COLUMNS).index("V28")
    p2.named_steps["scaler"].scale_[j] = 1e-3
    f = base_features()
    f["V28"] = 1e306
    with pytest.raises(ScoringNumericalError):
        model_scores(p2, features_to_frame(f))
    f["V28"] = 1e300  # still scores (the standardised value 1e303 is finite and the score saturates)
    s = score_or_controlled(p2, f)
    assert s is not None and 0.0 <= s <= 1.0


def test_modest_values_still_score_normally_and_match_the_original_pipeline_call(pipe):
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(500, 28)), columns=list(C.PCA_COLUMNS))
    X[C.AMOUNT_COLUMN] = np.abs(rng.normal(80, 40, 500))
    ref = pipe.predict_proba(X)[:, 1]
    assert np.array_equal(model_scores(pipe, X), ref)  # the explicit two-stage path is bit-identical for batches


# ------------------------------------------------------------------------- narrowness: nothing else is reclassified
def test_unrelated_faults_are_not_turned_into_validation_errors(pipe, monkeypatch):
    clf = pipe.named_steps["classifier"]
    f = features_to_frame(base_features())
    monkeypatch.setattr(type(clf), "decision_function", lambda self, X: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        model_scores(pipe, f)
    monkeypatch.undo()
    monkeypatch.setattr(type(clf), "predict_proba", lambda self, X: (_ for _ in ()).throw(ValueError("unrelated")))
    with pytest.raises(ValueError, match="unrelated") as ei:
        model_scores(pipe, f)
    assert not isinstance(ei.value, DataValidationError)


def test_scoring_does_not_change_global_numpy_error_state_or_warning_filters(pipe):
    err_before, filters_before = np.geterr(), list(warnings.filters)
    model_scores(pipe, features_to_frame(base_features()))
    with pytest.raises(ScoringNumericalError):
        bad = base_features()
        sc = pipe.named_steps["scaler"]
        clf = pipe.named_steps["classifier"]
        x = np.sign(clf.coef_[0]) * 1e308 * np.minimum(sc.scale_, 1.0)
        x[-1] = abs(x[-1])
        model_scores(pipe, features_to_frame({k: float(v) for k, v in zip(C.FEATURE_COLUMNS, x)}))
    assert np.geterr() == err_before and list(warnings.filters) == filters_before


# ------------------------------------------------------------------------- through the HTTP API
def overflow_vector(pipe):
    clf, sc = pipe.named_steps["classifier"], pipe.named_steps["scaler"]
    x = np.sign(clf.coef_[0]) * 1e308 * np.minimum(sc.scale_, 1.0)
    x[-1] = abs(x[-1])
    return {k: float(v) for k, v in zip(C.FEATURE_COLUMNS, x)}


def test_api_returns_422_numerical_overflow_not_500(synthetic_run, pipe):
    with TestClient(create_app(synthetic_run), raise_server_exceptions=False) as c:
        r = c.post("/score", json={"features": overflow_vector(pipe)})
        assert r.status_code == 422, r.text
        assert r.json()["error"]["code"] == "numerical_overflow"
        assert "too extreme to score" in r.json()["error"]["message"]


@pytest.mark.parametrize("feature", list(C.FEATURE_COLUMNS))
def test_api_extreme_single_feature_is_200_with_valid_score_or_controlled_422(synthetic_run, feature):
    with TestClient(create_app(synthetic_run), raise_server_exceptions=False) as c:
        for v in EXTREMES + tuple(-e for e in EXTREMES):
            if feature == C.AMOUNT_COLUMN and v < 0:
                continue
            f = base_features()
            f[feature] = v
            r = c.post("/score", content=json.dumps({"features": f}).encode())
            assert r.status_code in (200, 422), (feature, v, r.status_code, r.text[:150])
            if r.status_code == 200:
                s = r.json()["model_score"]
                assert np.isfinite(s) and 0.0 <= s <= 1.0
            else:
                assert r.json()["error"]["code"] == "numerical_overflow"


def test_api_reproduces_the_reported_case_on_a_tiny_variance_feature(synthetic_run, pipe):
    """The reported 500: constructed example with V28 = 1e308 (the real V28 has a small scale)."""
    import dataclasses

    app = create_app(synthetic_run)
    with TestClient(app, raise_server_exceptions=False) as c:
        p2 = copy.deepcopy(pipe)
        p2.named_steps["scaler"].scale_[list(C.FEATURE_COLUMNS).index("V28")] = 0.5
        app.state.bundle = dataclasses.replace(app.state.bundle, pipeline=p2)
        f = base_features()
        f["V28"] = 1e308
        r = c.post("/score", json={"features": f})
        assert r.status_code == 422 and r.json()["error"]["code"] == "numerical_overflow"


def test_api_unrelated_server_fault_is_still_a_500(synthetic_run, monkeypatch):
    import fraudrisk.api as api_mod

    monkeypatch.setattr(api_mod, "model_scores", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with TestClient(create_app(synthetic_run), raise_server_exceptions=False) as c:
        r = c.post("/score", json={"features": base_features()})
        assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
