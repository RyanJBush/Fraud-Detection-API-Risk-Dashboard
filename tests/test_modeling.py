import joblib
import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C
from fraudrisk.errors import DataValidationError, TrainingNotConvergedError
from fraudrisk.modeling import (
    ConstantScoreBaseline,
    build_pipeline,
    fit_on_training_rows,
    model_scores,
    prepare_scoring_frame,
)


def _xy(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 28)), columns=list(C.PCA_COLUMNS))
    X[C.AMOUNT_COLUMN] = np.abs(rng.normal(50, 10, size=n))
    y = (rng.random(n) < 0.2).astype(int)
    X.loc[y == 1, "V1"] += 2.0
    return X, y


def test_feature_allowlist_and_order_are_explicit():
    assert C.FEATURE_COLUMNS == tuple([f"V{i}" for i in range(1, 29)] + ["Amount"])
    assert len(C.FEATURE_COLUMNS) == 29
    assert "Class" not in C.FEATURE_COLUMNS and "Time" not in C.FEATURE_COLUMNS


def test_scoring_frame_reorders_by_name_and_scores_identically():
    X, y = _xy()
    pipe, _, _ = fit_on_training_rows(X, y, np.array(["train"] * len(X), dtype=object))
    shuffled = X[list(reversed(X.columns))]
    assert list(prepare_scoring_frame(shuffled).columns) == list(C.FEATURE_COLUMNS)
    assert np.array_equal(model_scores(pipe, X), model_scores(pipe, shuffled))


@pytest.mark.parametrize("extra", ["Class", "Time", "row_id", "mystery_post_transaction_field"])
def test_forbidden_or_unknown_fields_rejected(extra):
    X, _ = _xy(10)
    X[extra] = 1.0
    with pytest.raises(DataValidationError, match="outside the feature allowlist"):
        prepare_scoring_frame(X)


def test_missing_feature_nonfinite_negative_amount_and_arrays_rejected():
    X, _ = _xy(10)
    with pytest.raises(DataValidationError, match="missing required"):
        prepare_scoring_frame(X.drop(columns=["V9"]))
    bad = X.copy()
    bad.loc[0, "V2"] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        prepare_scoring_frame(bad)
    bad = X.copy()
    bad.loc[0, "Amount"] = -1.0
    with pytest.raises(DataValidationError, match="negative Amount"):
        prepare_scoring_frame(bad)
    with pytest.raises(DataValidationError, match="DataFrame"):
        prepare_scoring_frame(X.to_numpy())


def test_classifier_configuration_matches_spec():
    clf = build_pipeline().named_steps["classifier"]
    assert (clf.C, clf.solver, clf.max_iter, clf.class_weight) == (1.0, "lbfgs", 2000, None)
    assert type(build_pipeline().named_steps["scaler"]).__name__ == "StandardScaler"


def test_scaler_and_classifier_fit_on_training_rows_only():
    X, y = _xy(300)
    assign = np.array(["train"] * 180 + ["validation"] * 60 + ["test"] * 60, dtype=object)
    held = X.copy()
    held.loc[assign != "train", :] = held.loc[assign != "train", :] + 1000.0  # absurdly different held-out values
    pipe, base, info = fit_on_training_rows(held, y, assign)
    scaler = pipe.named_steps["scaler"]
    train_X = held.loc[assign == "train", list(C.FEATURE_COLUMNS)]
    assert np.array_equal(scaler.mean_, train_X.to_numpy().mean(axis=0))
    assert not np.allclose(scaler.mean_, held[list(C.FEATURE_COLUMNS)].to_numpy().mean(axis=0))
    assert info["n_train_rows"] == 180
    assert base.prevalence == y[assign == "train"].mean()
    # the same fit is obtained if held-out rows are removed entirely
    pipe2, _, _ = fit_on_training_rows(held.loc[assign == "train"], y[assign == "train"],
                                       assign[assign == "train"])
    assert np.array_equal(pipe.named_steps["classifier"].coef_, pipe2.named_steps["classifier"].coef_)


def test_non_convergence_is_an_error_not_a_warning():
    X, y = _xy(200)
    with pytest.raises(TrainingNotConvergedError):
        fit_on_training_rows(X, y, np.array(["train"] * len(X), dtype=object), max_iter=1)


def test_constant_baseline_assigns_training_prevalence_to_every_row():
    b = ConstantScoreBaseline.fit(np.array([0, 0, 0, 1]))
    assert b.prevalence == 0.25
    s = b.score(7)
    assert s.shape == (7,) and np.all(s == 0.25)


def test_predictions_identical_after_serialization(tmp_path):
    X, y = _xy(300)
    pipe, _, _ = fit_on_training_rows(X, y, np.array(["train"] * len(X), dtype=object))
    before = model_scores(pipe, X)
    p = tmp_path / "m.joblib"
    joblib.dump(pipe, p)
    after = model_scores(joblib.load(p), X)  # our own artifact written in this test
    assert np.array_equal(before, after)  # bitwise identical


def test_threshold_float_roundtrips_exactly_via_json_and_hex():
    import json

    t = 0.1 + 0.2  # 0.30000000000000004
    assert json.loads(json.dumps(t)) == t
    assert float.fromhex(t.hex()) == t
