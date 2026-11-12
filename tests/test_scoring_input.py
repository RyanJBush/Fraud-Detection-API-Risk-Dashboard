"""Shared scoring-input validation (used by offline scoring AND the API): strict numeric types only."""
import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C
from fraudrisk.errors import DataValidationError
from fraudrisk.modeling import (
    coerce_feature_value,
    features_to_frame,
    fit_on_training_rows,
    model_scores,
    prepare_scoring_frame,
)


def valid_mapping(seed=0):
    rng = np.random.default_rng(seed)
    m = {f"V{i}": float(rng.normal()) for i in range(1, 29)}
    m["Amount"] = float(abs(rng.normal(50, 10)))
    return m


@pytest.mark.parametrize("bad, needle", [
    ("0.0", "strings are not accepted"),
    ("1e3", "strings are not accepted"),
    (b"1", "strings are not accepted"),
    (True, "boolean"),
    (False, "boolean"),
    (np.bool_(True), "boolean"),
    (None, "null"),
    (pd.NA, "null"),
    (float("nan"), "not finite"),
    (float("inf"), "not finite"),
    (float("-inf"), "not finite"),
    (10**400, "too large"),
    (-(10**400), "too large"),
    ([1.0], "unsupported type"),
    ({"x": 1}, "unsupported type"),
    (1 + 2j, "unsupported type"),
])
def test_bad_scalar_rejected_with_controlled_error(bad, needle):
    with pytest.raises(DataValidationError, match=needle):
        coerce_feature_value(bad, "V1")


@pytest.mark.parametrize("bad", ["0.0", True, None, float("nan"), float("inf"), 10**400, [1.0]])
@pytest.mark.parametrize("feature", ["V1", "V28", "Amount"])
def test_bad_value_rejected_through_the_mapping_path(bad, feature):
    m = valid_mapping()
    m[feature] = bad
    with pytest.raises(DataValidationError, match="invalid scoring input"):
        features_to_frame(m)


def test_overflow_never_leaks_as_overflowerror():
    m = valid_mapping()
    m["V3"] = 10**400
    try:
        features_to_frame(m)
    except OverflowError:  # pragma: no cover
        pytest.fail("OverflowError leaked")
    except DataValidationError:
        pass


def test_negative_amount_missing_duplicate_and_unknown_fields_rejected():
    m = valid_mapping()
    with pytest.raises(DataValidationError, match="negative Amount"):
        features_to_frame({**m, "Amount": -0.01})
    with pytest.raises(DataValidationError, match="missing required"):
        features_to_frame({k: v for k, v in m.items() if k != "V9"})
    for extra in ("Class", "Time", "row_id", "mystery"):
        with pytest.raises(DataValidationError, match="outside the feature allowlist"):
            features_to_frame({**m, extra: 1.0})
    dup = pd.DataFrame([list(m.values()) + [1.0]], columns=list(m) + ["V1"])  # duplicate column name
    with pytest.raises(DataValidationError, match="duplicate"):
        prepare_scoring_frame(dup)
    with pytest.raises(DataValidationError, match="object"):
        features_to_frame([1, 2, 3])


def test_dataframe_paths_reject_same_cases():
    m = valid_mapping()
    base = pd.DataFrame([m])
    for col, bad in (("V2", True), ("V2", "0.0"), ("V2", None), ("V2", 10**400)):
        df = base.astype({c: object for c in base.columns})
        df.loc[0, col] = bad
        with pytest.raises(DataValidationError):
            prepare_scoring_frame(df)
    boolcol = base.copy()
    boolcol["V4"] = pd.Series([True])  # a real bool-dtype column
    with pytest.raises(DataValidationError, match="boolean"):
        prepare_scoring_frame(boolcol)
    strcol = base.copy()
    strcol["V4"] = pd.Series(["0.5"])
    with pytest.raises(DataValidationError, match="strings"):
        prepare_scoring_frame(strcol)
    na = base.copy()
    na["V4"] = pd.array([pd.NA], dtype="Float64")
    with pytest.raises(DataValidationError, match="null"):
        prepare_scoring_frame(na)
    nan = base.copy()
    nan.loc[0, "V4"] = np.nan
    with pytest.raises(DataValidationError, match="non-finite"):
        prepare_scoring_frame(nan)


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_ordinary_ints_and_floats_accepted_and_scores_identical():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 28)), columns=list(C.PCA_COLUMNS))
    X[C.AMOUNT_COLUMN] = np.abs(rng.normal(50, 10, 300))
    y = (rng.random(300) < 0.2).astype(int)
    X.loc[y == 1, "V1"] += 2
    pipe, _, _ = fit_on_training_rows(X, y, np.array(["train"] * 300, dtype=object))
    # reference: the fitted pipeline called directly on the raw array, bypassing all validation
    ref = pipe.predict_proba(X.to_numpy())[:, 1]
    assert np.array_equal(model_scores(pipe, X), ref)
    for i in (0, 7, 123, 299):
        row = {k: float(v) for k, v in X.iloc[i].items()}
        single = model_scores(pipe, features_to_frame(row))
        # A one-row call and a 300-row call may differ in the last bits (BLAS summation order), never more.
        assert np.allclose(single, ref[[i]], rtol=1e-13, atol=0.0)
        # ...but the same single-row input is bit-for-bit repeatable
        assert np.array_equal(single, model_scores(pipe, features_to_frame(row)))
    # integers, numpy scalars and float64 of the same value give bit-identical scores
    m = {k: 0 for k in C.FEATURE_COLUMNS}
    m["Amount"] = 50
    as_int = model_scores(pipe, features_to_frame(m))
    as_float = model_scores(pipe, features_to_frame({k: float(v) for k, v in m.items()}))
    as_np = model_scores(pipe, features_to_frame({k: np.float64(v) for k, v in m.items()}))
    as_npint = model_scores(pipe, features_to_frame({k: np.int64(v) for k, v in m.items()}))
    assert np.array_equal(as_int, as_float) and np.array_equal(as_int, as_np) and np.array_equal(as_int, as_npint)
    assert coerce_feature_value(2**70, "V1") == float(2**70)  # large but representable ints are fine
    assert coerce_feature_value(-3, "V1") == -3.0
    assert coerce_feature_value(0, "Amount") == 0.0
