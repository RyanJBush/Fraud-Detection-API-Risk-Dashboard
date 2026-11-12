"""Scoring-feature validation, constant-score baseline, and the logistic-regression pipeline."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import constants as C
from .errors import DataValidationError, ScoringNumericalError, TrainingNotConvergedError


def prepare_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The one place scoring inputs are validated and ordered (used for fit, validation and test).

    * must be a named-column DataFrame (no bare arrays)
    * columns must be EXACTLY the allowlist V1..V28 + Amount: Class, Time, row ids and any
      other field are rejected, not silently dropped
    * columns are put into the saved explicit order by name
    * values must be finite; Amount must be nonnegative
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("scoring input must be a pandas DataFrame with named columns")
    cols = list(df.columns)
    if len(set(cols)) != len(cols):
        raise DataValidationError("scoring input has duplicate column names")
    extra = [c for c in cols if c not in C.FEATURE_COLUMNS]
    if extra:
        forbidden = [c for c in extra if c in C.KNOWN_FORBIDDEN_SCORING_FIELDS]
        raise DataValidationError(
            f"scoring input contains fields outside the feature allowlist: {extra}"
            + (f" (forbidden outcome/partition/identifier fields: {forbidden})" if forbidden else "")
        )
    missing = [c for c in C.FEATURE_COLUMNS if c not in cols]
    if missing:
        raise DataValidationError(f"scoring input is missing required features: {missing}")
    data, problems = {}, []
    for name in C.FEATURE_COLUMNS:
        series = df[name]
        if isinstance(series.dtype, np.dtype) and series.dtype.kind in "iuf":  # plain numpy numeric column
            data[name] = series.to_numpy(dtype="float64")
            if not np.all(np.isfinite(data[name])):
                problems.append(f"{name}: missing or non-finite value")
            continue
        vals = []  # object / bool / string / extension dtypes: every cell is checked individually
        for v in series.tolist():
            try:
                vals.append(coerce_feature_value(v, name))
            except DataValidationError as e:
                problems.append(str(e))
                break
        else:
            data[name] = np.asarray(vals, dtype="float64")
    if problems:
        raise DataValidationError("invalid scoring input: " + "; ".join(problems[:10]))
    X = pd.DataFrame(data, index=df.index, columns=list(C.FEATURE_COLUMNS))
    if (X[C.AMOUNT_COLUMN] < 0).any():
        raise DataValidationError("invalid scoring input: negative Amount")
    return X


def coerce_feature_value(value, name: str) -> float:
    """One scoring value -> finite float. Accepts ordinary ints and floats only.

    Rejected with a controlled DataValidationError: booleans, numeric strings, bytes, None/NA, NaN/inf, any
    other type, and integers too large for a finite float64 (OverflowError is converted, never leaked).
    """
    if value is None or value is pd.NA or value is pd.NaT:
        raise DataValidationError(f"{name}: null is not allowed")
    if isinstance(value, (bool, np.bool_)):
        raise DataValidationError(f"{name}: boolean is not a number")
    if isinstance(value, (str, bytes)):
        raise DataValidationError(f"{name}: strings are not accepted, even numeric ones")
    if isinstance(value, (int, np.integer)):
        try:
            f = float(value)
        except OverflowError:
            raise DataValidationError(f"{name}: integer is too large for a finite float64") from None
    elif isinstance(value, (float, np.floating)):
        f = float(value)
    else:
        raise DataValidationError(f"{name}: unsupported type {type(value).__name__}")
    if not math.isfinite(f):
        raise DataValidationError(f"{name}: value is not finite")
    return f


def features_to_frame(mapping) -> pd.DataFrame:
    """Single-row scoring frame from a name->value mapping (the API path). Same rules as offline scoring."""
    if not isinstance(mapping, dict):
        raise DataValidationError("features must be an object mapping feature names to numbers")
    frame = pd.DataFrame({k: [v] for k, v in mapping.items()}, dtype=object)
    return prepare_scoring_frame(frame)


@dataclass(frozen=True)
class ConstantScoreBaseline:
    """Every row gets the training partition's fraud prevalence as its score."""

    prevalence: float

    @classmethod
    def fit(cls, y_train: np.ndarray) -> "ConstantScoreBaseline":
        y = np.asarray(y_train)
        if len(y) == 0:
            raise DataValidationError("cannot fit baseline on empty training labels")
        return cls(prevalence=float(y.sum()) / float(len(y)))

    def score(self, n_rows: int) -> np.ndarray:
        return np.full(int(n_rows), self.prevalence, dtype="float64")


def build_pipeline(max_iter: int | None = None) -> Pipeline:
    cfg = C.MODEL_CONFIG
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=cfg["C"],
                    solver=cfg["solver"],
                    max_iter=cfg["max_iter"] if max_iter is None else max_iter,
                    class_weight=cfg["class_weight"],
                ),
            ),
        ]
    )


def fit_on_training_rows(
    X_all: pd.DataFrame, y_all: np.ndarray, assignment: np.ndarray, max_iter: int | None = None
) -> tuple[Pipeline, ConstantScoreBaseline, dict]:
    """Fit scaler + classifier + baseline on rows assigned to 'train' ONLY.

    X_all/y_all may contain other partitions; they are sliced away here and never seen by `fit`.
    """
    X_all = prepare_scoring_frame(X_all)
    mask = np.asarray(assignment) == "train"
    if not mask.any():
        raise DataValidationError("no training rows")
    X_tr = X_all.loc[mask]
    y_tr = np.asarray(y_all)[mask]
    pipe = build_pipeline(max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X_tr, y_tr)
    conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    clf = pipe.named_steps["classifier"]
    n_iter = int(np.max(clf.n_iter_))
    if conv:
        raise TrainingNotConvergedError(
            f"LogisticRegression did not converge ({conv[0].message}); n_iter={n_iter}, max_iter={clf.max_iter}"
        )
    if n_iter >= clf.max_iter:
        raise TrainingNotConvergedError(f"n_iter={n_iter} reached max_iter={clf.max_iter}; treating as unresolved")
    if list(clf.classes_) != list(C.LABELS):
        raise DataValidationError(f"training labels must contain both classes; got classes_={list(clf.classes_)}")
    baseline = ConstantScoreBaseline.fit(y_tr)
    info = {
        "n_train_rows": int(mask.sum()),
        "n_train_fraud": int(y_tr.sum()),
        "n_iter": n_iter,
        "converged": True,
        "other_fit_warnings": [str(w.message) for w in caught if not issubclass(w.category, ConvergenceWarning)],
        "baseline_prevalence": baseline.prevalence,
        "scaler_mean_first3": [float(v) for v in pipe.named_steps["scaler"].mean_[:3]],
    }
    return pipe, baseline, info


def model_scores(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Model score = predicted probability of Class 1, full float64 precision. Not calibrated."""
    X = prepare_scoring_frame(df)
    clf = pipe.named_steps["classifier"]
    if list(clf.classes_) != list(C.LABELS):
        raise DataValidationError("unexpected classes_ in fitted classifier")
    scaler = pipe.named_steps["scaler"]
    try:
        # Same two stages the Pipeline runs (transform, then the classifier), made explicit so numerical failure in
        # either stage is detected here. The errstate is local to these calls; no global warning filter is touched.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            Z = scaler.transform(X)
            z_ok = bool(np.all(np.isfinite(Z)))
            decision = clf.decision_function(Z) if z_ok else None
    except FloatingPointError:
        raise ScoringNumericalError(
            "feature values are finite but too extreme to score: standardisation or the linear score overflowed"
        ) from None
    if not z_ok or not np.all(np.isfinite(decision)):
        raise ScoringNumericalError(
            "feature values are finite but too extreme to score: standardisation or the linear score is not finite"
        )
    p = clf.predict_proba(Z)[:, 1]
    if not (np.all(np.isfinite(p)) and np.all((p >= 0.0) & (p <= 1.0))):
        raise ScoringNumericalError("score is not a finite number in [0, 1]")
    return p
