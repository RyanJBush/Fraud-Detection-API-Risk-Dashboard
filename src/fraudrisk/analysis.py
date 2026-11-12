"""Error-analysis examples. Reports observed values only; assigns no business meaning to PCA components."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C
from .thresholds import review_flags


def _example(frame: pd.DataFrame, pos: int, row_id: str, y: int, score: float, threshold: float) -> dict:
    return {
        "row_id": row_id,
        "label": int(y),
        "model_score": float(score),
        "model_score_hex": float(score).hex(),
        "threshold": float(threshold),
        "score_minus_threshold": float(score) - float(threshold),
        "Time": float(frame.iloc[pos][C.TIME_COLUMN]),
        "feature_values": {c: float(frame.iloc[pos][c]) for c in C.FEATURE_COLUMNS},
    }


def earliest_errors(
    frame: pd.DataFrame, row_ids: list[str], row_numbers: np.ndarray, y: np.ndarray,
    scores: np.ndarray, threshold: float,
) -> dict:
    """Earliest false positive and false negative by (Time, original row number) in `frame`."""
    y = np.asarray(y)
    flags = review_flags(scores, threshold)
    time = frame[C.TIME_COLUMN].to_numpy(dtype="float64")
    order = np.lexsort((np.asarray(row_numbers), time))
    out = {}
    for name, mask in (("earliest_false_positive", flags & (y == 0)), ("earliest_false_negative", ~flags & (y == 1))):
        hits = [i for i in order if mask[i]]
        out[name] = _example(frame, hits[0], row_ids[hits[0]], y[hits[0]], scores[hits[0]], threshold) if hits else None
    out["note"] = (
        "Feature values are the anonymised V1..V28 components and Amount as recorded in the source file. "
        "No business meaning is assigned to any component."
    )
    return out
