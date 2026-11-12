"""Data access and arithmetic for the Streamlit dashboard (pure functions, unit-tested).

The dashboard reads only AGGREGATE saved artifacts: the validation threshold-candidate table (counts per
threshold), `frozen_config.json`, `threshold_selection_validation.json`, `test_metrics.json`. It never reads
row-level scores, labels or features, so real row-level data stays private. Exploring a validation threshold
here reads the candidate table only; it cannot reach the frozen threshold or the test results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from . import constants as C
from .provenance import read_json, sha256_file
from .strict_json import StrictJSONError, parse_strict_json

EXAMPLE_NAME = "Constructed example (not a real transaction)"
EXAMPLE_DESCRIPTION = (
    "Every anonymised component V1..V28 is set to 0.0 and Amount to 25.00. The vector was written by hand to show the "
    "request format. It is not a row of the dataset and not a 'typical' transaction: the training mean of each "
    "component is only approximately 0, and nothing is known about what any component means."
)


def example_features() -> dict:
    ex = {name: 0.0 for name in C.PCA_COLUMNS}
    ex[C.AMOUNT_COLUMN] = 25.0
    return ex


@dataclass(frozen=True)
class DashboardData:
    artifact_dir: Path
    frozen: dict
    frozen_sha256: str
    selection: dict
    candidates: pd.DataFrame       # highest threshold first; columns threshold,tp,fp,fn,tn,f1,...
    candidates_baseline: pd.DataFrame
    test: dict
    data_kind: str
    notice: str | None
    frozen_index: int              # row of `candidates` that holds the frozen threshold
    consistent: bool               # test_metrics.json was produced against this frozen_config.json


def load_dashboard_data(artifact_dir: str | Path) -> DashboardData:
    d = Path(artifact_dir)
    frozen = read_json(d / "frozen_config.json")
    cand = pd.read_csv(d / "threshold_candidates_model.csv.gz")
    base = pd.read_csv(d / "threshold_candidates_baseline.csv.gz")
    test = read_json(d / "test_metrics.json")
    sel = read_json(d / "threshold_selection_validation.json")
    thr = float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
    hits = np.flatnonzero(cand["threshold"].to_numpy() == thr)
    if len(hits) != 1:
        raise ValueError("the frozen threshold is not exactly one row of the validation candidate table")
    fsha = sha256_file(d / "frozen_config.json")
    return DashboardData(d, frozen, fsha, sel, cand, base, test, frozen.get("data_kind", "unknown"),
                         frozen.get("notice"), int(hits[0]), test.get("frozen_config_sha256") == fsha)


def row_metrics(cand: pd.DataFrame, i: int) -> dict:
    """Validation confusion counts and rates at candidate row i (rule: flag when score >= threshold)."""
    r = cand.iloc[int(i)]
    tp, fp, fn, tn = (int(r[k]) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    flagged = tp + fp
    return {
        "threshold": float(r["threshold"]), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "flagged": flagged,
        "review_fraction": flagged / n,
        "precision": tp / flagged if flagged else None,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def pr_curve(cand: pd.DataFrame, max_points: int = 1500, keep: tuple[int, ...] = ()) -> pd.DataFrame:
    """Precision-recall points from the candidate table, thinned for plotting (always keeps `keep` rows)."""
    n = len(cand)
    idx = np.unique(np.concatenate([np.linspace(0, n - 1, min(n, max_points)).astype(int), np.asarray(keep, dtype=int)]))
    sub = cand.iloc[idx]
    flagged = sub["tp"] + sub["fp"]
    return pd.DataFrame({
        "threshold": sub["threshold"].to_numpy(), "recall": (sub["tp"] / (sub["tp"] + sub["fn"])).to_numpy(),
        "precision": (sub["tp"] / flagged).to_numpy(), "review_fraction": (flagged / (flagged + sub["fn"] + sub["tn"])).to_numpy(),
    })


def call_score_api(base_url: str, features: dict, timeout: float = 10.0) -> tuple[int, dict]:
    """POST one transaction to the local API; returns (status, parsed JSON)."""
    r = httpx.post(base_url.rstrip("/") + "/score", json={"features": features}, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:300]}
    return r.status_code, body


def submit_payload(base_url: str, payload_text: str) -> tuple[int | None, dict]:
    """Strictly parse the text the user typed, then (only if it is acceptable) send it to the API.

    Duplicate keys, NaN/Infinity, invalid JSON and over-long input are rejected HERE, before any request is made,
    so a duplicated key can never be silently collapsed to its last value and submitted as a valid transaction.
    Returns (status, body); status is None when nothing was submitted.
    """
    try:
        features = parse_strict_json(payload_text)
    except StrictJSONError as e:
        return None, {"error": f"not submitted: {e}", "submitted": False}
    try:
        return call_score_api(base_url, features)
    except httpx.HTTPError as e:
        return None, {"error": f"{type(e).__name__}: could not reach the API at {base_url}", "submitted": False}
