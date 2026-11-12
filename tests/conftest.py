from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_transactions.csv"


def make_frame(n: int = 40, seed: int = 0, fraud_every: int = 4) -> pd.DataFrame:
    """Small valid frame in source column order. Time = 5*i+1 (so it is not the row index)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=list(C.PCA_COLUMNS))
    df.insert(0, C.TIME_COLUMN, 5.0 * np.arange(n) + 1.0)
    df[C.AMOUNT_COLUMN] = np.round(np.abs(rng.normal(50, 20, size=n)), 2)
    df[C.LABEL_COLUMN] = ((np.arange(n) % fraud_every) == (fraud_every - 1)).astype(int)
    return df[list(C.SOURCE_COLUMNS)]


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def fixture_path():
    return FIXTURE


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory):
    """Full SYNTHETIC offline pipeline once per test session: data-check, split, train, evaluate."""
    from fraudrisk.cli import main

    out = tmp_path_factory.mktemp("synthetic_session_run")
    for cmd in ("data-check", "split", "train", "evaluate"):
        rc = main([cmd, "--data", str(FIXTURE), "--data-kind", "synthetic", "--out", str(out)])
        assert rc == 0, cmd
    return out
