"""Generate the small deterministic SYNTHETIC fixture tests/fixtures/synthetic_transactions.csv.

Seeded; values rounded to 6 decimals so the committed CSV is the source of truth (the CSV, not this
script, is what tests and CI read). Frauds are a shifted Gaussian blob so the pipeline has something to
learn. This says nothing about real fraud data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudrisk import constants as C  # noqa: E402

N = 1500
SEED = 20260101


def main() -> None:
    rng = np.random.default_rng(SEED)
    gaps = rng.integers(0, 4, size=N)               # 0 => tied timestamps on purpose
    time = np.cumsum(gaps).astype("float64")
    y = (rng.random(N) < 0.05).astype(int)
    X = rng.normal(size=(N, 28))
    shift = np.zeros(28)
    shift[:6] = [1.6, -1.4, 1.2, -1.0, 0.8, 0.6]
    X[y == 1] += shift
    amount = np.round(np.exp(rng.normal(3.0, 1.0, size=N)), 2)
    df = pd.DataFrame(np.round(X, 6), columns=list(C.PCA_COLUMNS))
    df.insert(0, C.TIME_COLUMN, time)
    df[C.AMOUNT_COLUMN] = amount
    df[C.LABEL_COLUMN] = y
    df = df[list(C.SOURCE_COLUMNS)]
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_transactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    n = len(df)
    b1, b2 = 60 * n // 100, 80 * n // 100
    print(f"wrote {out} rows={n} fraud={int(y.sum())} per-sorted-third fraud:",
          int(y[:b1].sum()), int(y[b1:b2].sum()), int(y[b2:].sum()))


if __name__ == "__main__":
    main()
