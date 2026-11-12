"""Loading ARFF/CSV transaction files and assigning stable row identifiers.

The loader never drops columns. In particular `Time` is kept even though OpenML
labels it as the row-identifier attribute: chronological partitioning needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import constants as C
from .errors import DataValidationError
from .provenance import sha256_file


@dataclass(frozen=True)
class LoadedData:
    frame: pd.DataFrame          # columns: SOURCE_COLUMNS; Class kept as raw strings until validated
    row_ids: list[str]           # parallel to frame rows; NOT part of the feature matrix
    source_path: str
    source_sha256: str
    source_bytes: int
    source_format: str


def make_row_ids(source_sha256: str, n_rows: int) -> list[str]:
    """Stable id = first 12 hex of the source file SHA-256 + 0-based original row number."""
    prefix = source_sha256[:12]
    return [f"{prefix}-{i:07d}" for i in range(n_rows)]


def _parse_arff_header(path: Path) -> tuple[list[tuple[str, str]], int]:
    attrs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for lineno, line in enumerate(f):
            s = line.strip()
            low = s.lower()
            if low.startswith("@attribute"):
                parts = s.split(None, 2)
                if len(parts) != 3:
                    raise DataValidationError(f"malformed @attribute line {lineno + 1}: {s!r}")
                attrs.append((parts[1], parts[2].strip()))
            elif low.startswith("@data"):
                return attrs, lineno + 1
            if lineno > 10_000:
                break
    raise DataValidationError("ARFF header: no @data section found")


def load_dataset(path: str | Path) -> LoadedData:
    """Load an ARFF (the pinned OpenML file) or a CSV (synthetic fixture)."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"data file not found: {path}")
    sha = sha256_file(path)
    size = path.stat().st_size
    suffix = path.suffix.lower()

    if suffix == ".arff":
        attrs, skip = _parse_arff_header(path)
        names = [a[0] for a in attrs]
        if names != list(C.SOURCE_COLUMNS):
            raise DataValidationError(
                "ARFF attributes do not match the documented schema.\n"
                f"  expected: {list(C.SOURCE_COLUMNS)}\n  found:    {names}"
            )
        types = dict(attrs)
        for col in (C.TIME_COLUMN, *C.PCA_COLUMNS, C.AMOUNT_COLUMN):
            if types[col].lower() != "numeric":
                raise DataValidationError(f"ARFF attribute {col} has type {types[col]!r}, expected numeric")
        if types[C.LABEL_COLUMN].replace(" ", "") != "{'0','1'}":
            raise DataValidationError(f"ARFF Class declaration unexpected: {types[C.LABEL_COLUMN]!r}")
        dtypes = {c: "float64" for c in C.SOURCE_COLUMNS if c != C.LABEL_COLUMN}
        dtypes[C.LABEL_COLUMN] = "str"
        df = pd.read_csv(
            path,
            skiprows=skip,
            header=None,
            names=list(C.SOURCE_COLUMNS),
            quotechar="'",
            na_values=["?"],
            dtype=dtypes,
        )
        fmt = "arff"
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype="float64", na_values=[""])
        # Class may arrive as integer-valued floats; keep a raw string form for uniform validation.
        if C.LABEL_COLUMN in df.columns:
            df[C.LABEL_COLUMN] = df[C.LABEL_COLUMN].map(
                lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else repr(float(v)))
            )
        fmt = "csv"
    else:
        raise DataValidationError(f"unsupported data file type: {suffix!r}")

    df = df.reset_index(drop=True)
    return LoadedData(
        frame=df,
        row_ids=make_row_ids(sha, len(df)),
        source_path=str(path),
        source_sha256=sha,
        source_bytes=size,
        source_format=fmt,
    )


def parse_fidelity_check(path: str | Path, frame: pd.DataFrame, k: int = 2000) -> dict:
    """Re-parse the first and last k data lines with Python float() and compare exactly.

    Independent of pandas' C float parser; confirms the loaded values are the
    values written in the file for the sampled rows.
    """
    path = Path(path)
    if path.suffix.lower() != ".arff":
        return {"performed": False, "reason": "not an ARFF file"}
    _, skip = _parse_arff_header(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()[skip:]
    lines = [ln for ln in lines if ln.strip()]
    n = len(lines)
    idx = list(range(min(k, n))) + list(range(max(k, n - k), n))
    mismatches = 0
    for i in idx:
        toks = lines[i].split(",")
        if len(toks) != len(C.SOURCE_COLUMNS):
            mismatches += 1
            continue
        bad = False
        for col, tok in zip(C.SOURCE_COLUMNS[:-1], toks[:-1]):
            if float(tok) != float(frame.at[i, col]):
                bad = True
                break
        if toks[-1].strip("'") != str(frame.at[i, C.LABEL_COLUMN]):
            bad = True
        mismatches += int(bad)
    return {
        "performed": True,
        "rows_compared": len(idx),
        "data_lines_in_file": n,
        "rows_in_frame": int(len(frame)),
        "mismatching_rows": mismatches,
    }
