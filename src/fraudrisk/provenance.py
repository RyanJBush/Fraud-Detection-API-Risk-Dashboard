"""Hashing, JSON writing and environment/provenance capture."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(obj: Any) -> Any:
    """Make an object strictly JSON-serialisable (no NaN/Inf, numpy scalars to python)."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float cannot be written to JSON: {obj!r}")
        return obj
    return obj


def canonical_json(obj: Any) -> str:
    """Canonical text form used to compare structures that went through JSON."""
    return json.dumps(_clean(obj), sort_keys=True, allow_nan=False)


def write_json(path: str | Path, obj: Any) -> str:
    """Write deterministic JSON (sorted keys, 2-space indent, LF) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv_gz(df, path: str | Path) -> str:
    """Write a gzip CSV with a fixed gzip mtime so bytes are reproducible; return SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return sha256_file(path)


def dependency_versions() -> dict[str, str]:
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def code_hashes() -> dict[str, str]:
    """SHA-256 of every source file in the package (stand-in for a commit hash)."""
    pkg = Path(__file__).resolve().parent
    return {p.name: sha256_file(p) for p in sorted(pkg.glob("*.py"))}
