"""Strict JSON parsing shared by the scoring API and the dashboard.

Python's ``json.loads`` silently keeps the LAST value of a duplicated key and accepts ``NaN``/``Infinity``. For a
scoring request that is dangerous (the value that was reviewed is not the value that is scored), so both entry points
use this parser: duplicate keys at any depth, non-standard constants, invalid UTF-8/JSON, over-long digit strings,
excessive nesting and oversized text are all rejected with ``StrictJSONError``.
"""
from __future__ import annotations

import json
from typing import Any

MAX_BODY_BYTES = 16 * 1024


class StrictJSONError(ValueError):
    """The text is not acceptable strict JSON. The message is safe to show to a user."""


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise StrictJSONError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _reject_constant(name):
    raise StrictJSONError(f"non-standard JSON constant {name}")


def parse_strict_json(data: bytes | str, max_bytes: int = MAX_BODY_BYTES) -> Any:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > max_bytes:
        raise StrictJSONError(f"request body larger than {max_bytes} bytes")
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except StrictJSONError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:  # JSONDecodeError and the int-digit limit are ValueErrors
        raise StrictJSONError(f"invalid JSON: {type(e).__name__}") from None
