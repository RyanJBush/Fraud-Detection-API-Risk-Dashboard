"""Local scoring service (FastAPI). Scores one transaction with the FROZEN offline pipeline.

Scope: local only. No model upload, no retraining, no threshold changes, no deployment. The artifact directory
is chosen by the application operator (``create_app(artifact_dir=...)`` or the ``FRAUDRISK_ARTIFACT_DIR``
environment variable read once at startup); no request can influence it.

Status codes for POST /score
  200  scored
  400  the body could not be read as strict JSON: too large (> 16 KiB), not UTF-8, invalid JSON, duplicate keys
       (at any level), NaN / Infinity / -Infinity literals, integers with more than 4300 digits, nesting too deep
  422  well-formed JSON that is not a valid scoring request (including finite values too extreme to score:
       standardisation or the linear score overflows, code numerical_overflow): envelope is not exactly {"features": {...}};
       features are not exactly V1..V28 + Amount (missing, extra, Class, Time, ...); a value is a string, boolean,
       null, non-number, non-finite (e.g. 1e999) or too large for float64; Amount is negative
  Client input never produces a 500.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from . import constants as C
from .errors import ArtifactIntegrityError, DataValidationError, ScoringNumericalError
from .modeling import features_to_frame, model_scores
from .provenance import canonical_json, read_json, sha256_bytes, sha256_file
from .strict_json import MAX_BODY_BYTES, StrictJSONError, parse_strict_json
from .thresholds import review_flags

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "real"
MODEL_FILE = "model_pipeline.joblib"
SCORE_NOTE = "Model score from a logistic regression; it is not a calibrated probability (calibration was not evaluated)."


# --------------------------------------------------------------------------- trusted artifact bundle
@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    threshold: float
    threshold_hex: str
    model_version: str
    threshold_version: str
    model_sha256: str
    frozen_config_sha256: str
    frozen: dict
    artifact_dir: Path


def load_bundle(artifact_dir: str | Path, expected_frozen_sha256: str | None = None) -> ModelBundle:
    """Load the frozen pipeline only if every checksum matches frozen_config.json. Raises ArtifactIntegrityError."""
    d = Path(artifact_dir)
    frozen_path = d / "frozen_config.json"
    if not frozen_path.is_file():
        raise ArtifactIntegrityError(f"frozen_config.json not found in {d}")
    frozen_sha = sha256_file(frozen_path)
    if expected_frozen_sha256 is not None and frozen_sha != expected_frozen_sha256:
        raise ArtifactIntegrityError("frozen_config.json does not match the pinned SHA-256")
    frozen = read_json(frozen_path)
    if frozen.get("model_file") != MODEL_FILE:
        raise ArtifactIntegrityError(f"frozen model_file must be {MODEL_FILE!r}")  # no path from config
    if not (frozen.get("frozen_before_test_scoring") or frozen.get("frozen_before_test_evaluation")):
        raise ArtifactIntegrityError("frozen_config.json does not declare the model frozen before test scoring")
    if list(frozen.get("feature_order", [])) != list(C.FEATURE_COLUMNS):
        raise ArtifactIntegrityError("frozen feature order differs from the code's allowlist")
    model_path, cfg_path = d / MODEL_FILE, d / "model_config.json"
    if not model_path.is_file() or not cfg_path.is_file():
        raise ArtifactIntegrityError("model_pipeline.joblib or model_config.json is missing")
    if sha256_file(model_path) != frozen["model_sha256"]:
        raise ArtifactIntegrityError("model_pipeline.joblib SHA-256 does not match frozen_config.json")
    if sha256_file(cfg_path) != frozen["model_config_sha256"]:
        raise ArtifactIntegrityError("model_config.json SHA-256 does not match frozen_config.json")
    t = frozen["thresholds"]["model"]
    thr = float(t["threshold"])
    if float.fromhex(t["threshold_hex"]) != thr:
        raise ArtifactIntegrityError("frozen threshold decimal and hex forms disagree")
    # The checksum matched, so this is the artifact `train` wrote; pickle is only loaded after that check.
    pipe = joblib.load(model_path)
    try:
        ok = (list(pipe.feature_names_in_) == list(C.FEATURE_COLUMNS) and list(pipe.named_steps["classifier"].classes_) == [0, 1])
    except Exception:  # noqa: BLE001 - any shape surprise means the artifact is not what was frozen
        ok = False
    if not ok:
        raise ArtifactIntegrityError("loaded pipeline does not expose the frozen feature names / classes")
    model_version = "lr-" + frozen["model_sha256"][:12]
    thr_doc = {"threshold_hex": t["threshold_hex"], "decision_rule": C.THRESHOLD_RULE, "model_sha256": frozen["model_sha256"]}
    threshold_version = "thr-" + sha256_bytes(canonical_json(thr_doc).encode())[:12]
    return ModelBundle(pipe, thr, t["threshold_hex"], model_version, threshold_version, frozen["model_sha256"],
                       frozen_sha, frozen, d)


# --------------------------------------------------------------------------- request parsing
def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def request_schema() -> dict:
    feat = {n: {"type": "number"} for n in C.FEATURE_COLUMNS}
    feat[C.AMOUNT_COLUMN] = {"type": "number", "minimum": 0}
    return {"type": "object", "additionalProperties": False, "required": ["features"], "properties": {
        "features": {"type": "object", "additionalProperties": False, "required": list(C.FEATURE_COLUMNS),
                     "properties": feat,
                     "description": "Exactly V1..V28 (anonymised PCA components) and Amount. JSON numbers only."}}}


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    model_score: float
    threshold: float
    threshold_version: str
    review_flag: bool
    score_note: str = SCORE_NOTE


# --------------------------------------------------------------------------- application
def create_app(artifact_dir: str | Path | None = None, expected_frozen_sha256: str | None = None) -> FastAPI:
    chosen = Path(artifact_dir or os.environ.get("FRAUDRISK_ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_bundle(chosen, expected_frozen_sha256)  # raises -> the service does not start
        yield

    app = FastAPI(title="Fraud risk scoring API (local, offline-trained model)", version="0.1.0", lifespan=lifespan,
                  description="Scores one transaction with the frozen logistic-regression pipeline. " + SCORE_NOTE)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):  # server fault only; message stays generic
        return _err(500, "internal_error", "unexpected server error")

    @app.get("/health")
    async def health(request: Request):
        b: ModelBundle = request.app.state.bundle
        return {"status": "ok", "model_version": b.model_version, "threshold_version": b.threshold_version}

    @app.get("/model")
    async def model_info(request: Request):
        b: ModelBundle = request.app.state.bundle
        return {
            "model_version": b.model_version, "threshold_version": b.threshold_version,
            "model_sha256": b.model_sha256, "frozen_config_sha256": b.frozen_config_sha256,
            "feature_order": list(C.FEATURE_COLUMNS), "decision_rule": C.THRESHOLD_RULE,
            "threshold": b.threshold, "threshold_hex": b.threshold_hex,
            "data_kind": b.frozen.get("data_kind"), "notice": b.frozen.get("notice"),
            "score_note": SCORE_NOTE,
            "threshold_note": "Frozen from the validation partition only; this service cannot change it.",
        }

    @app.post("/score", response_model=ScoreResponse,
              responses={400: {"description": "Body is not strict JSON, too large, or has duplicate keys / NaN / Infinity"},
                         422: {"description": "Valid JSON but not a valid scoring request"}},
              openapi_extra={"requestBody": {"required": True, "content": {"application/json": {
                  "schema": request_schema()}}}})
    async def score(request: Request):
        b: ModelBundle = request.app.state.bundle
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return _err(400, "request_too_large", f"request body larger than {MAX_BODY_BYTES} bytes")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return _err(400, "request_too_large", f"request body larger than {MAX_BODY_BYTES} bytes")
        try:
            doc = parse_strict_json(body)
        except StrictJSONError as e:
            return _err(400, "malformed_json", str(e))
        if not isinstance(doc, dict):
            return _err(422, "invalid_request", 'request must be a JSON object of the form {"features": {...}}')
        extra = sorted(set(doc) - {"features"})
        if extra:
            return _err(422, "invalid_request", f"unexpected top-level fields: {extra}")
        if "features" not in doc:
            return _err(422, "invalid_request", 'missing required field "features"')
        try:
            frame = features_to_frame(doc["features"])
            score_value = float(model_scores(b.pipeline, frame)[0])
        except ScoringNumericalError as e:  # finite inputs whose standardised/linear score overflows
            return _err(422, "numerical_overflow", str(e))
        except DataValidationError as e:
            return _err(422, "invalid_features", str(e))
        flag = bool(review_flags(np.array([score_value]), b.threshold)[0])  # exact full-precision >=
        return ScoreResponse(model_version=b.model_version, model_score=score_value, threshold=b.threshold,
                             threshold_version=b.threshold_version, review_flag=flag)

    return app
