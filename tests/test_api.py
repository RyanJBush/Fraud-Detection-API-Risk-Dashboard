"""Local scoring API tests. SYNTHETIC artifacts only; no network, no real data."""
import dataclasses
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudrisk import constants as C
from fraudrisk.api import MAX_BODY_BYTES, create_app, load_bundle
from fraudrisk.cli import main
from fraudrisk.errors import ArtifactIntegrityError
from fraudrisk.modeling import model_scores
from fraudrisk.provenance import sha256_file

from conftest import FIXTURE


@pytest.fixture(scope="module")
def client(synthetic_run):
    with TestClient(create_app(synthetic_run)) as c:
        yield c


def payload(row: dict) -> dict:
    return {"features": {k: row[k] for k in C.FEATURE_COLUMNS}}


def fixture_rows(n=12):
    df = pd.read_csv(FIXTURE)
    idx = np.linspace(0, len(df) - 1, n).astype(int)
    return df.iloc[idx]


# ------------------------------------------------------------------------------- agreement with offline
def test_api_scores_agree_with_offline_scoring_on_multiple_examples(client, synthetic_run):
    bundle = load_bundle(synthetic_run)
    df = fixture_rows(16)
    for _, r in df.iterrows():
        resp = client.post("/score", json=payload(r.to_dict()))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        offline_one = float(model_scores(bundle.pipeline, pd.DataFrame([r[list(C.FEATURE_COLUMNS)].to_dict()]))[0])
        assert body["model_score"] == offline_one                    # same code path, bit for bit
        assert body["threshold"] == bundle.threshold
        assert body["review_flag"] == (offline_one >= bundle.threshold)
        assert body["model_version"] == bundle.model_version and body["threshold_version"] == bundle.threshold_version
        assert "not a calibrated probability" in body["score_note"]
        assert set(body) == {"model_version", "model_score", "threshold", "threshold_version", "review_flag", "score_note"}


def test_api_scores_match_saved_batch_scores_within_float_noise(client, synthetic_run):
    """The batch scores saved by `evaluate` and one-row API scores differ at most in the last bits."""
    saved = pd.read_csv(synthetic_run / "test_scores.csv.gz")
    mem = pd.read_csv(synthetic_run / "split_membership.csv.gz").set_index("row_id")
    raw = pd.read_csv(FIXTURE)
    thr = load_bundle(synthetic_run).threshold
    for _, r in saved.sample(25, random_state=1).iterrows():
        feats = raw.iloc[int(mem.loc[r["row_id"], "row_number"])]
        body = client.post("/score", json=payload(feats.to_dict())).json()
        assert abs(body["model_score"] - r["model_score"]) <= 1e-13 * max(1.0, abs(r["model_score"]))
        if abs(r["model_score"] - thr) > 1e-9:  # away from the boundary the flags must agree exactly
            assert body["review_flag"] == bool(r["model_score"] >= thr)


def test_integers_and_floats_give_identical_scores(client):
    row = {k: 0 for k in C.FEATURE_COLUMNS}
    row["Amount"] = 50
    a = client.post("/score", json={"features": row}).json()["model_score"]
    b = client.post("/score", json={"features": {k: float(v) for k, v in row.items()}}).json()["model_score"]
    assert a == b


# ------------------------------------------------------------------------------- threshold boundary
def test_review_flag_uses_exact_inclusive_comparison(synthetic_run):
    app = create_app(synthetic_run)
    row = payload(fixture_rows(3).iloc[1].to_dict())
    with TestClient(app) as c:
        s = c.post("/score", json=row).json()["model_score"]
        base = app.state.bundle
        for thr, expected in ((s, True), (float(np.nextafter(s, np.inf)), False), (float(np.nextafter(s, -np.inf)), True)):
            app.state.bundle = dataclasses.replace(base, threshold=thr)
            body = c.post("/score", json=row).json()
            assert body["review_flag"] is expected, (thr, expected)
            assert body["threshold"] == thr and body["model_score"] == s


# ------------------------------------------------------------------------------- malformed requests
def good_features():
    return dict(payload(fixture_rows(2).iloc[0].to_dict())["features"])


def post_raw(client, body: bytes | str, ctype="application/json"):
    data = body.encode() if isinstance(body, str) else body
    return client.post("/score", content=data, headers={"content-type": ctype})


def with_value(name, literal: str) -> str:
    """JSON text of a valid request in which one feature's value is replaced by a raw JSON literal."""
    parts = []
    for k, v in good_features().items():
        parts.append(f'"{k}": {literal if k == name else json.dumps(v)}')
    return '{"features": {' + ", ".join(parts) + "}}"


@pytest.mark.parametrize("name", ["V1", "V17", "V28", "Amount"])
@pytest.mark.parametrize("literal, status", [
    ('"0.0"', 422), ('"abc"', 422), ("true", 422), ("false", 422), ("null", 422), ("[]", 422), ("{}", 422),
    ("1e999", 422), ("-1e999", 422), ("1" + "0" * 400, 422), ("-1" + "0" * 400, 422),
    ("NaN", 400), ("Infinity", 400), ("-Infinity", 400), ("1" + "0" * 5000, 400),
])
def test_bad_feature_values_are_controlled_4xx(client, name, literal, status):
    r = post_raw(client, with_value(name, literal))
    assert r.status_code == status, (r.status_code, r.text[:200])
    err = r.json()["error"]
    assert err["code"] and err["message"]


def test_negative_amount_rejected(client):
    assert post_raw(client, with_value("Amount", "-0.01")).status_code == 422
    assert post_raw(client, with_value("Amount", "0")).status_code == 200
    assert post_raw(client, with_value("Amount", "-0.0")).status_code == 200  # -0.0 equals 0, not negative


def test_missing_and_extra_fields_and_forbidden_outcome_fields(client):
    g = good_features()
    missing = {k: v for k, v in g.items() if k != "V9"}
    assert client.post("/score", json={"features": missing}).status_code == 422
    for extra in ("Class", "Time", "row_id", "label", "whatever"):
        r = client.post("/score", json={"features": {**g, extra: 1.0}})
        assert r.status_code == 422 and "outside the feature allowlist" in r.json()["error"]["message"], extra
    assert client.post("/score", json={"features": g, "Class": 0}).status_code == 422       # envelope extra
    assert client.post("/score", json={"features": g, "extra": 1}).status_code == 422
    assert client.post("/score", json={}).status_code == 422
    assert client.post("/score", json={"feature": g}).status_code == 422
    assert client.post("/score", json={"features": [1, 2]}).status_code == 422
    assert client.post("/score", json={"features": None}).status_code == 422
    assert client.post("/score", json=[g]).status_code == 422
    assert client.post("/score", json=3).status_code == 422
    assert client.post("/score", json="x").status_code == 422


@pytest.mark.parametrize("body, label", [
    ("", "empty"), ("{", "truncated"), ("not json", "garbage"), ("{'features': {}}", "single quotes"),
    ('{"features": {}} trailing', "trailing text"),
    (b"\xff\xfe\x00", "not utf-8"),
    ('{"features": {"V1": 1, "V1": 2}}', "duplicate feature key"),
    ('{"features": {}, "features": {}}', "duplicate envelope key"),
    ('{"features": {"V1": NaN}}', "NaN literal"),
    ("[" * 12000, "deep nesting within the size limit"),
    ("[" * 100000, "deep nesting"),
    ('{"features": ' + "[" * 100000, "deep nesting inside"),
])
def test_unreadable_bodies_are_400(client, body, label):
    r = post_raw(client, body)
    assert r.status_code == 400, (label, r.status_code, r.text[:200])
    assert r.json()["error"]["code"] in {"malformed_json", "request_too_large"}


def test_oversized_body_is_400(client):
    r = post_raw(client, '{"features": {"pad": "' + "x" * (MAX_BODY_BYTES + 10) + '"}}')
    assert r.status_code == 400 and r.json()["error"]["code"] == "request_too_large"


def test_wrong_method_and_content_type_do_not_500(client):
    assert client.get("/score").status_code == 405
    assert post_raw(client, json.dumps({"features": good_features()}), ctype="text/plain").status_code == 200
    assert client.post("/nonexistent").status_code == 404


def test_no_unexpected_500_for_a_battery_of_garbage(client):
    rng = np.random.default_rng(0)
    for i in range(60):
        n = int(rng.integers(0, 400))
        blob = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        assert client.post("/score", content=blob).status_code in (400, 422)
    for v in ("1e308", "1.7976931348623157e308", "-1.7976931348623157e308", "5e-324"):
        assert post_raw(client, with_value("V3", v)).status_code in (200, 422)


# ------------------------------------------------------------------------------- metadata and docs
def test_health_model_and_openapi(client, synthetic_run):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"].startswith("lr-") and h["threshold_version"].startswith("thr-")
    m = client.get("/model").json()
    assert m["feature_order"] == list(C.FEATURE_COLUMNS) and m["decision_rule"] == C.THRESHOLD_RULE
    assert m["data_kind"] == "synthetic" and m["notice"].startswith("SYNTHETIC FIXTURE")
    assert m["model_sha256"] == sha256_file(synthetic_run / "model_pipeline.joblib")
    spec = client.get("/openapi.json").json()
    post = spec["paths"]["/score"]["post"]
    assert {"200", "400", "422"} <= set(post["responses"])
    sch = post["requestBody"]["content"]["application/json"]["schema"]
    assert sch["additionalProperties"] is False and sch["properties"]["features"]["additionalProperties"] is False
    assert set(spec["paths"]) == {"/health", "/model", "/score"}  # no upload / retrain / threshold endpoints


def test_versions_are_derived_from_frozen_artifacts(synthetic_run):
    a, b = load_bundle(synthetic_run), load_bundle(synthetic_run)
    frozen = json.loads((synthetic_run / "frozen_config.json").read_text())
    assert a.model_version == b.model_version == "lr-" + frozen["model_sha256"][:12]
    assert a.threshold_version == b.threshold_version


# ------------------------------------------------------------------------------- artifact integrity
@pytest.fixture
def copy_of_run(synthetic_run, tmp_path):
    dst = tmp_path / "artifacts"
    shutil.copytree(synthetic_run, dst)
    return dst


def test_untouched_copy_starts(copy_of_run):
    with TestClient(create_app(copy_of_run)) as c:
        assert c.get("/health").status_code == 200


def test_tampered_model_prevents_startup(copy_of_run):
    with open(copy_of_run / "model_pipeline.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(ArtifactIntegrityError, match="model_pipeline.joblib SHA-256"):
        with TestClient(create_app(copy_of_run)):
            pass


def test_tampered_model_config_prevents_startup(copy_of_run):
    with open(copy_of_run / "model_config.json", "ab") as f:
        f.write(b" ")
    with pytest.raises(ArtifactIntegrityError, match="model_config.json"):
        with TestClient(create_app(copy_of_run)):
            pass


def test_edited_threshold_is_caught_by_pin_and_by_hex_check(copy_of_run):
    pinned = sha256_file(copy_of_run / "frozen_config.json")
    fz = json.loads((copy_of_run / "frozen_config.json").read_text())
    fz["thresholds"]["model"]["threshold"] = 0.5
    (copy_of_run / "frozen_config.json").write_text(json.dumps(fz))
    with pytest.raises(ArtifactIntegrityError, match="decimal and hex"):
        load_bundle(copy_of_run)
    fz["thresholds"]["model"]["threshold_hex"] = float(0.5).hex()  # a consistent but different threshold
    (copy_of_run / "frozen_config.json").write_text(json.dumps(fz))
    load_bundle(copy_of_run)  # internally consistent: only a pin can notice
    with pytest.raises(ArtifactIntegrityError, match="pinned"):
        load_bundle(copy_of_run, expected_frozen_sha256=pinned)


def test_missing_files_and_wrong_feature_order_prevent_startup(copy_of_run, tmp_path):
    with pytest.raises(ArtifactIntegrityError):
        load_bundle(tmp_path / "nowhere")
    (copy_of_run / "model_pipeline.joblib").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        load_bundle(copy_of_run)


def test_foreign_pipeline_with_matching_hash_is_rejected(copy_of_run):
    """A different pickle whose hash was also written into the config still has to expose the frozen features."""
    from sklearn.linear_model import LogisticRegression

    joblib.dump(LogisticRegression(), copy_of_run / "model_pipeline.joblib")
    fz = json.loads((copy_of_run / "frozen_config.json").read_text())
    fz["model_sha256"] = sha256_file(copy_of_run / "model_pipeline.joblib")
    (copy_of_run / "frozen_config.json").write_text(json.dumps(fz))
    with pytest.raises(ArtifactIntegrityError, match="frozen feature names"):
        load_bundle(copy_of_run)


def test_model_file_name_comes_from_code_not_config(copy_of_run):
    fz = json.loads((copy_of_run / "frozen_config.json").read_text())
    fz["model_file"] = "../elsewhere.joblib"
    (copy_of_run / "frozen_config.json").write_text(json.dumps(fz))
    with pytest.raises(ArtifactIntegrityError, match="model_file"):
        load_bundle(copy_of_run)


def test_serve_command_refuses_to_start_on_mismatch(copy_of_run):
    with open(copy_of_run / "model_pipeline.joblib", "ab") as f:
        f.write(b"x")
    assert main(["serve", "--artifact-dir", str(copy_of_run), "--port", "8999"]) == C.EXIT_ARTIFACT_MISMATCH
