"""The dashboard must not collapse duplicate JSON keys (or accept NaN/Infinity) and then submit the survivor as valid."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fraudrisk.dashboard_data as dd
from fraudrisk import constants as C
from fraudrisk.strict_json import StrictJSONError, parse_strict_json

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def valid_text(**over):
    f = dd.example_features()
    f.update(over)
    return json.dumps(f)


def dup_text(key="V1"):
    """A syntactically valid object whose KEY is repeated: json.loads would silently keep only the last value."""
    body = valid_text()
    value = json.dumps(dd.example_features()[key])
    out = body.replace(f'"{key}": {value}', f'"{key}": 999.0, "{key}": {value}', 1)
    assert out != body and json.loads(out)[key] == dd.example_features()[key]  # json.loads keeps only the last value
    return out


@pytest.fixture
def calls(monkeypatch):
    """Replace the network call with a recorder; submit_payload still runs for real."""
    seen = []

    def fake(base_url, features, timeout=10.0):
        seen.append(features)
        return 200, {"model_score": 0.0, "review_flag": False, "threshold": 0.5, "model_version": "x",
                     "threshold_version": "y"}

    monkeypatch.setattr(dd, "call_score_api", fake)
    return seen


def test_the_duplicate_text_really_is_one_json_loads_would_swallow():
    text = dup_text("V1")
    assert json.loads(text)["V1"] == 0.0  # the failure mode: 999.0 vanished without any error
    with pytest.raises(StrictJSONError, match="duplicate key 'V1'"):
        parse_strict_json(text)


@pytest.mark.parametrize("key", ["V1", "V17", "V28", "Amount"])
def test_submit_payload_refuses_duplicate_keys_and_sends_nothing(calls, key):
    status, body = dd.submit_payload("http://unused", dup_text(key))
    assert status is None and body["submitted"] is False and f"duplicate key '{key}'" in body["error"]
    assert calls == []


@pytest.mark.parametrize("text, needle", [
    ('{"V1": NaN}', "non-standard JSON constant NaN"),
    ('{"V1": Infinity}', "Infinity"),
    ("not json", "invalid JSON"),
    ("", "invalid JSON"),
    ('{"a": {"b": 1, "b": 2}}', "duplicate key 'b'"),
    ("[" * 100000, "larger than"),
])
def test_submit_payload_refuses_other_non_strict_input(calls, text, needle):
    status, body = dd.submit_payload("http://unused", text)
    assert status is None and body["submitted"] is False and needle in body["error"]
    assert calls == []


def test_valid_text_is_submitted_exactly_once_with_the_parsed_values(calls):
    status, body = dd.submit_payload("http://unused", valid_text(Amount=40.0, V3=1.25))
    assert status == 200 and len(calls) == 1
    assert calls[0]["Amount"] == 40.0 and calls[0]["V3"] == 1.25 and set(calls[0]) == set(C.FEATURE_COLUMNS)


def test_unreachable_api_is_reported_not_raised():
    status, body = dd.submit_payload("http://127.0.0.1:9", valid_text())
    assert status is None and body["submitted"] is False and "could not reach" in body["error"]


def run_app(monkeypatch, artifact_dir):
    monkeypatch.setenv("FRAUDRISK_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("FRAUDRISK_API_URL", "http://127.0.0.1:9")
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_shows_a_clear_error_and_does_not_submit_duplicate_keys(monkeypatch, synthetic_run, calls):
    at = run_app(monkeypatch, synthetic_run)
    at.text_area(key="payload_text").set_value(dup_text("V1")).run()
    at.button(key="score_button").click().run()
    assert not at.exception
    assert calls == []                                              # nothing reached the (fake) API
    assert any("Nothing was scored" in e.value and "duplicate key 'V1'" in e.value for e in at.error)
    shown = at.json[0].value
    shown = json.loads(shown) if isinstance(shown, str) else shown
    assert shown["submitted"] is False and "model_score" not in shown
    text = " ".join(m.value for m in at.markdown)
    assert "No API response" in text and "API threshold equals the frozen threshold" not in text


def test_app_still_submits_a_valid_payload(monkeypatch, synthetic_run, calls):
    at = run_app(monkeypatch, synthetic_run)
    at.button(key="score_button").click().run()
    assert not at.exception and len(calls) == 1 and not at.error
    at.text_area(key="payload_text").set_value('{"V1": NaN}').run()
    at.button(key="score_button").click().run()
    assert len(calls) == 1 and any("non-standard JSON constant" in e.value for e in at.error)
