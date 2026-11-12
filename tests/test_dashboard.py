"""Dashboard data logic and the Streamlit app (via AppTest) against a live local API. SYNTHETIC artifacts only."""
import json
import socket
import threading
import time
import urllib.request

import pandas as pd
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from fraudrisk.api import create_app
from fraudrisk.dashboard_data import call_score_api, example_features, load_dashboard_data, pr_curve, row_metrics
from fraudrisk.provenance import sha256_file

from pathlib import Path

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def live_api(synthetic_run):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(create_app(synthetic_run), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=10)


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------------ pure logic
def test_row_metrics_match_saved_validation_selection(synthetic_run):
    D = load_dashboard_data(synthetic_run)
    m = row_metrics(D.candidates, D.frozen_index)
    sel = D.selection["model"]
    assert m["threshold"] == sel["threshold"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (
        sel["validation_tp"], sel["validation_fp"], sel["validation_fn"], sel["validation_tn"])
    assert m["f1"] == pytest.approx(sel["validation_f1"], abs=1e-15)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n"] == sel["n_validation_rows"]
    assert D.consistent


def test_candidate_rows_equal_recount_from_validation_scores(synthetic_run):
    D = load_dashboard_data(synthetic_run)
    v = pd.read_csv(synthetic_run / "validation_scores.csv.gz")
    for i in (0, 1, 17, len(D.candidates) // 2, len(D.candidates) - 1):
        m = row_metrics(D.candidates, i)
        flag = v["model_score"] >= m["threshold"]
        assert m["tp"] == int((flag & (v["Class"] == 1)).sum())
        assert m["fp"] == int((flag & (v["Class"] == 0)).sum())
        assert m["review_fraction"] == pytest.approx(flag.mean())


def test_pr_curve_is_thinned_but_keeps_requested_points(synthetic_run):
    D = load_dashboard_data(synthetic_run)
    c = pr_curve(D.candidates, max_points=20, keep=(D.frozen_index, 3))
    assert len(c) <= 22 and D.candidates["threshold"].iloc[D.frozen_index] in set(c["threshold"])
    assert c["recall"].between(0, 1).all() and c["precision"].between(0, 1).all()


def test_dashboard_code_never_opens_row_level_files():
    import inspect

    import fraudrisk.dashboard_data as dd

    code = inspect.getsource(dd) + open(APP, encoding="utf-8").read()
    for forbidden in ("read_csv(d / \"validation", "test_scores.csv", "split_membership.csv", "creditcard.arff",
                      "cross_partition_overlap_rows.csv", "secondary_subset_row_ids.csv"):
        assert forbidden not in code, forbidden


# ------------------------------------------------------------------------ the Streamlit app
def run_app(monkeypatch, artifact_dir, api_url):
    monkeypatch.setenv("FRAUDRISK_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("FRAUDRISK_API_URL", api_url)
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_renders_all_sections_without_exceptions(monkeypatch, synthetic_run, live_api):
    at = run_app(monkeypatch, synthetic_run, live_api)
    assert not at.exception and not at.error
    heads = [h.value for h in at.header]
    assert len(heads) == 6 and heads[0].startswith("1.") and heads[-1].startswith("6.")
    text = " ".join([m.value for m in at.markdown] + [h.value for h in at.subheader])
    for needle in ("not a calibrated probability", "anonymised principal components", "have no stated meaning",
                   "Contains information from", "Open Database License", "Calibrating", "Constructed example",
                   "previously unseen feature-vector subset"):
        assert needle in text, needle
    assert at.warning and at.warning[0].value.startswith("SYNTHETIC FIXTURE")


def test_app_calls_the_live_api_and_shows_the_actual_response(monkeypatch, synthetic_run, live_api):
    at = run_app(monkeypatch, synthetic_run, live_api)
    at.button(key="score_button").click().run()
    assert not at.exception
    body = at.json[0].value
    body = json.loads(body) if isinstance(body, str) else body
    status, direct = call_score_api(live_api, example_features())
    assert status == 200 and body == direct  # what the page shows is what the API returns
    assert set(body) >= {"model_version", "model_score", "threshold", "threshold_version", "review_flag"}
    frozen = json.loads((synthetic_run / "frozen_config.json").read_text())
    assert body["threshold"] == float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
    assert "API threshold equals the frozen threshold in frozen_config.json: **True**" in " ".join(
        m.value for m in at.markdown)


def test_app_survives_bad_input_and_a_down_api(monkeypatch, synthetic_run, live_api):
    at = run_app(monkeypatch, synthetic_run, live_api)
    at.text_area(key="payload_text").set_value('{"V1": true}').run()
    at.button(key="score_button").click().run()
    assert not at.exception
    assert "422" in " ".join(m.value for m in at.markdown)  # the API's controlled rejection is shown
    at.text_area(key="payload_text").set_value("not json").run()
    at.button(key="score_button").click().run()
    assert not at.exception and "error" in str(at.json[0].value)
    down = run_app(monkeypatch, synthetic_run, "http://127.0.0.1:9")
    down.button(key="score_button").click().run()
    assert not down.exception and "error" in str(down.json[0].value)


def test_validation_exploration_cannot_change_frozen_threshold_or_test_results(monkeypatch, synthetic_run, live_api):
    files = ["frozen_config.json", "test_metrics.json", "model_pipeline.joblib", "threshold_selection_validation.json"]
    before = {f: sha256_file(synthetic_run / f) for f in files}
    api_before = call_score_api(live_api, example_features())[1]
    model_before = get_json(live_api + "/model")

    at = run_app(monkeypatch, synthetic_run, live_api)
    metrics0 = [(m.label, m.value) for m in at.metric]
    tables0 = [t.value.to_csv() for t in at.table]
    assert len(tables0) == 8  # explored-validation, frozen-validation, then 6 test tables
    top = at.slider(key="val_rank").max
    start = at.slider(key="val_rank").value
    moved_explored_table = False
    for target in (0, int(top // 2), int(top)):
        at.slider(key="val_rank").set_value(int(target)).run()
        assert not at.exception
        assert at.slider(key="val_rank").value == target
        assert [(m.label, m.value) for m in at.metric] == metrics0       # "What is frozen" never moves
        tables = [t.value.to_csv() for t in at.table]
        assert tables[1] == tables0[1]                                    # frozen validation view unchanged
        assert tables[2:] == tables0[2:]                                  # every test table unchanged
        moved_explored_table |= tables[0] != tables0[0]
    assert moved_explored_table and start != 0                            # the explored view did respond to the slider
    assert {f: sha256_file(synthetic_run / f) for f in files} == before   # no artifact was touched
    assert call_score_api(live_api, example_features())[1] == api_before  # the API still scores identically
    assert get_json(live_api + "/model") == model_before
