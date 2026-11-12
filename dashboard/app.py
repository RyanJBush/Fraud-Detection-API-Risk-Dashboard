"""Streamlit dashboard for the offline fraud-score milestone. Local only.

    streamlit run dashboard/app.py

Environment (read at startup, never from the browser):
    FRAUDRISK_ARTIFACT_DIR  artifact directory (default: artifacts/real under the project)
    FRAUDRISK_API_URL       scoring API base URL (default: http://127.0.0.1:8000)

The validation-threshold slider reads only the saved validation candidate table. It cannot change the API's
frozen threshold or any test number: those come from frozen_config.json, the API, and test_metrics.json.
"""
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fraudrisk import constants as C
from fraudrisk.dashboard_data import (
    EXAMPLE_DESCRIPTION,
    EXAMPLE_NAME,
    submit_payload,
    example_features,
    load_dashboard_data,
    pr_curve,
    row_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("FRAUDRISK_ARTIFACT_DIR", ROOT / "artifacts" / "real"))
API_URL = os.environ.get("FRAUDRISK_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fraud risk: offline model dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def _load(path: str):
    return load_dashboard_data(path)


def pct(x, digits=4):
    return "undefined" if x is None else f"{100 * x:.{digits}f}%"


def num(x, digits=4):
    return "undefined" if x is None else f"{x:.{digits}f}"


def confusion_table(tp, fp, fn, tn) -> pd.DataFrame:
    return pd.DataFrame(
        {"flagged for review": [tp, fp], "not flagged": [fn, tn]},
        index=["labeled fraud", "labeled non-fraud"],
    )


try:
    D = _load(str(ARTIFACT_DIR))
except Exception as e:  # noqa: BLE001 - show a clear message instead of a stack trace
    st.error(f"Could not load the saved artifacts from {ARTIFACT_DIR}: {e}")
    st.stop()

frozen_thr = float.fromhex(D.frozen["thresholds"]["model"]["threshold_hex"])

st.title("Fraud risk: offline model, frozen threshold, honest limits")
if D.data_kind == "synthetic":
    st.warning(D.notice or "SYNTHETIC FIXTURE: not evidence of real-world performance.")
st.markdown(
    "A logistic-regression **model score** for card transactions, trained offline on a historical dataset. The score is "
    "**not a calibrated probability** (calibration was not evaluated). A transaction is flagged for review when "
    f"`{C.THRESHOLD_RULE}`. The threshold is demonstration-grade: it maximises F1 on a validation period, which is not a "
    "business cost model or an analyst-capacity limit. No savings are claimed."
)

# ------------------------------------------------------------------------------------------------ frozen settings
st.header("1. What is frozen")
c1, c2, c3 = st.columns(3)
c1.metric("Model", "lr-" + D.frozen["model_sha256"][:12])
c2.metric("Frozen threshold (from validation only)", f"{frozen_thr:.10f}")
c3.metric("Data", D.data_kind)
st.caption(
    "The model, the feature list and the threshold were fixed and hashed before any test row was scored. Nothing on this "
    f"page can change them. frozen_config.json SHA-256: `{D.frozen_sha256}`"
)
if not D.consistent:
    st.error("test_metrics.json was not produced against this frozen_config.json. Do not trust the test section.")

# ------------------------------------------------------------------------------------------------ example + API
st.header("2. Score one transaction through the local API")
st.markdown(f"**{EXAMPLE_NAME}.** {EXAMPLE_DESCRIPTION}")
payload_text = st.text_area("Request features (JSON object with exactly V1..V28 and Amount)",
                            json.dumps(example_features(), indent=1), height=260, key="payload_text")
st.caption(f"Calls `POST {API_URL}/score` with `{{\"features\": ...}}`. Start the API first (see README).")
if st.button("Score with the local API", key="score_button"):
    # Strict parsing first (shared with the API): duplicate keys, NaN/Infinity and invalid JSON are refused here and
    # nothing is sent, so a repeated key can never be collapsed to its last value and scored as if it were valid.
    st.session_state["api_result"] = submit_payload(API_URL, payload_text)
if "api_result" in st.session_state:
    status, body = st.session_state["api_result"]
    if status is None:
        st.error("Nothing was scored. " + str(body.get("error", "")))
    st.markdown(f"**Actual API response** (HTTP status `{status}`):" if status is not None
                else "**No API response** (the request was not sent or could not be completed):")
    st.json(body)
    if status == 200:
        same = body["threshold"] == frozen_thr
        st.markdown(
            f"model score `{body['model_score']!r}` {'>=' if body['review_flag'] else '<'} threshold "
            f"`{body['threshold']!r}`, so `review_flag = {body['review_flag']}`. "
            f"API threshold equals the frozen threshold in frozen_config.json: **{same}**."
        )

st.subheader("What the inputs are (and are not)")
st.markdown(
    "- **Amount**: the transaction amount as recorded in the source data. The currency and units are not documented.\n"
    "- **V1 to V28**: anonymised principal components produced by a PCA the data providers ran for confidentiality. "
    "**They have no stated meaning**, and nothing here assigns one. How that PCA was fitted is undocumented, and "
    "scaling inside this project (training rows only) does not resolve that uncertainty.\n"
    "- **Time** and **Class** are not model inputs. The API rejects them, and any other unknown field."
)

# ------------------------------------------------------------------------------------------------ validation exploration
st.header("3. Explore thresholds on the validation period (exploration only)")
n_cand = len(D.candidates)
st.markdown(
    "Each point is one distinct validation score used as a threshold, with transactions flagged when "
    "`score >= threshold`. **Moving the slider changes only this section.** It does not change the API's threshold, the "
    "frozen threshold, or any test result below, and no test data is used here."
)
rank = st.slider(f"Validation threshold, by rank (0 = highest of {n_cand:,} candidates; frozen threshold is rank {D.frozen_index:,})",
                 0, n_cand - 1, D.frozen_index, key="val_rank")
m = row_metrics(D.candidates, rank)
fz = row_metrics(D.candidates, D.frozen_index)
curve = pr_curve(D.candidates, keep=(rank, D.frozen_index))
line = alt.Chart(curve).mark_line().encode(
    x=alt.X("recall:Q", title="Recall (validation)"), y=alt.Y("precision:Q", title="Precision (validation)"),
    tooltip=["threshold", "recall", "precision"])
pts = pd.DataFrame({"recall": [fz["recall"], m["recall"]], "precision": [fz["precision"], m["precision"]],
                    "which": ["frozen threshold", "explored threshold"]})
marks = alt.Chart(pts).mark_point(size=140, filled=True).encode(x="recall:Q", y="precision:Q", color="which:N")
st.altair_chart(line + marks, width="stretch")

cc1, cc2 = st.columns(2)
with cc1:
    st.markdown(f"**Explored threshold** `{m['threshold']!r}`")
    st.table(confusion_table(m["tp"], m["fp"], m["fn"], m["tn"]))
    st.markdown(f"review fraction **{pct(m['review_fraction'])}** ({m['flagged']:,} of {m['n']:,} validation transactions) · "
                f"precision {num(m['precision'])} · recall {num(m['recall'])} · F1 {num(m['f1'])}")
with cc2:
    st.markdown(f"**Frozen threshold** `{fz['threshold']!r}` (validation view)")
    st.table(confusion_table(fz["tp"], fz["fp"], fz["fn"], fz["tn"]))
    st.markdown(f"review fraction {pct(fz['review_fraction'])} · precision {num(fz['precision'])} · recall {num(fz['recall'])} · "
                f"F1 {num(fz['f1'])}")
st.caption(
    "These are validation numbers at the period used to pick the threshold, so they are optimistic. With only "
    f"{fz['tp'] + fz['fn']} labeled-fraud validation transactions, they are noisy."
)

# ------------------------------------------------------------------------------------------------ final test
st.header("4. Final test results at the frozen threshold")
st.markdown(
    "Computed once, after the model and thresholds were frozen. Both scorers are evaluated on exactly the same rows, "
    "with their own frozen thresholds chosen on validation. The **constant baseline** gives every transaction the training "
    "fraud rate as its score, so at its threshold it flags everything."
)


def test_block(title, pop):
    st.subheader(title)
    rows = []
    for key, name in (("model", "logistic regression"), ("constant_score_baseline", "constant baseline")):
        t = pop[key]
        rows.append({"scorer": name, "rows": t["n_rows"], "labeled fraud": t["n_fraud"],
                     "fraud prevalence": pct(t["fraud_prevalence"]), "review fraction": pct(t["fraction_flagged"]),
                     "precision": num(t["precision"]), "recall": num(t["recall"]), "F1": num(t["f1"]),
                     "average precision": num(t["average_precision"])})
    st.table(pd.DataFrame(rows).set_index("scorer"))
    cols = st.columns(2)
    for col, (key, name) in zip(cols, (("model", "logistic regression"), ("constant_score_baseline", "constant baseline"))):
        t = pop[key]
        col.markdown(f"Confusion matrix, {name}")
        col.table(confusion_table(t["tp"], t["fp"], t["fn"], t["tn"]))


pops = D.test["populations"]
sec_key = "secondary_previously_unseen_feature_vector_subset"
test_block(f"Primary: {C.PRIMARY_LABEL}", pops["primary"])
test_block(f"Secondary: {C.SECONDARY_LABEL}", pops[sec_key])
exc = pops["excluded_test_rows"]
st.markdown(
    f"Excluded from the secondary view: {exc['n_rows']:,} test rows whose exact feature vector also appears in training or "
    f"validation (labeled fraud among them: {exc['model']['n_fraud']}). The model flagged {exc['model']['n_flagged']:,} of them; "
    f"the constant baseline flagged {exc['constant_score_baseline']['n_flagged']:,}. The two views reconcile exactly: "
    f"**{D.test['reconciliation']['all_reconcile']}**."
)
st.info(
    "Because only labeled non-fraud rows are excluded, the secondary view has a higher fraud prevalence, so its precision "
    "and average precision are not a pure measure of leakage. It makes no claim of entity independence, duplicate removal "
    "or leakage removal; it removes only exact repeats of a training or validation feature vector."
)
st.warning(
    "**Batch evaluation versus single-row serving.** Saved evaluation tables use whole-partition batch scores. "
    "The API scores one transaction at a time, and a one-row call can differ from the batch score in the last bits "
    "because of floating-point summation order. The rule `score >= threshold` stays exact, with no rounding or "
    "tolerance. A batch score exactly on the threshold can therefore be flagged differently when served singly. "
    + ("In the reviewed real run, one validation flag and no test flags differed. " if D.data_kind == "real" else "")
    + "The test tables above are batch results."
)

# ------------------------------------------------------------------------------------------------ limitations
st.header("5. Limits you should know")
st.markdown(
    "- **Short and historical**: two days of card transactions from September 2013, split in time (about 60/20/20). Fraud "
    "prevalence differs across the periods, and a threshold chosen on one period faces another.\n"
    f"- **Few positives**: {pops['primary']['model']['n_fraud']} labeled-fraud test transactions and "
    f"{fz['tp'] + fz['fn']} in validation. No confidence intervals were computed; every figure is a point estimate.\n"
    "- **Labels**: how fraud was confirmed, and how long confirmation took, is not documented.\n"
    "- **No customer or card identifiers**: separation of the same customer across train, validation and test cannot be "
    "established or tested.\n"
    "- **Anonymised, PCA-transformed inputs**: no business interpretation of any component is offered.\n"
    "- **Not calibrated, not a cost model**: do not read the score as a probability or the F1 threshold as an operating policy.\n"
    "- **Local demo only**: not deployed, not monitored, no retraining."
)

# ------------------------------------------------------------------------------------------------ attribution
st.header("6. Data source and attribution")
st.markdown(
    "Contains information from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), "
    "which is made available here under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/); "
    "contents under the [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/). Collected and analysed "
    "by Worldline and the Machine Learning Group of the Universite Libre de Bruxelles (ULB). Copy used: OpenML dataset 1597, "
    "version 1. Please cite: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi, *Calibrating "
    "Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015. This dashboard shows aggregate results "
    "only; no real transaction rows are displayed. The dataset's licence is separate from this project's code."
)
