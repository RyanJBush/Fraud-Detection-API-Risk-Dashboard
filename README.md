# Fraud Detection API & Risk Dashboard (local, offline-trained)

An offline transaction-classification pipeline that shows the tradeoff between catching labeled fraud and flagging
labeled non-fraud transactions, plus a **local** scoring API (FastAPI) and a small dashboard (Streamlit) over the frozen
model. The local MVP is complete and published. The original code is MIT-licensed ([LICENSE](LICENSE)); the dataset is not
covered by that licence (see [Licence](#licence)).

> **Status: local MVP accepted after ChatGPT review (v4); published under MIT.**
> - The reviewed local run passed **309 tests**, including the private real-data boundary test. In a clean public checkout,
>   that one test is intentionally skipped because real data and model artifacts are not included.
> - Numerical-overflow rejection, strict API/dashboard JSON, artifact checks, loopback binding, and validation-slider
>   isolation were independently checked. The frozen real model and thresholds were preserved.
> - **Hosted GitHub Actions: passed** on commit `4ca5a0e` ([run 43294204336](https://github.com/RyanJBush/Fraud-Detection-API-Risk-Dashboard/actions/runs/43294204336),
>   `windows-latest`, CPython 3.13.17): install from the pinned requirements, **308 tests passed, 1 skipped** (the private-data test),
>   the synthetic pipeline, and the tampered-model refusal (exit 9). Later commits have their own runs; see the Actions tab.
>   Nothing is deployed.
> - Licensing is decided: MIT for original code, the existing dataset notice and attribution for the data; see
>   [publication notes](docs/publication.md).
> - Outputs are **model scores**. Calibration was not evaluated.

## Verified held-out results (real data, batch evaluation)

Both scorers use the same 56,962 chronological test rows, with **75 labeled frauds** (prevalence **0.131667%**).
Thresholds were selected on validation only by maximizing F1; ties select the highest threshold.

| Scorer | Average precision | Precision | Recall | F1 | TP | FP | FN | TN | Flagged for review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.696381 | 0.675000 | 0.720000 | 0.696774 | 54 | 26 | 21 | 56,861 | 80 (0.140445%) |
| Constant-score baseline | 0.001317 | 0.001317 | 1.000000 | 0.002630 | 75 | 56,887 | 0 | 0 | 56,962 (100%) |

The model threshold is `0.11129646331946615`; the baseline score/threshold is the training prevalence,
`0.002106716915766435`. AP means `sklearn.metrics.average_precision_score`, not trapezoidal PR-AUC.
These are point estimates from 75 positives, not business-savings estimates or evidence of deployment performance.
The documented batch/single-row boundary exception still applies.

| Partition | Rows | Labeled fraud | Labeled non-fraud | Fraud prevalence |
|---|---:|---:|---:|---:|
| Train | 170,882 | 360 | 170,522 | 0.210672% |
| Validation | 56,963 | 57 | 56,906 | 0.100065% |
| Test | 56,962 | 75 | 56,887 | 0.131667% |

The secondary test subset contains 55,240 previously unseen feature-vector rows and all 75 positives. Its model counts
are TP 54 / FP 26 / FN 21 / TN 55,139, AP 0.696431. It excludes 1,722 labeled non-fraud rows and changes prevalence;
it does not establish entity independence or leakage removal.
Full aggregate values, provenance hashes, and attribution are in [reports/heldout_results.json](reports/heldout_results.json)
and [reports/README.md](reports/README.md). No real transaction rows are included.

> Contains information from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which is made
> available here under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/) (contents under the
> [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/)). The results above are derived from that dataset.

For the no-download walkthrough, follow [the short synthetic demo](docs/demo.md).

## Dataset and limits

ULB / Worldline "Credit Card Fraud Detection", pinned to **OpenML dataset 1597, version 1, file 1673544** (ARFF). Original
source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud . Download, SHA-256 and OpenML MD5 check:
`data/manifest.json`. Raw data lives in `data/raw/` and is git-ignored; real row-level files are never committed or shown
by the dashboard. Attribution, licence declarations and conditions: [NOTICE_DATA.md](NOTICE_DATA.md) (the dataset's licence is
separate from any code licence). Cite: Dal Pozzolo, Caelen, Johnson, Bontempi, *Calibrating Probability with Undersampling
for Unbalanced Classification*, IEEE CIDM 2015.

Unresolved, not claimed away: the upstream PCA fitting procedure, how labels were confirmed, label-availability delays,
no customer/card identifiers (entity separation is **not** established), and a two-day historical window with 492 labeled
frauds. Training-only scaling does not resolve uncertainty about the upstream PCA. F1 is a demonstration objective, not a
cost model or analyst-capacity limit. No business savings are claimed.

## Setup (one environment, used by every command below)

Use a **short** venv path on Windows. Every later command assumes this environment is **activated**, so `python` means the
venv's interpreter.

```powershell
python -m venv C:\fdv
C:\fdv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Git Bash: `source C:/fdv/Scripts/activate` instead of the activation line. Dependency versions are pinned in
`requirements.txt` (Python 3.13.7, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, FastAPI 0.162.4, Streamlit 1.67.0, ...).

## Run the local API and dashboard

A fresh checkout contains code, a **synthetic** fixture, and aggregate real-data results. It contains no trained model,
raw real data, or real row-level artifacts. Generate synthetic artifacts first; these commands never train on real data.
Run from the repository root with the environment above activated.

**Terminal 1 (PowerShell): generate synthetic artifacts and start the API.**

```powershell
$env:FRAUDRISK_ARTIFACT_DIR = Join-Path (Get-Location).Path "artifacts\synthetic"
$env:FRAUDRISK_API_URL = "http://127.0.0.1:8000"
foreach ($stage in @("data-check", "split", "train", "evaluate")) {
    python -m fraudrisk $stage --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out $env:FRAUDRISK_ARTIFACT_DIR
    if ($LASTEXITCODE -ne 0) { throw "Synthetic stage failed: $stage" }
}
python -m fraudrisk serve --artifact-dir $env:FRAUDRISK_ARTIFACT_DIR --port 8000
```

**Terminal 2 (PowerShell): activate the same environment, select the SAME artifacts, and start the dashboard.**

```powershell
C:\fdv\Scripts\Activate.ps1
$env:FRAUDRISK_ARTIFACT_DIR = Join-Path (Get-Location).Path "artifacts\synthetic"
$env:FRAUDRISK_API_URL = "http://127.0.0.1:8000"
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open <http://127.0.0.1:8501>. The dashboard must show **SYNTHETIC FIXTURE**; it is a code demonstration, not real-data performance.
It calls the local API and reads only aggregate synthetic artifacts. The published real-data table above remains a separate result.
For exact scoring commands and a three-minute walkthrough, see [docs/demo.md](docs/demo.md).

The API binds to `127.0.0.1`. Keep the explicit dashboard `--server.address 127.0.0.1` flag; the project config provides
the same default when launched from its root. Check both live listeners on Windows with:

```powershell
python scripts/verify_loopback.py 8000 8501
```

To use the real model instead, first run the real-data pipeline below and set `FRAUDRISK_ARTIFACT_DIR` to the absolute
`artifacts\real` path in **both** terminals. Use a separate output directory when reproducing an already-frozen run.
The API checks model/config checksums before loading; neither service retrains or changes a threshold.
Strict JSON validation rejects duplicate keys and non-standard constants before any dashboard request is sent.

### `POST /score`

Request body: `{"features": {"V1": ..., ..., "V28": ..., "Amount": ...}}`, JSON numbers only, exactly those 29 names.

```bash
python -c "import json; f={f'V{i}':0.0 for i in range(1,29)}; f['Amount']=25.0; print(json.dumps({'features':f}))" > request.json
curl -s -X POST http://127.0.0.1:8000/score -H "content-type: application/json" --data-binary @request.json
```

Response (200): `model_version`, `model_score`, `threshold`, `threshold_version`, `review_flag`, `score_note`.
`review_flag` is `model_score >= threshold` at full float64 precision. Versions come from the frozen artifacts
(`lr-` + first 12 hex of the model SHA-256; `thr-` + a hash of the frozen threshold, rule and model hash). `GET /model`
returns the metadata and full hashes; `GET /health` a liveness check. There is no upload, retrain or threshold endpoint.

| status | when |
|---|---|
| 400 | body not strict JSON: over 16 KiB, not UTF-8, invalid JSON, duplicate keys at any level, `NaN`/`Infinity` literals, integers over 4300 digits, nesting too deep |
| 422 | valid JSON, invalid request: envelope not exactly `{"features": {...}}`; features not exactly V1..V28 + Amount (missing, extra, `Class`, `Time`, ...); a value that is a string (even numeric), boolean, null, array/object, non-finite (`1e999`) or too large for float64; negative Amount; **finite values too extreme to score** (standardisation or the linear score overflows, error code `numerical_overflow`; e.g. `V28 = 1e308` on the real model) |

Overflow inside scoring is detected where it happens (the scaler, then the linear score), not only on the raw inputs; only that
condition becomes a 422, any other fault stays a server error. Successful responses always hold a finite score in [0, 1]; absurdly
large inputs can saturate the score to exactly 0.0 or 1.0. The verified malformed-input and numerical-overflow cases return controlled 4xx responses; unrelated server faults remain 500s. Errors look like `{"error": {"code": "...", "message": "..."}}`.

### Batch evaluation versus single-row serving (confirmed numerical limitation)

Offline and API scoring run the same code (`fraudrisk.modeling`), and the API equals **offline single-row** scoring bit for bit.
But a one-row `predict_proba` and a many-row (batch) call can differ in the last bits (BLAS summation order). All saved evaluation
numbers (validation selection, test metrics, the dashboard tables) are **batch** results; the API is **single-row**.

The rule stays exactly `model_score >= threshold`: no rounding, no epsilon, no threshold change. A batch score
exactly on the threshold can fall just below it when scored singly. The private v4 review reproduced one such
validation case; this public package does not publish that transaction's identifier, label, or per-row score.
The optional real-data regression test derives the boundary case from the private validation artifact when available.

The earlier spot check covered 1,206 real test rows only. A reported full scan of the same frozen model found:

| Partition | Rows | Scores differing in the last bits | Maximum relative difference | Review-flag differences |
|---|---:|---:|---:|---:|
| Validation | 56,963 | 8,910 | about 7.2e-15 | 1 |
| Test | 56,962 | 8,516 | about 7.2e-15 | 0 |

Thus the reviewed test review set is unchanged under single-row serving. At the frozen threshold, single-row
validation scoring has 41 true positives rather than the batch view's 42. These aggregate scan results were reported
in v4; the numerical boundary behavior was independently reproduced. No threshold was changed or reselected.

## Pipeline commands (offline)

Each stage recomputes the leakage audit and the secondary subset and stops unless they equal the saved ones and the reviewed
policy (`policy/overlap_policy_real_v1.json`).

```bash
python scripts/fetch_data.py
python -m fraudrisk data-check --data data/raw/creditcard.arff --data-kind real --data-manifest data/manifest.json --out artifacts/real
python -m fraudrisk split      --data data/raw/creditcard.arff --data-kind real --data-manifest data/manifest.json --out artifacts/real --overlap-policy policy/overlap_policy_real_v1.json
python -m fraudrisk train      --data data/raw/creditcard.arff --data-kind real --data-manifest data/manifest.json --out artifacts/real --overlap-policy policy/overlap_policy_real_v1.json
python -m fraudrisk evaluate   --data data/raw/creditcard.arff --data-kind real --data-manifest data/manifest.json --out artifacts/real --overlap-policy policy/overlap_policy_real_v1.json
python -I scripts/independent_split_check.py   data/raw/creditcard.arff artifacts/real
python -I scripts/independent_results_check.py data/raw/creditcard.arff artifacts/real
```

Re-running `train` rewrites `frozen_config.json` and the model with new timestamps; the verified artifacts in this project were
produced once and are left untouched by the API and dashboard. Without `--overlap-policy`, `split` exits 4 on this data.

Synthetic fixture (no download; what CI runs):

```bash
python -m pytest
python -m fraudrisk data-check --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out artifacts/synthetic
python -m fraudrisk split      --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out artifacts/synthetic
python -m fraudrisk train      --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out artifacts/synthetic
python -m fraudrisk evaluate   --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out artifacts/synthetic
python -m fraudrisk serve --artifact-dir artifacts/synthetic --port 8000     # a SYNTHETIC model; the dashboard says so
```

Exit codes: `0` ok, `2` validation failed, `3` conflicting labels, `4` repeated feature vectors cross partitions and no
policy given, `5` partitions unusable, `6` training did not converge, `7` evaluation prerequisite failed (a frozen artifact
changed, a partition has no positives), `8` findings differ from the reviewed overlap policy, `9` artifact checksum mismatch
(the API will not start).

## Rules implemented

- **Scoring features:** `V1..V28, Amount` in that saved order. Excluded: `Class`, `Time` (partitioning only), row ids,
  anything else. One function (`modeling.prepare_scoring_frame`, with `coerce_feature_value` per cell) validates and orders
  features for training, offline scoring and the API: ordinary numbers only; booleans, numeric strings, nulls, non-finite
  values, negative Amount, missing, duplicate and unknown fields are rejected, and numeric overflow becomes a controlled error.
- **Row ids:** `<first 12 hex of source SHA-256>-<0-based original row number>`; kept outside the feature matrix.
- **Split:** sort by `Time`, ties by original row number; boundaries at `floor(0.60N)` and `floor(0.80N)` (integer
  arithmetic), each moved to the first row of its timestamp group; fail (never random) if a partition is empty or has no
  positive or no negative labels. Validation and test are not rebalanced.
- **Leakage audit and secondary subset:** SHA-256 fingerprints of `V1..V28 + Amount` (Time excluded) are counted within
  and across partitions. Nothing is dropped, re-split or reweighted. The **previously unseen feature-vector subset** (test
  rows whose exact vector is in neither training nor validation; features only, no rounding) is saved at split time, before
  fitting. It makes no claim of entity independence, duplicate removal or leakage removal.
- **Baseline:** every row scores the training prevalence. **Classifier:** `StandardScaler` +
  `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)`, fit on training rows only; non-convergence
  is an error.
- **Decision rule:** `review_flag = model_score >= threshold`, full float64 comparison.
- **Threshold:** maximise validation F1 = `2TP/(2TP+FP+FN)` over every distinct validation score; exact integer comparison;
  ties go to the highest threshold; the baseline gets its own threshold (its single score, so it flags every row). Frozen
  (with the model hash) before test scoring; no refit on train+validation. The data file is read earlier, for validation,
  partitioning and overlap audits; test rows are excluded from fitting, validation scoring and threshold selection
  (see `docs/provenance_notes.md` for the wording of the already-frozen artifacts).
- **Metrics:** `sklearn.metrics.average_precision_score`, precision (JSON `null` plus
  `"precision_status": "undefined_no_rows_flagged"` when nothing is flagged), recall, F1, TP/TN/FP/FN, fraction flagged.

## Artifacts (per `--out` directory)

`data_quality_report.json`, `split_manifest.json`, `split_membership.csv.gz`, `leakage_audit.json`,
`cross_partition_overlap_rows.csv.gz`, `secondary_subset_row_ids.csv.gz`, `secondary_subset_manifest.json`,
`model_pipeline.joblib`, `model_config.json`, `validation_scores.csv.gz`, `threshold_selection_validation.json`
(+ candidate tables), `frozen_config.json`, `test_metrics.json`, `test_scores.csv.gz`, `error_analysis.json`. Every synthetic
artifact carries a `SYNTHETIC FIXTURE` notice. `model_pipeline.joblib` is a pickle: the API and `evaluate` load it only after
its SHA-256 matches `frozen_config.json`. Row-level files stay private.

## Layout

`src/fraudrisk/` (pipeline, `api.py`, `dashboard_data.py`) · `dashboard/app.py` · `tests/` (synthetic fixture) · `scripts/`
(fetch, fixture, logged runner, mutation checks, independent checks, demos) · `data/manifest.json` + `data/meta/` ·
`policy/` · `licenses/` · `LICENSE` · `NOTICE_DATA.md` · `docs/` · `.github/workflows/ci.yml`.

## Licence

- **Code:** the original code, tests, scripts and documentation are released under the [MIT licence](LICENSE).
- **Data:** the MIT licence does **not** cover the Credit Card Fraud Detection dataset or anything derived from its contents.
  The dataset is published by Worldline and the Machine Learning Group of ULB under the Open Database License (database) and the
  Database Contents License (contents). Attribution, the conditions that matter here, and the licence texts are in
  [NOTICE_DATA.md](NOTICE_DATA.md) and [licenses/](licenses/). Raw data and row-level derivatives are not in this repository.
- Third-party Python packages keep their own licences.
