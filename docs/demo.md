# Three-minute local demo (synthetic, no download)

Run from the repository root on Windows with Python 3.13. The checked-in 1,500-row fixture is synthetic.
This walkthrough demonstrates the code; it does **not** reproduce the real-data performance in the README.
Raw real data, real model artifacts, row-level exports, and private review packets are absent from the public package.

## 1. Prepare the environment

```powershell
python -m venv C:\fdv
C:\fdv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

Use the same activated environment in each terminal below. If you choose another venv location, substitute its
activation path consistently. Dependency installation may need internet access; the synthetic demo itself does not.

## 2. Terminal 1: prepare the fixture artifacts and start the API

```powershell
$env:FRAUDRISK_ARTIFACT_DIR = Join-Path (Get-Location).Path "artifacts\synthetic"
$env:FRAUDRISK_API_URL = "http://127.0.0.1:8000"
foreach ($stage in @("data-check", "split", "train", "evaluate")) {
    python -m fraudrisk $stage --data tests/fixtures/synthetic_transactions.csv --data-kind synthetic --out $env:FRAUDRISK_ARTIFACT_DIR
    if ($LASTEXITCODE -ne 0) { throw "Synthetic stage failed: $stage" }
}
python -m fraudrisk serve --artifact-dir $env:FRAUDRISK_ARTIFACT_DIR --port 8000
```

The `train` stage reproduces the existing synthetic-fixture pipeline. It does not read or modify the frozen real run.
Leave this terminal running.

## 3. Terminal 2: select the SAME artifacts and start the dashboard

```powershell
C:\fdv\Scripts\Activate.ps1
$env:FRAUDRISK_ARTIFACT_DIR = Join-Path (Get-Location).Path "artifacts\synthetic"
$env:FRAUDRISK_API_URL = "http://127.0.0.1:8000"
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open <http://127.0.0.1:8501>. Confirm the **SYNTHETIC FIXTURE** warning. The API and dashboard must use the same
artifact directory; starting the dashboard against its default `artifacts/real` would fail in a fresh checkout.

## 4. Terminal 3: score a constructed example and check listeners

```powershell
C:\fdv\Scripts\Activate.ps1
Invoke-RestMethod http://127.0.0.1:8000/health
$features = [ordered]@{}
foreach ($i in 1..28) { $features["V$i"] = 0.0 }
$features["Amount"] = 25.0
$body = @{ features = $features } | ConvertTo-Json -Depth 3
Invoke-RestMethod http://127.0.0.1:8000/score -Method Post -ContentType "application/json" -Body $body
python scripts/verify_loopback.py 8000 8501
```

The score response includes model/threshold versions, the full-precision score, the frozen threshold, and a review flag.
Explain that `review_flag = model_score >= threshold` and that calibration has not been evaluated.

## 5. Walk through the dashboard

1. Show the model version, frozen threshold, and synthetic label.
2. Click **Score with the local API**. Show the actual response for the constructed example, not a real transaction.
3. Move the validation-threshold slider. Explain how precision, recall, and the review fraction change.
4. Show that the frozen validation view and final test tables stay unchanged.
5. Add a repeated key such as `"V1": 999.0, "V1": 0.0` to the feature JSON and click score. The page must say
   **Nothing was scored**, identify the duplicate, and send no API request. Restore the valid example afterward.
6. Distinguish this synthetic demo from the independently verified real-data batch results in `reports/heldout_results.json`.
   Mention the two-day historical window, 75 real test positives, unknown PCA meanings, unavailable entity identifiers,
   and the documented batch/single-row boundary exception. Make no business-savings claim.

## 6. Stop and verify

Press **Ctrl+C** in terminals 1 and 2 to stop the servers. Optional local verification:

```powershell
python -m pytest
```

A checkout without private real data is expected to skip `test_confirmed_real_boundary_case_is_reproduced_and_api_matches_offline_single_row`.
The accepted private v4 run passed all 309 tests; the hosted GitHub Actions run on the first public commit passed 308 with the same 1 expected
skip (see the README for the run link). A local run is not itself a hosted result.

