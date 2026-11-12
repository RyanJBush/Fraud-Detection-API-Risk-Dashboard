#!/usr/bin/env bash
# The CI workflow's "Local API refuses a tampered model" step, runnable locally: bash scripts/ci_tamper_step.sh DIR
set -e
SRC="${1:-artifacts/synthetic}"
python - "$SRC" <<'PY'
import pathlib, shutil, subprocess, sys
src = sys.argv[1]
d = pathlib.Path(src + "_tampered")
if d.exists(): shutil.rmtree(d)
shutil.copytree(src, d)
with open(d / "model_pipeline.joblib", "ab") as f: f.write(b"x")
rc = subprocess.run([sys.executable, "-m", "fraudrisk", "serve", "--artifact-dir", str(d)]).returncode
print("serve exit code:", rc)
shutil.rmtree(d)
sys.exit(0 if rc == 9 else 1)
PY
