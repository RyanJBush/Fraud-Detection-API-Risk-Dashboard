"""Hand-written mutation checks: break the code in a copy, confirm the test suite notices.

    python scripts/mutation_checks.py          # writes evidence/mutation_checks.json

Each mutation edits ONE expression in a temporary copy of src/ and tests/ (the working tree is never
modified) and runs pytest there. A mutation is KILLED if pytest exits non-zero. A SURVIVOR is reported
honestly, not hidden.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("threshold uses > instead of >=", "thresholds.py",
     "return np.asarray(scores, dtype=\"float64\") >= float(threshold)",
     "return np.asarray(scores, dtype=\"float64\") > float(threshold)"),
    ("F1 tie keeps LOWEST threshold", "thresholds.py",
     "if best is None or nj * best[2] > best[1] * dj:",
     "if best is None or nj * best[2] >= best[1] * dj:"),
    ("F1 compared with wrong direction (minimise)", "thresholds.py",
     "if best is None or nj * best[2] > best[1] * dj:",
     "if best is None or nj * best[2] < best[1] * dj:"),
    ("boundary not moved to timestamp-group start", "split.py",
     "return int(np.searchsorted(sorted_time, sorted_time[b], side=\"left\"))",
     "return b"),
    ("ties broken by reversed row number", "split.py",
     "order = np.lexsort((rn, time))", "order = np.lexsort((-rn, time))"),
    ("scaler/classifier fit on train+validation", "modeling.py",
     "mask = np.asarray(assignment) == \"train\"", "mask = np.asarray(assignment) != \"test\""),
    ("extra scoring fields silently allowed", "modeling.py", "    if extra:\n", "    if False:\n"),
    ("ConvergenceWarning check alone disabled (redundant guard: EXPECTED survivor)", "modeling.py",
     "    if conv:\n", "    if False:\n"),
    ("max_iter-exhaustion check alone disabled (redundant guard: EXPECTED survivor)", "modeling.py",
     "    if n_iter >= clf.max_iter:\n", "    if False:\n"),
    ("BOTH convergence guards disabled", "modeling.py",
     ["    if conv:\n", "    if n_iter >= clf.max_iter:\n"], ["    if False:\n", "    if False:\n"]),
    ("baseline score is 0.5 instead of train prevalence", "modeling.py",
     "return cls(prevalence=float(y.sum()) / float(len(y)))", "return cls(prevalence=0.5)"),
    ("-0.0 not normalised in fingerprints", "validation.py", "to_numpy(dtype=\"float64\") + 0.0)",
     "to_numpy(dtype=\"float64\"))"),
    ("precision reported as 0.0 when nothing flagged", "thresholds.py",
     "        precision = None\n", "        precision = 0.0\n"),
    ("average precision replaced by prevalence", "thresholds.py",
     "float(average_precision_score(y, s))", "float(y.mean())"),
    ("audit fingerprint includes Time", "split.py",
     "fp = fingerprints(frame, C.FEATURE_COLUMNS)\n    df = pd.DataFrame",
     "fp = fingerprints(frame, [C.TIME_COLUMN, *C.FEATURE_COLUMNS])\n    df = pd.DataFrame"),
    ("overlap does not block training without a policy", "cli.py",
     "        if audit_now[\"review_required\"]:\n            raise CrossPartitionOverlapError(",
     "        if False:\n            raise CrossPartitionOverlapError("),
    ("policy findings not enforced", "cli.py", "        enforce(policy, observed)\n", "        pass\n"),
    ("secondary subset only excludes training matches (validation ignored)", "split.py",
     "seen = set(fp[assignment != \"test\"])", "seen = set(fp[assignment == \"train\"])"),
    ("secondary subset ignores training matches", "split.py",
     "seen = set(fp[assignment != \"test\"])", "seen = set(fp[assignment == \"validation\"])"),
    ("policy comparison ignores mismatches", "policy.py",
     "    return all(r[\"match\"] for r in rows), rows", "    return True, rows"),
    ("evaluate skips frozen-artifact hash check", "cli.py", "    if bad:\n", "    if False:\n"),
    ("feature order changed (Amount first)", "constants.py",
     "FEATURE_COLUMNS = (*PCA_COLUMNS, AMOUNT_COLUMN)", "FEATURE_COLUMNS = (AMOUNT_COLUMN, *PCA_COLUMNS)"),
]


def run_pytest(workdir: Path) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=str(workdir / "src"))
    p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                       cwd=workdir, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-600:]


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="mut_") as tmp:
        for i, (name, fname, old, new) in enumerate([("UNMUTATED control", None, None, None)] + MUTATIONS):
            work = Path(tmp) / f"w{i}"
            shutil.copytree(ROOT / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            shutil.copytree(ROOT / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
            applied = True
            if fname:
                path = work / "src" / "fraudrisk" / fname
                text = path.read_text(encoding="utf-8")
                olds, news = (old, new) if isinstance(old, list) else ([old], [new])
                if any(text.count(o) != 1 for o in olds):
                    applied = False
                else:
                    for o, n in zip(olds, news):
                        text = text.replace(o, n)
                    path.write_text(text, encoding="utf-8")
            if not applied:
                results.append({"mutation": name, "file": fname, "status": "NOT APPLIED (pattern count != 1)"})
                continue
            code, tail = run_pytest(work)
            if fname is None:
                status = "control passes" if code == 0 else "CONTROL FAILED"
            else:
                status = "KILLED" if code != 0 else "SURVIVED"
            results.append({"mutation": name, "file": fname, "pytest_exit": code, "status": status,
                            "first_failure_tail": tail.strip().splitlines()[-1] if code else ""})
            print(f"{status:15s} {name}")
    out = ROOT / "evidence" / "mutation_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    bad = [r for r in results
           if r["status"] not in ("KILLED", "control passes") and "EXPECTED survivor" not in r["mutation"]]
    print(f"{len(results) - 1} mutations; killed={sum(r['status'] == 'KILLED' for r in results)}; "
          f"unexpected survivors/not-applied={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
