"""Run a command and record it as executed evidence.

    python scripts/run_logged.py NAME -- COMMAND [ARGS...]

Writes evidence/logs/NAME.log containing the exact argv, working directory, UTC start/end,
exit code and the combined stdout/stderr. The wrapper itself exits with the command's exit code.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if "--" not in sys.argv[1:]:
        print(__doc__)
        return 64
    i = sys.argv.index("--")
    name, cmd = sys.argv[1], sys.argv[i + 1:]
    started = now()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finished = now()
    log = ROOT / "evidence" / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"status: EXECUTED\nname: {name}\nargv: {cmd!r}\ncwd: {ROOT.name}\n"
        f"started_utc: {started}\nfinished_utc: {finished}\nexit_code: {p.returncode}\n"
        f"----- stdout -----\n{p.stdout}\n----- stderr -----\n{p.stderr}\n",
        encoding="utf-8",
    )
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    print(f"[run_logged] {name}: exit_code={p.returncode}")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
