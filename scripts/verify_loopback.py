"""Verify that the given TCP ports are listening ONLY on loopback (Windows: parses `netstat -ano -p TCP`).

    python scripts/verify_loopback.py 8000 8501

Exit 0 only if every port has at least one LISTENING socket and every one is bound to 127.0.0.1 or [::1]
(never 0.0.0.0, [::] or a LAN address).
"""
import subprocess
import sys

ports = {int(p) for p in sys.argv[1:]}
if not ports:
    sys.exit("usage: verify_loopback.py PORT [PORT ...]")
out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
seen: dict[int, list[str]] = {p: [] for p in ports}
for line in out.splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[0] == "TCP" and parts[3].upper() == "LISTENING":
        addr, _, port = parts[1].rpartition(":")
        if port.isdigit() and int(port) in ports:
            seen[int(port)].append(addr)
ok = True
for p, addrs in sorted(seen.items()):
    loop = bool(addrs) and all(a in ("127.0.0.1", "[::1]") for a in addrs)
    ok &= loop
    print(f"port {p}: listening on {addrs or 'NOTHING'} -> {'loopback only' if loop else 'NOT RESTRICTED TO LOOPBACK (or not listening)'}")
sys.exit(0 if ok else 1)
