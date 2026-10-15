#!/usr/bin/env python3
"""CPU smoke test of make_reference.py and checker.py against benchmark/smoke/fake_server.py.

  1. make_reference.py against a clean fake server writes a reference.
  2. checker.py passes the same clean server,
  3. passes it with small logit noise (0.02 nats),
  4. fails it with large logit noise (1.5 nats),
  5. fails a server that answers one canned token for every prompt,
  6. fails a server that does not report cached tokens (no prefix-cache hit).

usage: smoke_test.py --text-file <enwik9>
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE = HERE.parent / "benchmark" / "smoke" / "fake_server.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def with_server(flags: list[str], command: list[str]) -> subprocess.CompletedProcess[str]:
    port = free_port()
    server = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port), *flags],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert server.stdout is not None
        server.stdout.readline()
        argv = [a.replace("{URL}", f"http://127.0.0.1:{port}") for a in command]
        return subprocess.run(argv, capture_output=True, text=True, timeout=900, check=False)
    finally:
        server.terminate()
        server.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    ns = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="checker_smoke-"))
    reference = root / "reference.json"
    made = with_server([], [sys.executable, str(HERE / "make_reference.py"), "--base-url", "{URL}",
                            "--text-file", ns.text_file, "--output", str(reference)])
    print(made.stdout[-400:], made.stderr[-400:])
    assert made.returncode == 0 and reference.is_file(), made
    print(f"reference: {reference.stat().st_size} bytes")
    checker = [sys.executable, str(HERE / "checker.py"), "--base-url", "{URL}", "--reference", str(reference)]
    cases = [("clean", [], 0), ("small-noise", ["--noise", "0.02"], 0), ("large-noise", ["--noise", "1.5"], 1),
             ("canned", ["--canned"], 1), ("no-cache-report", ["--no-cache-report"], 1)]
    for name, flags, expected in cases:
        result = with_server(flags, checker)
        verdict = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        brief = [line for line in result.stdout.splitlines() if '"failures"' in line or "near_ties" in line
                 or "mean_abs" in line]
        print(f"--- {name}: exit {result.returncode} ({verdict}) {' '.join(s.strip() for s in brief)}")
        if result.returncode == 1:
            failures = result.stdout[result.stdout.find('"failures"'):][:300]
            print(f"    {failures}")
        assert result.returncode == expected, (name, result.stdout[-2000:], result.stderr[-2000:])
    print("ALL CHECKER SMOKE CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
