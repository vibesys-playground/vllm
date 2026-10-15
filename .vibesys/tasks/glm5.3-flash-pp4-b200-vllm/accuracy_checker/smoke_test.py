#!/usr/bin/env python3
"""CPU smoke test of make_reference.py and checker.py against benchmark/smoke/fake_server.py.

  1. make_reference.py against a clean fake server writes a reference.
  2. checker.py passes the same clean server,
  3. passes it with small logit noise (0.02 nats),
  4. fails it with large logit noise (1.5 nats),
  5. fails a server that answers one canned token for every prompt,
  6. fails a server that does not report cached tokens (no prefix-cache hit),
  7. fails a server that answers replay-shaped (streamed) requests differently,
  8. passes two checker runs on one server (per-run cache_salt),
  9. passes a server whose cached-prefix resume changes the answer, with a reference
     made through the cached path, and fails it with a reference made cold,
 10. trusted builds: a second honest layout (--alt-numerics 3: a confident flip on
     one prompt in three) fails against the first build's reference alone, and
     passes once make_reference.py --merge adds its answers; the first build still
     passes the merged reference; a constant-token server, a shuffled-answer server
     and large noise all fail it.

usage: smoke_test.py --text-file <enwik9>
"""
from __future__ import annotations

import argparse
import json
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


def with_server(flags: list[str], command: list[str], runs: int = 1) -> subprocess.CompletedProcess[str]:
    port = free_port()
    server = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port), *flags],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert server.stdout is not None
        server.stdout.readline()
        argv = [a.replace("{URL}", f"http://127.0.0.1:{port}") for a in command]
        for _ in range(runs):
            result = subprocess.run(argv, capture_output=True, text=True, timeout=900, check=False)
            if result.returncode != 0:
                break
        return result
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
                            "--text-file", ns.text_file, "--build-id", "a", "--output", str(reference)])
    print(made.stdout[-400:], made.stderr[-400:])
    assert made.returncode == 0 and reference.is_file(), made
    print(f"reference: {reference.stat().st_size} bytes")
    checker = [sys.executable, str(HERE / "checker.py"), "--base-url", "{URL}", "--reference", str(reference)]
    cases = [("clean", [], 0), ("small-noise", ["--noise", "0.02"], 0), ("large-noise", ["--noise", "1.5"], 1),
             ("canned", ["--canned"], 1), ("no-cache-report", ["--no-cache-report"], 1),
             ("fake-replay", ["--fake-replay"], 1)]
    # A checker run, then a second one on the same server (its cache kept): both pass.
    cases.append(("repeat-run", [], 0))
    for name, flags, expected in cases:
        result = with_server(flags, checker, runs=2 if name == "repeat-run" else 1)
        verdict = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        brief = [line for line in result.stdout.splitlines() if '"failures"' in line or "near_ties" in line
                 or "mean_abs" in line]
        print(f"--- {name}: exit {result.returncode} ({verdict}) {' '.join(s.strip() for s in brief)}")
        if result.returncode == 1:
            failures = result.stdout[result.stdout.find('"failures"'):][:300]
            print(f"    {failures}")
        assert result.returncode == expected, (name, result.stdout[-2000:], result.stderr[-2000:])
        if name == "fake-replay":
            assert "the replay-shaped request answered 4242" in result.stdout, result.stdout[-2000:]
    # A server whose cached-prefix resume answers differently: the reference made
    # through the cached path still passes it.
    shifted = root / "reference_shifted.json"
    made = with_server(["--cached-path-shift"], [sys.executable, str(HERE / "make_reference.py"), "--base-url",
                                                  "{URL}", "--text-file", ns.text_file, "--build-id", "a",
                                                  "--output", str(shifted)])
    assert made.returncode == 0, made
    result = with_server(["--cached-path-shift"], [*checker[:-1], str(shifted)])
    print(f"--- cached-path-shift: exit {result.returncode}")
    assert result.returncode == 0, result.stdout[-2000:]
    stale = with_server(["--cached-path-shift"], checker)
    print(f"--- cached-path-shift against a cold-path reference: exit {stale.returncode}")
    assert stale.returncode == 1, stale.stdout[-2000:]
    # Trusted builds: a second honest layout added with --merge.
    alt = ["--alt-numerics", "3"]
    alone = with_server(alt, checker)
    print(f"--- alt-numerics against build a alone: exit {alone.returncode}")
    assert alone.returncode == 1 and "mismatch" in alone.stdout, alone.stdout[-2000:]
    merged = root / "reference_merged.json"
    made = with_server(alt, [sys.executable, str(HERE / "make_reference.py"), "--base-url", "{URL}", "--build-id",
                             "b", "--merge", str(reference), "--output", str(merged)])
    assert made.returncode == 0, (made.stdout[-1000:], made.stderr[-1000:])
    body = json.loads(merged.read_text())
    assert [b["id"] for b in body["builds"]] == ["a", "b"], body["builds"]
    base = json.loads(reference.read_text())
    assert [c.get("prompt", c.get("round0")) for c in body["cases"]] == \
        [c.get("prompt", c.get("round0")) for c in base["cases"]], "--merge changed the prompts"
    duplicate = with_server(alt, [sys.executable, str(HERE / "make_reference.py"), "--base-url", "{URL}",
                                  "--build-id", "a", "--merge", str(reference), "--output", str(root / "dup.json")])
    assert duplicate.returncode != 0 and "already has build" in duplicate.stderr, duplicate.stderr[-500:]
    on_merged = [*checker[:-1], str(merged)]
    for name, flags, expected in (("alt-numerics", alt, 0), ("clean", [], 0), ("canned", ["--canned"], 1),
                                  ("shuffled", ["--shuffled"], 1), ("large-noise", ["--noise", "1.5"], 1)):
        result = with_server(flags, on_merged)
        print(f"--- {name} against builds a+b: exit {result.returncode}")
        assert result.returncode == expected, (name, result.stdout[-2000:])
        if name == "alt-numerics":
            report = json.loads(result.stdout[result.stdout.rindex("\n{\n"):result.stdout.rindex("}") + 1])
            assert report["answers_by_build"]["b"] > 0, report["answers_by_build"]
    print("ALL CHECKER SMOKE CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
