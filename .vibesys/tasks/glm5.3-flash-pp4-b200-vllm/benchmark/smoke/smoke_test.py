#!/usr/bin/env python3
"""CPU smoke test of benchmark/session_replay.py against benchmark/smoke/fake_server.py.

Runs the real session_runner on a tiny synthetic trace (16 sessions, a
20-second window) and checks each outcome the benchmark can report:

  pass        a healthy server: one `result` row with every declared metric, slo_met 1
  slo         every 5th request takes 6.5 s: a scored `result` with slo_met 0 and
              a negative slo_margin_s (the 6 s SLO is reported, not gated)
  abort       every 2nd request takes 9 s, over the smoke workload's 8 s
              rejection bound (60 s in the real workload): rejected by the early abort
  failure     HTTP 500 after 40 requests: rejected for a failed request
  shortfall   no generated token ids after 40 requests: rejected for prefix_shortfall_tokens
  no-cache    no cached-token report: rejected at screening
  slow-screen every request takes 9 s, over the 8 s bound: rejected at screening
  big-block   prefix-cache blocks of 1,536 tokens: rejected at screening, before
              session_runner's 1,024-token preflight would abort
  record      recording mode on a healthy server: replays the seeded session sample
              (8 of the 15 reachable sessions, at most 12 rounds each) through a
              derived trace and writes the replay reference the scored cases below
              are checked against; their prompts match it only if the derived
              trace keeps session_runner's prompts (session ordinals, token pool)
  faked-replay streamed (replay-shaped) requests get a wrong token: rejected by the
              replay-reference agreement check
  truncation  prompts over 9,000 tokens are cut: rejected for usage.prompt_tokens
  no-ids      text but no token ids after 40 requests (the client re-encodes, so no
              shortfall shows): rejected for missing token ids
  starvation  four requests never answer: they count in p99 as outstanding rounds,
              which puts p99 over the 8 s bound: rejected, naming them
  cross-eval  the replay twice on one server, cache kept: the second run is rejected
              for cache hits from outside the evaluation
  crash       an unexpected exception inside run(): an `error` record, not no outcome
  judge       a finished run with p99 7 s: scored, slo_met 0; with p99 9 s: rejected
              by the end-of-run gate (the 8 s smoke bound)

usage:
  smoke_test.py --engine <session_runner> --tokenizer <tokenizer.json> --text-file <text> [--case NAME ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import session_replay as sr  # noqa: E402

FAKE = HERE / "fake_server.py"
SMOKE_REJECT_S = 8.0


def make_trace(path: Path, sessions: int = 16, rounds: int = 30, seed: int = 7) -> str:
    rng = random.Random(seed)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len",
                         "input_len", "output_len", "tool_wait_after_ms"])
        for s in range(sessions):
            arrival = s * 500.0
            prompt = 0
            for r in range(rounds):
                prefix = 0 if r == 0 else prompt + 1
                new = rng.randrange(2_000, 8_000) if r == 0 else rng.randrange(100, 3_000)
                writer.writerow([f"session_{s}_round_{r:06d}", s, r, f"{arrival:.6f}", prefix, new, 1,
                                 f"{rng.uniform(300, 800):.6f}"])
                prompt = prefix + new
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def smoke_workload(trace_sha: str) -> sr.Workload:
    return sr.Workload(
        trace_name="smoke.csv", trace_sha256=trace_sha, manifest=None, text_bytes=None, text_md5=None,
        max_concurrency=8, window_s=20.0, grace_s=3.0,
        screen_gate_tokens=2_048, screen_info_tokens=4_096, screen_append_tokens=256, preflight_probe_tokens=1_024,
        startup_timeout_s=120.0, stall_timeout_s=30.0, request_timeout_s=30.0,
        # The real bound is 60 s; 8 s keeps the abort and screening cases short
        # while staying above the 6.5 s slow requests of the `slo` case.
        reject_p99_ttft_s=SMOKE_REJECT_S,
        # 8 sampled sessions: their round 0s alone match; 30 needs some later rounds too.
        replay_min_agreement=0.95, replay_min_matched_rounds=30,
        replay_sample=sr.ReplaySample(seed=11, sessions=8, round_cap=12, concurrency=4, time_limit_s=60.0),
        round0_max_cached_tokens=16,  # the fake server's block
    )


def run_case(name: str, server_flags: list[str], ns: argparse.Namespace, root: Path, sha: str,
             mode: list[str] | None = None, runs: int = 1) -> list[dict]:
    port = free_port()
    server = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port), *server_flags],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert server.stdout is not None
        server.stdout.readline()  # the "fake server on" line: listening
        for run in range(runs):
            records = run_one(f"{name}{'' if run == 0 else f'-{run}'}", port, ns, root, sha, mode)
        return records
    finally:
        server.terminate()
        server.wait()


def run_one(name: str, port: int, ns: argparse.Namespace, root: Path, sha: str,
            mode: list[str] | None) -> list[dict]:
    out = root / name / "vs_output.jsonl"
    args = sr.parse_args([
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--request-factory-engine", ns.engine,
        "--text-file", ns.text_file,
        "--asset-dir", str(root),
        "--tokenizer", ns.tokenizer,
        "--vs-output", str(out),
        *(mode if mode is not None else ["--replay-reference", str(root / "replay_reference.json")]),
    ])
    started = time.time()
    code = sr.run(args, smoke_workload(sha), sr.ProtocolReport(out))
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    print(f"--- {name}: exit {code} in {time.time() - started:.0f} s; records:")
    for record in records:
        print("   ", json.dumps(record)[:600])
    return records


def expect_error(records: list[dict], *needles: str) -> None:
    assert records[0]["kind"] == "hello", records
    assert records[-1]["kind"] == "error", records
    message = records[-1]["message"]
    for needle in needles:
        assert needle in message, f"{needle!r} not in {message!r}"


def case_judge() -> None:
    workload = smoke_workload("0" * 64)
    watch = sr.LogWatch(over_bound_limit=10, window_s=20.0, ttft_bound_s=workload.reject_p99_ttft_s)
    facts = sr.TraceFacts(rounds=100, sessions=4, ceiling_rounds=100, ceiling_new_tokens=100_000)
    screened = {"screen_ttft_16k_s": 0.1, "screen_ttft_32k_s": 0.2}
    metrics = {"rounds": 90, "rounds_per_s": 4.5, "new_tokens_per_s": 4000.0, "prompt_tokens_per_s": 1.0,
               "computed_tokens_per_s": 1.0, "prefix_hit_share": 0.9,
               "ttft_s": {"p50": 0.1, "p90": 0.2, "p99": 7.0, "max": 9.0}, "rounds_ttft_over_60s": 0}
    agree = sr.Agreement(reached=110, matched=100, agreed=99, first_disagreement=None)
    values = sr.judge(metrics, watch, facts, workload, screened, agree)
    assert set(values) == set(sr.METRICS), set(sr.METRICS) ^ set(values)
    assert values["slo_met"] == 0.0 and abs(values["slo_margin_s"] + 1.0) < 1e-9, values
    metrics["ttft_s"]["p99"] = 5.0
    values = sr.judge(metrics, watch, facts, workload, screened, agree)
    assert values["slo_met"] == 1.0 and abs(values["slo_margin_s"] - 1.0) < 1e-9, values
    metrics["ttft_s"]["p99"] = 9.0
    try:
        sr.judge(metrics, watch, facts, workload, screened, agree)
    except sr.Rejected as exc:
        assert "p99 TTFT 9.00 s is over the 8 s rejection bound" in str(exc), exc
        assert exc.partial["name"] == "p99_ttft_s" and exc.partial["target"] == SMOKE_REJECT_S
        print(f"--- judge: rejected as expected: {exc}")
    else:
        raise AssertionError("judge accepted a p99 over the rejection bound")
    metrics["ttft_s"]["p99"] = 5.0
    for bad, needle in ((sr.Agreement(100, 100, 94, "x"), "disagree with the reference build"),
                        (sr.Agreement(29, 29, 29, None), "fewer than 30")):
        try:
            sr.judge(metrics, watch, facts, workload, screened, bad)
        except sr.Rejected as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"judge accepted {bad}")
    # The early-abort count is a property of the quantile and the ceiling, not of the bound.
    assert sr.rounds_over_bound_that_fail(41_444) == 416
    assert sr.WORKLOAD.reject_p99_ttft_s == 60.0
    print("--- judge: p99 7 s scored with slo_met 0, 5 s with slo_met 1, 9 s rejected; "
          "early-abort count for the real trace = 416 rounds over 60 s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--case", action="append", default=None)
    ns = parser.parse_args()
    cases = ns.case or ["judge", "record", "faked-replay", "truncation", "no-ids", "starvation", "cross-eval",
                        "crash", "pass", "slo", "abort", "failure", "shortfall", "no-cache", "slow-screen", "big-block"]
    root = Path(tempfile.mkdtemp(prefix="session_replay_smoke-"))
    sha = make_trace(root / "smoke.csv")
    print(f"smoke root {root}")
    for name in cases:
        if name == "judge":
            case_judge()
        elif name == "record":
            ref = root / "replay_reference.json"
            records = run_case(name, [], ns, root, sha, ["--record-replay-reference", str(ref)])
            assert [r["kind"] for r in records] == ["hello"], records  # not a scored run
            body = json.loads(ref.read_text())
            sample = smoke_workload(sha).replay_sample
            assert body["sampling"] == sample.describe(), body["sampling"]
            chosen = sr.sample_rounds(sr.ceiling_depths(sr.trace_sessions(root / "smoke.csv"), smoke_workload(sha)),
                                      sample)
            planned = {f"{sid}/{r}" for sid, n in chosen.items() for r in range(n)}
            assert set(body["rounds"]) == planned, sorted(set(body["rounds"]) ^ planned)
            assert body["recording"]["complete"] and body["recording"]["recorded_rounds"] == len(planned)
            print(f"--- record: {len(planned)} rounds of {len(chosen)} sampled sessions in the replay reference")
        elif name == "faked-replay":
            expect_error(run_case(name, ["--fake-replay"], ns, root, sha), "disagree with the reference build")
        elif name == "truncation":
            expect_error(run_case(name, ["--truncate-over", "9000"], ns, root, sha), "usage.prompt_tokens differs")
        elif name == "no-ids":
            expect_error(run_case(name, ["--text-only-after", "40"], ns, root, sha),
                         "streamed no generated token ids")
        elif name == "starvation":
            records = run_case(name, ["--hang-at", "40,45,50,55"], ns, root, sha)
            expect_error(records, "rejection bound")
            outstanding = int(records[-1]["message"].split(" outstanding rounds")[0].rsplit(" ", 1)[-1])
            assert outstanding >= 4, records[-1]
        elif name == "cross-eval":
            expect_error(run_case(name, [], ns, root, sha, runs=2), "cache state carried in from outside")
        elif name == "crash":
            out = root / name / "vs_output.jsonl"
            args = sr.parse_args(["--base-url", "http://127.0.0.1:1/v1", "--request-factory-engine", ns.engine,
                                  "--text-file", ns.text_file, "--asset-dir", str(root), "--vs-output", str(out),
                                  "--replay-reference", str(root / "replay_reference.json")])
            args.tokenizer = 12345  # Path(12345) raises TypeError inside run()
            assert sr.run(args, smoke_workload(sha), sr.ProtocolReport(out)) == 1
            records = [json.loads(line) for line in out.read_text().splitlines()]
            expect_error(records, "benchmark error: TypeError")
            print(f"--- crash: {records[-1]['message']}")
        elif name == "pass":
            records = run_case(name, [], ns, root, sha)
            assert [r["kind"] for r in records] == ["hello", "result"], records
            values = records[-1]["values"]
            assert set(values) == set(sr.METRICS)
            assert values["new_tokens_per_s"] > 0 and values["p99_ttft_s"] < 6
            assert values["slo_met"] == 1.0 and values["slo_margin_s"] > 0, values
            assert values["prefix_hit_share"] > 0.5, values
            assert values["replay_token_agreement"] == 1.0 and values["replay_matched_rounds"] >= 30, values
            assert values["replay_agreement_gate"] == 0.95 and values["replay_min_matched_rounds"] == 30, values
            assert values["outstanding_rounds"] <= 8, values
        elif name == "slo":
            records = run_case(name, ["--slow-every", "5", "--slow-ms", "6500"], ns, root, sha)
            assert [r["kind"] for r in records] == ["hello", "result"], records
            values = records[-1]["values"]
            assert values["p99_ttft_s"] > 6 and values["slo_met"] == 0.0, values
            assert abs(values["slo_margin_s"] - (6.0 - values["p99_ttft_s"])) < 1e-9, values
        elif name == "abort":
            records = run_case(name, ["--slow-every", "2", "--slow-ms", "9000"], ns, root, sha)
            expect_error(records, "run stopped early", "took over 8 s to the first token",
                         "rejection bound")
        elif name == "failure":
            expect_error(run_case(name, ["--fail-after", "40"], ns, root, sha), "a request failed")
        elif name == "shortfall":
            # The id-less round itself is caught (no token ids) unless its successor's
            # shortfall lands in the same 2 s poll; either rejection is right.
            records = run_case(name, ["--drop-token-ids-after", "40"], ns, root, sha)
            expect_error(records)
            assert any(k in records[-1]["message"] for k in ("prefix_shortfall_tokens > 0", "no generated token ids"))
        elif name == "no-cache":
            expect_error(run_case(name, ["--no-cache-report"], ns, root, sha), "cached_tokens")
        elif name == "big-block":
            expect_error(run_case(name, ["--block", "1536"], ns, root, sha), "preflight-probe.patch")
        elif name == "slow-screen":
            records = run_case(name, ["--base-ms", "9000"], ns, root, sha)
            expect_error(records, "screening")
            assert records[-1]["partial"]["name"] == "screen_ttft_16k_s"
        else:
            raise SystemExit(f"unknown case {name}")
    print("ALL SMOKE CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
