#!/usr/bin/env python3
"""20-minute coding-agent session replay for GLM-5.3-Flash NVFP4 prefill on 4 x B200.

The benchmark of `reference/1_contract.md` §2.2 at N = 500: the trace
`sessions_2ps.csv` (2,457 sessions arriving at 2/s, at most 500 active),
replayed closed-loop with request-factory's `session_runner` (rev 7f6aa9d),
token-ID prompts built from enwik9, temperature 0, one output token per round.
Scored with the handoff's `metrics.py` (copied verbatim next to this file).

Phases:
  1. Inputs. Verify the trace's sha256 and its manifest, enwik9's md5, the
     tokenizer revision, and that the server serves `--model`.
  2. Screening (seconds on a healthy server). A lone fresh 16k prompt twice,
     a session-shaped follow-up that must hit the prefix cache, and a lone 32k
     prompt. Rejects a server that cannot return token ids, miscounts the
     prompt, does not report cached tokens, or whose best lone 16k TTFT is over
     60 s, the rejection bound (README.md "Screening" says why that proves the
     run would be rejected).
  3. Replay. session_runner under a 20-minute window measured from the first
     request's submit (t = 0 of §3), plus a grace for in-flight responses.
     The request log is read while it grows; the run stops early when failure
     is already certain (README.md "Early abort").
  4. Score. `metrics.py` on the request log, then the gates: no failed
     request; on every successful round `prefix_shortfall_tokens` 0, streamed
     token ids, `usage.prompt_tokens` equal to the prompt sent, and no cache
     hit beyond the round's own prefix (none past one block on a round 0);
     first-token agreement with the recorded reference replay
     (`--replay-reference`); p99 TTFT <= 60 s, where rounds still unanswered at
     the window's end count at least their wait (`outstanding_rounds`).
     The 6 s SLO is not a gate: `slo_met` (1/0) and `slo_margin_s` (6 s minus
     p99) report it, and p99 TTFT is the primary objective.

Result: an evaluator result protocol v2 stream written to `--vs-output`
(VibeSys appends the flag): `hello`, then one `result` row, or one `error`
record (a rejected run) carrying what was measured before the stop.

The workload is fixed by this file's `WORKLOAD` and is not configurable from
the command line; tests construct their own `Workload` through `run()`.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import heapq
import http.client
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
METRICS_PY = HERE / "metrics.py"
PINNED_MANIFEST = HERE / "sessions_2ps.manifest.json"

MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
MODEL_REVISION = "da920bb0b9f4a06727223a349e55468e38352348"
ENWIK9_MD5 = "e206c3450ac99950df65bf70ef61a12d"
ENWIK9_BYTES = 1_000_000_000
TRACE_FORMAT = "text-generation-session-execution-v2"
MAX_MODEL_LEN = 1_048_576

# The contract's SLO. Reported (`slo_met`, `slo_margin_s`), not a gate.
SLO_P99_TTFT_S = 6.0
# A run whose p99 TTFT is over this is rejected (an `error`): hopeless, and the
# early abort saves its GPU time. WORKLOAD.reject_p99_ttft_s carries it.
REJECT_P99_TTFT_S = 60.0
# The handoff design's simulated result at N = 500 (reference/README.md, "Final: sessions").
SIM_TARGET_NEW_TOKENS_PER_S = 118_800.0
SIM_TARGET_P99_TTFT_S = 4.94

PROTOCOL_VERSION = 2
METRICS: dict[str, dict[str, Any]] = {
    "new_tokens_per_s": {"unit": "new prompt tokens/s", "direction": "max"},
    "p99_ttft_s": {"unit": "s", "direction": "min"},
    "slo_met": {"unit": "1 if p99 TTFT <= 6 s, else 0", "direction": "max"},
    "slo_margin_s": {"unit": "s (6 s minus p99 TTFT; negative = SLO missed)", "direction": "max"},
    "p50_ttft_s": {"unit": "s", "direction": "min"},
    "p90_ttft_s": {"unit": "s", "direction": "min"},
    "max_ttft_s": {"unit": "s", "direction": "min"},
    "rounds_per_s": {"unit": "rounds/s", "direction": "max"},
    "computed_tokens_per_s": {"unit": "prompt tokens computed/s"},
    "prefix_hit_share": {"unit": "fraction", "direction": "max"},
    "rounds_ttft_over_60s": {"unit": "rounds", "direction": "min"},
    "fraction_of_sim_target": {"unit": "fraction", "direction": "max"},
    "fraction_of_workload_ceiling": {"unit": "fraction", "direction": "max"},
    "screen_ttft_16k_s": {"unit": "s", "direction": "min"},
    "screen_ttft_32k_s": {"unit": "s", "direction": "min"},
    "replay_token_agreement": {"unit": "fraction of prompt-matched rounds whose first token equals the reference's",
                               "direction": "max"},
    "replay_matched_rounds": {"unit": "rounds"},
    "outstanding_rounds": {"unit": "rounds submitted before the window's end and not answered inside it",
                           "direction": "min"},
}

REPLAY_REFERENCE_FORMAT = "glm53-session-replay-reference-v1"


@dataclasses.dataclass(frozen=True)
class Workload:
    """Everything that defines the measured workload."""

    trace_name: str
    trace_sha256: str
    # The pinned manifest the asset dir's copy must equal; None skips the check.
    manifest: Path | None
    # enwik9's size and md5; None skips the check (tests use a small corpus).
    text_bytes: int | None
    text_md5: str | None
    max_concurrency: int
    window_s: float
    # Time after the window for responses already in flight to be logged. Rounds
    # whose first token arrives after the window are not counted by metrics.py.
    grace_s: float
    # Screening prompt lengths, in tokens.
    screen_gate_tokens: int
    screen_info_tokens: int
    screen_append_tokens: int
    # session_runner's prefix-cache preflight sends one probe of this many tokens
    # twice and aborts unless the second reports cached tokens (runner.rs,
    # PREFLIGHT_PROBE_TOKENS: 8,192 at rev 7f6aa9d).
    preflight_probe_tokens: int
    # session_runner must log its first round within this time of starting
    # (corpus tokenization and the prefix-cache preflight come first).
    startup_timeout_s: float
    # A run with rounds outstanding and nothing logged for this long has stalled.
    # session_runner's own per-request stream idle timeout is 600 s.
    stall_timeout_s: float
    request_timeout_s: float
    # p99 TTFT over this rejects the run; the early abort fires once it is certain,
    # and screening rejects a lone 16k prompt slower than this.
    reject_p99_ttft_s: float
    # The replay-reference check (README.md, "Replay answers"): over rounds whose
    # exact prompt matches the reference run's, at least this share of first
    # tokens must equal the reference's, and at least this many rounds must match.
    replay_min_agreement: float = 0.95
    replay_min_matched_rounds: int = 1_000
    # A round 0 carries no prefix of its own session, so a cache hit on it is
    # cache state carried in from outside this evaluation; one block of slack.
    round0_max_cached_tokens: int = 8_576


WORKLOAD = Workload(
    trace_name="sessions_2ps.csv",
    trace_sha256="a1a1e78b231b7e5bbe02b7e6db7a83cd85d2e9de98262b89a70263c274eb3e72",
    manifest=PINNED_MANIFEST,
    text_bytes=ENWIK9_BYTES,
    text_md5=ENWIK9_MD5,
    max_concurrency=500,
    window_s=1200.0,
    grace_s=15.0,
    screen_gate_tokens=16_384,
    screen_info_tokens=32_768,
    screen_append_tokens=1_024,
    preflight_probe_tokens=8_192,
    startup_timeout_s=900.0,
    stall_timeout_s=600.0,
    request_timeout_s=180.0,
    reject_p99_ttft_s=REJECT_P99_TTFT_S,
    replay_min_agreement=0.95,
    # Every admitted session's round 0 is built from the corpus alone, so its
    # prompt matches the reference's whatever the server answered before; each
    # later round matches until the session's first first-token flip. The
    # zero-latency replay admits 1,679 sessions in the window and counts 41,444
    # rounds, so an honest run matches tens of thousands; 1,000 still makes the
    # 5% tolerance about 50 rounds, and a server that answers the replay wrongly
    # matches little more than its round 0s.
    replay_min_matched_rounds=1_000,
    # vLLM's prefix-cache block for this model (KDA state plus attention), ~8.5k tokens.
    round0_max_cached_tokens=8_576,
)


class Rejected(Exception):
    """The run is rejected: an `error` record with an optional partial measurement."""

    def __init__(self, message: str, partial: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial = partial


# ----------------------------------------------------------------------------- output


class ProtocolReport:
    """The record stream at `--vs-output`; `hello` is written before measuring."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._write({"kind": "hello", "protocol": PROTOCOL_VERSION, "metrics": METRICS}, mode="w")

    def emit(self, values: dict[str, float]) -> None:
        missing = sorted(set(METRICS) - set(values))
        if missing:
            raise ValueError(f"result row lacks declared metrics {missing}")
        self._write({"kind": "result", "label": "", "values": {k: values[k] for k in METRICS}})

    def fail(self, message: str, partial: dict[str, Any] | None) -> None:
        record: dict[str, Any] = {"kind": "error", "message": message or "benchmark failed"}
        if partial is not None:
            record["partial"] = partial
        self._write(record)

    def _write(self, record: dict[str, Any], mode: str = "a") -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def log(message: str) -> None:
    print(f"[session_replay {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- inputs


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_trace(asset_dir: Path, workload: Workload) -> Path:
    trace = asset_dir / workload.trace_name
    if not trace.is_file():
        raise Rejected(f"--asset-dir {asset_dir} has no {workload.trace_name}")
    actual = file_digest(trace, "sha256")
    if actual != workload.trace_sha256:
        raise Rejected(f"{trace}: sha256 {actual}, expected {workload.trace_sha256}")
    manifest = workload.manifest
    if manifest is not None:
        shipped = asset_dir / manifest.name
        if not shipped.is_file():
            raise Rejected(f"--asset-dir {asset_dir} has no {manifest.name}")
        if json.loads(shipped.read_text()) != json.loads(manifest.read_text()):
            raise Rejected(f"{shipped} differs from the pinned {manifest}")
    return trace


def verify_text(path: Path, workload: Workload) -> None:
    if not path.is_file():
        raise Rejected(f"--text-file {path} does not exist")
    size = path.stat().st_size
    if workload.text_bytes is not None and size != workload.text_bytes:
        raise Rejected(f"--text-file {path}: {size} bytes, enwik9 has {workload.text_bytes}")
    if workload.text_md5 is not None:
        actual = file_digest(path, "md5")
        if actual != workload.text_md5:
            raise Rejected(f"--text-file {path}: md5 {actual}, enwik9 is {workload.text_md5}")


def resolve_tokenizer(explicit: Path | None) -> Path:
    """tokenizer.json of the pinned checkpoint revision in the Hugging Face cache."""
    if explicit is not None:
        if not explicit.is_file():
            raise Rejected(f"--tokenizer {explicit} is not a file")
        return explicit
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub is None:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        hub = str(Path(hf_home) / "hub")
    snapshot = Path(hub) / f"models--{MODEL.replace('/', '--')}" / "snapshots" / MODEL_REVISION
    candidate = snapshot / "tokenizer.json"
    if not candidate.is_file():
        raise Rejected(
            f"tokenizer.json of {MODEL}@{MODEL_REVISION} not found at {candidate}; "
            "pass --tokenizer or set HF_HOME / HF_HUB_CACHE"
        )
    return candidate


@dataclasses.dataclass(frozen=True)
class TraceFacts:
    rounds: int
    sessions: int
    ceiling_rounds: int
    ceiling_new_tokens: int


def trace_facts(trace: Path, workload: Workload) -> TraceFacts:
    """The trace's size and its zero-latency ceiling.

    The ceiling replays the trace's closed loop with every TTFT at zero: a
    session takes a slot at its arrival or when one frees (first come, first
    served), each round follows the previous one after its tool wait, and the
    slot frees after the last round's wait. Rounds submitted inside the window
    are the most any server can have counted: a positive TTFT only delays every
    later event. It bounds the counted rounds for the early-abort rule, and its
    new tokens are the throughput no candidate can exceed.
    """
    sessions: dict[str, list[tuple[int, float, float, int]]] = {}
    rounds = 0
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rounds += 1
            sessions.setdefault(row["session_id"], []).append(
                (
                    int(row["round_idx"]),
                    float(row["arrival_time_ms"]),
                    float(row["tool_wait_after_ms"]),
                    int(row["input_len"]),
                )
            )
    window_ms = workload.window_s * 1e3
    free = [0.0] * workload.max_concurrency
    ceiling_rounds = 0
    ceiling_new = 0
    # session_runner admits sessions in trace order (its AdmissionOrder).
    for steps in sessions.values():
        steps.sort()
        t = max(steps[0][1], heapq.heappop(free))
        for _, _, wait_ms, input_len in steps:
            if t < window_ms:
                ceiling_rounds += 1
                ceiling_new += input_len
            t += wait_ms
        heapq.heappush(free, t)
    return TraceFacts(rounds, len(sessions), ceiling_rounds, ceiling_new)


def rounds_over_bound_that_fail(max_rounds: int, quantile: float = 0.99) -> int:
    """Fewest rounds over a TTFT bound that put the p99 over it for any count <= max_rounds.

    Independent of the bound's value (used with the 60 s rejection bound).
    metrics.py takes numpy's linear-interpolation percentile over n counted
    TTFTs: position h = q (n - 1) of the sorted values. When at least
    n - floor(h) values exceed the bound, the value at floor(h) does, and so does
    the interpolated percentile. That count grows with n, so its value at the
    largest possible n holds for every smaller one.
    """
    n = max(max_rounds, 1)
    return n - int(quantile * (n - 1))


# ----------------------------------------------------------------------------- HTTP


@dataclasses.dataclass(frozen=True)
class Completion:
    ttft_s: float
    token_ids: list[int]
    prompt_tokens: int | None
    cached_tokens: int | None


def split_base_url(base_url: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise Rejected(f"--base-url must be http://host:port/v1, got {base_url!r}")
    return parsed.hostname, parsed.port or 80, parsed.path.rstrip("/")


def served_models(base_url: str, timeout_s: float) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout_s) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise Rejected(f"GET {base_url}/models failed: {exc}") from exc
    return [entry.get("id") for entry in body.get("data", [])]


def stream_completion(base_url: str, model: str, prompt: list[int], timeout_s: float) -> Completion:
    """One streamed /completions request with token-ID prompt, 1 output token, temperature 0.

    The same request shape session_runner sends (backend `openai`).
    """
    host, port, path = split_base_url(base_url)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        start = time.perf_counter()
        conn.request(
            "POST", f"{path}/completions", body=json.dumps(payload), headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise Rejected(f"POST {path}/completions: HTTP {resp.status}: {resp.read()[:300]!r}")
        ttft: float | None = None
        token_ids: list[int] = []
        usage: dict[str, Any] | None = None
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            for choice in event.get("choices") or []:
                ids = choice.get("token_ids") or []
                if ids and ttft is None:
                    ttft = time.perf_counter() - start
                token_ids.extend(int(i) for i in ids)
            if event.get("usage"):
                usage = event["usage"]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise Rejected(f"screening request failed: {exc!r}") from exc
    finally:
        conn.close()
    if ttft is None or not token_ids:
        raise Rejected(
            "screening: the server returned no generated token ids; it must honour "
            "`return_token_ids` on streamed /v1/completions"
        )
    details = (usage or {}).get("prompt_tokens_details") or {}
    return Completion(
        ttft_s=ttft,
        token_ids=token_ids,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        cached_tokens=details.get("cached_tokens"),
    )


# ----------------------------------------------------------------------------- screening


def fresh_prompt(rng: random.Random, length: int) -> list[int]:
    # Ordinary vocabulary ids, clear of the special tokens at the top of the vocabulary.
    return [rng.randrange(1_000, 150_000) for _ in range(length)]


def screen(base_url: str, model: str, workload: Workload) -> dict[str, float]:
    """A few lone prompts on the idle server; rejects a server that cannot pass the run."""
    # Seeded by the clock: no screening prompt repeats across evaluations or
    # shares a prefix with the session prompts.
    rng = random.Random(time.time_ns())
    gate, info, append = workload.screen_gate_tokens, workload.screen_info_tokens, workload.screen_append_tokens
    timeout = workload.request_timeout_s

    first_prompt = fresh_prompt(rng, gate)
    first = stream_completion(base_url, model, first_prompt, timeout)
    if first.prompt_tokens != gate:
        raise Rejected(f"screening: usage.prompt_tokens {first.prompt_tokens} for a {gate}-token prompt")
    log(f"screen: lone {gate}-token prompt, TTFT {first.ttft_s:.3f} s")

    # Round 1 of a session: the previous prompt, its output token, then new tokens.
    follow = first_prompt + first.token_ids[:1] + fresh_prompt(rng, append - 1)
    second = stream_completion(base_url, model, follow, timeout)
    if second.cached_tokens is None:
        raise Rejected(
            "screening: the server reports no usage.prompt_tokens_details.cached_tokens "
            "(vLLM: --enable-prompt-tokens-details); the run needs it"
        )
    if not 0 < second.cached_tokens <= gate:
        raise Rejected(
            f"screening: a follow-up round on a {gate}-token prefix reported {second.cached_tokens} "
            f"cached tokens; expected a prefix-cache hit in (0, {gate}]"
        )
    log(f"screen: session follow-up, {second.cached_tokens} cached tokens, TTFT {second.ttft_s:.3f} s")

    # The same check session_runner's preflight makes, so its failure has a precise reason.
    probe = fresh_prompt(rng, workload.preflight_probe_tokens)
    stream_completion(base_url, model, probe, timeout)
    repeat = stream_completion(base_url, model, probe, timeout)
    if not repeat.cached_tokens:
        raise Rejected(
            f"screening: an identical {len(probe)}-token prompt sent twice reported no cached tokens the "
            f"second time, while a session follow-up on a {gate}-token prefix hit "
            f"{second.cached_tokens}. The prefix cache resumes only at blocks longer than "
            f"{len(probe)} tokens, so session_runner's prefix-cache preflight ({len(probe)}-token probe) "
            "will abort the replay. Either serve hits below that block size, or build session_runner "
            "with benchmark/request-factory-preflight-probe.patch (benchmark/README.md)."
        )
    log(f"screen: repeated {len(probe)}-token probe, {repeat.cached_tokens} cached tokens")

    third = stream_completion(base_url, model, fresh_prompt(rng, gate), timeout)
    best_gate = min(first.ttft_s, third.ttft_s)
    log(f"screen: second lone {gate}-token prompt, TTFT {third.ttft_s:.3f} s")
    bound = workload.reject_p99_ttft_s
    if best_gate > bound:
        raise Rejected(
            f"screening: the best of two lone {gate}-token prompts on the idle server took "
            f"{best_gate:.2f} s to the first token, above the {bound:.0f} s p99 bound that rejects a run. "
            f"Over 1% of the run's rounds compute at least {gate} new tokens under load, so the run's "
            "p99 would exceed it too (benchmark/README.md, Screening). The server is broken or far "
            "too slow (a healthy build takes about 0.4 s).",
            {"name": "screen_ttft_16k_s", "value": best_gate, "direction": "min", "unit": "s",
             "target": bound},
        )
    fourth = stream_completion(base_url, model, fresh_prompt(rng, info), timeout)
    log(f"screen: lone {info}-token prompt, TTFT {fourth.ttft_s:.3f} s")
    return {"screen_ttft_16k_s": best_gate, "screen_ttft_32k_s": fourth.ttft_s}


# ----------------------------------------------------------------------------- replay


@dataclasses.dataclass
class LogWatch:
    """Folds the growing request log; decides when failure is already certain."""

    over_bound_limit: int
    window_s: float
    ttft_bound_s: float
    rounds: int = 0
    successes: int = 0
    failures: int = 0
    shortfall_rounds: int = 0
    over_bound: int = 0
    min_submit: float | None = None
    first_failure: str | None = None
    first_shortfall: str | None = None
    new_tokens: int = 0
    # "<session_id>/<round_idx>" -> (prompt hash, first output token id) of each
    # successful round, for the replay-reference check and for recording one.
    answers: dict[str, tuple[str, int]] = dataclasses.field(default_factory=dict)
    # Successful rounds the patched session_runner should have hashed but did not.
    unhashed_rounds: int = 0
    # Successful rounds that break a correctness rule: reason -> (count, first example).
    invalid: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    round0_max_cached_tokens: int = 0
    _buffer: str = ""
    _offset: int = 0

    def poll(self, path: Path) -> int:
        """Read lines appended since the last poll; return how many records were new."""
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line.strip():
                self.add(json.loads(line))
        return len([line for line in lines if line.strip()])

    def add(self, record: dict[str, Any]) -> None:
        self.rounds += 1
        outcome = record.get("outcome", {})
        source = (record.get("source") or {}).get("data") or {}
        submit = outcome.get("submit_timestamp")
        if submit is not None:
            self.min_submit = submit if self.min_submit is None else min(self.min_submit, submit)
        if str(outcome.get("status")).upper() != "SUCCESS":
            self.failures += 1
            if self.first_failure is None:
                self.first_failure = (
                    f"{outcome.get('request_id')}: status {outcome.get('status')}: {outcome.get('error')}"
                )
        else:
            self.successes += 1
            self.new_tokens += int(source.get("input_len") or 0)
            self._check_round(outcome, source)
            prompt_hash = outcome.get("prompt_fnv1a64")
            first_id = outcome.get("first_output_token_id")
            if prompt_hash is None:
                self.unhashed_rounds += 1
            elif first_id is not None:
                self.answers[f"{source.get('session_id')}/{source.get('round_idx')}"] = (prompt_hash, int(first_id))
            ttft_ms = outcome.get("first_token_id_ms")
            if ttft_ms is not None and outcome.get("post_timestamp") is not None:
                first_token = outcome["post_timestamp"] + ttft_ms / 1e3
                # Only rounds metrics.py can count: first token inside the window.
                counted = self.min_submit is not None and first_token < self.min_submit + self.window_s
                if counted and first_token - submit > self.ttft_bound_s:
                    self.over_bound += 1
        shortfall = int(source.get("prefix_shortfall_tokens") or 0)
        if shortfall > 0:
            self.shortfall_rounds += 1
            if self.first_shortfall is None:
                self.first_shortfall = f"{outcome.get('request_id')}: {shortfall} tokens"

    def _invalid(self, reason: str, example: str) -> None:
        entry = self.invalid.setdefault(reason, [0, example])
        entry[0] += 1

    def _check_round(self, outcome: dict[str, Any], source: dict[str, Any]) -> None:
        """The per-round correctness rules of a successful round (README.md, "Scoring and gates")."""
        rid = outcome.get("request_id")
        if outcome.get("first_token_id_ms") is None or outcome.get("first_output_token_id", 0) is None:
            # session_runner then carries re-encoded text forward, so no shortfall shows.
            self._invalid("streamed no generated token ids (`return_token_ids`); its TTFT is unmeasured",
                          f"{rid}")
        usage = outcome.get("server_usage") or {}
        expected = int(source.get("derived_prefix_len") or 0) + int(source.get("derived_append_len") or 0)
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens != expected:
            self._invalid("usage.prompt_tokens differs from the prompt sent (truncated or miscounted)",
                          f"{rid}: {prompt_tokens}, sent {expected}")
        cached = usage.get("cached_prompt_tokens") or 0
        prefix = int(source.get("derived_prefix_len") or 0)
        if int(source.get("round_idx") or 0) == 0:
            if cached > self.round0_max_cached_tokens:
                self._invalid(
                    f"a session's round 0 hit more than {self.round0_max_cached_tokens} cached tokens: cache state "
                    "carried in from outside this evaluation (each evaluation must start with an empty cache)",
                    f"{rid}: {cached} cached")
        elif cached > prefix + 1:
            self._invalid("more cached tokens than the round's reusable prefix: cache state carried in from "
                          "outside this evaluation", f"{rid}: {cached} cached, prefix {prefix}")

    def verdict(self) -> str | None:
        """Why the run has already failed, or None while it can still pass."""
        for reason, (count, example) in self.invalid.items():
            return f"{count} successful round(s) {reason}; first: {example}"
        if self.unhashed_rounds:
            return (
                f"{self.unhashed_rounds} round(s) logged without `prompt_fnv1a64`: --request-factory-engine "
                "must be built with benchmark/request-factory-preflight-probe.patch"
            )
        if self.failures:
            return f"a request failed ({self.failures} so far); first: {self.first_failure}"
        if self.shortfall_rounds:
            return (
                f"prefix_shortfall_tokens > 0 on {self.shortfall_rounds} round(s); first: "
                f"{self.first_shortfall}. A round's prompt must carry the previous round's real "
                "output token (`return_token_ids`)"
            )
        if self.over_bound >= self.over_bound_limit:
            return (
                f"{self.over_bound} rounds already took over {self.ttft_bound_s:.0f} s to the first token; "
                f"with at most the zero-latency ceiling of rounds counted, {self.over_bound_limit} such "
                f"rounds put the p99 TTFT over the {self.ttft_bound_s:.0f} s rejection bound"
            )
        return None


# ----------------------------------------------------------------------------- replay reference


def write_replay_reference(path: Path, watch: LogWatch, workload: Workload, model: str) -> None:
    """The reference run's answers: prompt hash and first token per session round."""
    body = {
        "format": REPLAY_REFERENCE_FORMAT,
        "model": model,
        "trace_sha256": workload.trace_sha256,
        "recorded_at_unix_s": time.time(),
        "rounds": {key: [h, token] for key, (h, token) in sorted(watch.answers.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body) + "\n")
    tmp.replace(path)
    log(f"replay reference: {len(watch.answers)} rounds written to {path}")


def load_replay_reference(path: Path, workload: Workload, model: str) -> dict[str, tuple[str, int]]:
    if not path.is_file():
        raise Rejected(f"--replay-reference {path} does not exist (record it with --record-replay-reference "
                       "against the reference build; README.md, Replay answers)")
    try:
        body = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Rejected(f"--replay-reference {path}: not JSON: {exc}") from exc
    if body.get("format") != REPLAY_REFERENCE_FORMAT:
        raise Rejected(f"--replay-reference {path}: format {body.get('format')!r}, expected {REPLAY_REFERENCE_FORMAT!r}")
    if body.get("trace_sha256") != workload.trace_sha256:
        raise Rejected(f"--replay-reference {path} was recorded on trace {body.get('trace_sha256')}, "
                       f"this workload is {workload.trace_sha256}")
    if body.get("model") != model:
        raise Rejected(f"--replay-reference {path} was recorded for model {body.get('model')!r}, not {model!r}")
    return {key: (str(h), int(token)) for key, (h, token) in body["rounds"].items()}


@dataclasses.dataclass(frozen=True)
class Agreement:
    matched: int
    agreed: int
    first_disagreement: str | None

    @property
    def share(self) -> float:
        return self.agreed / self.matched if self.matched else 0.0


def replay_agreement(answers: dict[str, tuple[str, int]], reference: dict[str, tuple[str, int]]) -> Agreement:
    """First-token agreement over rounds whose exact prompt the reference run also sent.

    One differing token changes every later prompt of that session, so those
    rounds have no reference answer and are left out rather than counted wrong.
    """
    matched = agreed = 0
    first: str | None = None
    for key, (prompt_hash, token) in answers.items():
        expected = reference.get(key)
        if expected is None or expected[0] != prompt_hash:
            continue
        matched += 1
        if token == expected[1]:
            agreed += 1
        elif first is None:
            first = f"{key}: token {token}, reference {expected[1]}"
    return Agreement(matched, agreed, first)


def check_agreement(agreement: Agreement, workload: Workload) -> None:
    detail = (f"{agreement.agreed} of {agreement.matched} prompt-matched rounds gave the reference's first token "
              f"({agreement.share:.1%}); first disagreement: {agreement.first_disagreement}")
    partial = {"name": "replay_token_agreement", "value": agreement.share, "direction": "max",
               "unit": "fraction", "target": workload.replay_min_agreement}
    if agreement.share < workload.replay_min_agreement:
        raise Rejected(
            f"run rejected: replay answers disagree with the reference build: {detail}; at least "
            f"{workload.replay_min_agreement:.0%} must agree (README.md, Replay answers)", partial)
    if agreement.matched < workload.replay_min_matched_rounds:
        raise Rejected(
            f"run rejected: only {agreement.matched} rounds sent the same prompt as the reference run, fewer "
            f"than {workload.replay_min_matched_rounds}: the replay answers cannot be checked. {detail}", partial)
    log(f"replay reference: {detail}")


def session_runner_argv(
    engine: Path, trace: Path, text_file: Path, tokenizer: Path, base_url: str, model: str,
    workload: Workload, work_dir: Path,
) -> list[str]:
    """The replay of reference/1_contract.md §2.2 as session_runner flags."""
    return [
        str(engine),
        "--trace", str(trace),
        "--input-file-format", TRACE_FORMAT,
        "--text-file", str(text_file),
        "--tokenizer", str(tokenizer),
        "--base-url", base_url,
        "--model", model,
        "--backend", "openai",
        "--temperature", "0",
        "--arrival-mode", "trace-timed",
        "--max-concurrency", str(workload.max_concurrency),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--log-path", str(work_dir / "requests.jsonl"),
        "--summary-path", str(work_dir / "session_runner_summary.json"),
        "--timeline", "false",
        "--timeline-path", str(work_dir / "timeline.parquet"),
    ]


def stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def stderr_tail(path: Path, limit: int = 3000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="replace")


def progress_partial(watch: LogWatch, facts: TraceFacts, elapsed_s: float) -> dict[str, Any] | None:
    if watch.successes == 0 or elapsed_s <= 0:
        return None
    return {
        "name": "new_tokens_per_s_so_far",
        "value": watch.new_tokens / elapsed_s,
        "direction": "max",
        "unit": "new prompt tokens/s",
        "target": SIM_TARGET_NEW_TOKENS_PER_S,
        "progress": {"completed": min(watch.successes, facts.ceiling_rounds),
                     "required": max(facts.ceiling_rounds, 1), "unit": "rounds"},
    }


def replay(argv: list[str], work_dir: Path, workload: Workload, facts: TraceFacts,
           poll_s: float = 2.0, early_abort: bool = True) -> LogWatch:
    """Run session_runner for the window and return the folded request log.

    `early_abort=False` (recording a reference) keeps running past a certain p99
    rejection; correctness failures still stop the run.
    """
    log_path = work_dir / "requests.jsonl"
    err_path = work_dir / "session_runner.stderr"
    limit = rounds_over_bound_that_fail(facts.ceiling_rounds) if early_abort else sys.maxsize
    watch = LogWatch(over_bound_limit=limit, window_s=workload.window_s, ttft_bound_s=workload.reject_p99_ttft_s,
                     round0_max_cached_tokens=workload.round0_max_cached_tokens)
    log(f"replay: {' '.join(argv)}")
    started = time.time()
    last_record = started
    with err_path.open("wb") as err, (work_dir / "session_runner.stdout").open("wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
    try:
        while True:
            exited = proc.poll() is not None
            new = watch.poll(log_path)
            now = time.time()
            if new:
                last_record = now
            why = watch.verdict()
            if why is not None:
                elapsed = now - (watch.min_submit or started)
                raise Rejected(f"run stopped early: {why}", progress_partial(watch, facts, elapsed))
            if watch.min_submit is None:
                if exited:
                    raise Rejected(
                        f"session_runner exited with code {proc.returncode} before any round "
                        f"completed:\n{stderr_tail(err_path)}"
                    )
                if now - started > workload.startup_timeout_s:
                    raise Rejected(
                        f"no round completed within {workload.startup_timeout_s:.0f} s of starting "
                        f"session_runner:\n{stderr_tail(err_path)}"
                    )
            else:
                if now >= watch.min_submit + workload.window_s + workload.grace_s:
                    log(f"replay: window closed after {now - watch.min_submit:.0f} s; stopping session_runner")
                    break
                if exited and proc.returncode == 0:
                    log("replay: session_runner finished the whole workload inside the window")
                    break
                if exited:
                    raise Rejected(
                        f"session_runner exited with code {proc.returncode} "
                        f"{now - watch.min_submit:.0f} s into the {workload.window_s:.0f} s window:\n"
                        f"{stderr_tail(err_path)}",
                        progress_partial(watch, facts, now - watch.min_submit),
                    )
                if now - last_record > workload.stall_timeout_s:
                    raise Rejected(
                        f"no round completed for {workload.stall_timeout_s:.0f} s; the server stalled",
                        progress_partial(watch, facts, now - watch.min_submit),
                    )
            time.sleep(poll_s)
    finally:
        stop(proc)
    watch.poll(log_path)
    return watch


def percentile(values: list[float], q: float) -> float:
    """numpy's default (linear interpolation) percentile, as metrics.py takes it."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = int(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


@dataclasses.dataclass(frozen=True)
class SessionPlan:
    session_id: str
    arrival_ms: float
    tool_waits_ms: tuple[float, ...]


def session_plans(trace: Path) -> list[SessionPlan]:
    """The trace's sessions in file order (session_runner's admission order)."""
    rows: dict[str, list[tuple[int, float, float]]] = {}
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["session_id"], []).append(
                (int(row["round_idx"]), float(row["arrival_time_ms"]), float(row["tool_wait_after_ms"])))
    plans = []
    for sid, steps in rows.items():
        steps.sort()
        plans.append(SessionPlan(sid, steps[0][1], tuple(wait for _, _, wait in steps)))
    return plans


def outstanding_waits(records: list[dict[str, Any]], plans: list[SessionPlan], window_s: float,
                      max_concurrency: int) -> list[float]:
    """A lower bound on the TTFT of every round submitted before the window's end and not answered in it.

    metrics.py counts only rounds whose first token is inside the window, so a
    round still waiting at the end would vanish from the percentiles and a
    server could starve its slowest rounds. Each such round counts at least
    (window end - its submit), or its real TTFT when it was logged later:
      - logged rounds submitted in the window whose first token came after it;
      - a logged session's next round, submitted (estimate) at its last logged
        round's completion plus that round's tool wait;
      - a session with no logged round that session_runner had admitted by the
        window's end: admission replayed in trace order over `max_concurrency`
        slots from the logged sessions' real admissions and releases. An
        unlogged session ahead of a logged one was admitted no later than it
        (admission is first come, first served).
    """
    rounds = [r for r in records if (r.get("source") or {}).get("type") == "session_round"]
    if not rounds:
        return []
    t0 = min(r["outcome"]["submit_timestamp"] for r in records)
    end = t0 + window_s
    waits: list[float] = []
    by_session: dict[str, dict[int, dict[str, Any]]] = {}
    for record in rounds:
        source, outcome = record["source"]["data"], record["outcome"]
        by_session.setdefault(str(source["session_id"]), {})[int(source["round_idx"])] = outcome
        submit = outcome["submit_timestamp"]
        post, ttft_ms = outcome.get("post_timestamp"), outcome.get("first_token_id_ms")
        first = post + ttft_ms / 1e3 if post is not None and ttft_ms is not None else None
        if submit < end and (first is None or first >= end):
            waits.append(max(first - submit, end - submit) if first is not None else end - submit)
    arrival0 = min((plan.arrival_ms for plan in plans), default=0.0)
    # Each known session's admission (its round 0 submit) and slot release.
    admission: dict[str, float] = {}
    release: dict[str, float] = {}
    for plan in plans:
        logged = by_session.get(plan.session_id)
        if not logged:
            continue
        admission[plan.session_id] = min(o["submit_timestamp"] for o in logged.values())
        last = max(logged)
        free_at = logged[last]["complete_timestamp"] + plan.tool_waits_ms[last] / 1e3
        if last + 1 < len(plan.tool_waits_ms):
            if free_at < end:
                waits.append(end - free_at)
            release[plan.session_id] = float("inf")
        else:
            release[plan.session_id] = free_at
    # Latest admission any unlogged session ahead of index i can have had.
    cap = [float("inf")] * (len(plans) + 1)
    for i in range(len(plans) - 1, -1, -1):
        cap[i] = min(cap[i + 1], admission.get(plans[i].session_id, float("inf")))
    slots = [t0] * max_concurrency
    for i, plan in enumerate(plans):
        slot = heapq.heappop(slots)
        if plan.session_id in admission:
            heapq.heappush(slots, release[plan.session_id])
            continue
        admitted = min(max(t0 + (plan.arrival_ms - arrival0) / 1e3, slot), cap[i + 1])
        if admitted >= end:
            break
        waits.append(end - admitted)
        heapq.heappush(slots, float("inf"))
    return waits


def ttft_with_outstanding(log_path: Path, plans: list[SessionPlan], workload: Workload,
                          metrics: dict[str, Any]) -> tuple[dict[str, float], int]:
    """TTFT p50/p90/p99/max over the counted rounds plus the outstanding ones, and their count.

    With no outstanding round this is metrics.py's TTFT block, checked here.
    """
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    t0 = min(r["outcome"]["submit_timestamp"] for r in records)
    counted = []
    for record in records:
        o = record["outcome"]
        if (o.get("first_token_id_ms") is None or o.get("post_timestamp") is None
                or str(o.get("status")).upper() != "SUCCESS"):
            continue
        first = o["post_timestamp"] + o["first_token_id_ms"] / 1e3
        if first - t0 < workload.window_s:
            counted.append(first - o["submit_timestamp"])
    waits = outstanding_waits(records, plans, workload.window_s, workload.max_concurrency)
    every = counted + waits
    ttft = {"p50": percentile(every, 50), "p90": percentile(every, 90), "p99": percentile(every, 99),
            "max": max(every)}
    if not waits:
        for key, value in ttft.items():
            if abs(value - metrics["ttft_s"][key]) > 1e-3:
                raise Rejected(f"internal: TTFT {key} {value} differs from metrics.py's {metrics['ttft_s'][key]}")
    return ttft, len(waits)


def score(log_path: Path, workload: Workload) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(METRICS_PY), str(log_path), "--minutes", str(workload.window_s / 60)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise Rejected(f"metrics.py failed (exit {result.returncode}): {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def judge(metrics: dict[str, Any], watch: LogWatch, facts: TraceFacts, workload: Workload,
          screened: dict[str, float], agreement: Agreement,
          ttft: dict[str, float] | None = None, outstanding: int = 0) -> dict[str, float]:
    """The run's result row, or `Rejected` when a gate fails.

    Gates: no failed request, `prefix_shortfall_tokens` 0 on every round, at
    least one counted round, and p99 TTFT within the rejection bound (60 s; the
    same rule the early abort enforces, so the outcome does not depend on when
    it is detected). The 6 s SLO is reported, not gated. `ttft` (default:
    metrics.py's) includes the `outstanding` rounds unanswered at the window's end.
    """
    why = watch.verdict()
    if why is not None:
        raise Rejected(f"run rejected: {why}", progress_partial(watch, facts, workload.window_s))
    if metrics["rounds"] == 0:
        raise Rejected("no round was counted in the window")
    check_agreement(agreement, workload)
    ttft = ttft if ttft is not None else metrics["ttft_s"]
    p99 = ttft["p99"]
    bound = workload.reject_p99_ttft_s
    if p99 > bound:
        raise Rejected(
            f"run rejected: p99 TTFT {p99:.2f} s is over the {bound:.0f} s rejection bound "
            f"(new tokens/s {metrics['new_tokens_per_s']:.0f}, p50/p90 {ttft['p50']:.2f}/"
            f"{ttft['p90']:.2f} s, {metrics['rounds']} rounds counted, {outstanding} outstanding rounds "
            "counted at least their wait at the window's end)",
            {"name": "p99_ttft_s", "value": p99, "direction": "min", "unit": "s", "target": bound,
             "progress": {"completed": min(metrics["rounds"], facts.ceiling_rounds),
                          "required": max(facts.ceiling_rounds, 1), "unit": "rounds"}},
        )
    return {
        "new_tokens_per_s": metrics["new_tokens_per_s"],
        "p99_ttft_s": p99,
        "slo_met": 1.0 if p99 <= SLO_P99_TTFT_S else 0.0,
        "slo_margin_s": SLO_P99_TTFT_S - p99,
        "p50_ttft_s": ttft["p50"],
        "p90_ttft_s": ttft["p90"],
        "max_ttft_s": ttft["max"],
        "rounds_per_s": metrics["rounds_per_s"],
        "computed_tokens_per_s": metrics["computed_tokens_per_s"],
        "prefix_hit_share": metrics["prefix_hit_share"],
        "rounds_ttft_over_60s": float(metrics["rounds_ttft_over_60s"]),
        "fraction_of_sim_target": metrics["new_tokens_per_s"] / SIM_TARGET_NEW_TOKENS_PER_S,
        "fraction_of_workload_ceiling": metrics["new_tokens_per_s"]
        / max(facts.ceiling_new_tokens / workload.window_s, 1.0),
        "replay_token_agreement": agreement.share,
        "replay_matched_rounds": float(agreement.matched),
        "outstanding_rounds": float(outstanding),
        **screened,
    }


# ----------------------------------------------------------------------------- main


def run(args: argparse.Namespace, workload: Workload, report: ProtocolReport) -> int:
    work_dir = Path(args.work_dir) if args.work_dir else (
        args.vs_output.parent / "session_replay" if args.vs_output else Path(tempfile.mkdtemp(prefix="session_replay-"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work_dir}")
    if args.preflight_probe_tokens is not None:
        # A property of the session_runner build, not of the workload.
        workload = dataclasses.replace(workload, preflight_probe_tokens=args.preflight_probe_tokens)
    try:
        engine = Path(args.request_factory_engine)
        if not os.access(engine, os.X_OK):
            raise Rejected(f"--request-factory-engine {engine} is not an executable")
        trace = verify_trace(Path(args.asset_dir), workload)
        verify_text(Path(args.text_file), workload)
        tokenizer = resolve_tokenizer(Path(args.tokenizer) if args.tokenizer else None)
        facts = trace_facts(trace, workload)
        log(
            f"trace: {facts.sessions} sessions, {facts.rounds} rounds; zero-latency ceiling in the window: "
            f"{facts.ceiling_rounds} rounds, {facts.ceiling_new_tokens / workload.window_s:.0f} new tokens/s; "
            f"early abort at {rounds_over_bound_that_fail(facts.ceiling_rounds)} rounds over "
            f"{workload.reject_p99_ttft_s:.0f} s"
        )
        models = served_models(args.base_url, workload.request_timeout_s)
        if args.model not in models:
            raise Rejected(f"the server serves {models}, not --model {args.model!r}")

        recording = args.record_replay_reference is not None
        if recording == (args.replay_reference is not None):
            raise Rejected("pass exactly one of --replay-reference PATH (a scored run) and "
                           "--record-replay-reference OUT (recording on the reference build)")
        reference = None if recording else load_replay_reference(Path(args.replay_reference), workload, args.model)

        screened = screen(args.base_url, args.model, workload)

        argv = session_runner_argv(
            engine, trace, Path(args.text_file), tokenizer, args.base_url, args.model, workload, work_dir
        )
        watch = replay(argv, work_dir, workload, facts, early_abort=not recording)
        if recording:
            # Correctness gates only: a reference must not record broken rounds,
            # but a slow reference build still records valid answers.
            why = watch.verdict()
            if why is not None:
                raise Rejected(f"recording rejected, nothing written: {why}")
            write_replay_reference(Path(args.record_replay_reference), watch, workload, args.model)
            agreement = Agreement(len(watch.answers), len(watch.answers), None)
            workload = dataclasses.replace(workload, replay_min_matched_rounds=0)
        else:
            assert reference is not None
            agreement = replay_agreement(watch.answers, reference)
        metrics = score(work_dir / "requests.jsonl", workload)
        (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        log(f"metrics.py: {json.dumps(metrics)}")

        ttft, outstanding = ttft_with_outstanding(work_dir / "requests.jsonl", session_plans(trace), workload,
                                                  metrics)
        log(f"TTFT with {outstanding} outstanding rounds: {json.dumps(ttft)}")
        values = judge(metrics, watch, facts, workload, screened, agreement, ttft, outstanding)
        report.emit(values)
        print(json.dumps(values, indent=2))
        return 0
    except Rejected as exc:
        log(f"REJECTED: {exc}")
        report.fail(str(exc), exc.partial)
        return 1
    except Exception as exc:  # noqa: BLE001 - every exit must leave an outcome record (no NO_OUTCOME)
        log(f"REJECTED (unexpected error): {exc!r}")
        report.fail(f"benchmark error: {type(exc).__name__}: {exc}", None)
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL, http://127.0.0.1:PORT/v1")
    parser.add_argument("--model", default=MODEL, help="served model name (vibesys_serve.sh sets it)")
    parser.add_argument("--request-factory-engine", required=True,
                        help="session_runner built from request-factory 7f6aa9d8e36f3b81ae1f6c76c69f460f95f9c84c")
    parser.add_argument("--text-file", required=True, help="enwik9 (1,000,000,000 bytes)")
    parser.add_argument("--asset-dir", required=True,
                        help="directory holding sessions_2ps.csv and sessions_2ps.manifest.json")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer.json; default: the pinned revision in the Hugging Face cache")
    parser.add_argument("--work-dir", default=None,
                        help="where the request log and stderr go; default: next to --vs-output")
    parser.add_argument("--preflight-probe-tokens", type=int, default=None,
                        help="probe length of the session_runner build: 8192 at rev 7f6aa9d (default), "
                             "32768 with benchmark/request-factory-preflight-probe.patch")
    parser.add_argument("--replay-reference", default=None,
                        help="operator path of the reference build's replay answers (made with "
                             "--record-replay-reference); required for a scored run")
    parser.add_argument("--record-replay-reference", default=None, metavar="OUT",
                        help="recording mode, on the reference build only: run the replay and write its "
                             "per-round prompt hashes and first tokens to OUT")
    parser.add_argument("--vs-output", type=Path, default=None, help="result protocol v2 stream (VibeSys)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = ProtocolReport(args.vs_output)
    return run(args, WORKLOAD, report)


if __name__ == "__main__":
    sys.exit(main())
