# Benchmark: 20-minute session replay

`session_replay.py` measures the session benchmark of `reference/1_contract.md`
§2.2 at N = 500 and reports it in the VibeSys evaluator result protocol
(version 2). `metrics.py` is the handoff's scorer, copied unchanged;
`sessions_2ps.manifest.json` is the handoff's trace manifest, the pin the
operator's copy must equal.

## Interface

```bash
python session_replay.py \
  --base-url http://127.0.0.1:PORT/v1 \
  --request-factory-engine <session_runner built at request-factory 7f6aa9d8e36f3b81ae1f6c76c69f460f95f9c84c> \
  --text-file <enwik9> \
  --asset-dir <dir with sessions_2ps.csv and sessions_2ps.manifest.json> \
  [--model nvidia/GLM-5.3-Flash-NVFP4] [--tokenizer <tokenizer.json>] [--work-dir DIR] \
  --preflight-probe-tokens 32768 \
  --vs-output <path>        # appended by VibeSys
```

- The server must already be running. The script never starts one.
- `--tokenizer` defaults to `tokenizer.json` of the pinned revision in the
  Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
  `~/.cache/huggingface/hub`).
- Checked before anything is sent: the trace's sha256
  (`a1a1e78b…eb3e72`), the operator's manifest equal to the pinned one,
  enwik9's size and md5 (`e206c345…a12d`), and that `/v1/models` lists `--model`.
- `--preflight-probe-tokens 32768` is the operator setting: session_runner is
  built with `request-factory-preflight-probe.patch` ("The preflight probe"
  below). Without the flag the script assumes the unpatched 8,192-token probe.
- `--work-dir` (default: `session_replay/` next to `--vs-output`) receives
  `requests.jsonl` (req-frontend's per-request log), `session_runner.stderr`,
  the session_runner summary, and `metrics.json`.
- Exit 0 with a `result` row; exit 1 with an `error` record.

## The replay

session_runner flags, mapped from the contract's YAML:
`--input-file-format text-generation-session-execution-v2 --backend openai
--temperature 0 --arrival-mode trace-timed --max-concurrency 500
--max-model-len 1048576`, token pool at session_runner's default (100M tokens
of enwik9), `--timeline false` (no per-event Parquet; the request log carries
every number `metrics.py` reads). session_runner runs its own prefix-cache
preflight (an 8,192-token probe sent twice) before the first session.

**The 20-minute window.** The contract runs the client under `timeout 20m`
and counts rounds whose first token arrives in [0, 20 min) from the first
request's submit. `timeout 20m` would also spend corpus tokenization and the
preflight (about 1-3 minutes) out of the 20 minutes. This script instead
stops session_runner 20 minutes plus 15 s after the first submit, read from the
request log, which session_runner flushes per record. The 15 s let rounds
answered just before the window's end reach the log; `metrics.py` drops every
round whose first token is past the window, so the grace adds no rounds.

## Scoring and gates

`metrics.py <work dir>/requests.jsonl` computes rounds, rounds/s, new tokens/s,
computed tokens/s, prefix hit share, TTFT p50/p90/p99/max and rounds over 60 s.
The 6 s p99 SLO is **not** a gate: a run that misses it is scored, and
`slo_met` / `slo_margin_s` say by how much. The run is rejected when:

- any request failed (`status` not `SUCCESS`);
- any round has `prefix_shortfall_tokens > 0` (its prompt could not carry the
  previous round's real output token);
- screening fails (no token ids, wrong prompt count, no cached-token report, a
  prefix cache that does not hit, or a lone 16k prompt over 60 s);
- p99 TTFT exceeds 60 s (hopeless; the early abort below usually catches it
  first), or no round was counted.

Reported metrics: `p99_ttft_s` (primary objective) and `new_tokens_per_s`
(second objective), `slo_met` (1 when p99 TTFT <= 6 s, else 0), `slo_margin_s`
(6 s minus p99 TTFT), then `p50_ttft_s`, `p90_ttft_s`, `max_ttft_s`, `rounds_per_s`,
`computed_tokens_per_s`, `prefix_hit_share`, `rounds_ttft_over_60s`,
`fraction_of_sim_target` (new tokens/s over the design's 118.8k),
`fraction_of_workload_ceiling` (over the zero-latency ceiling below),
`screen_ttft_16k_s` and `screen_ttft_32k_s`.

**Workload ceiling.** Replaying the trace's closed loop with every TTFT at
zero (first-come-first-served slots, tool waits, 500 slots) admits 41,444
rounds and 120.3k new tokens/s inside the window. No server exceeds it, since a
positive TTFT only delays every later event. The design's 118.8k is 98.8% of it.

## Screening

Before the replay, on the idle server, with fresh pseudo-random token ids:

1. a lone 16,384-token prompt: generated token id returned, `usage.prompt_tokens`
   equal to 16,384;
2. a session-shaped follow-up (that prompt, its output token, 1,023 new tokens):
   must report cached tokens in (0, 16,384];
3. a fresh 8,192-token prompt sent twice (the shape of session_runner's own
   preflight): the repeat must report cached tokens;
4. a second lone 16,384-token prompt;
5. a lone 32,768-token prompt (informational).

The run is rejected when the better of the two 16k TTFTs exceeds 60 s, the same
bound that rejects a finished run. Why that proves the run would be rejected:
2.3% of the trace's rounds carry at least 16k new tokens (half of all round 0s
do, median 16.7k), and a round that computes 16k fresh tokens under load is no
faster than a lone one on idle GPUs. More than 1% of counted rounds would then
exceed 60 s, so p99 would too. The gate therefore never rejects a server the
replay would have scored; it only stops a broken or pathologically slow one
(about 270 tokens/s) in about a minute instead of after the 20-minute window. A
healthy build takes about 0.4 s (the design's lone 16k TTFT is 0.43 s).

**The preflight probe (step 3).** session_runner aborts before the replay
unless an 8,192-token prompt, sent twice, reports cached tokens the second
time. vLLM serves this hybrid model (KDA linear attention plus DSA) with an
attention block grown to hold a KDA state, about 8.5k tokens, and resumes a
cached prefix only at block boundaries: an 8,192-token prompt has no complete
block and gets no hit, though 16k-prefix session rounds do. Step 3 diagnoses this
in seconds with a clear message instead of a session_runner abort.
`request-factory-preflight-probe.patch` raises session_runner's probe to
32,768 tokens; an operator who builds session_runner with it passes
`--preflight-probe-tokens 32768` so step 3 checks the same length. The patch
changes only the preflight, not the replay.

## Early abort

The request log is read every 2 s. The run stops at once, rejected, when:

- a request failed or a round has a prefix shortfall: either rejects the run;
- **p99 TTFT over 60 s is already certain**: the counted rounds can be at most
  the 41,444 of the zero-latency ceiling, and `metrics.py`'s p99 (numpy linear
  interpolation) exceeds a bound B for any count up to 41,444 once 416 counted
  rounds exceed B. The count is n - floor(0.99 (n - 1)) at n = 41,444: with
  h = 0.99 (n - 1) the sorted position the percentile interpolates from, at
  least n - floor(h) values over B put the value at floor(h), and so the
  interpolated p99, over B. It grows with n, so the value at the ceiling holds
  for every smaller count, and it does not depend on B: the bound moved from
  6 s to 60 s with the same 416 rounds. A run stops once 416 counted rounds took
  over 60 s. The plain-scheduler baseline the simulator predicts (p99 about 22 s)
  is scored, not aborted;
- session_runner exits with an error, logs no round within 15 minutes of
  starting, or logs nothing for 10 minutes while the window is open (its own
  per-request stream timeout is 10 minutes).

An early stop's `error` record carries `new_tokens_per_s_so_far` and the rounds
completed out of the ceiling; a p99 over 60 s at the end carries `p99_ttft_s`
against its 6 s goal.

## Time

Screening takes seconds; corpus tokenization about 40 s on a warm page cache
and up to a few minutes cold; the window 20 min 15 s; scoring under a minute. A
passing run takes about 23-25 minutes; the manifest's timeout is 45 minutes.

## Smoke test (CPU)

`smoke/fake_server.py` is an OpenAI-compatible stand-in (token ids, streaming,
cached-token usage, fault injection). `smoke/smoke_test.py` runs
`session_replay.run()` on a 16-session synthetic trace with a 20-second window
and checks a passing run, an SLO miss that is still scored, and every rejection
path (the smoke workload lowers the 60 s bound to 8 s to keep cases short):

```bash
python smoke/smoke_test.py --engine <session_runner> --tokenizer <tokenizer.json> --text-file <text>
```
