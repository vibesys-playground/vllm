# glm5.3-flash-pp4-b200-vllm

A VibeSys task built from the ServingStudioSim sim-to-real handoff
`glm53_flash_pp4_b200_v1` (`reference/`): the candidate is this vLLM tree
serving the prefill of `nvidia/GLM-5.3-Flash-NVFP4` with pipeline parallelism
on 4 x B200, scored on the handoff's 20-minute coding-agent session replay at
500 sessions. `OBJECTIVE.md` is what the agents read.

| Path | Owner | Content |
|---|---|---|
| `/vibesys_serve.sh` (repository root) | candidate (agent-editable) | the server launch; baseline PP4 [12,11,11,11] |
| `vibesys.input.toml`, `objectives.toml` | task | commands, timeouts, objectives |
| `benchmark/` | task | `session_replay.py`, the handoff's `metrics.py`, the trace manifest pin, CPU smoke test |
| `accuracy_checker/` | task | `checker.py`, `make_reference.py` (`reference.json` is generated once on GPUs and kept at an operator path) |
| `reference/` | task | the handoff's README, contract and pass-rule docs (4-7), small target files under `data/` |

## Interface

Run VibeSys with `--interface service`. The operator owns the server and the
inputs; the task files hold no machine paths.

**Server.** A service runs `vibesys_serve.sh --port PORT` from the candidate
root, with the candidate's `vllm` package first on `PYTHONPATH`, a fresh
process per evaluation (empty caches), and waits for `/v1/models` before
accuracy and benchmark start. The script serves `--served-model-name
nvidia/GLM-5.3-Flash-NVFP4`, which is the default `--model` of both commands.
The checkpoint must be in the Hugging Face cache at revision
`da920bb0b9f4a06727223a349e55468e38352348`.

**Accuracy** (`accuracy_checker/checker.py`), operator appends:
`--base-url http://127.0.0.1:PORT` (no `/v1`) and
`--reference <operator path>/reference.json`, made once with
`make_reference.py` on the reference build (`accuracy_checker/README.md`).
Keep it outside the candidate tree.

**Benchmark** (`benchmark/session_replay.py`), operator appends:

```
--base-url http://127.0.0.1:PORT/v1
--request-factory-engine <session_runner, request-factory rev 7f6aa9d8e36f3b81ae1f6c76c69f460f95f9c84c
                          + benchmark/request-factory-preflight-probe.patch>
--text-file <enwik9, md5 e206c3450ac99950df65bf70ef61a12d>
--asset-dir <dir with sessions_2ps.csv (sha256 a1a1e78b...eb3e72) and sessions_2ps.manifest.json>
--preflight-probe-tokens 32768
--replay-reference <operator path>/replay_reference.json
```

`--replay-reference` is the reference build's replay answers, recorded once
with `--record-replay-reference` on the TP4+EP reference server
(`benchmark/README.md`, "Replay answers"). Keep it outside the candidate tree.

and, when the tokenizer is not in the default Hugging Face cache,
`--tokenizer <tokenizer.json>` or `HF_HUB_CACHE`/`HF_HOME` in the environment.
VibeSys itself appends `--vs-output`.

`--preflight-probe-tokens 32768` with the patched session_runner is the
operator setting. vLLM's hybrid allocator gives this model prefix-cache blocks
of about 8.5k tokens, so the unpatched client's 8,192-token cache preflight
never hits and aborts the replay. The patch raises that probe to 32,768 tokens
and changes nothing else; the flag makes screening check the same length
(`benchmark/README.md`, "The preflight probe").

Both commands run with `uv run --no-project --with-requirements requirements.txt`
(numpy only; the checker uses the standard library).

The trace and its manifest are at ServingStudioSim `pp-main`, commit
`820adda0b719e587b2bdc2ad95f446e11bee64db` (the commit that adds the handoff;
the design it freezes is `7bb224a6`), path
`doc/handoff/glm53_flash_pp4_b200_v1/data/`. session_runner is the
`session_runner` binary of github.com/uw-syfi/request-factory at the rev above
(`cargo build --release --features runtime --bin session_runner`). This rev,
not VibeSys's shared pin 89dce4a, has the fix that carries a round's real output
token id into the next prompt.

## Differences from the handoff contract

None changes the trace or the metric definitions. See `benchmark/README.md`
for the exact replay.

1. **The 6 s SLO is the goal, not a gate.** p99 TTFT is the primary objective
   and `slo_met` / `slo_margin_s` report the SLO; a run is rejected only when
   p99 exceeds 60 s. The contract treats p99 <= 6 s as a pass rule. With the
   SLO as a gate, the baseline (simulated p99 about 22 s) and every early
   candidate would be unscored, and throughput alone barely separates
   candidates (the design is at 98.8% of the workload ceiling).
2. **N = 500 only.** The contract also runs N = 400 (the "low latency" row).
3. **Window start = first request's submit**, not the client process start
   under `timeout 20m`, which would spend corpus tokenization (about 40 s to a
   few minutes) out of the 20 minutes. Rounds are still counted by first-token
   time in [0, 20 min) exactly as `1_contract.md` §3 defines.
4. **Screening before the run**: six prompts (about 98k tokens: 16k, a
   session-shaped follow-up, an 8k-or-32k probe sent twice, 16k, 32k). They stay
   in the cache and warm lazily compiled kernels. The contract runs no warm-up.
5. **Patched session_runner preflight.** session_runner at rev 7f6aa9d checks
   the prefix cache with an 8,192-token probe sent twice before the replay; the
   operator build raises it to 32,768 tokens
   (`benchmark/request-factory-preflight-probe.patch`). Only the preflight
   changes, never the replayed requests.
6. **Full-memory benchmark only (§2.2).** The cache-tier benchmark (§2.3) needs
   the contract's DRAM and SSD per GPU. Candidates may still use host DRAM and
   SSD.
7. **Correctness on HBM-cached prefixes only.** §4 also asks for prefixes read
   back from DRAM and SSD; the checker cannot force a tier.
8. **Correctness tolerance.** A flipped first token is accepted only where the
   reference's own top-2 gap is at most 0.3 nats (on at most 20% of answers),
   with bounded logprob drift (`accuracy_checker/README.md`). §4 asks to log
   every mismatch.
9. **No server logs, stage busy or microbatch fit.** `microbatches.jsonl` and
   `reads.jsonl` (§3) need server instrumentation; they are diagnostics, not
   the score.
10. **Placement.** §5 asks for an otherwise idle node and `uptime` before each
    run; the operator owns placement and the benchmark does not record it.
11. **Handoff commit.** The data files are at ServingStudioSim `pp-main`
    `820adda0`, the commit that adds the handoff; the frozen design commit
    `7bb224a6` has no `doc/handoff/`. The trace sha256 matches the contract.

## Smoke tests (CPU)

```bash
python benchmark/smoke/smoke_test.py --engine <session_runner> --tokenizer <tokenizer.json> --text-file <text>
python accuracy_checker/smoke_test.py --text-file <enwik9>
```
