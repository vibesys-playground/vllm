# Objective - GLM-5.3-Flash NVFP4 prefill, pipeline-parallel on 4x B200 (vLLM)

Serve `nvidia/GLM-5.3-Flash-NVFP4` (revision `da920bb0b9f4a06727223a349e55468e38352348`)
from this vLLM repository on 4x NVIDIA B200 in one NVLink domain, on a
20-minute replay of coding-agent sessions at N = 500 concurrent sessions.
**First meet a 99th-percentile time to first token (TTFT) of at most 6 s, then
maximize the new prompt tokens per second.**

The workload is closed-loop, so throughput is nearly saturated before latency
is fixed: with zero TTFT the replay admits 41,444 rounds in the window, 120.3k
new tokens/s, and no server can exceed that. The known design below reaches
118.8k (98.8% of the ceiling). Latency is the lever: p99 TTFT is the primary
objective, and lower TTFT on long rounds is also where the last percent of
throughput comes from.

The four GPUs only run prefill: every request asks for one output token
(decode runs elsewhere in the modeled system). The baseline launch,
`vibesys_serve.sh` at the repository root, serves the model with pipeline
parallelism (PP4, layers split 12/11/11/11), FP8 KV cache, prefix caching and
8,192-token chunked prefill.

## Workload

`benchmark/session_replay.py` replays `sessions_2ps.csv`
(`reference/1_contract.md` §2.2) with request-factory's `session_runner`:

- 2,457 coding-agent sessions arriving at 2 per second for 20 minutes, at
  most 500 active at once; a later session waits for a free slot.
- Closed loop: a session's next round is sent after the previous round's first
  token plus the trace's tool wait. Its prompt is the previous prompt, the real
  output token, then new tokens: 95% of a prompt is the session's own prefix.
  The rounds served in 20 minutes average about 74k prompt tokens, 3.5k new.
- Prompts are token IDs drawn from enwik9, sent to `/v1/completions` with
  streaming, `temperature=0`, `max_tokens=1`, `return_token_ids=true`, and
  usage reporting. Contexts reach 1M tokens.
- A fresh server per evaluation: every run starts with empty caches.
- Before the replay, a screening step sends a few lone 16k and 32k prompts and a
  session-shaped follow-up; it rejects a broken server, or one whose idle 16k
  TTFT already exceeds 60 s, in seconds. It also sends a 32,768-token prompt
  twice and requires the repeat to report cached tokens: the replay client runs
  that same check before it starts, so a prefix cache that resumes only at
  blocks longer than 32,768 tokens cannot be evaluated.

## Metrics

- `p99_ttft_s` (primary, minimize): p99 TTFT over the rounds whose first token
  arrives within 20 minutes of the first request. The goal is 6 s or less.
- `new_tokens_per_s` (second, maximize): the trace's new tokens of those
  rounds divided by 1,200 s. It counts work the workload adds, independent of
  the cache.
- `slo_met` (1 when p99 TTFT <= 6 s, else 0) and `slo_margin_s` (6 s minus
  p99 TTFT; negative when the SLO is missed).
- Reported for diagnosis: p50/p90/max TTFT, rounds/s, computed tokens/s,
  `prefix_hit_share`, rounds over 60 s, `fraction_of_sim_target` (against
  118.8k), `fraction_of_workload_ceiling`, and the screening TTFTs.

A run that misses the 6 s SLO is still scored, with `slo_met = 0`; compare
candidates by p99 first. A run is **rejected** (an error, not a score) when any
request fails, when any round comes up short of its prefix
(`prefix_shortfall_tokens > 0`, a missing or wrong output token id), when the
prefix cache does not work (screening), or when p99 TTFT exceeds 60 s. The run
stops early once one of these is certain (`benchmark/README.md`).

## Correctness

`accuracy_checker/checker.py` compares the greedy first token and top-10
logprobs against a reference build (vLLM TP4 with expert parallelism, same
checkpoint): lone prompts of 1k to 128k tokens, session rounds on a cached
prefix, and session rounds sent cold. Only near-ties of the reference may
flip; see `accuracy_checker/README.md`.

## Rules

- Do not change the workload, the benchmark, the accuracy checker or anything
  under `.vibesys/`. Do not detect the benchmark or special-case its prompts,
  sessions or request ids.
- The server must run the real model on every prompt: no canned or
  prompt-independent outputs, no skipped layers, no cached answers across
  evaluations.
- Keep the checkpoint and its NVFP4/BF16 weights as shipped, the FP8 KV cache
  format, `max_model_len` 1,048,576, and MTP/speculative decoding off.
- The server must listen on `127.0.0.1:<PORT>` from `vibesys_serve.sh --port PORT`,
  accept token-ID prompts on `/v1/completions`, honour `return_token_ids`, and
  report cached prompt tokens (`usage.prompt_tokens_details.cached_tokens`)
  truthfully.
- Prefix-cache contents must come from this run: the evaluation starts a fresh
  server; do not persist or preload KV across server starts.

## What may change

Anything in this vLLM tree and in `vibesys_serve.sh`: launch flags, the layer
partition (`VLLM_PP_LAYER_PARTITION`), the parallel layout on the 4 GPUs, the
scheduler, the KV-cache manager and prefix cache (including CPU offload of
prefix KV), the pipeline-parallel executor, CUDA graphs, and kernels in Python,
Triton, TileLang, CuTe DSL or other JIT paths. The evaluation installs the
candidate's Python tree over a prebuilt vLLM wheel: **changes under `csrc/` are
not rebuilt**, so a new CUDA kernel must ship as a JIT-compiled kernel or a
prebuilt extension committed to the tree.

## A known-good design to beat

A simulator handoff (`reference/README.md`, frozen design `glm53_flash_pp4_b200_v1`)
serves this exact workload at **118.8k new tokens/s, TTFT p50/p90/p99
0.16/0.44/4.94 s**, at stage busy 92/88/91/91%. Its main points:

- **Layout.** PP4, whole layers per stage, split [12,11,11,11] (stage 0:
  embedding + dense layers 0-2; the slowest stage). One block pool shared by
  the pipeline, sized by the stages with 3 sparse-attention (DSA) layers: 61.3M
  tokens. Each request holds 6 blocks for its linear-attention (KDA) states.
- **Prefix cache.** Exact hits on the previous round's whole prompt, resuming the
  KDA state at its last token (no block-boundary recompute); whole-session LRU
  eviction; HBM holds the active contexts at 500 sessions.
- **Scheduler** (on stage 0, `reference/6_scheduler.md`): at most 4 microbatches
  in flight; a chunk's progress is committed when it is scheduled; started
  prompts shortest-remaining-first and queued prompts shortest-first; budget =
  min(load budget 4,096 tokens rising to 32,768 as the backlog grows from 1M to
  3M tokens, backlog / 4 clamped to [512, 32,768]); a request waiting 60 s goes
  first; whole footprint reserved at admission, no preemption.
- **Kernels** (`reference/README.md`, "Core kernels"): DeepGEMM `mega_mhc` and
  `fp8_mqa_logits`, FlashInfer trtllm sparse MLA over the FP8 cache,
  `recurrent_kda`, `trtllm_fp4_block_scale_moe` and `mm_fp4`, CUDA graphs with
  no host gaps inside a stage.

With the plain chunked-prefill scheduler and HBM-only cache, the simulator
overloads at 500 sessions (TTFT p99 22 s, `reference/5_prefix_cache.md` §4):
expect the baseline to miss the SLO (`slo_met = 0`) until scheduling and the
cache improve. `reference/` holds the per-stage, per-microbatch and scheduler targets
(`data/*.json`) for diagnosing where a candidate falls short. The targets are
not a ceiling: faster kernels or schedules are welcome.
