# Accuracy checker

`checker.py` compares the candidate's greedy first token, and its top-10
logprobs at that position, with `reference.json`: answers of the contract's
reference build (`reference/1_contract.md` §4: vLLM serving the same checkpoint
without pipeline parallelism, TP4 with expert parallelism, FP8 KV cache).
Exit 0 = pass, 1 = fail, 2 = `reference.json` missing or of another version.

```bash
python checker.py --base-url http://127.0.0.1:PORT --reference <operator path>/reference.json \
  [--model nvidia/GLM-5.3-Flash-NVFP4] [--report report.json]
```

`--reference` is an operator path outside the candidate tree, so agents cannot
read the expected answers. Without it the checker falls back to
`accuracy_checker/reference.json` and warns.

The server must already be running. Each prompt is sent twice. First as a
request shaped exactly like a benchmark replay round (streamed, `max_tokens=1`,
`temperature=0`, `ignore_eos`, `return_token_ids`, the usage chunk, no logprobs,
an opaque 32-hex-digit id in `rid` and `x-request-id`): its cached tokens are
the ones checked, and its token must equal the second request's. Then
non-streaming with `logprobs=10`, `return_tokens_as_token_ids=true`,
`return_token_ids=true`, judged against the reference. A server therefore
cannot answer the benchmark's request shape differently (NOTES.md D15).
Per-request timeout: 1,200 s.

## Cases

Every prompt is enwik9 text, tokenized by the reference server and stored as
token ids (taken past byte 600M of enwik9, so no checker prompt shares a prefix
with the benchmark's prompts, which come from the first ~100M tokens).

| Cases | Answers | What it covers |
|---|---|---|
| 22 lone prompts: 4 each of 1k, 2k, 4k, 8k; 2 each of 16k, 32k; 1 each of 64k, 128k | 22 | the §2.1 bring-up lengths: pipeline wiring, chunked prefill, long-context DSA |
| 6 sessions sent as round 0, then round 1 (P = 4k-128k prefix, A = 1k-8k new) | 12 | round 0: a session round without a prefix; round 1: a hit on the cached prefix, resuming the KDA state and the cached DSA KV |
| 3 sessions whose round 1 is sent cold (P = 8k-64k) | 3 | the same round-1 prompt with no prefix in the cache |

Round 1 is round 0's prompt, the reference's greedy token, then A - 1 new
tokens. About 1M prompt tokens in all; the baseline should take 1-3 minutes.

The reference sends every request with its own `cache_salt`, so each of its
answers is computed with no cache, and asks each lone prompt twice to record
its own run-to-run noise (`self_noise_*` in the file).

## Pass rule and why

For each answer:

- **match**: the candidate's token is the reference's;
- **near tie**: it differs, but the reference ranked it within **0.3 nats** of
  its own top token (a probability ratio of 1.35). NVFP4 weights with a
  different parallel layout change the order of reductions, so two candidates
  the reference almost tied can swap. At most 20% of answers may be near ties;
- **mismatch**: anything else. **One mismatch fails the check.**

And over the logprobs:

- the reference's token must be in the candidate's top 10;
- the candidate's logprob of it within **1.0 nat** of the reference's on every
  answer, and within **0.25 nats** on average.

The server must also count every prompt token in `usage.prompt_tokens`, return
the generated token id, report cached tokens for each cached round 1 with at
least 32k tokens of prefix (below that, a hybrid model's cache may resume only
at block boundaries of several thousand tokens), and report no more cached
tokens than the prefix holds.

**Canned or prompt-independent outputs fail**: the reference's tokens vary with
the prompt (make_reference.py refuses a reference with fewer distinct tokens
than a third of its answers), so a canned answer mismatches almost every case,
and the checker also requires at least 80% as many distinct answers as the
reference has.

**Calibration.** The thresholds are set from first principles, not from a
measurement of this checkpoint. After generating `reference.json`, run the
checker against the reference server itself (expect `mean_abs_dlogprob` near
`self_noise_max_abs_dlogprob`, no flips) and against the baseline
`vibesys_serve.sh`, with `--report`. If the baseline fails only on near-tie or
logprob thresholds, widen those (record the measured values here); if it
mismatches outright, the baseline itself is wrong.

## Generating reference.json

Once, on 4 x B200, against the reference build:

```bash
vllm serve nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348 \
  --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 \
  --max-model-len 1048576 --enable-prompt-tokens-details --port 8000
python make_reference.py --base-url http://127.0.0.1:8000 --text-file <enwik9> \
  --reference-build "<vLLM commit, flags>" --output reference.json
```

The file is about 4 MB. Store it at an operator path outside the candidate tree
and pass it with `--reference`; do not commit it.

## Smoke test (CPU)

```bash
python smoke_test.py --text-file <enwik9>
```

Builds a reference from `../benchmark/smoke/fake_server.py`, then checks that
the clean server and one with 0.02-nat logit noise pass, and that one with
1.5-nat noise, one with a canned answer, and one without cached-token reports
fail.
