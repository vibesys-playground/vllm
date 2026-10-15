# Accuracy checker

`checker.py` compares the candidate's greedy first token, and its top-10
logprobs at that position, with `reference.json`: the answers of a set of
trusted builds ("Trusted builds" below), first the contract's reference build
(`reference/1_contract.md` §4: vLLM serving the same checkpoint without
pipeline parallelism, TP4 with expert parallelism, FP8 KV cache).
Exit 0 = pass, 1 = fail, 2 = `reference.json` missing, of another version, or
malformed (the message names the case and key).

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

`make_reference.py` asks every answer `--repeats` times (default 3), each
repeat under a fresh `cache_salt`, and stores per answer the tokens the
repeats gave (`tokens`) and the range of the first repeat's token's logprob
(`lp_range`): the reference build is bit-stable up to about 2k tokens and
noisy beyond (a 64k answer flipped between two tokens across runs; logprobs
swing ~0.3 nats at 16k). Lone prompts and cold round 1s are computed from
scratch. A cached session's round 1 is recorded through the cached path, round
0 then round 1 under one salt, because resuming a cached prefix can change the
answer (on both builds, `session_p32k_a1k_cached` round 1 answers 2999 cached
and 5726 cold).

The checker's logprob requests in one run share a random `cache_salt`, so a
cached round 1 resumes this run's round 0 and never an earlier run's cache, and
the checker can run several times on one server.

## Trusted builds

Two honest parallel layouts of the same unmodified engine can disagree
confidently on a prompt: the unmodified PP4 baseline answers `single_1k_0`
with 2937 where TP4+EP answers 19439 (TP4 without EP also gives 2937: an EP
kernel difference), and `single_2k_1` with 312 where TP4+EP gives 16248 with
a logprob 1.43 nats apart (PP/TP1 numerics, independent of the PP stage
split). Each build is bit-stable on both over repeats. A single-build
reference would fail every candidate on these, so `reference.json` (format 3)
holds, per answer slot, one record per trusted build, keyed by build id, and
`builds` lists each build's id, description, repeats and own noise
(`self_noise_max_abs_dlogprob`, `self_noise_token_flips`).

An answer is judged against every build and passes if it fits **any one**
build within that build's own repeat noise (the rules below, applied with that
build's tokens, top logprobs and `lp_range`); the report names the build each
answer was judged against (`build`, `answers_by_build`). Builds may differ per
answer, since each layout's numerics flip different prompts. A wrong server
still fails: it must fit some trusted build on every answer, and the builds
agree on almost all of them.

Trusted builds are unmodified engine builds of the checkpoint in a supported
parallel layout, run by the operator; never a candidate.

## Pass rule and why

For each answer, against each trusted build:

- **match**: the candidate's token is one the reference gave over its repeats;
- **near tie**: it differs, but the reference ranked it within **0.3 nats** of
  its own top token (a probability ratio of 1.35). NVFP4 weights with a
  different parallel layout change the order of reductions, so two candidates
  the reference almost tied can swap. At most 20% of answers may be near ties;
- **mismatch**: anything else. **One mismatch fails the check.**

And over the logprobs:

- the reference's token must be in the candidate's top 10;
- the candidate's logprob of it within **1.0 nat** of the range the reference's
  repeats spanned on every answer, and within **0.25 nats** on average (the
  distance is 0 inside the range).

The replay-shaped request's token must be the logprob request's, a token the
reference gave, or a near tie of the reference's top token. It carries no
salt (the replay sends none), so on a server that already ran the checker it
may resume the earlier run's cache, hence the tolerance instead of equality.

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
python make_reference.py --base-url http://127.0.0.1:8000 --text-file <enwik9> --build-id tp4ep \
  --reference-build "<vLLM commit, flags>" --repeats 3 --output reference.json
```

Then once per further trusted build, against that build's server, on the
same prompts (taken from the file; no corpus):

```bash
python make_reference.py --base-url http://127.0.0.1:8000 --build-id pp4-baseline \
  --reference-build "<commit, flags>" --repeats 3 --merge reference.json --output reference.merged.json
```

A format-2 file (one unnamed build) converts with `--merge OLD
--existing-build-id ID`.

The file is about 4 MB. Store it at an operator path outside the candidate tree
and pass it with `--reference`; do not commit it.

## Smoke test (CPU)

```bash
python smoke_test.py --text-file <enwik9>
```

Builds a reference from `../benchmark/smoke/fake_server.py`, then checks that
the clean server and one with 0.02-nat logit noise pass, and that one with
1.5-nat noise, one with a canned answer, and one without cached-token reports
fail. Then trusted builds: a second honest layout (`--alt-numerics 3`, a
confident flip on one prompt in three) fails against the first build alone and
passes once `--merge` adds its answers; the first build still passes the
merged file; constant-token, shuffled-answer and 1.5-nat-noise servers fail it.

`test_trusted_builds.py` (no server; pytest or plain python) checks the
format-3 validation, the format-2 conversion, and that an answer passes
exactly when it fits some build, over generated inputs.
