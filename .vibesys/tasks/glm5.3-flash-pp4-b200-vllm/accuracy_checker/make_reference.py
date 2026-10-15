#!/usr/bin/env python3
"""Write accuracy_checker/reference.json from the contract's reference build.

Run once, against vLLM serving the same checkpoint without pipeline
parallelism (reference/1_contract.md §4), for example:

    vllm serve nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348 \\
        --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 \\
        --max-model-len 1048576 --enable-prompt-tokens-details --port 8000
    python make_reference.py --base-url http://127.0.0.1:8000 --text-file enwik9 --output reference.json

Every prompt is text from enwik9 (past TEXT_START), tokenized by the server's
/tokenize, and stored as token ids, so the checker needs neither the text nor a
tokenizer.

Every answer is asked `--repeats` times (default 3), each repeat under a fresh
`cache_salt`, and the file keeps the tokens observed and the range of the
first repeat's token's logprob over the repeats: the reference's own
run-to-run noise, which the checker tolerates per case. Lone prompts and cold
round 1s are computed from scratch. A cached session's round 1 is recorded the
way the checker and the replay send it: round 0, then round 1 under the same
salt, so round 1 resumes round 0's cached prefix (resuming can change the
answer, so an uncached answer is the wrong reference for it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODEL, MODEL_REVISION, REFERENCE_VERSION, SESSIONS, SINGLES, TEXT_START, TEXT_STRIDE, TOP_K, Answer,
    RequestError, complete, get_json, pack_ids, post_json,
)


class Text:
    """Successive non-overlapping enwik9 regions, tokenized by the server."""

    def __init__(self, path: Path, base_url: str, model: str) -> None:
        self.path, self.base_url, self.model = path, base_url, model
        self.offset = TEXT_START

    def take(self, tokens: int) -> list[int]:
        start = self.offset
        self.offset += max(TEXT_STRIDE, tokens * 8)
        size = tokens * 6
        while True:
            with self.path.open("rb") as handle:
                handle.seek(start)
                text = handle.read(size).decode("utf-8", errors="ignore")
            ids = post_json(f"{self.base_url}/tokenize",
                            {"model": self.model, "prompt": text, "add_special_tokens": False}, 600)["tokens"]
            if len(ids) >= tokens:
                return ids[:tokens]
            size *= 2


def ask(base_url: str, model: str, prompt: list[int], salt: str | None = None, cached_ok: bool = False) -> Answer:
    answer = complete(base_url, model, prompt, 1800, cache_salt=salt or uuid.uuid4().hex)
    if answer.prompt_tokens != len(prompt):
        raise SystemExit(f"reference counted {answer.prompt_tokens} prompt tokens for {len(prompt)}")
    if answer.cached_tokens and not cached_ok:
        raise SystemExit(f"reference reused {answer.cached_tokens} cached tokens despite cache_salt")
    if answer.token not in answer.top:
        raise SystemExit("reference greedy token is not in its own top logprobs")
    return answer


def record(answers: list[Answer]) -> dict:
    """One answer slot over its repeats: the first repeat's token and top logprobs,
    every top-1 token seen, and the range of the first token's logprob."""
    first = answers[0]
    top = sorted(first.top.items(), key=lambda kv: -kv[1])
    lps = [a.top[first.token] for a in answers if first.token in a.top]
    return {"token": first.token, "top": [[t, lp] for t, lp in top], "seconds": round(first.seconds, 3),
            "tokens": sorted({a.token for a in answers}), "lp_range": [min(lps), max(lps)],
            "repeats": len(answers)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="http://host:port of the reference server (no /v1)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text-file", type=Path, required=True, help="enwik9")
    parser.add_argument("--reference-build", default="vLLM TP4 + expert parallel, FP8 KV cache",
                        help="free text recorded in the file: what served the reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="times each answer is asked (>= 2)")
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2: the noise range needs two answers")
    n = args.repeats
    base = args.base_url.rstrip("/")
    served = [m.get("id") for m in get_json(f"{base}/v1/models", 30).get("data", [])]
    if args.model not in served:
        raise SystemExit(f"server serves {served}, not {args.model!r}")
    text = Text(args.text_file, base, args.model)
    cases = []
    noise = []
    started = time.time()
    try:
        for case in SINGLES:
            prompt = text.take(case.tokens)
            answers = [ask(base, args.model, prompt) for _ in range(n)]
            expect = record(answers)
            noise.append(expect["lp_range"][1] - expect["lp_range"][0])
            cases.append({"name": case.name, "kind": "single", "prompt": pack_ids(prompt),
                          "prompt_tokens": len(prompt), "expect": expect,
                          "repeat_token": answers[1].token})
            print(f"{case.name}: tokens {expect['tokens']} lp range {expect['lp_range']} "
                  f"({answers[0].seconds:.2f} s)", flush=True)
        for case in SESSIONS:
            round0 = text.take(case.prefix)
            first0 = ask(base, args.model, round0)
            round1 = round0 + [first0.token] + text.take(case.append - 1)
            answers0, answers1 = [first0], []
            for r in range(n):
                if case.cached:
                    # Round 0 then round 1 under one salt: round 1 resumes the cached prefix.
                    salt = uuid.uuid4().hex
                    if r > 0:
                        answers0.append(ask(base, args.model, round0, salt))
                    else:
                        ask(base, args.model, round0, salt)
                    answers1.append(ask(base, args.model, round1, salt, cached_ok=True))
                else:
                    if r > 0:
                        answers0.append(ask(base, args.model, round0))
                    answers1.append(ask(base, args.model, round1))
            entry = {"name": case.name, "kind": "session", "cached": case.cached,
                     "prefix_tokens": case.prefix, "append_tokens": case.append,
                     "round0": pack_ids(round0), "round1_append": pack_ids(round1[len(round0):]),
                     "expect_round0": record(answers0), "expect_round1": record(answers1),
                     "round1_cached_tokens": [a.cached_tokens for a in answers1]}
            cases.append(entry)
            print(f"{case.name}: round 0 tokens {entry['expect_round0']['tokens']}, round 1 tokens "
                  f"{entry['expect_round1']['tokens']} (round 1 cached {entry['round1_cached_tokens']})", flush=True)
    except RequestError as exc:
        raise SystemExit(str(exc)) from exc
    tokens = [c["expect"]["token"] for c in cases if c["kind"] == "single"]
    tokens += [c[k]["token"] for c in cases if c["kind"] == "session" for k in ("expect_round0", "expect_round1")]
    if len(set(tokens)) < len(tokens) // 3:
        raise SystemExit(f"only {len(set(tokens))} distinct greedy tokens over {len(tokens)} answers; "
                         "the reference looks prompt-independent")
    reference = {
        "version": REFERENCE_VERSION,
        "model": args.model,
        "revision": MODEL_REVISION,
        "reference_build": args.reference_build,
        "top_k": TOP_K,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
        "repeats": n,
        "self_noise_max_abs_dlogprob": max(noise),
        "self_noise_token_flips": sum(len(c["expect"]["tokens"]) > 1 for c in cases if c["kind"] == "single")
        + sum(len(c[k]["tokens"]) > 1 for c in cases if c["kind"] == "session"
              for k in ("expect_round0", "expect_round1")),
        "cases": cases,
    }
    args.output.write_text(json.dumps(reference) + "\n")
    print(f"wrote {args.output}: {len(cases)} cases, self noise {max(noise):.4f} nats, "
          f"{reference['self_noise_token_flips']} repeat flips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
