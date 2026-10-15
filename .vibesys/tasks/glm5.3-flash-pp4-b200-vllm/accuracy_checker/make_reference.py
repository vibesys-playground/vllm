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
tokenizer. Every request carries its own `cache_salt`, so the reference never
reuses its prefix cache: each answer is the whole prompt computed from scratch.
Each lone prompt is asked twice; the spread between the two answers is the
reference's own run-to-run noise, recorded in the file.
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


def ask(base_url: str, model: str, prompt: list[int]) -> Answer:
    answer = complete(base_url, model, prompt, 1800, cache_salt=uuid.uuid4().hex)
    if answer.prompt_tokens != len(prompt):
        raise SystemExit(f"reference counted {answer.prompt_tokens} prompt tokens for {len(prompt)}")
    if answer.cached_tokens:
        raise SystemExit(f"reference reused {answer.cached_tokens} cached tokens despite cache_salt")
    if answer.token not in answer.top:
        raise SystemExit("reference greedy token is not in its own top logprobs")
    return answer


def record(answer: Answer) -> dict:
    top = sorted(answer.top.items(), key=lambda kv: -kv[1])
    return {"token": answer.token, "top": [[t, lp] for t, lp in top], "seconds": round(answer.seconds, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="http://host:port of the reference server (no /v1)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text-file", type=Path, required=True, help="enwik9")
    parser.add_argument("--reference-build", default="vLLM TP4 + expert parallel, FP8 KV cache",
                        help="free text recorded in the file: what served the reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
            first, second = ask(base, args.model, prompt), ask(base, args.model, prompt)
            shared = set(first.top) & set(second.top)
            noise.append(max((abs(first.top[t] - second.top[t]) for t in shared), default=0.0))
            cases.append({"name": case.name, "kind": "single", "prompt": pack_ids(prompt),
                          "prompt_tokens": len(prompt), "expect": record(first),
                          "repeat_token": second.token})
            print(f"{case.name}: token {first.token} ({first.seconds:.2f} s)", flush=True)
        for case in SESSIONS:
            round0 = text.take(case.prefix)
            first = ask(base, args.model, round0)
            round1 = round0 + [first.token] + text.take(case.append - 1)
            second = ask(base, args.model, round1)
            entry = {"name": case.name, "kind": "session", "cached": case.cached,
                     "prefix_tokens": case.prefix, "append_tokens": case.append,
                     "round0": pack_ids(round0), "round1_append": pack_ids(round1[len(round0):]),
                     "expect_round0": record(first), "expect_round1": record(second)}
            cases.append(entry)
            print(f"{case.name}: tokens {first.token}, {second.token}", flush=True)
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
        "self_noise_max_abs_dlogprob": max(noise),
        "self_noise_token_flips": sum(c["repeat_token"] != c["expect"]["token"]
                                      for c in cases if c["kind"] == "single"),
        "cases": cases,
    }
    args.output.write_text(json.dumps(reference) + "\n")
    print(f"wrote {args.output}: {len(cases)} cases, self noise {max(noise):.4f} nats, "
          f"{reference['self_noise_token_flips']} repeat flips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
