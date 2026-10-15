#!/usr/bin/env python3
"""Write accuracy_checker/reference.json: the answers of one or more trusted builds.

First build, against vLLM serving the same checkpoint without pipeline
parallelism (reference/1_contract.md §4), for example:

    vllm serve nvidia/GLM-5.3-Flash-NVFP4 --revision da920bb0b9f4a06727223a349e55468e38352348 \\
        --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 \\
        --max-model-len 1048576 --enable-prompt-tokens-details --port 8000
    python make_reference.py --base-url http://127.0.0.1:8000 --text-file enwik9 --build-id tp4ep \\
        --output reference.json

Each further trusted build (another parallel layout of an unmodified engine)
adds its answers on the same prompts:

    python make_reference.py --base-url http://127.0.0.1:8000 --build-id pp4-baseline \\
        --merge reference.json --output reference.merged.json

`--merge` takes the prompts from the existing file (no corpus, no /tokenize)
and keeps its builds; a format-2 file (one unnamed build) is converted, its
build named by `--existing-build-id`. The checker accepts an answer that
matches any build within that build's own repeat noise (README.md, "Trusted
builds").

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
    ReferenceError, RequestError, answer_slots, complete, get_json, pack_ids, post_json, unpack_ids, upgrade_v2,
    validate_reference,
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


def fresh_cases(text: Text) -> list[dict]:
    """The cases' prompts, new from the corpus (the first build)."""
    cases: list[dict] = []
    for case in SINGLES:
        prompt = text.take(case.tokens)
        cases.append({"name": case.name, "kind": "single", "prompt": pack_ids(prompt), "prompt_tokens": len(prompt)})
    for case in SESSIONS:
        round0 = text.take(case.prefix)
        # Round 1 appends the round-0 answer, which is build-specific; this build's
        # first round-0 answer fixes it for every build (the prompt must be shared).
        cases.append({"name": case.name, "kind": "session", "cached": case.cached, "prefix_tokens": case.prefix,
                      "append_tokens": case.append, "round0": pack_ids(round0),
                      "round1_append_fresh": text.take(case.append - 1)})
    return cases


def measure(base: str, model: str, case: dict, n: int) -> dict[str, object]:
    """This build's answer slots for one case, each asked `n` times."""
    if case["kind"] == "single":
        prompt = unpack_ids(case["prompt"])
        answers = [ask(base, model, prompt) for _ in range(n)]
        return {"expect": record(answers)}
    round0 = unpack_ids(case["round0"])
    first0 = ask(base, model, round0)
    if "round1_append" not in case:
        case["round1_append"] = pack_ids([first0.token, *case.pop("round1_append_fresh")])
    round1 = round0 + unpack_ids(case["round1_append"])
    answers0, answers1 = [first0], []
    for r in range(n):
        if case["cached"]:
            # Round 0 then round 1 under one salt: round 1 resumes the cached prefix.
            salt = uuid.uuid4().hex
            if r > 0:
                answers0.append(ask(base, model, round0, salt))
            else:
                ask(base, model, round0, salt)
            answers1.append(ask(base, model, round1, salt, cached_ok=True))
        else:
            if r > 0:
                answers0.append(ask(base, model, round0))
            answers1.append(ask(base, model, round1))
    return {"expect_round0": record(answers0), "expect_round1": record(answers1),
            "round1_cached_tokens": [a.cached_tokens for a in answers1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="http://host:port of the trusted build's server (no /v1)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text-file", type=Path, default=None, help="enwik9 (first build only)")
    parser.add_argument("--build-id", required=True, help="short id of this build, unique in the file (e.g. tp4ep)")
    parser.add_argument("--reference-build", default="vLLM TP4 + expert parallel, FP8 KV cache",
                        help="free text recorded with the build: what served it")
    parser.add_argument("--merge", type=Path, default=None,
                        help="existing reference: reuse its prompts and keep its builds")
    parser.add_argument("--existing-build-id", default=None,
                        help="with a format-2 --merge file: the id to give its one build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="times each answer is asked (>= 2)")
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2: the noise range needs two answers")
    if (args.merge is None) == (args.text_file is None):
        raise SystemExit("pass exactly one of --text-file (a new reference) and --merge (add a build)")
    n = args.repeats
    base = args.base_url.rstrip("/")
    served = [m.get("id") for m in get_json(f"{base}/v1/models", 30).get("data", [])]
    if args.model not in served:
        raise SystemExit(f"server serves {served}, not {args.model!r}")
    if args.merge is not None:
        existing = json.loads(args.merge.read_text())
        try:
            if existing.get("version") == 2:
                if not args.existing_build_id:
                    raise SystemExit(f"{args.merge} is format 2: name its build with --existing-build-id")
                existing = upgrade_v2(existing, args.existing_build_id)
            builds = validate_reference(existing)
        except ReferenceError as exc:
            raise SystemExit(f"--merge {args.merge}: {exc}") from exc
        if args.build_id in builds:
            raise SystemExit(f"--merge {args.merge} already has build {args.build_id!r}")
        if existing["model"] != args.model:
            raise SystemExit(f"--merge {args.merge} is for model {existing['model']!r}, not {args.model!r}")
        cases = existing["cases"]
    else:
        existing = None
        cases = fresh_cases(Text(args.text_file, base, args.model))
    noise: list[float] = []
    flips = 0
    started = time.time()
    try:
        for case in cases:
            slots = measure(base, args.model, case, n)
            for key, value in slots.items():
                case.setdefault(key, {})[args.build_id] = value
                if key.startswith("expect"):
                    noise.append(value["lp_range"][1] - value["lp_range"][0])
                    flips += len(value["tokens"]) > 1
            print(f"{case['name']}: " + ", ".join(f"{k} tokens {v['tokens']} lp range {v['lp_range']}"
                                                   for k, v in slots.items() if k.startswith("expect")), flush=True)
    except RequestError as exc:
        raise SystemExit(str(exc)) from exc
    tokens = [case[key][args.build_id]["token"] for case in cases for key in answer_slots(case)]
    if len(set(tokens)) < len(tokens) // 3:
        raise SystemExit(f"only {len(set(tokens))} distinct greedy tokens over {len(tokens)} answers; "
                         "the build looks prompt-independent")
    build = {
        "id": args.build_id,
        "reference_build": args.reference_build,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
        "repeats": n,
        "self_noise_max_abs_dlogprob": max(noise),
        "self_noise_token_flips": flips,
    }
    reference = {
        "version": REFERENCE_VERSION,
        "model": args.model,
        "revision": MODEL_REVISION,
        "top_k": TOP_K,
        "builds": [*(existing["builds"] if existing else []), build],
        "cases": cases,
    }
    validate_reference(reference)
    args.output.write_text(json.dumps(reference) + "\n")
    print(f"wrote {args.output}: {len(cases)} cases, builds {[b['id'] for b in reference['builds']]}; "
          f"{args.build_id} self noise {max(noise):.4f} nats, {flips} repeat flips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
