#!/usr/bin/env python3
"""Accuracy gate: the candidate's greedy first tokens against the reference build.

    python checker.py --base-url http://127.0.0.1:PORT --reference <operator path>/reference.json
                      [--model NAME] [--report report.json]

Exit status 0 = PASS, 1 = FAIL, 2 = the checker could not run (no reference.json).

Cases (reference/1_contract.md §4, accuracy_checker/README.md): lone prompts of
1k to 128k tokens; two-round sessions whose round 1 runs on the cached round-0
prefix; and round-1 prompts sent cold, without a prefix in the cache. Every
prompt is enwik9 text as token ids, stored in reference.json.

Per answer (the first generated token and its top-10 logprobs), against each
trusted build in the reference (format 3: each build asked each answer
several times), judged by the build it fits best; an answer passes if it fits
any one build within that build's own repeat noise:
  match     the candidate's greedy token is one the reference gave in a repeat.
  near tie  it differs, but it is in the reference's top 10 and the reference
            ranked it within NEAR_TIE_NATS of its own top token.
  mismatch  anything else. One mismatch fails the check.
  logprob   the candidate's logprob of the reference's top token should be within
            MAX_ABS_DLOGPROB of the range the reference's repeats spanned. A
            matching token outside it is a "logprob-only miss": allowed for at
            most MAX_LOGPROB_MISS_SHARE of the answers (cached-path answers are
            not repeatable even on the reference build). The mean |dlogprob|
            over all answers must stay within MEAN_ABS_DLOGPROB, so a server
            whose logprobs are systematically shifted fails. The reference
            token missing from the candidate's top 10 is a problem (fails).
The report names the build each answer was judged against (`build`).
The logprob requests of one checker run share a random `cache_salt`, so a
cached round 1 resumes this run's round 0 and never a previous run's cache: the
checker can run repeatedly on one server.
Every prompt is also sent first as a request shaped exactly like a benchmark
replay round (streamed, no logprobs, opaque id); its token must equal the
logprob request's, so a server cannot serve the benchmark's request shape a
cheaper, different answer (README.md, "Threat model").
Also: the server must count every prompt token, return the generated token id,
report a prefix-cache hit for each cached round 1 of at least 32k tokens of
prefix, and its answers must vary with the prompt like the reference's do.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODEL, Answer, ReferenceError, RequestError, complete, get_json, replay_shaped_token, unpack_ids,
    validate_reference,
)

# Fallback only: the operator passes --reference, a path outside the candidate
# tree, so agents cannot read the expected answers (NOTES.md D15, R5).
REFERENCE = Path(__file__).with_name("reference.json")

# A flip is accepted only where the reference itself barely prefers its token:
# a logprob gap of 0.3 nats is a probability ratio of 1.35 between the two.
NEAR_TIE_NATS = 0.3
# At most this share of answers may be near-tie flips.
MAX_NEAR_TIE_SHARE = 0.2
MAX_ABS_DLOGPROB = 1.0
# At most this share of answers may be logprob-only misses: the token matches a
# trusted build's accepted tokens but a logprob is beyond MAX_ABS_DLOGPROB.
MAX_LOGPROB_MISS_SHARE = 0.2
MEAN_ABS_DLOGPROB = 0.25
# Below this prefix a cache hit is not required: a hybrid model's prefix cache
# may only resume at block boundaries of several thousand tokens.
MIN_PREFIX_FOR_REQUIRED_HIT = 32 * 1024
# A cold 128k prompt on a slow or broken-but-honest candidate; 600 s was tight.
REQUEST_TIMEOUT_S = 1200.0


@dataclasses.dataclass
class Verdict:
    name: str
    # The trusted build this answer was judged against (the one it fits best).
    build: str
    expected: int
    got: int
    kind: str  # match, near_tie, mismatch
    ref_gap_nats: float | None
    dlogprob: float | None
    cached_tokens: int | None
    seconds: float
    problems: list[str]
    # Token matches, but the reference token's logprob is beyond MAX_ABS_DLOGPROB of the range.
    logprob_miss: bool = False


def judge(name: str, build: str, expect: dict, answer: Answer, prompt_len: int) -> Verdict:
    ref_top = {int(t): float(lp) for t, lp in expect["top"]}
    expected = int(expect["token"])
    accepted = {int(t) for t in expect.get("tokens", [expected])}
    lo, hi = expect.get("lp_range", [ref_top[expected], ref_top[expected]])
    problems: list[str] = []
    if answer.prompt_tokens != prompt_len:
        problems.append(f"usage.prompt_tokens {answer.prompt_tokens} for a {prompt_len}-token prompt")
    gap = None
    if answer.token in accepted:
        kind = "match"
    elif answer.token in ref_top and ref_top[expected] - ref_top[answer.token] <= NEAR_TIE_NATS:
        kind = "near_tie"
        gap = ref_top[expected] - ref_top[answer.token]
    else:
        kind = "mismatch"
        gap = ref_top[expected] - ref_top[answer.token] if answer.token in ref_top else None
    dlp = None
    logprob_miss = False
    if expected in answer.top:
        # Distance to the range the reference's own repeats spanned (0 inside it).
        got = answer.top[expected]
        dlp = got - min(max(got, lo), hi)
        if abs(dlp) > MAX_ABS_DLOGPROB and kind == "match":
            logprob_miss = True
        elif abs(dlp) > MAX_ABS_DLOGPROB:
            problems.append(f"logprob of the reference token differs by {dlp:+.3f} nats")
    else:
        problems.append("the reference's greedy token is not in the candidate's top logprobs")
    return Verdict(name, build, expected, answer.token, kind, gap, dlp, answer.cached_tokens, answer.seconds,
                   problems, logprob_miss)


KIND_RANK = {"match": 0, "near_tie": 1, "mismatch": 2}


def judge_builds(name: str, expects: dict[str, dict], answer: Answer, prompt_len: int) -> Verdict:
    """The answer judged against every trusted build; the verdict of the build it fits best.

    Best = passes (not a mismatch, no problem) first, then match before near
    tie, then within-tolerance before logprob-only miss, then the smallest logprob distance to that build's repeat range.
    """
    verdicts = [judge(name, build, expect, answer, prompt_len) for build, expect in expects.items()]
    return min(verdicts, key=lambda v: (v.kind == "mismatch" or bool(v.problems), KIND_RANK[v.kind],
                                        v.logprob_miss, abs(v.dlogprob) if v.dlogprob is not None else float("inf")))


def replay_shaped_kind(expects: dict[str, dict], token: int, logprob_token: int) -> str | None:
    """None when the replay-shaped answer is acceptable: the logprob request's token,
    a token some trusted build gave, or a near tie of some build's top token."""
    if token == logprob_token:
        return None
    for expect in expects.values():
        ref_top = {int(t): float(lp) for t, lp in expect["top"]}
        expected = int(expect["token"])
        if token in {int(t) for t in expect.get("tokens", [expected])}:
            return None
        if token in ref_top and ref_top[expected] - ref_top[token] <= NEAR_TIE_NATS:
            return None
    return (f"the replay-shaped request answered {token}, the logprob request {logprob_token}: "
            "neither the reference's answer nor a near tie")


def run(base_url: str, model: str, reference: dict, log=print) -> tuple[bool, dict]:
    base = base_url.rstrip("/")
    served = [m.get("id") for m in get_json(f"{base}/v1/models", 30).get("data", [])]
    if model not in served:
        return False, {"error": f"server serves {served}, not {model!r}"}
    verdicts: list[Verdict] = []
    hit_problems: list[str] = []
    run_salt = secrets.token_hex(16)

    def check(name: str, expects: dict[str, dict], prompt: list[int]) -> Verdict:
        # The replay-shaped request carries no salt (the replay sends none), so on a
        # server that already ran the checker it may resume an earlier run's cache:
        # it is judged against the reference with the near-tie tolerance, not by
        # equality. The salted logprob request's cached tokens are the ones checked.
        shaped, shaped_prompt_tokens, _ = replay_shaped_token(base, model, prompt, REQUEST_TIMEOUT_S)
        answer = complete(base, model, prompt, REQUEST_TIMEOUT_S, cache_salt=run_salt)
        verdict = judge_builds(name, expects, answer, len(prompt))
        shaped_problem = replay_shaped_kind(expects, shaped, verdict.got)
        if shaped_problem is not None:
            verdict.problems.append(shaped_problem)
        if shaped_prompt_tokens != len(prompt):
            verdict.problems.append(f"replay-shaped request: usage.prompt_tokens {shaped_prompt_tokens} for a "
                                    f"{len(prompt)}-token prompt")
        verdicts.append(verdict)
        flag = "" if verdict.kind == "match" and not verdict.problems else f"  <-- {verdict.kind} {verdict.problems}"
        log(f"{name}: expected {verdict.expected} ({verdict.build}) got {verdict.got} "
            f"dlogprob {verdict.dlogprob if verdict.dlogprob is None else round(verdict.dlogprob, 4)} "
            f"cached {verdict.cached_tokens} ({verdict.seconds:.2f} s){flag}")
        return verdict

    for case in reference["cases"]:
        if case["kind"] == "single":
            check(case["name"], case["expect"], unpack_ids(case["prompt"]))
            continue
        round0 = unpack_ids(case["round0"])
        round1 = round0 + unpack_ids(case["round1_append"])
        if case["cached"]:
            check(f"{case['name']}/round0", case["expect_round0"], round0)
            verdict = check(f"{case['name']}/round1", case["expect_round1"], round1)
            if case["prefix_tokens"] >= MIN_PREFIX_FOR_REQUIRED_HIT and not verdict.cached_tokens:
                hit_problems.append(f"{case['name']}: round 1 reported no cached prefix tokens")
            if verdict.cached_tokens is not None and verdict.cached_tokens > len(round0) + 1:
                hit_problems.append(f"{case['name']}: {verdict.cached_tokens} cached tokens exceed the "
                                    f"{len(round0) + 1}-token prefix")
        else:
            check(f"{case['name']}/round1_cold", case["expect_round1"], round1)

    return summarize(verdicts, hit_problems, reference)


def summarize(verdicts: list[Verdict], hit_problems: list[str], reference: dict) -> tuple[bool, dict]:
    """The run's verdict from its per-answer verdicts: the shares, the mean, the failures."""
    near = [v for v in verdicts if v.kind == "near_tie"]
    mismatches = [v for v in verdicts if v.kind == "mismatch"]
    misses = [v for v in verdicts if v.logprob_miss]
    problems = [f"{v.name}: {p}" for v in verdicts for p in v.problems] + hit_problems
    dlps = [abs(v.dlogprob) for v in verdicts if v.dlogprob is not None]
    mean_dlp = sum(dlps) / len(dlps) if dlps else float("inf")
    expected_distinct = len({v.expected for v in verdicts})
    got_distinct = len({v.got for v in verdicts})
    failures = []
    if mismatches:
        failures.append(f"{len(mismatches)} mismatch(es): " + ", ".join(v.name for v in mismatches))
    if len(near) > MAX_NEAR_TIE_SHARE * len(verdicts):
        failures.append(f"{len(near)} near-tie flips of {len(verdicts)} answers, above "
                        f"{MAX_NEAR_TIE_SHARE:.0%}")
    if len(misses) > MAX_LOGPROB_MISS_SHARE * len(verdicts):
        failures.append(f"{len(misses)} logprob-only misses of {len(verdicts)} answers, above "
                        f"{MAX_LOGPROB_MISS_SHARE:.0%}: " + ", ".join(v.name for v in misses))
    if mean_dlp > MEAN_ABS_DLOGPROB:
        failures.append(f"mean |dlogprob| {mean_dlp:.3f} nats above {MEAN_ABS_DLOGPROB}")
    if got_distinct < expected_distinct * 0.8:
        failures.append(f"{got_distinct} distinct answers where the reference has {expected_distinct}: "
                        "outputs do not follow the prompt")
    failures += problems
    summary = {
        "pass": not failures,
        "answers": len(verdicts),
        "matches": sum(v.kind == "match" for v in verdicts),
        "near_ties": len(near),
        "mismatches": len(mismatches),
        "logprob_only_misses": len(misses),
        "logprob_only_miss_names": [v.name for v in misses],
        "max_logprob_miss_share": MAX_LOGPROB_MISS_SHARE,
        "max_near_tie_share": MAX_NEAR_TIE_SHARE,
        "max_abs_dlogprob_tolerance": MAX_ABS_DLOGPROB,
        "mean_abs_dlogprob_limit": MEAN_ABS_DLOGPROB,
        "mean_abs_dlogprob": mean_dlp,
        "max_abs_dlogprob": max(dlps, default=float("inf")),
        "builds": {b["id"]: b.get("reference_build") for b in reference["builds"]},
        "answers_by_build": {b["id"]: sum(v.build == b["id"] for v in verdicts) for b in reference["builds"]},
        "reference_self_noise_max_abs_dlogprob": {b["id"]: b.get("self_noise_max_abs_dlogprob")
                                                  for b in reference["builds"]},
        "failures": failures,
        "verdicts": [dataclasses.asdict(v) for v in verdicts],
    }
    return not failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:PORT of the candidate (no /v1)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reference", type=Path, default=None,
                        help="the reference build's answers (operator path, outside the candidate tree); "
                             "default: accuracy_checker/reference.json, with a warning")
    parser.add_argument("--report", type=Path, default=None, help="write the per-answer report here")
    args = parser.parse_args()
    if args.reference is None:
        args.reference = REFERENCE
        print(f"WARNING: no --reference; falling back to the in-tree {REFERENCE}, which agents can read. "
              "Pass the operator's copy.", file=sys.stderr)
    if not args.reference.is_file():
        print(f"FAIL: {args.reference} is missing; generate it with make_reference.py "
              "(accuracy_checker/README.md)", file=sys.stderr)
        return 2
    reference = json.loads(args.reference.read_text())
    try:
        validate_reference(reference)
    except ReferenceError as exc:
        print(f"FAIL: {args.reference}: {exc}")
        return 2
    try:
        ok, summary = run(args.base_url, args.model, reference)
    except RequestError as exc:
        print(f"FAIL: {exc}")
        return 1
    if args.report is not None:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    brief = {k: v for k, v in summary.items() if k != "verdicts"}
    print(json.dumps(brief, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
