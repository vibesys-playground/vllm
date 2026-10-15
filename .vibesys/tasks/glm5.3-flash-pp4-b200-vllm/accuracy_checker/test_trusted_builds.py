#!/usr/bin/env python3
"""CPU unit tests of the trusted-build reference (format 3): validation, format-2 conversion, and judging an
answer against several builds. No server. Each test checks a property over seeded, generated inputs; runs
under pytest or as a script.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checker  # noqa: E402
from common import Answer, ReferenceError, upgrade_v2, validate_reference  # noqa: E402

CASES = 500


def random_record(rng: random.Random) -> dict:
    """One build's answer slot: its top logprobs, the tokens its repeats gave, its logprob range."""
    tokens = rng.sample(range(100, 140), 10)
    lps = sorted((rng.uniform(-8, 0) for _ in tokens), reverse=True)
    top = [[t, lp] for t, lp in zip(tokens, lps, strict=True)]
    spread = rng.choice([0.0, 0.0, rng.uniform(0, 0.8)])
    seen = [tokens[0]] + ([tokens[1]] if rng.random() < 0.2 else [])
    return {"token": tokens[0], "top": top, "tokens": seen, "lp_range": [lps[0] - spread, lps[0]]}


def random_answer(rng: random.Random, records: list[dict]) -> Answer:
    source = rng.choice(records)
    top = {int(t): lp + rng.gauss(0, rng.choice([0.01, 0.5, 2.0])) for t, lp in source["top"]}
    if rng.random() < 0.3:
        top[rng.randrange(100, 140)] = rng.uniform(-1, 0)
    token = max(top, key=top.get)
    return Answer(token=token, top=top, prompt_tokens=8, cached_tokens=0, seconds=0.1)


def fits(verdict: checker.Verdict) -> bool:
    return verdict.kind != "mismatch" and not verdict.problems


def test_an_answer_passes_exactly_when_it_fits_some_build() -> None:
    rng = random.Random(1)
    for _ in range(CASES):
        records = {f"b{i}": random_record(rng) for i in range(rng.randint(1, 3))}
        answer = random_answer(rng, list(records.values()))
        per_build = {b: checker.judge("c", b, r, answer, 8) for b, r in records.items()}
        best = checker.judge_builds("c", records, answer, 8)
        assert fits(best) == any(fits(v) for v in per_build.values())
        assert best == per_build[best.build]
        if fits(best) and best.kind == "near_tie":
            assert not any(v.kind == "match" and fits(v) for v in per_build.values())


def test_adding_a_build_never_turns_a_pass_into_a_fail() -> None:
    rng = random.Random(2)
    for _ in range(CASES):
        records = {f"b{i}": random_record(rng) for i in range(rng.randint(1, 3))}
        answer = random_answer(rng, list(records.values()))
        before = fits(checker.judge_builds("c", records, answer, 8))
        after = fits(checker.judge_builds("c", {**records, "new": random_record(rng)}, answer, 8))
        assert after or not before


def test_a_build_answer_fits_its_own_record() -> None:
    """A build's own greedy token at its own logprobs passes, whatever the other builds say."""
    rng = random.Random(3)
    for _ in range(CASES):
        records = {f"b{i}": random_record(rng) for i in range(rng.randint(1, 4))}
        own = rng.choice(list(records))
        top = {int(t): lp for t, lp in records[own]["top"]}
        answer = Answer(token=records[own]["token"], top=top, prompt_tokens=8, cached_tokens=0, seconds=0.1)
        verdict = checker.judge_builds("c", records, answer, 8)
        assert fits(verdict) and verdict.kind == "match", verdict


def test_replay_shaped_answer_accepted_iff_some_build_accepts_it() -> None:
    rng = random.Random(4)
    for _ in range(CASES):
        records = {f"b{i}": random_record(rng) for i in range(rng.randint(1, 3))}
        token, logprob_token = rng.randrange(100, 140), rng.randrange(100, 140)
        single = [checker.replay_shaped_kind({b: r}, token, logprob_token) is None for b, r in records.items()]
        assert (checker.replay_shaped_kind(records, token, logprob_token) is None) == any(single)


def v2_reference(rng: random.Random) -> dict:
    cases = [{"name": "single_1k_0", "kind": "single", "prompt": "AAAA", "prompt_tokens": 1,
              "expect": random_record(rng), "repeat_token": 5},
             {"name": "session_x", "kind": "session", "cached": True, "prefix_tokens": 4, "append_tokens": 1,
              "round0": "AAAA", "round1_append": "AAAA", "expect_round0": random_record(rng),
              "expect_round1": random_record(rng), "round1_cached_tokens": [0, 0, 0]}]
    return {"version": 2, "model": "m", "revision": "r", "reference_build": "tp4ep text", "top_k": 10,
            "created": "t", "seconds": 1.0, "repeats": 3, "self_noise_max_abs_dlogprob": 0.5,
            "self_noise_token_flips": 0, "cases": cases}


def test_format_2_converts_to_one_named_build_and_validates() -> None:
    rng = random.Random(5)
    old = v2_reference(rng)
    new = upgrade_v2(old, "tp4ep")
    assert validate_reference(new) == ["tp4ep"]
    assert new["builds"][0]["reference_build"] == "tp4ep text"
    for before, after in zip(old["cases"], new["cases"], strict=True):
        for key in ("expect", "expect_round0", "expect_round1"):
            if key in before:
                assert after[key] == {"tp4ep": before[key]}
        assert after.get("prompt") == before.get("prompt") and after.get("round0") == before.get("round0")


def test_validation_names_a_slot_missing_a_build() -> None:
    rng = random.Random(6)
    ref = upgrade_v2(v2_reference(rng), "a")
    ref["builds"].append({"id": "b"})
    for case in ref["cases"]:
        for key in ("expect", "expect_round0", "expect_round1", "round1_cached_tokens"):
            if key in case:
                case[key]["b"] = case[key]["a"]
    assert validate_reference(ref) == ["a", "b"]
    for case_index, key in ((0, "expect"), (1, "expect_round1"), (1, "round1_cached_tokens")):
        broken = {**ref, "cases": [dict(c) for c in ref["cases"]]}
        broken["cases"][case_index] = {**broken["cases"][case_index],
                                       key: {"a": broken["cases"][case_index][key]["a"]}}
        try:
            validate_reference(broken)
        except ReferenceError as exc:
            assert key in str(exc) and broken["cases"][case_index]["name"] in str(exc), exc
        else:
            raise AssertionError(f"accepted a reference whose {key} lacks build b")
    for builds in ([], [{"id": "a"}, {"id": "a"}]):
        try:
            validate_reference({**ref, "builds": builds})
        except ReferenceError:
            pass
        else:
            raise AssertionError(f"accepted builds {builds}")
    try:
        validate_reference(v2_reference(rng))
    except ReferenceError as exc:
        assert "--existing-build-id" in str(exc)
    else:
        raise AssertionError("accepted a format-2 reference")


def verdict_set(n: int, *, wrong: int = 0, misses: int = 0, shift: float = 0.0) -> list[checker.Verdict]:
    """n distinct-token answers judged against single-build records: the first `wrong` answer a token
    outside the reference, the next `misses` have a matching token but a logprob 2 nats off its range,
    and every answer's logprob is moved by `shift` nats."""
    rng = random.Random(7)
    out = []
    for i in range(n):
        record = random_record(rng)
        record["lp_range"] = [record["top"][0][1]] * 2
        top = {int(t): lp for t, lp in record["top"]}
        token = record["token"]
        if i < wrong:
            token = 999_000 + i
            top[token] = 1.0
        elif i < wrong + misses:
            top[record["token"]] -= 2.0
        top[record["token"]] += -shift if i >= wrong + misses else 0.0
        answer = Answer(token=token, top=top, prompt_tokens=8, cached_tokens=0, seconds=0.1)
        out.append(checker.judge_builds(f"a{i}", {"b": record}, answer, 8))
    return out


def summary_of(verdicts: list[checker.Verdict]) -> tuple[bool, dict]:
    return checker.summarize(verdicts, [], {"builds": [{"id": "b"}]})


def test_one_logprob_only_miss_in_37_passes_and_is_reported() -> None:
    ok, summary = summary_of(verdict_set(37, misses=1))
    assert ok, summary["failures"]
    assert summary["logprob_only_misses"] == 1 and summary["mismatches"] == 0
    assert summary["logprob_only_miss_names"] == ["a0"]
    assert summary["max_logprob_miss_share"] == checker.MAX_LOGPROB_MISS_SHARE == 0.2
    assert summary["mean_abs_dlogprob_limit"] == checker.MEAN_ABS_DLOGPROB == 0.25


def test_a_wrong_token_still_fails() -> None:
    ok, summary = summary_of(verdict_set(37, wrong=1))
    assert not ok and summary["mismatches"] == 1
    assert any("mismatch" in f for f in summary["failures"])


def test_logprob_only_misses_above_the_share_fail() -> None:
    n = 40
    limit = int(checker.MAX_LOGPROB_MISS_SHARE * n)
    ok, summary = summary_of(verdict_set(n, misses=limit + 1))
    assert not ok and any("logprob-only misses" in f for f in summary["failures"])
    ok, summary = summary_of(verdict_set(n, misses=limit))
    assert not any("logprob-only misses" in f for f in summary["failures"])


def test_systematically_shifted_logprobs_fail_on_the_mean_with_matching_tokens() -> None:
    ok, summary = summary_of(verdict_set(37, shift=0.5))
    # 0.5 nats is inside the 1.0 per-answer tolerance: no misses, every token matches.
    assert summary["logprob_only_misses"] == 0 and summary["mismatches"] == 0
    assert not ok and any("mean |dlogprob|" in f for f in summary["failures"])
    ok, summary = summary_of(verdict_set(37, shift=0.1))
    assert ok, summary["failures"]


def test_a_constant_token_server_fails() -> None:
    rng = random.Random(8)
    verdicts = []
    for i in range(37):
        record = random_record(rng)
        top = {int(t): lp for t, lp in record["top"]}
        answer = Answer(token=4242, top={**top, 4242: -0.1}, prompt_tokens=8, cached_tokens=0, seconds=0.1)
        verdicts.append(checker.judge_builds(f"c{i}", {"b": record}, answer, 8))
    ok, summary = summary_of(verdicts)
    assert not ok and summary["mismatches"] == 37


def test_a_token_outside_every_build_is_never_a_logprob_only_miss() -> None:
    rng = random.Random(9)
    for _ in range(CASES):
        records = {f"b{i}": random_record(rng) for i in range(rng.randint(1, 3))}
        answer = random_answer(rng, list(records.values()))
        best = checker.judge_builds("c", records, answer, 8)
        accepted = {int(t) for r in records.values() for t in r["tokens"]}
        if best.logprob_miss:
            assert best.kind == "match" and best.got in accepted


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"ALL {len(tests)} UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
