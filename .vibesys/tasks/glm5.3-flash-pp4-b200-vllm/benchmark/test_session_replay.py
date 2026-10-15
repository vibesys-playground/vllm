#!/usr/bin/env python3
"""CPU unit tests of session_replay.py's recording sample and replay-reference gate (no server, no engine).

Each test checks a property over many seeded, generated inputs. Runs under
pytest or as a script (`python benchmark/test_session_replay.py`).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_replay as sr  # noqa: E402

CASES = 200


def random_trace(rng: random.Random, sessions: int) -> sr.TraceSessions:
    """A canonical session trace (session_runner's rules): contiguous sessions, nondecreasing arrivals."""
    out: sr.TraceSessions = {}
    arrival = 0.0
    for s in range(sessions):
        sid = str(rng.randrange(10**6)) + f"x{s}"
        rows, context = [], 0
        for r in range(rng.randint(1, 40)):
            prefix = 0 if r == 0 else context + 1
            new = rng.randint(1 if r == 0 else 0, 5_000) or (0 if prefix else 1)
            rows.append({"request_id": f"session_{sid}_round_{r:06d}", "session_id": sid, "round_idx": str(r),
                         "arrival_time_ms": f"{arrival:.6f}", "prefix_len": str(prefix), "input_len": str(new),
                         "output_len": "1", "tool_wait_after_ms": f"{rng.uniform(0, 30_000):.6f}"})
            context = prefix + new
        out[sid] = rows
        arrival += rng.expovariate(1 / 500)
    return out


def random_sample(rng: random.Random) -> sr.ReplaySample:
    return sr.ReplaySample(seed=rng.randrange(2**32), sessions=rng.randint(1, 60), round_cap=rng.randint(1, 40),
                           concurrency=rng.randint(1, 64), time_limit_s=600.0)


def small_workload(rng: random.Random) -> sr.Workload:
    return dataclasses.replace(sr.WORKLOAD, max_concurrency=rng.randint(1, 50), window_s=rng.uniform(10, 600))


def test_sample_is_deterministic_and_seeded() -> None:
    rng = random.Random(1)
    differ = 0
    for _ in range(CASES):
        depths = {str(i): rng.randint(0, 30) for i in range(rng.randint(1, 300))}
        sample = random_sample(rng)
        first = sr.sample_rounds(depths, sample)
        assert first == sr.sample_rounds(dict(depths), sample)
        assert list(first) == list(sr.sample_rounds(depths, sample))  # same order too
        other = sr.sample_rounds(depths, dataclasses.replace(sample, seed=sample.seed + 1))
        differ += first != other
    assert differ > CASES // 2, differ  # the seed matters wherever a stratum has a choice


def test_sample_spreads_one_session_per_stratum_of_reachable_sessions() -> None:
    rng = random.Random(2)
    for _ in range(CASES):
        depths = {str(i): rng.choice([0, 0, rng.randint(1, 80)]) for i in range(rng.randint(1, 400))}
        sample = random_sample(rng)
        chosen = sr.sample_rounds(depths, sample)
        population = [sid for sid, depth in depths.items() if depth > 0]
        count = min(sample.sessions, len(population))
        assert len(chosen) == count
        # Trace order, and exactly one session from each contiguous stratum.
        assert list(chosen) == [sid for sid in population if sid in chosen]
        for k in range(count):
            stratum = population[k * len(population) // count:(k + 1) * len(population) // count]
            assert sum(sid in chosen for sid in stratum) == 1, (k, stratum, chosen)
        for sid, rounds in chosen.items():
            # Unreachable sessions are never chosen; rounds start at 0 and stop at the ceiling or the cap.
            assert rounds == min(depths[sid], sample.round_cap) >= 1


def test_sample_trace_keeps_ordinals_and_replays_chosen_sessions_from_round_0() -> None:
    rng = random.Random(3)
    for _ in range(CASES // 4):
        sessions = random_trace(rng, rng.randint(1, 60))
        depths = sr.ceiling_depths(sessions, small_workload(rng))
        chosen = sr.sample_rounds(depths, random_sample(rng))
        rows = sr.sample_trace_rows(sessions, chosen)
        order = list(dict.fromkeys(row["session_id"] for row in rows))
        # Ordinal of every kept session = its position in the full trace.
        assert order == list(sessions)[:len(order)]
        assert (not chosen and not rows) or order[-1] == list(chosen)[-1]
        by_session: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(row)
            assert row["tool_wait_after_ms"] == "0.000000"
            assert row["request_id"] == f"session_{row['session_id']}_round_{int(row['round_idx']):06d}"
        for sid, kept in by_session.items():
            if sid in chosen:
                assert len(kept) == chosen[sid]
                for row, original in zip(kept, sessions[sid], strict=False):
                    same = ("round_idx", "prefix_len", "input_len", "output_len", "arrival_time_ms")
                    assert all(row[k] == original[k] for k in same)
            else:
                assert [(r["round_idx"], r["prefix_len"], r["input_len"]) for r in kept] == [("0", "0", "1")]
        arrivals = [float(rows[i]["arrival_time_ms"]) for i in range(len(rows))]
        assert arrivals == sorted(arrivals) and (not rows or arrivals[0] == 0.0)


def test_sample_trace_file_round_trips_and_matches_the_ceiling() -> None:
    rng = random.Random(4)
    with tempfile.TemporaryDirectory() as tmp:
        for case in range(10):
            sessions = random_trace(rng, rng.randint(1, 40))
            full = Path(tmp) / f"full{case}.csv"
            sr.write_trace(full, [row for rows in sessions.values() for row in rows])
            assert sr.trace_sessions(full) == sessions
            workload = small_workload(rng)
            facts = sr.trace_facts(full, workload)
            assert facts.ceiling_rounds == sum(sr.ceiling_depths(sessions, workload).values())
            assert facts.rounds == sum(len(rows) for rows in sessions.values())
            with full.open(newline="") as handle:
                assert next(csv.reader(handle)) == sr.TRACE_COLUMNS


def agreement_cases(rng: random.Random):
    for _ in range(CASES):
        reference = {f"{s}/{r}": (f"{rng.randrange(4):x}", rng.randrange(3)) for s in range(20) for r in range(5)}
        answers = {key: (rng.choice([h, "f"]), rng.choice([t, 9])) for key, (h, t) in reference.items()
                   if rng.random() < 0.8}
        answers.update({f"x/{i}": ("0", 0) for i in range(rng.randrange(5))})  # rounds the reference lacks
        yield reference, answers


def test_agreement_counts_reached_matched_and_agreed() -> None:
    rng = random.Random(5)
    for reference, answers in agreement_cases(rng):
        got = sr.replay_agreement(answers, reference)
        reached = [k for k in answers if k in reference]
        matched = [k for k in reached if answers[k][0] == reference[k][0]]
        agreed = [k for k in matched if answers[k][1] == reference[k][1]]
        assert (got.reached, got.matched, got.agreed) == (len(reached), len(matched), len(agreed))
        assert (got.first_disagreement is None) == (len(agreed) == len(matched))


def test_gate_passes_exactly_when_agreement_and_matched_rounds_reach_their_minimums() -> None:
    rng = random.Random(6)
    for _ in range(CASES * 5):
        matched = rng.randint(0, 3_000)
        agreement = sr.Agreement(reached=matched + rng.randint(0, 50), matched=matched,
                                 agreed=rng.randint(0, matched), first_disagreement="k")
        workload = dataclasses.replace(sr.WORKLOAD, replay_min_agreement=rng.choice([0.8, 0.9, 0.93, 1.0]),
                                       replay_min_matched_rounds=rng.choice([0, 1, 500, 1_000, 2_000]))
        should_pass = agreement.share >= workload.replay_min_agreement and matched >= workload.replay_min_matched_rounds
        try:
            sr.check_agreement(agreement, workload)
            passed, partial = True, None
        except sr.Rejected as exc:
            passed, partial = False, exc.partial
            assert f"gate {workload.replay_min_agreement:.0%}" in str(exc)
        assert passed == should_pass, (agreement, workload.replay_min_agreement, workload.replay_min_matched_rounds)
        if partial is not None:
            failed = ("replay_token_agreement" if agreement.share < workload.replay_min_agreement
                      else "replay_matched_rounds")
            assert partial["name"] == failed


def test_a_server_that_dodges_matching_is_rejected() -> None:
    """Wrong answers make every later prompt unmatched: only round 0s and first flips match, all wrong."""
    rng = random.Random(7)
    for _ in range(CASES):
        sessions, depth = rng.randint(1, 300), rng.randint(1, 32)
        reference = {f"{s}/{r}": (f"h{s}.{r}", 1) for s in range(sessions) for r in range(depth)}
        honest_rounds = rng.randint(0, depth - 1)  # right answers before the server starts skipping work
        answers = {}
        for s in range(sessions):
            for r in range(depth):
                matched_prompt = r <= honest_rounds
                answers[f"{s}/{r}"] = (f"h{s}.{r}" if matched_prompt else "other", 1 if r < honest_rounds else 2)
        agreement = sr.replay_agreement(answers, reference)
        try:
            sr.check_agreement(agreement, sr.WORKLOAD)
        except sr.Rejected:
            continue
        # Passing needs >= 90% agreement over >= the floor: honest on almost every checked round.
        assert agreement.matched >= sr.REPLAY_MIN_MATCHED_ROUNDS and honest_rounds >= 9, agreement


def test_result_reports_the_gate_and_the_defaults_are_the_named_values() -> None:
    assert sr.WORKLOAD.replay_min_agreement == sr.REPLAY_MIN_AGREEMENT
    assert sr.WORKLOAD.replay_min_matched_rounds == sr.REPLAY_MIN_MATCHED_ROUNDS
    assert sr.WORKLOAD.replay_sample == sr.REPLAY_SAMPLE
    assert {"replay_token_agreement", "replay_matched_rounds", "replay_agreement_gate",
            "replay_min_matched_rounds"} <= set(sr.METRICS)
    metrics = {"rounds": 90, "rounds_per_s": 4.5, "new_tokens_per_s": 4000.0, "computed_tokens_per_s": 1.0,
               "prefix_hit_share": 0.9, "ttft_s": {"p50": 0.1, "p90": 0.2, "p99": 5.0, "max": 9.0},
               "rounds_ttft_over_60s": 0}
    watch = sr.LogWatch(over_bound_limit=10, window_s=1200.0, ttft_bound_s=60.0)
    facts = sr.TraceFacts(rounds=100, sessions=4, ceiling_rounds=100, ceiling_new_tokens=100_000)
    agreement = sr.Agreement(reached=1_200, matched=1_100, agreed=1_050, first_disagreement=None)
    values = sr.judge(metrics, watch, facts, sr.WORKLOAD, {"screen_ttft_16k_s": 0.1, "screen_ttft_32k_s": 0.2},
                      agreement)
    assert set(values) == set(sr.METRICS)
    assert values["replay_agreement_gate"] == sr.REPLAY_MIN_AGREEMENT
    assert values["replay_min_matched_rounds"] == sr.REPLAY_MIN_MATCHED_ROUNDS
    assert values["replay_matched_rounds"] == 1_100


def test_both_modes_pin_the_token_pool() -> None:
    """A session's prompts depend on the pool size, so the derived recording trace must not change it."""
    for mode in ("trace-timed", "saturated"):
        argv = sr.session_runner_argv(Path("e"), Path("t"), Path("x"), Path("k"), "http://h/v1", "m", sr.WORKLOAD,
                                      Path("w"), arrival_mode=mode)
        assert argv[argv.index("--token-pool-limit") + 1] == str(sr.TOKEN_POOL_TOKENS)
        assert argv[argv.index("--arrival-mode") + 1] == mode


def test_reference_round_trips_with_its_sampling_and_old_formats_are_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        answers = {"3/0": ("ab", 7), "3/1": ("cd", 8)}
        sr.write_replay_reference(path, answers, sr.WORKLOAD, sr.MODEL, {"recorded_rounds": 2})
        body = json.loads(path.read_text())
        assert body["sampling"] == dataclasses.asdict(sr.REPLAY_SAMPLE)
        assert sr.load_replay_reference(path, sr.WORKLOAD, sr.MODEL) == answers
        path.write_text(json.dumps(dict(body, format="glm53-session-replay-reference-v1")))
        try:
            sr.load_replay_reference(path, sr.WORKLOAD, sr.MODEL)
        except sr.Rejected as exc:
            assert "re-record" in str(exc)
        else:
            raise AssertionError("a v1 (unsampled) reference was accepted")


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"ALL {len(tests)} UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
