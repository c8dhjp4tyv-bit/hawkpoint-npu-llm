#!/usr/bin/env python3
"""Host-side sampling tests. Pure NumPy: no NPU, no converted model."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.sampling import (  # noqa: E402
    GREEDY,
    Sampler,
    SamplingError,
    SamplingParams,
    apply_penalties,
    filter_top_k,
    filter_top_p,
)


LOGITS = np.array([0.5, 3.0, -1.0, 2.0, 1.0], dtype=np.float32)


def test_default_is_greedy():
    assert GREEDY.greedy
    assert GREEDY.deterministic
    sampler = Sampler()
    assert sampler.greedy
    assert sampler.seed is None
    assert sampler.select(LOGITS) == int(np.argmax(LOGITS)) == 1
    assert sampler.stats()["mode"] == "greedy"


def test_zero_temperature_ignores_truncation():
    params = SamplingParams.build(temperature=0.0, top_p=0.1, top_k=3)
    assert params.greedy
    assert Sampler(params).select(LOGITS) == 1


def test_top_k_one_is_greedy():
    params = SamplingParams.build(temperature=1.5, top_k=1)
    assert params.greedy
    for _ in range(20):
        assert Sampler(params).select(LOGITS) == 1


def test_seed_makes_sampling_reproducible():
    params = SamplingParams.build(temperature=1.0, seed=1234)
    first = [Sampler(params).select(LOGITS) for _ in range(8)]
    second = [Sampler(params).select(LOGITS) for _ in range(8)]
    assert first == second
    assert Sampler(params).seed == 1234
    assert params.deterministic

    # One sampler's stream advances, so consecutive draws are not locked
    # together the way independently seeded samplers are.
    stream = Sampler(params)
    assert len({stream.select(LOGITS) for _ in range(64)}) > 1


def test_unseeded_sampling_reports_its_seed():
    params = SamplingParams.build(temperature=1.0)
    sampler = Sampler(params)
    assert sampler.seed is not None
    assert not params.deterministic
    # Replaying the reported seed reproduces the run.
    replay = Sampler(SamplingParams.build(temperature=1.0, seed=sampler.seed))
    assert [sampler.select(LOGITS) for _ in range(8)] == [
        replay.select(LOGITS) for _ in range(8)
    ]


def test_temperature_widens_the_distribution():
    cold = Sampler(SamplingParams.build(temperature=0.1, seed=7))
    hot = Sampler(SamplingParams.build(temperature=2.0, seed=7))
    cold_draws = {cold.select(LOGITS) for _ in range(200)}
    hot_draws = {hot.select(LOGITS) for _ in range(200)}
    assert cold_draws == {1}
    assert len(hot_draws) == LOGITS.shape[0]


def test_top_k_and_top_p_bound_the_candidate_set():
    masked = filter_top_k(LOGITS.copy(), 2)
    assert np.isfinite(masked).sum() == 2
    assert sorted(np.flatnonzero(np.isfinite(masked)).tolist()) == [1, 3]

    # An unreachable nucleus still keeps the single most likely token.
    nucleus = filter_top_p(LOGITS.copy(), 1e-6)
    assert np.flatnonzero(np.isfinite(nucleus)).tolist() == [1]

    # top_p == 1 keeps everything; a mid nucleus keeps a strict subset.
    assert np.isfinite(filter_top_p(LOGITS.copy(), 1.0)).all()
    partial = np.isfinite(filter_top_p(LOGITS.copy(), 0.9)).sum()
    assert 1 < partial < LOGITS.shape[0]

    sampler = Sampler(SamplingParams.build(temperature=1.0, top_k=2, seed=3))
    assert {sampler.select(LOGITS) for _ in range(200)} == {1, 3}


def test_repetition_penalty_demotes_seen_tokens_of_either_sign():
    params = SamplingParams.build(repetition_penalty=2.0)
    penalized = apply_penalties(LOGITS.copy(), [1, 2], params)
    assert penalized[1] == 1.5  # positive logit divided
    assert penalized[2] == -2.0  # negative logit multiplied
    assert penalized[3] == LOGITS[3]  # untouched
    # Strong enough to change the greedy choice.
    assert Sampler(params).select(LOGITS, [1]) == 3


def test_frequency_and_presence_penalties_use_openai_semantics():
    params = SamplingParams.build(frequency_penalty=0.5, presence_penalty=0.25)
    penalized = apply_penalties(LOGITS.copy(), [1, 1, 1, 3], params)
    assert np.isclose(penalized[1], 3.0 - 1.5 - 0.25)
    assert np.isclose(penalized[3], 2.0 - 0.5 - 0.25)
    assert penalized[0] == LOGITS[0]


def test_penalties_ignore_out_of_range_history():
    params = SamplingParams.build(repetition_penalty=2.0)
    penalized = apply_penalties(LOGITS.copy(), [-4, 99, 1], params)
    assert penalized[1] == 1.5
    assert penalized.tolist()[2:] == LOGITS.tolist()[2:]


def test_penalties_leave_argmax_intact_when_neutral():
    params = SamplingParams.build(repetition_penalty=1.0)
    assert apply_penalties(LOGITS.copy(), [1, 2], params).tolist() == LOGITS.tolist()


def test_penalized_greedy_decoding_stays_deterministic():
    params = SamplingParams.build(temperature=0.0, repetition_penalty=1.6)
    assert not params.greedy  # penalties must still be applied
    assert params.deterministic
    assert [Sampler(params).select(LOGITS, [1]) for _ in range(4)] == [3] * 4


def test_request_parsing_and_validation():
    params = SamplingParams.from_request(
        {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "presence_penalty": 0.5,
            "seed": 11,
        }
    )
    assert params.temperature == 0.7
    assert params.top_k == 40
    assert params.frequency_penalty == 0.0
    assert SamplingParams.from_request({}) == GREEDY

    rejected = [
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"temperature": "hot"},
        {"temperature": float("nan")},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": -1},
        {"top_k": 1.5},
        {"top_k": True},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": 3.0},
        {"presence_penalty": -3.0},
        {"frequency_penalty": 9.0},
        {"seed": -1},
        {"seed": "x"},
    ]
    for request in rejected:
        try:
            SamplingParams.from_request(request)
        except SamplingError:
            continue
        raise AssertionError(f"{request} was accepted")
    assert issubclass(SamplingError, ValueError)


def test_transport_round_trip_revalidates():
    params = SamplingParams.build(temperature=0.8, top_k=5, seed=2)
    assert SamplingParams.from_dict(params.to_dict()) == params
    assert SamplingParams.from_dict(None) == GREEDY
    for bad in ({"temperature": 99.0}, {"nope": 1}, "not-an-object"):
        try:
            SamplingParams.from_dict(bad)
        except SamplingError:
            continue
        raise AssertionError(f"{bad!r} survived transport validation")


def test_stats_report_the_effective_configuration():
    stats = Sampler(SamplingParams.build(temperature=0.9, top_p=0.8, seed=5)).stats()
    assert stats["mode"] == "sampled"
    assert stats["seed"] == 5
    assert stats["temperature"] == 0.9
    assert stats["top_p"] == 0.8


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} sampling tests")


if __name__ == "__main__":
    main()
