#!/usr/bin/env python3
"""Decode-loop tests for token selection, run without an NPU.

``runtime.generate`` imports MLIR-AIE and every IRON design at module scope, so
this test installs import stubs for the ``aie`` and ``designs`` packages before
importing it. Nothing here executes a kernel: the loop is driven with a fake
``decode_token`` that returns canned logits, which is exactly the surface that
decides *which* token a request emits.
"""

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUBBED_PACKAGES = ("aie", "designs")


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


class _StubFinder(MetaPathFinder, Loader):
    """Resolve the hardware-only imports to inert placeholder modules."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root not in STUBBED_PACKAGES:
            return None
        return ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []


sys.meta_path.insert(0, _StubFinder())

from runtime.generate import NPUDecoder  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402


EOS = 9
VOCAB = 10


class FakeTokenizer:
    eos_id = EOS

    def __init__(self, prompt_ids):
        self.prompt_ids = list(prompt_ids)

    def encode_chat(self, messages):
        return list(self.prompt_ids)

    def encode_chat_window(self, messages, budget):
        return list(self.prompt_ids), {"dropped_messages": 0}

    def decode(self, ids):
        return "".join(f"<{token}>" for token in ids)

    def token_bytes(self, token_id):
        return self.decode([token_id]).encode()


class SplitCharacterTokenizer(FakeTokenizer):
    """Token 7 carries the first byte of "ç" and token 8 the second."""

    PIECES = {7: b"\xc3", 8: b"\xa7"}

    def decode(self, ids):
        data = b"".join(self.PIECES.get(token, b"<%d>" % token) for token in ids)
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token_id):
        return self.PIECES.get(token_id, b"<%d>" % token_id)


def build_decoder(prompt_ids, logits_for, *, context_length=64, tokenizer=None):
    """An NPUDecoder whose decode step is a pure function of the input token."""
    decoder = object.__new__(NPUDecoder)
    decoder.context_length = context_length
    decoder.tokenizer = (tokenizer or FakeTokenizer)(prompt_ids)
    decoder.npu_layers = 0
    decoder.layers = 0
    decoder.calls = []
    decoder._cached_prefix = []
    decoder._prefix_cache_enabled = True

    def decode_token(
        token_id,
        position,
        *,
        diagnostics=False,
        select=None,
        compute_logits=True,
    ):
        logits = np.asarray(logits_for(token_id), dtype=np.float32)
        decoder.calls.append(
            (token_id, position, select is not None, compute_logits)
        )
        if not compute_logits:
            return None, 0.001
        chosen = int(np.argmax(logits)) if select is None else int(select(logits))
        return chosen, 0.001

    decoder.decode_token = decode_token
    return decoder


def run(decoder, max_new_tokens=4, sampling=None, **options):
    text = []
    stats = None
    for piece, final_stats, *_ in decoder.generate_messages(
        [{"role": "user", "content": "hi"}],
        max_new_tokens,
        sampling=sampling,
        **options,
    ):
        text.append(piece)
        if final_stats is not None:
            stats = final_stats
    return "".join(text), stats


def _ramp(token_id):
    """Deterministic logits: the next token is always ``token_id + 1``."""
    logits = np.zeros(VOCAB, dtype=np.float32)
    logits[(token_id + 1) % VOCAB] = 5.0
    return logits


def test_greedy_default_is_unchanged():
    decoder = build_decoder([1, 2, 3], _ramp)
    text, stats = run(decoder)
    assert stats["generated_token_ids"] == [4, 5, 6, 7]
    assert text == "<4><5><6><7>"
    assert stats["finish_reason"] == "length"
    assert stats["sampling"]["mode"] == "greedy"
    assert stats["decode_steps"] == 3
    assert stats["decode_seconds"] == 0.003
    # No prompt position engaged a sampler.
    assert [engaged for _, _, engaged, _ in decoder.calls] == [False] * 6
    assert [enabled for _, _, _, enabled in decoder.calls] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_sampler_runs_once_per_emitted_token():
    decoder = build_decoder([1, 2, 3], _ramp)
    _, stats = run(decoder, sampling=SamplingParams.build(temperature=1.0, seed=5))
    engaged = [engaged for _, _, engaged, _ in decoder.calls]
    # Three prompt positions, but only the last one produces a used token.
    assert engaged == [False, False, True, True, True, True]
    assert stats["sampling"]["mode"] == "sampled"
    assert stats["sampling"]["seed"] == 5


def test_seeded_generation_is_reproducible_and_seeds_differ():
    params = SamplingParams.build(temperature=1.0, seed=99)
    flat = lambda token_id: np.ones(VOCAB, dtype=np.float32)  # noqa: E731
    first = run(build_decoder([1], flat), max_new_tokens=8, sampling=params)[1]
    second = run(build_decoder([1], flat), max_new_tokens=8, sampling=params)[1]
    assert first["generated_token_ids"] == second["generated_token_ids"]

    other = SamplingParams.build(temperature=1.0, seed=100)
    third = run(build_decoder([1], flat), max_new_tokens=8, sampling=other)[1]
    assert third["generated_token_ids"] != first["generated_token_ids"]


def test_repetition_penalty_sees_prompt_and_generated_history():
    # Every step scores token 4 highest and token 5 second, so unpenalized
    # greedy decoding repeats 4 forever.
    def sticky(token_id):
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[4] = 4.0
        logits[5] = 3.0
        logits[6] = 2.5
        return logits

    _, greedy = run(build_decoder([1, 2], sticky), max_new_tokens=3)
    assert greedy["generated_token_ids"] == [4, 4, 4]

    _, penalized = run(
        build_decoder([1, 2], sticky),
        max_new_tokens=3,
        sampling=SamplingParams.build(repetition_penalty=2.0),
    )
    assert penalized["generated_token_ids"] == [4, 5, 6]

    # A token already present in the prompt is penalized from the first step.
    _, prompt_penalized = run(
        build_decoder([4, 4], sticky),
        max_new_tokens=1,
        sampling=SamplingParams.build(repetition_penalty=2.0),
    )
    assert prompt_penalized["generated_token_ids"] == [5]


def test_eos_stops_a_sampled_generation():
    def to_eos(token_id):
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[EOS if token_id == 7 else 7] = 5.0
        return logits

    _, stats = run(
        build_decoder([1], to_eos),
        max_new_tokens=6,
        sampling=SamplingParams.build(temperature=0.01, seed=1),
    )
    assert stats["generated_token_ids"] == [7]
    assert stats["finish_reason"] == "stop"


def test_invalid_sampling_is_rejected_by_the_decoder():
    decoder = build_decoder([1], _ramp)
    for bad in ({"temperature": 99.0}, {"top_k": -1}, {"unknown": 1}):
        try:
            run(decoder, sampling=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted by generate_messages")


def test_shared_prompt_prefix_is_not_recomputed():
    decoder = build_decoder([1, 2, 3], _ramp)
    _, first = run(decoder)
    assert first["cached_prompt_tokens"] == 0
    # The generated tokens were written to the cache after the prompt.
    assert decoder._cached_prefix == [1, 2, 3, 4, 5, 6]

    # A follow-up prompt extends the first conversation.
    decoder.tokenizer.prompt_ids = [1, 2, 3, 4, 5, 2]
    decoder.calls.clear()
    _, second = run(decoder)
    assert second["cached_prompt_tokens"] == 5
    assert second["prompt_tokens"] == 6
    assert [position for _, position, _, _ in decoder.calls][:1] == [5]
    assert second["generated_token_ids"] == [3, 4, 5, 6]

    # An identical prompt still recomputes its last position for the logits.
    decoder.calls.clear()
    _, repeat = run(decoder)
    assert repeat["cached_prompt_tokens"] == 5
    assert repeat["generated_token_ids"] == second["generated_token_ids"]

    # A prompt that diverges early reuses only the common prefix.
    decoder.tokenizer.prompt_ids = [1, 7, 3]
    decoder.calls.clear()
    _, diverged = run(decoder)
    assert diverged["cached_prompt_tokens"] == 1
    assert decoder.calls[0][:2] == (7, 1)


def test_prefix_cache_can_be_disabled_and_reset():
    decoder = build_decoder([1, 2, 3], _ramp)
    run(decoder)
    decoder.reset_prefix_cache()
    decoder.calls.clear()
    _, stats = run(decoder)
    assert stats["cached_prompt_tokens"] == 0
    assert decoder.calls[0][:2] == (1, 0)

    decoder._prefix_cache_enabled = False
    decoder.calls.clear()
    _, stats = run(decoder)
    assert stats["cached_prompt_tokens"] == 0
    assert len(decoder.calls) == 6


def test_failed_step_invalidates_its_position():
    decoder = build_decoder([1, 2, 3], _ramp)
    run(decoder)
    original = decoder.decode_token

    def failing(token_id, position, **kwargs):
        if position == 1:
            raise RuntimeError("kernel failure")
        return original(token_id, position, **kwargs)

    decoder.decode_token = failing
    decoder.tokenizer.prompt_ids = [1, 9, 9, 9]
    try:
        run(decoder)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failing step did not raise")
    assert decoder._cached_prefix == [1]


def test_stop_sequence_ends_generation_without_emitting_it():
    decoder = build_decoder([1], _ramp)
    text, stats = run(decoder, max_new_tokens=8, stop=["<4><"])
    assert text == "<2><3>"
    assert stats["finish_reason"] == "stop"
    assert stats["stop_sequence"] == "<4><"
    # Generation stopped at the token that completed the sequence.
    assert stats["generated_token_ids"] == [2, 3, 4, 5]

    # A partial match that is ruled out is released unchanged.
    text, stats = run(build_decoder([1], _ramp), max_new_tokens=3, stop="<3>x")
    assert text == "<2><3><4>"
    assert stats["finish_reason"] == "length"
    assert stats["stop_sequence"] is None


def test_split_utf8_character_is_emitted_whole():
    def spell(token_id):
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[{1: 7, 7: 8}.get(token_id, EOS)] = 5.0
        return logits

    decoder = build_decoder([1], spell, tokenizer=SplitCharacterTokenizer)
    pieces = [
        piece
        for piece, stats in decoder.generate_messages(
            [{"role": "user", "content": "hi"}], 4
        )
        if stats is None
    ]
    assert pieces == ["ç"]


def test_logprobs_follow_the_emitted_text():
    def spell(token_id):
        logits = np.full(VOCAB, -1.0, dtype=np.float32)
        logits[{1: 7, 7: 8, 8: 3}.get(token_id, EOS)] = 5.0
        return logits

    decoder = build_decoder([1], spell, tokenizer=SplitCharacterTokenizer)
    chunks = list(
        decoder.generate_messages(
            [{"role": "user", "content": "hi"}], 8, logprobs=2
        )
    )
    assert all(len(chunk) == 3 for chunk in chunks)
    text_chunks = [(text, entries) for text, stats, entries in chunks if stats is None]
    # The first byte's entry waits for the chunk that completes the character.
    assert [text for text, _ in text_chunks] == ["ç", "<3>"]
    assert [len(entries) for _, entries in text_chunks] == [2, 1]
    entry = text_chunks[0][1][0]
    assert entry["bytes"] == [0xC3]
    assert entry["logprob"] < 0 and entry["logprob"] > -0.1
    assert len(entry["top_logprobs"]) == 2
    assert entry["top_logprobs"][0]["logprob"] == entry["logprob"]
    stats = chunks[-1][1]
    # The end-of-turn token is not reported.
    assert stats["finish_reason"] == "stop"
    assert stats["generated_token_ids"] == [7, 8, 3]


def test_logprobs_do_not_change_greedy_tokens():
    plain = run(build_decoder([1, 2], _ramp), max_new_tokens=5)[1]
    scored = run(build_decoder([1, 2], _ramp), max_new_tokens=5, logprobs=0)[1]
    assert scored["generated_token_ids"] == plain["generated_token_ids"]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} decode-loop tests")


if __name__ == "__main__":
    main()
