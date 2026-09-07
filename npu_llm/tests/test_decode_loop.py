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

    def decode(self, ids):
        return "".join(f"<{token}>" for token in ids)


def build_decoder(prompt_ids, logits_for, *, context_length=64):
    """An NPUDecoder whose decode step is a pure function of the input token."""
    decoder = object.__new__(NPUDecoder)
    decoder.context_length = context_length
    decoder.tokenizer = FakeTokenizer(prompt_ids)
    decoder.npu_layers = 0
    decoder.layers = 0
    decoder.calls = []

    def decode_token(token_id, position, *, diagnostics=False, select=None):
        logits = np.asarray(logits_for(token_id), dtype=np.float32)
        decoder.calls.append((token_id, position, select is not None))
        chosen = int(np.argmax(logits)) if select is None else int(select(logits))
        return chosen, 0.001

    decoder.decode_token = decode_token
    return decoder


def run(decoder, max_new_tokens=4, sampling=None):
    text = []
    stats = None
    for piece, final_stats in decoder.generate_messages(
        [{"role": "user", "content": "hi"}], max_new_tokens, sampling=sampling
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
    # No prompt position engaged a sampler.
    assert [engaged for _, _, engaged in decoder.calls] == [False] * 7


def test_sampler_runs_once_per_emitted_token():
    decoder = build_decoder([1, 2, 3], _ramp)
    _, stats = run(decoder, sampling=SamplingParams.build(temperature=1.0, seed=5))
    engaged = [engaged for _, _, engaged in decoder.calls]
    # Three prompt positions, but only the last one produces a used token.
    assert engaged == [False, False, True, True, True, True, True]
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
