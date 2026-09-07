"""Hardware acceptance test for the complete NPU chatbot."""

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.generate import NPUDecoder
from runtime.sampling import SamplingParams


def _token_ids(decoder, prompt, sampling):
    """Generate once and return only the generated token ids."""
    for _, final_stats in decoder.generate(
        prompt, max_new_tokens=32, sampling=sampling
    ):
        if final_stats is not None:
            return final_stats["generated_token_ids"]
    raise AssertionError("generation produced no statistics")


def main():
    model_dir = Path(
        os.environ.get(
            "HAWKPOINT_MODEL_DIR",
            ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16",
        )
    )

    smoke = NPUDecoder(model_dir)
    first_token, cold_seconds = smoke.decode_token(2, 0)
    second_token, warm_seconds = smoke.decode_token(2, 1)
    assert first_token == 198, (
        f"reference-logit argmax mismatch: expected 198, got {first_token}"
    )

    decoder = NPUDecoder(model_dir)
    prompt = (
        "Explain in simple terms why the sky looks blue during the day. "
        "Give a detailed answer."
    )
    pieces = []
    stats = None
    for text, final_stats in decoder.generate(prompt, max_new_tokens=32):
        pieces.append(text)
        if final_stats is not None:
            stats = final_stats
    answer = "".join(pieces)
    assert stats is not None
    assert stats["generated_tokens"] == 32
    assert "sky" in answer.lower() and "blue" in answer.lower()
    assert len(answer.split()) >= 15

    # Sampling gate: an explicit greedy request must reproduce the greedy
    # tokens exactly, and a seeded sampled request must be reproducible.
    greedy_ids = stats["generated_token_ids"]
    explicit_greedy = _token_ids(
        decoder, prompt, SamplingParams.build(temperature=0.0)
    )
    assert explicit_greedy == greedy_ids, (
        "temperature=0 diverged from the greedy reference path"
    )

    seeded = SamplingParams.build(temperature=0.8, top_p=0.95, seed=20260907)
    first_sampled = _token_ids(decoder, prompt, seeded)
    second_sampled = _token_ids(decoder, prompt, seeded)
    assert first_sampled == second_sampled, (
        "a seeded sampled run was not reproducible"
    )
    assert len(first_sampled) == 32

    print("PASS NPU end-to-end")
    print(f"argmax={first_token}")
    print(f"cold_seconds={cold_seconds:.4f}")
    print(f"warm_seconds={warm_seconds:.4f}")
    print(f"answer={answer!r}")
    print(f"stats={stats}")
    print(f"greedy_token_ids={greedy_ids}")
    print(f"seeded_sampled_token_ids={first_sampled}")


if __name__ == "__main__":
    main()
