"""Hardware acceptance test for the complete NPU chatbot."""

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from npu_llm.runtime.generate import NPUDecoder


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

    print("PASS NPU end-to-end")
    print(f"argmax={first_token}")
    print(f"cold_seconds={cold_seconds:.4f}")
    print(f"warm_seconds={warm_seconds:.4f}")
    print(f"answer={answer!r}")
    print(f"stats={stats}")


if __name__ == "__main__":
    main()
