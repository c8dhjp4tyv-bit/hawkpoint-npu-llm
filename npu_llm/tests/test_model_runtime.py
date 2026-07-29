from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.model import XDNA1Model
from runtime.tokenizer import SmolLMTokenizer


def main():
    model_dir = Path(
        os.environ.get(
            "HAWKPOINT_MODEL_DIR",
            ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16",
        )
    )
    model = XDNA1Model(model_dir)
    assert model.layer(0)["qkv"].shape == (960, 576)
    assert model.quantized("lm_head").shape == (49152, 576)
    tokenizer = SmolLMTokenizer(model_dir)
    ids = tokenizer.encode_chat([{"role": "user", "content": "Hello"}])
    assert len(ids) > 4
    assert tokenizer.eos_id is not None
    print(f"PASS! prompt_tokens={len(ids)}, eos_id={tokenizer.eos_id}")


if __name__ == "__main__":
    main()
