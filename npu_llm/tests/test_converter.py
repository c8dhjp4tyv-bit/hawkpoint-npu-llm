import importlib.util
import hashlib
from pathlib import Path
import tempfile

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "tools/convert_smollm2.py"
SPEC = importlib.util.spec_from_file_location("convert_smollm2", MODULE)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def test_per_channel_quantization():
    rng = np.random.default_rng(7)
    weight = rng.normal(0, 0.2, (32, 64)).astype(np.float32)
    q, scale = CONVERTER.quantize_per_output_channel(weight)
    restored = q.astype(np.float32) * scale[:, None]
    error = np.abs(restored - weight)
    assert q.dtype == np.int8
    assert scale.dtype == np.float32
    assert np.all(error.max(axis=1) <= scale / 2 + 1e-6)


def test_architecture_validation():
    valid = {
        **CONVERTER.EXPECTED_ARCHITECTURE,
        "model_type": "llama",
    }
    assert CONVERTER._validate_architecture(valid) == "llama"
    invalid = {**valid, "hidden_size": 960}
    try:
        CONVERTER._validate_architecture(invalid)
        raise AssertionError("incompatible architecture unexpectedly accepted")
    except ValueError as exc:
        assert "hidden_size=960" in str(exc)
    qwen = {
        **CONVERTER.QWEN_ARCHITECTURE,
        "model_type": "qwen2",
    }
    assert CONVERTER._validate_architecture(qwen) == "qwen2"


def test_atomic_publish_and_failed_conversion_preserves_previous():
    original = CONVERTER._convert_into
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        (output / "old.bin").write_bytes(b"old")

        def successful(_source, staging, **_metadata):
            data = b"complete-model"
            (staging / "weights.bin").write_bytes(data)
            return {
                "files": {
                    "weights.bin": {
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                }
            }

        CONVERTER._convert_into = successful
        CONVERTER.convert(source, output)
        assert not (output / "old.bin").exists()
        assert (output / "weights.bin").read_bytes() == b"complete-model"

        def failing(_source, staging, **_metadata):
            (staging / "partial.bin").write_bytes(b"partial")
            raise OSError("simulated disk-full failure")

        CONVERTER._convert_into = failing
        try:
            CONVERTER.convert(source, output)
            raise AssertionError("failed conversion unexpectedly succeeded")
        except OSError:
            pass
        assert (output / "weights.bin").read_bytes() == b"complete-model"
        assert not (output / "partial.bin").exists()
    CONVERTER._convert_into = original


if __name__ == "__main__":
    test_per_channel_quantization()
    test_architecture_validation()
    test_atomic_publish_and_failed_conversion_preserves_previous()
    print("PASS!")
