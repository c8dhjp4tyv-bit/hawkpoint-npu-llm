import importlib.util
from pathlib import Path

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


if __name__ == "__main__":
    test_per_channel_quantization()
    print("PASS!")
