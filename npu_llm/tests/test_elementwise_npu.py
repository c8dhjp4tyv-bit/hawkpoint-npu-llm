from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from designs.elementwise import dequantize, quantize, residual_add, swiglu


def close(actual, expected, atol=0.08):
    np.testing.assert_allclose(
        actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=0.08
    )


def main():
    set_current_device(from_name("npu"))
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.5, 576).astype(bfloat16)
    X = iron.tensor(x, dtype=bfloat16, device="npu")
    Q = iron.zeros(576, dtype=np.int16, device="npu")
    quantize(X, Q, size=576)
    np.testing.assert_allclose(Q.numpy(), x.astype(np.float32) * 256, atol=1.1)

    acc = rng.integers(-10000, 10000, 960, dtype=np.int32)
    scale = rng.uniform(0.001, 0.01, 960).astype(np.float32)
    A = iron.tensor(acc, dtype=np.int32, device="npu")
    S = iron.tensor(scale, dtype=np.float32, device="npu")
    D = iron.zeros(960, dtype=bfloat16, device="npu")
    dequantize(A, S, D, size=960)
    close(D.numpy(), (acc * scale / 256).astype(bfloat16))

    y = rng.normal(0, 0.5, 576).astype(bfloat16)
    Y = iron.tensor(y, dtype=bfloat16, device="npu")
    R = iron.zeros(576, dtype=bfloat16, device="npu")
    residual_add(X, Y, R, size=576)
    close(R.numpy(), (x.astype(np.float32) + y.astype(np.float32)).astype(bfloat16))

    gu = rng.normal(0, 0.5, 3072).astype(bfloat16)
    GU = iron.tensor(gu, dtype=bfloat16, device="npu")
    SW = iron.zeros(1536, dtype=bfloat16, device="npu")
    swiglu(GU, SW, size=1536)
    gate, up = np.split(gu.astype(np.float32), 2)
    expected = (gate / (1 + np.exp(-gate)) * up).astype(bfloat16)
    close(SW.numpy(), expected, atol=0.12)
    print("PASS!")


if __name__ == "__main__":
    main()
