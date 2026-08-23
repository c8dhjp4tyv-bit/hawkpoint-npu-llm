"""Validate the Qwen2.5 grouped-query RoPE/attention layout on XDNA1."""

from pathlib import Path
import sys

from ml_dtypes import bfloat16
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device
from npu_llm.designs.attention_block import attention_block
from npu_llm.designs.elementwise import swiglu
from npu_llm.designs.project import project
from npu_llm.designs.project_bf16 import project_bf16
from npu_llm.designs.qkv_rope import qkv_rope
from npu_llm.designs.rmsnorm import rmsnorm


Q_HEADS = 14
KV_HEADS = 2
Q_PER_KV = 7
HEAD_DIM = 64
CONTEXT = 64


def main():
    set_current_device(from_name("npu", n_cols=4))
    rng = np.random.default_rng(25)
    hidden_np = rng.normal(0, 0.2, 896).astype(bfloat16)
    gamma_np = rng.normal(1, 0.05, 896).astype(bfloat16)
    hidden = iron.tensor(hidden_np, dtype=bfloat16, device="npu")
    gamma = iron.tensor(gamma_np, dtype=bfloat16, device="npu")
    normalized = iron.zeros(896, dtype=bfloat16, device="npu")
    rmsnorm(hidden, gamma, normalized, size=896, epsilon=1e-6)
    hidden_f32 = hidden_np.astype(np.float32)
    expected_norm = (
        hidden_f32
        / np.sqrt(np.mean(hidden_f32 * hidden_f32) + 1e-6)
        * gamma_np.astype(np.float32)
    )
    np.testing.assert_allclose(
        normalized.numpy().astype(np.float32),
        expected_norm,
        atol=3e-2,
        rtol=3e-2,
    )

    weight_np = rng.integers(-4, 5, (32, 896), dtype=np.int8)
    scale_np = np.full(32, 0.01, dtype=np.float32)
    weight = iron.tensor(weight_np, dtype=np.int8, device="npu")
    scale = iron.tensor(scale_np, dtype=np.float32, device="npu")
    projected = iron.zeros(32, dtype=bfloat16, device="npu")
    project(weight, scale, normalized, projected, M=32, K=896)
    expected_project = (
        weight_np.astype(np.float32)
        @ normalized.numpy().astype(np.float32)
    ) * scale_np
    np.testing.assert_allclose(
        projected.numpy().astype(np.float32),
        expected_project,
        atol=8e-2,
        rtol=8e-2,
    )

    bf16_weight_np = rng.normal(0, 0.03, (32, 896)).astype(bfloat16)
    bf16_weight = iron.tensor(
        bf16_weight_np, dtype=bfloat16, device="npu"
    )
    bf16_projected = iron.zeros(32, dtype=bfloat16, device="npu")
    project_bf16(
        bf16_weight,
        normalized,
        bf16_projected,
        M=32,
        K=896,
        rows=32,
    )
    expected_bf16 = (
        bf16_weight_np.astype(np.float32)
        @ normalized.numpy().astype(np.float32)
    )
    np.testing.assert_allclose(
        bf16_projected.numpy().astype(np.float32),
        expected_bf16,
        atol=8e-2,
        rtol=8e-2,
    )

    gate_up_np = rng.normal(0, 3, 128).astype(bfloat16)
    gate_up = iron.tensor(gate_up_np, dtype=bfloat16, device="npu")
    activated = iron.zeros(64, dtype=bfloat16, device="npu")
    swiglu(gate_up, activated, size=64)
    gate, up = np.split(gate_up_np.astype(np.float32), 2)
    expected_activation = (
        gate / (1.0 + np.exp(-gate)) * up
    ).astype(bfloat16).astype(np.float32)
    np.testing.assert_allclose(
        activated.numpy().astype(np.float32),
        expected_activation,
        atol=1.5e-1,
        rtol=8e-2,
    )

    qkv_np = rng.normal(
        0,
        0.2,
        (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM,
    ).astype(bfloat16)
    angle = 3 * (1.0 / (1_000_000.0 ** (np.arange(32) / 32.0)))
    lut_np = np.concatenate([np.cos(angle), np.sin(angle)]).astype(bfloat16)

    qkv = iron.tensor(qkv_np, dtype=bfloat16, device="npu")
    lut = iron.tensor(lut_np, dtype=bfloat16, device="npu")
    packed = iron.zeros(
        (KV_HEADS, (Q_PER_KV + 2) * HEAD_DIM),
        dtype=bfloat16,
        device="npu",
    )
    qkv_rope(
        qkv,
        lut,
        packed,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )

    packed_np = packed.numpy().astype(np.float32)
    values = qkv_np[
        (Q_HEADS + KV_HEADS) * HEAD_DIM :
    ].astype(np.float32).reshape(KV_HEADS, HEAD_DIM)
    np.testing.assert_allclose(
        packed_np[:, (Q_PER_KV + 1) * HEAD_DIM :],
        values,
        atol=2e-2,
        rtol=2e-2,
    )

    cache = iron.zeros(
        (1, KV_HEADS, 2 * CONTEXT * HEAD_DIM),
        dtype=bfloat16,
        device="npu",
    )
    attended = iron.zeros(
        (KV_HEADS, Q_PER_KV * HEAD_DIM),
        dtype=bfloat16,
        device="npu",
    )
    attention_block(
        packed,
        cache,
        attended,
        position=0,
        kv_heads=KV_HEADS,
        q_per_kv=Q_PER_KV,
        head_dim=HEAD_DIM,
    )
    expected = np.repeat(values[:, None, :], Q_PER_KV, axis=1)
    np.testing.assert_allclose(
        attended.numpy().astype(np.float32).reshape(
            KV_HEADS, Q_PER_KV, HEAD_DIM
        ),
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    print("PASS Qwen2.5 RoPE/attention on NPU")


if __name__ == "__main__":
    main()
