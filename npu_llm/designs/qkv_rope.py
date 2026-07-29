"""Fused QKV split and Llama split-half RoPE in one xclbin."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/rope_bf16.cc"


@iron.jit
def qkv_rope(
    QKV: In,
    LUT: In,
    PackedQKV: Out,
    *,
    q_heads: CompileTime[int] = 9,
    kv_heads: CompileTime[int] = 3,
    head_dim: CompileTime[int] = 64,
):
    q_per_kv = q_heads // kv_heads
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size
    group_width = (q_per_kv + 2) * head_dim
    qkv_ty = np.ndarray[(qkv_size,), np.dtype[bfloat16]]
    lut_ty = np.ndarray[(64,), np.dtype[bfloat16]]
    packed_ty = np.ndarray[(qkv_size,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
    rope_fn = ExternalFunction(
        "rope_bf16",
        source_file=str(SRC),
        arg_types=[row_ty, row_ty, row_ty],
        include_dirs=[config.cxx_header_path()],
    )
    fq, fk, fv = ObjectFifo(row_ty), ObjectFifo(row_ty), ObjectFifo(row_ty)
    flq, flk = ObjectFifo(row_ty), ObjectFifo(row_ty)
    foq, fok, fov = ObjectFifo(row_ty), ObjectFifo(row_ty), ObjectFifo(row_ty)

    def rotate(of_x, of_lut, of_y, fn, heads):
        for _ in range_(heads):
            x, lut, y = of_x.acquire(1), of_lut.acquire(1), of_y.acquire(1)
            fn(x, lut, y)
            of_x.release(1)
            of_lut.release(1)
            of_y.release(1)

    def copy_v(of_x, of_y):
        for _ in range_(kv_heads):
            x, y = of_x.acquire(1), of_y.acquire(1)
            for j in range_(head_dim):
                y[j] = x[j]
            of_x.release(1)
            of_y.release(1)

    workers = [
        Worker(rotate, [fq.cons(), flq.cons(), foq.prod(), rope_fn, q_heads]),
        Worker(rotate, [fk.cons(), flk.cons(), fok.prod(), rope_fn, kv_heads]),
        Worker(copy_v, [fv.cons(), fov.prod()]),
    ]
    tap = lambda offset, size: TensorAccessPattern(
        (qkv_size,), offset, [1, 1, 1, size], [0, 0, 0, 1]
    )
    lut_q = TensorAccessPattern(
        (64,), 0, [q_heads, 1, 1, 64], [0, 0, 0, 1]
    )
    lut_k = TensorAccessPattern(
        (64,), 0, [kv_heads, 1, 1, 64], [0, 0, 0, 1]
    )
    rt = Runtime()
    q_pack = TensorAccessPattern(
        (qkv_size,),
        0,
        [kv_heads, q_per_kv, 1, head_dim],
        [group_width, head_dim, 0, 1],
    )
    k_pack = TensorAccessPattern(
        (qkv_size,),
        q_per_kv * head_dim,
        [kv_heads, 1, 1, head_dim],
        [group_width, 0, 0, 1],
    )
    v_pack = TensorAccessPattern(
        (qkv_size,),
        (q_per_kv + 1) * head_dim,
        [kv_heads, 1, 1, head_dim],
        [group_width, 0, 0, 1],
    )
    with rt.sequence(qkv_ty, lut_ty, packed_ty) as (qkv, lut, packed):
        rt.start(*workers)
        rt.fill(fq.prod(), qkv, tap(0, q_size), tile=Tile(0, 0))
        rt.fill(flq.prod(), lut, lut_q, tile=Tile(0, 0))
        rt.drain(foq.cons(), packed, q_pack, wait=True, tile=Tile(0, 0))
        rt.fill(fk.prod(), qkv, tap(q_size, kv_size), tile=Tile(1, 0))
        rt.fill(flk.prod(), lut, lut_k, tile=Tile(1, 0))
        rt.drain(fok.cons(), packed, k_pack, wait=True, tile=Tile(1, 0))
        rt.fill(fv.prod(), qkv, tap(q_size + kv_size, kv_size), tile=Tile(2, 0))
        rt.drain(fov.cons(), packed, v_pack, wait=True, tile=Tile(2, 0))
    return Program(iron.get_current_device(), rt).resolve_program()
