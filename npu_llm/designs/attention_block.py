"""Fused KV-cache update and grouped-query attention for one decode token."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/attention_block_bf16.cc"
KV_HEADS = 3
HEAD_DIM = 64
Q_PER_KV = 3
CONTEXT = 64


@iron.jit
def attention_block(
    PackedQKV: In,
    Cache: InOut,
    O: Out,
    *,
    position: CompileTime[int],
):
    packed_ty = np.ndarray[(KV_HEADS, (Q_PER_KV + 2) * HEAD_DIM), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(KV_HEADS, 2 * CONTEXT * HEAD_DIM), np.dtype[bfloat16]]
    out_ty = np.ndarray[(KV_HEADS, Q_PER_KV * HEAD_DIM), np.dtype[bfloat16]]
    packed_slice = np.ndarray[((Q_PER_KV + 2) * HEAD_DIM,), np.dtype[bfloat16]]
    cache_slice = np.ndarray[(2 * CONTEXT * HEAD_DIM,), np.dtype[bfloat16]]
    out_slice = np.ndarray[(Q_PER_KV * HEAD_DIM,), np.dtype[bfloat16]]
    kernel = ExternalFunction(
        "attention_block_bf16",
        source_file=str(SRC),
        arg_types=[
            packed_slice,
            cache_slice,
            np.int32,
            cache_slice,
            out_slice,
        ],
        include_dirs=[config.cxx_header_path()],
    )

    def split(parent, child_ty, width, name):
        return parent.cons().split(
            [head * width for head in range(KV_HEADS)],
            obj_types=[child_ty] * KV_HEADS,
            names=[f"{name}_{head}" for head in range(KV_HEADS)],
            depths=[1] * KV_HEADS,
        )

    packed_parent = ObjectFifo(packed_ty, name="packed_parent", depth=1)
    cache_in_parent = ObjectFifo(cache_ty, name="cache_in_parent", depth=1)
    packed_children = split(
        packed_parent, packed_slice, (Q_PER_KV + 2) * HEAD_DIM, "packed"
    )
    cache_in_children = split(
        cache_in_parent, cache_slice, 2 * CONTEXT * HEAD_DIM, "cache_in"
    )

    def join(parent, child_ty, width, name):
        return parent.prod().join(
            [head * width for head in range(KV_HEADS)],
            obj_types=[child_ty] * KV_HEADS,
            names=[f"{name}_{head}" for head in range(KV_HEADS)],
            depths=[1] * KV_HEADS,
        )

    cache_out_parent = ObjectFifo(cache_ty, name="cache_out_parent", depth=1)
    o_parent = ObjectFifo(out_ty, name="o_parent", depth=1)
    cache_out_children = join(
        cache_out_parent, cache_slice, 2 * CONTEXT * HEAD_DIM, "cache_out"
    )
    o_children = join(o_parent, out_slice, Q_PER_KV * HEAD_DIM, "o")

    workers = []
    for head in range(KV_HEADS):
        packed_f = packed_children[head]
        cache_in_f = cache_in_children[head]
        cache_out_f, of = cache_out_children[head], o_children[head]

        def core_fn(packed_port, cache_in_port, cache_out_port, o_port, fn):
            packed = packed_port.acquire(1)
            cache_in = cache_in_port.acquire(1)
            cache_out = cache_out_port.acquire(1)
            o = o_port.acquire(1)
            fn(packed, cache_in, position, cache_out, o)
            packed_port.release(1)
            cache_in_port.release(1)
            cache_out_port.release(1)
            o_port.release(1)

        workers.append(
            Worker(
                core_fn,
                [
                    packed_f.cons(), cache_in_f.cons(), cache_out_f.prod(),
                    of.prod(), kernel,
                ],
            )
        )

    rt = Runtime()
    with rt.sequence(packed_ty, cache_ty, out_ty) as (packed, cache, out):
        rt.start(*workers)
        rt.fill(packed_parent.prod(), packed, tile=Tile(0, 0))
        rt.fill(cache_in_parent.prod(), cache, tile=Tile(1, 0))
        rt.drain(cache_out_parent.cons(), cache, wait=True, tile=Tile(2, 0))
        rt.drain(o_parent.cons(), out, wait=True, tile=Tile(3, 0))
    return Program(iron.get_current_device(), rt).resolve_program()
