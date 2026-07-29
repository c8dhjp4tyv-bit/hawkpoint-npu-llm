"""NPU DMA-backed BF16 tensor slicing without host tensor arithmetic."""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_


TILE = 64


@iron.jit
def slice_bf16(
    A: In,
    C: Out,
    *,
    total_size: CompileTime[int],
    offset: CompileTime[int],
    size: CompileTime[int],
):
    in_ty = np.ndarray[(total_size,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(size,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    fi, fo = ObjectFifo(tile_ty), ObjectFifo(tile_ty)

    def core(of_i, of_o):
        for _ in range_(size // TILE):
            i, o = of_i.acquire(1), of_o.acquire(1)
            for j in range_(TILE):
                o[j] = i[j]
            of_i.release(1)
            of_o.release(1)

    worker = Worker(core, [fi.cons(), fo.prod()])
    tap = TensorAccessPattern(
        (total_size,), offset, [1, 1, 1, size], [0, 0, 0, 1]
    )
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a, c):
        rt.start(worker)
        rt.fill(fi.prod(), a, tap)
        rt.drain(fo.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
