"""Tiled AIE2 elementwise operations used by the decoder."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/decoder_elementwise.cc"
TILE = 64


def _extern(name, arg_types):
    runtime_headers = Path(config.root_path()) / "aie_runtime_lib/AIE2"
    return ExternalFunction(
        name,
        source_file=str(SRC),
        arg_types=arg_types,
        include_dirs=[config.cxx_header_path(), str(runtime_headers)],
        compile_flags=[f"-DTILE_SIZE={TILE}"],
    )


def _unary_design(size, in_dtype, out_dtype, symbol):
    in_ty = np.ndarray[(size,), np.dtype[in_dtype]]
    out_ty = np.ndarray[(size,), np.dtype[out_dtype]]
    in_tile = np.ndarray[(TILE,), np.dtype[in_dtype]]
    out_tile = np.ndarray[(TILE,), np.dtype[out_dtype]]
    fn = _extern(symbol, [in_tile, out_tile])
    fi, fo = ObjectFifo(in_tile), ObjectFifo(out_tile)

    def core(of_i, of_o, kernel):
        for _ in range_(size // TILE):
            i, o = of_i.acquire(1), of_o.acquire(1)
            kernel(i, o)
            of_i.release(1)
            of_o.release(1)

    worker = Worker(core, [fi.cons(), fo.prod(), fn])
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (i, o):
        rt.start(worker)
        rt.fill(fi.prod(), i)
        rt.drain(fo.cons(), o, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def quantize(A: In, C: Out, *, size: CompileTime[int]):
    return _unary_design(size, bfloat16, np.int16, "quantize_bf16_i16")


@iron.jit
def dequantize(A: In, Scale: In, C: Out, *, size: CompileTime[int]):
    a_ty = np.ndarray[(size,), np.dtype[np.int32]]
    s_ty = np.ndarray[(size,), np.dtype[np.float32]]
    c_ty = np.ndarray[(size,), np.dtype[bfloat16]]
    at = np.ndarray[(TILE,), np.dtype[np.int32]]
    st = np.ndarray[(TILE,), np.dtype[np.float32]]
    ct = np.ndarray[(TILE,), np.dtype[bfloat16]]
    fn = _extern("dequant_i32_bf16", [at, st, ct])
    fa, fs, fc = ObjectFifo(at), ObjectFifo(st), ObjectFifo(ct)

    def core(of_a, of_s, of_c, kernel):
        for _ in range_(size // TILE):
            a, s, c = of_a.acquire(1), of_s.acquire(1), of_c.acquire(1)
            kernel(a, s, c)
            of_a.release(1)
            of_s.release(1)
            of_c.release(1)

    worker = Worker(core, [fa.cons(), fs.cons(), fc.prod(), fn])
    rt = Runtime()
    with rt.sequence(a_ty, s_ty, c_ty) as (a, s, c):
        rt.start(worker)
        rt.fill(fa.prod(), a)
        rt.fill(fs.prod(), s)
        rt.drain(fc.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def residual_add(A: In, B: In, C: Out, *, size: CompileTime[int]):
    ty = np.ndarray[(size,), np.dtype[bfloat16]]
    tile = np.ndarray[(TILE,), np.dtype[bfloat16]]
    fn = _extern("residual_add_bf16", [tile, tile, tile])
    fa, fb, fc = ObjectFifo(tile), ObjectFifo(tile), ObjectFifo(tile)

    def core(of_a, of_b, of_c, kernel):
        for _ in range_(size // TILE):
            a, b, c = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
            kernel(a, b, c)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = Worker(core, [fa.cons(), fb.cons(), fc.prod(), fn])
    rt = Runtime()
    with rt.sequence(ty, ty, ty) as (a, b, c):
        rt.start(worker)
        rt.fill(fa.prod(), a)
        rt.fill(fb.prod(), b)
        rt.drain(fc.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def swiglu(GateUp: In, C: Out, *, size: CompileTime[int]):
    input_ty = np.ndarray[(2 * size,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(size,), np.dtype[bfloat16]]
    tile = np.ndarray[(TILE,), np.dtype[bfloat16]]
    fn = _extern("swiglu_split_bf16", [tile, tile, tile])
    fg, fu, fc = ObjectFifo(tile), ObjectFifo(tile), ObjectFifo(tile)

    def core(of_g, of_u, of_c, kernel):
        for _ in range_(size // TILE):
            g, u, c = of_g.acquire(1), of_u.acquire(1), of_c.acquire(1)
            kernel(g, u, c)
            of_g.release(1)
            of_u.release(1)
            of_c.release(1)

    worker = Worker(core, [fg.cons(), fu.cons(), fc.prod(), fn])
    gate_tap = TensorAccessPattern(
        (2 * size,), 0, [1, 1, 1, size], [0, 0, 0, 1]
    )
    up_tap = TensorAccessPattern(
        (2 * size,), size, [1, 1, 1, size], [0, 0, 0, 1]
    )
    rt = Runtime()
    with rt.sequence(input_ty, output_ty) as (gate_up, c):
        rt.start(worker)
        rt.fill(fg.prod(), gate_up, gate_tap)
        rt.fill(fu.prod(), gate_up, up_tap)
        rt.drain(fc.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()
