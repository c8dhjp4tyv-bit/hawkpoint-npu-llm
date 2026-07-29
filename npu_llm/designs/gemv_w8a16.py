"""NPU test design for INT8-weight / INT16-activation GEMV."""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


KERNEL_SRC = Path(__file__).resolve().parents[1] / "kernels/mv_w8a16.cc"
ZERO_SRC = Path(__file__).resolve().parents[1] / "kernels/zero_i32.cc"


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemv_w8a16(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    max_chunk_tiles: CompileTime[int] = 64,
    vectorized: CompileTime[bool] = True,
):
    A_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    B_ty = np.ndarray[(1, K), np.dtype[np.int16]]
    C_ty = np.ndarray[(1, M), np.dtype[np.int32]]
    a_tile_ty = np.ndarray[(m, k), np.dtype[np.int8]]
    b_tile_ty = np.ndarray[(k,), np.dtype[np.int16]]
    c_tile_ty = np.ndarray[(m,), np.dtype[np.int32]]

    compile_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}"]
    kernel = ExternalFunction(
        "matvec_w8a16_vectorized" if vectorized else "matvec_w8a16_scalar",
        source_file=str(KERNEL_SRC),
        arg_types=[a_tile_ty, b_tile_ty, c_tile_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=compile_flags,
    )
    zero = ExternalFunction(
        "zero_w8a16_i32",
        source_file=str(ZERO_SRC),
        arg_types=[c_tile_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=compile_flags,
    )

    A_fifo = ObjectFifo(a_tile_ty, name="A")
    B_fifo = ObjectFifo(b_tile_ty, name="B")
    C_fifo = ObjectFifo(c_tile_ty, name="C")

    def core_fn(of_a, of_b, of_c, zero_fn, mv_fn):
        for _ in range_(M // m):
            elem_c = of_c.acquire(1)
            zero_fn(elem_c)
            for _ in range_(K // k):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                mv_fn(elem_a, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [A_fifo.cons(), B_fifo.cons(), C_fifo.prod(), zero, kernel],
    )
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(worker)
        for start in range(0, M // m, max_chunk_tiles):
            tiles = min(max_chunk_tiles, M // m - start)
            rows = tiles * m
            a_tap = TensorAccessPattern(
                (M, K),
                start * m * K,
                [tiles, K // k, m, k],
                [m * K, k, K, 1],
            )
            b_tap = TensorAccessPattern(
                (1, K), 0, [tiles, 1, 1, K], [0, 0, 0, 1]
            )
            c_tap = TensorAccessPattern(
                (1, M), start * m, [1, 1, 1, rows], [0, 0, 0, 1]
            )
            tg = rt.task_group()
            rt.fill(A_fifo.prod(), a_in, a_tap, task_group=tg)
            rt.fill(B_fifo.prod(), b_in, b_tap, task_group=tg)
            rt.drain(
                C_fifo.cons(), c_out, c_tap, wait=True, task_group=tg
            )
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


def _parser():
    p = argparse.ArgumentParser()
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=2048)
    p.add_argument("-K", type=int, default=576)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("--scalar", action="store_true")
    p.add_argument("--chunk-tiles", type=int, default=64)
    p.add_argument("--emit-mlir", action="store_true")
    add_benchmark_args(p)
    return p


def _validate(o):
    if o.M % o.m or o.K % o.k:
        sys.exit("M and K must divide evenly by m and k")
    if not 1 <= o.chunk_tiles <= 64:
        sys.exit("--chunk-tiles must be in [1, 64]")


def _kwargs(o):
    return dict(
        M=o.M,
        K=o.K,
        m=o.m,
        k=o.k,
        max_chunk_tiles=o.chunk_tiles,
        vectorized=not o.scalar,
    )


def _run(o):
    rng = np.random.default_rng(20260729)
    A_np = rng.integers(-127, 128, (o.M, o.K), dtype=np.int8)
    B_np = rng.integers(-255, 256, o.K, dtype=np.int16)
    A = iron.tensor(A_np, dtype=np.int8, device="npu")
    B = iron.tensor(B_np, dtype=np.int16, device="npu")
    C = iron.zeros(o.M, dtype=np.int32, device="npu")
    bench = run_iters(
        gemv_w8a16,
        A,
        B,
        C,
        **_kwargs(o),
        warmup=o.warmup,
        iters=o.iters,
    )
    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    assert_close_with_benchmark(
        C.numpy().reshape(o.M),
        expected,
        bench=bench,
        ops=2.0 * o.M * o.K,
        gflops_fmt=".4f",
        fail_msg="W8A16 output does not match NumPy",
    )


def main():
    o = _parser().parse_args()
    run_design_cli(
        gemv_w8a16,
        o,
        compile_kwargs=_kwargs,
        run_and_verify=_run,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
