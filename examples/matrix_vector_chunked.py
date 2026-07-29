#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Chunked multi-core matrix-vector multiply for large M on NPU1/AIE2."""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


MAX_DMA_REPEAT = 64


def _tap(tensor_dims, offset, sizes, strides):
    return TensorAccessPattern(tensor_dims, offset, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def matrix_vector_chunked(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n_cores: CompileTime[int],
    max_chunk_tiles: CompileTime[int] = MAX_DMA_REPEAT,
    vectorized: CompileTime[bool] = True,
):
    rows_per_core = M // n_cores
    row_tiles_per_core = rows_per_core // m
    k_tiles = K // k

    matvec_kernel = kernels.mv(dim_m=m, dim_k=k, vectorized=vectorized)
    zero_kernel = matvec_kernel.zero

    dtype_in = np.dtype[np.int16]
    dtype_out = np.dtype[np.int32]
    A_ty = np.ndarray[(M, K), dtype_in]
    B_ty = np.ndarray[(1, K), dtype_in]
    C_ty = np.ndarray[(1, M), dtype_out]
    inA_ty = np.ndarray[(m, k), dtype_in]
    inB_ty = np.ndarray[(k,), dtype_in]
    outC_ty = np.ndarray[(m,), dtype_out]
    a_dims_from_stream = [(m, 2), (k // 2, 2 * m), (2, 1)] if vectorized else None

    def core_fn(of_a, of_b, of_c, zero, matvec):
        for _ in range_(row_tiles_per_core):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(k_tiles):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matvec(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    B_fifo = ObjectFifo(inB_ty, name="B_broadcast")
    memA_fifos = []
    outC_fifos = []
    workers = []
    for i in range(n_cores):
        mem_a = ObjectFifo(inA_ty, name=f"memA{i}")
        core_a = mem_a.cons().forward(
            name=f"coreA{i}", dims_from_stream=a_dims_from_stream
        )
        out_c = ObjectFifo(outC_ty, name=f"outC{i}")
        memA_fifos.append(mem_a)
        outC_fifos.append(out_c)
        workers.append(
            Worker(
                core_fn,
                [
                    core_a.cons(),
                    B_fifo.cons(),
                    out_c.prod(),
                    zero_kernel,
                    matvec_kernel,
                ],
            )
        )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(*workers)
        for chunk_start in range(0, row_tiles_per_core, max_chunk_tiles):
            chunk_tiles = min(max_chunk_tiles, row_tiles_per_core - chunk_start)
            chunk_rows = chunk_tiles * m
            b_tap = _tap(
                (1, K),
                0,
                [chunk_tiles, 1, 1, K],
                [0, 0, 0, 1],
            )
            tg = rt.task_group()
            rt.fill(
                B_fifo.prod(), b_in, b_tap, task_group=tg, tile=Tile(0, 0)
            )
            for i in range(n_cores):
                row_start = i * rows_per_core + chunk_start * m
                a_tap = _tap(
                    (M, K),
                    row_start * K,
                    [chunk_tiles, k_tiles, m, k],
                    [m * K, k, K, 1],
                )
                c_tap = _tap(
                    (1, M),
                    row_start,
                    [1, 1, 1, chunk_rows],
                    [0, 0, 0, 1],
                )
                rt.fill(
                    memA_fifos[i].prod(),
                    a_in,
                    a_tap,
                    task_group=tg,
                    tile=Tile(i, 0),
                )
                rt.drain(
                    outC_fifos[i].cons(),
                    c_out,
                    c_tap,
                    wait=True,
                    task_group=tg,
                    tile=Tile(i, 0),
                )
            rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Chunked Matrix-Vector Multiplication")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=3072)
    p.add_argument("-K", type=int, default=576)
    p.add_argument("-N", type=int, default=1, help="accepted but unused")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-c", "--cores", type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument("--chunk-tiles", type=int, default=MAX_DMA_REPEAT)
    p.add_argument("--dtype_in", choices=["i16"], default="i16")
    p.add_argument("--dtype_out", choices=["i32"], default="i32")
    p.add_argument("--scalar", action="store_true")
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print resolved MLIR without compiling or running",
    )
    add_benchmark_args(p)
    return p


def _validate(opts):
    if opts.use_chess:
        sys.exit("--use-chess 1 is unavailable in this Peano setup")
    if not 1 <= opts.chunk_tiles <= MAX_DMA_REPEAT:
        sys.exit(f"--chunk-tiles must be in [1, {MAX_DMA_REPEAT}]")
    if opts.M % (opts.m * opts.cores) != 0:
        sys.exit(
            f"-M {opts.M} must be divisible by -m * --cores "
            f"({opts.m * opts.cores})"
        )
    if opts.K % opts.k != 0:
        sys.exit(f"-K {opts.K} must be divisible by -k {opts.k}")


def _run_and_verify(opts):
    rng = np.random.default_rng(1726250518)
    A_np = rng.integers(-1000, 1000, size=(opts.M, opts.K), dtype=np.int16)
    B_np = rng.integers(-1000, 1000, size=(opts.K,), dtype=np.int16)
    A_t = iron.tensor(A_np.reshape(-1), dtype=np.int16, device="npu")
    B_t = iron.tensor(B_np, dtype=np.int16, device="npu")
    C_t = iron.zeros(opts.M, dtype=np.int32, device="npu")

    bench = run_iters(
        matrix_vector_chunked,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        m=opts.m,
        k=opts.k,
        n_cores=opts.cores,
        max_chunk_tiles=opts.chunk_tiles,
        vectorized=not opts.scalar,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    actual = C_t.numpy().reshape(opts.M)
    chunks = iron.ceildiv(opts.M // (opts.m * opts.cores), opts.chunk_tiles)
    print(
        f"Configuration                  : {opts.cores} core(s), "
        f"{chunks} DMA chunk(s)/core"
    )
    print(f"Maximum DMA pattern repeat     : {opts.chunk_tiles} / 64")
    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K,
        gflops_fmt=".4f",
        fail_msg="chunked output does not match A @ b",
    )


def _compile_kwargs(opts):
    return {
        "M": opts.M,
        "K": opts.K,
        "m": opts.m,
        "k": opts.k,
        "n_cores": opts.cores,
        "max_chunk_tiles": opts.chunk_tiles,
        "vectorized": not opts.scalar,
    }


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        matrix_vector_chunked,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=o.cores),
        validate=_validate,
    )


if __name__ == "__main__":
    main()
