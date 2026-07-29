"""General BF16 projection kernels for the Qwen XDNA1 path."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/project_bf16.cc"


def _linear_tap(total, offset, size):
    return TensorAccessPattern(
        (total,), offset, [1, 1, 1, size], [0, 0, 0, 1]
    )


def _weight_tap(M, K, start, blocks, rows):
    return TensorAccessPattern(
        (M, K),
        start * rows * K,
        [blocks, 1, 1, rows * K],
        [rows * K, 0, 0, 1],
    )


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def project_bf16(
    W: In,
    X: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    rows: CompileTime[int] = 32,
):
    w_ty = np.ndarray[(M, K), np.dtype[bfloat16]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    wb_ty = np.ndarray[(rows, K), np.dtype[bfloat16]]
    cb_ty = np.ndarray[(rows,), np.dtype[bfloat16]]
    fn = ExternalFunction(
        "project_bf16_block_precise",
        source_file=str(SRC),
        arg_types=[wb_ty, x_ty, cb_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DDIM_M={rows}", f"-DDIM_K={K}"],
    )
    fw = ObjectFifo(wb_ty, depth=1)
    fx = ObjectFifo(x_ty, depth=1)
    fc = ObjectFifo(cb_ty, depth=1)

    def core(wp, xp, cp, kernel):
        x = xp.acquire(1)
        for _ in range_(M // rows):
            w, c = wp.acquire(1), cp.acquire(1)
            kernel(w, x, c)
            wp.release(1)
            cp.release(1)
        xp.release(1)

    worker = Worker(core, [fw.cons(), fx.cons(), fc.prod(), fn])
    rt = Runtime()
    with rt.sequence(w_ty, x_ty, c_ty) as (w, x, c):
        rt.start(worker)
        for start in range(0, M // rows, 64):
            blocks = min(64, M // rows - start)
            elements = blocks * rows
            tg = rt.task_group()
            rt.fill(
                fw.prod(),
                w,
                _weight_tap(M, K, start, blocks, rows),
                task_group=tg,
            )
            if start == 0:
                rt.fill(fx.prod(), x, task_group=tg)
            rt.drain(
                fc.cons(),
                c,
                _linear_tap(M, start * rows, elements),
                wait=True,
                task_group=tg,
            )
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def project_bf16_residual(
    W: In,
    X: In,
    Residual: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    rows: CompileTime[int] = 32,
):
    w_ty = np.ndarray[(M, K), np.dtype[bfloat16]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    wb_ty = np.ndarray[(rows, K), np.dtype[bfloat16]]
    cb_ty = np.ndarray[(rows,), np.dtype[bfloat16]]
    project_fn = ExternalFunction(
        "project_bf16_block_precise",
        source_file=str(SRC),
        arg_types=[wb_ty, x_ty, cb_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DDIM_M={rows}", f"-DDIM_K={K}"],
    )
    add_fn = Kernel(
        "project_bf16_residual",
        project_fn.object_file_name,
        [cb_ty, cb_ty, cb_ty],
    )
    fw = ObjectFifo(wb_ty, depth=1)
    fx = ObjectFifo(x_ty, depth=1)
    fr = ObjectFifo(cb_ty, depth=1)
    fp = ObjectFifo(cb_ty, depth=1)
    fc = ObjectFifo(cb_ty, depth=1)

    def project_core(wp, xp, pp, kernel):
        x = xp.acquire(1)
        for _ in range_(M // rows):
            w, p = wp.acquire(1), pp.acquire(1)
            kernel(w, x, p)
            wp.release(1)
            pp.release(1)
        xp.release(1)

    def add_core(pp, rp, cp, kernel):
        for _ in range_(M // rows):
            p, r, c = pp.acquire(1), rp.acquire(1), cp.acquire(1)
            kernel(p, r, c)
            pp.release(1)
            rp.release(1)
            cp.release(1)

    workers = [
        Worker(project_core, [fw.cons(), fx.cons(), fp.prod(), project_fn]),
        Worker(add_core, [fp.cons(), fr.cons(), fc.prod(), add_fn]),
    ]
    rt = Runtime()
    with rt.sequence(w_ty, x_ty, c_ty, c_ty) as (w, x, residual, c):
        rt.start(*workers)
        for start in range(0, M // rows, 64):
            blocks = min(64, M // rows - start)
            elements = blocks * rows
            linear = _linear_tap(M, start * rows, elements)
            tg = rt.task_group()
            rt.fill(
                fw.prod(),
                w,
                _weight_tap(M, K, start, blocks, rows),
                task_group=tg,
            )
            rt.fill(fr.prod(), residual, linear, task_group=tg)
            if start == 0:
                rt.fill(fx.prod(), x, task_group=tg)
            rt.drain(fc.cons(), c, linear, wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()
