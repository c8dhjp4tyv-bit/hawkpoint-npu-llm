"""Fused-xclbin W8/BF16 projection with NPU quantization and dequantization."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/project_w8bf16.cc"
RMS_SRC = Path(__file__).resolve().parents[1] / "kernels/rmsnorm_bf16.cc"
ROWS = 32


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def project(
    W: In,
    Scale: In,
    X: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
):
    w_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    s_ty = np.ndarray[(M,), np.dtype[np.float32]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    wb_ty = np.ndarray[(ROWS, K), np.dtype[np.int8]]
    sb_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    cb_ty = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    flags = [f"-DDIM_M={ROWS}", f"-DDIM_K={K}"]
    acc_fn = ExternalFunction(
        "project_w8bf16_acc32",
        source_file=str(SRC),
        arg_types=[wb_ty, x_ty, acc_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=flags,
    )
    dequant_fn = Kernel(
        "project_dequant_bf16",
        acc_fn.object_file_name,
        [acc_ty, sb_ty, cb_ty],
    )
    fw, fx = ObjectFifo(wb_ty, depth=1), ObjectFifo(x_ty, depth=1)
    fs, facc, fc = (
        ObjectFifo(sb_ty, depth=1),
        ObjectFifo(acc_ty, depth=1),
        ObjectFifo(cb_ty, depth=1),
    )

    def accumulate(of_w, of_x, of_acc, fn):
        x = of_x.acquire(1)
        for _ in range_(M // ROWS):
            w, acc = of_w.acquire(1), of_acc.acquire(1)
            fn(w, x, acc)
            of_w.release(1)
            of_acc.release(1)
        of_x.release(1)

    def dequant(of_acc, of_s, of_c, fn):
        for _ in range_(M // ROWS):
            acc, s, c = of_acc.acquire(1), of_s.acquire(1), of_c.acquire(1)
            fn(acc, s, c)
            of_acc.release(1)
            of_s.release(1)
            of_c.release(1)

    workers = [
        Worker(accumulate, [fw.cons(), fx.cons(), facc.prod(), acc_fn]),
        Worker(dequant, [facc.cons(), fs.cons(), fc.prod(), dequant_fn]),
    ]
    rt = Runtime()
    with rt.sequence(w_ty, s_ty, x_ty, c_ty) as (w, s, x, c):
        rt.start(*workers)
        for start in range(0, M // ROWS, 64):
            blocks = min(64, M // ROWS - start)
            elements = blocks * ROWS
            w_tap = TensorAccessPattern(
                (M, K),
                start * ROWS * K,
                [blocks, 1, 1, ROWS * K],
                [ROWS * K, 0, 0, 1],
            )
            s_tap = TensorAccessPattern(
                (M,), start * ROWS, [1, 1, 1, elements], [0, 0, 0, 1]
            )
            c_tap = TensorAccessPattern(
                (M,), start * ROWS, [1, 1, 1, elements], [0, 0, 0, 1]
            )
            tg = rt.task_group()
            rt.fill(fw.prod(), w, w_tap, task_group=tg)
            rt.fill(fs.prod(), s, s_tap, task_group=tg)
            if start == 0:
                rt.fill(fx.prod(), x, task_group=tg)
            rt.drain(fc.cons(), c, c_tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def norm_project(
    W: In,
    Scale: In,
    X: In,
    Gamma: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
):
    w_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    s_ty = np.ndarray[(M,), np.dtype[np.float32]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    wb_ty = np.ndarray[(ROWS, K), np.dtype[np.int8]]
    sb_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    cb_ty = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    flags = [f"-DDIM_M={ROWS}", f"-DDIM_K={K}"]
    acc_fn = ExternalFunction(
        "project_w8bf16_acc32",
        source_file=str(SRC),
        arg_types=[wb_ty, x_ty, acc_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=flags,
    )
    dequant_fn = Kernel(
        "project_dequant_bf16",
        acc_fn.object_file_name,
        [acc_ty, sb_ty, cb_ty],
    )
    rms_fn = ExternalFunction(
        "rmsnorm_bf16",
        source_file=str(RMS_SRC),
        arg_types=[x_ty, x_ty, x_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    fw, fs = ObjectFifo(wb_ty, depth=1), ObjectFifo(sb_ty, depth=1)
    fhidden, fgamma = ObjectFifo(x_ty, depth=1), ObjectFifo(x_ty, depth=1)
    fnorm = ObjectFifo(x_ty, depth=1)
    facc, fc = ObjectFifo(acc_ty, depth=1), ObjectFifo(cb_ty, depth=1)

    def normalize(of_x, of_g, of_norm, fn):
        x, g, norm = of_x.acquire(1), of_g.acquire(1), of_norm.acquire(1)
        fn(x, g, norm, K)
        of_x.release(1)
        of_g.release(1)
        of_norm.release(1)

    def accumulate(of_w, of_x, of_acc, fn):
        x = of_x.acquire(1)
        for _ in range_(M // ROWS):
            w, acc = of_w.acquire(1), of_acc.acquire(1)
            fn(w, x, acc)
            of_w.release(1)
            of_acc.release(1)
        of_x.release(1)

    def dequant(of_acc, of_s, of_c, fn):
        for _ in range_(M // ROWS):
            acc, s, c = of_acc.acquire(1), of_s.acquire(1), of_c.acquire(1)
            fn(acc, s, c)
            of_acc.release(1)
            of_s.release(1)
            of_c.release(1)

    workers = [
        Worker(normalize, [fhidden.cons(), fgamma.cons(), fnorm.prod(), rms_fn]),
        Worker(accumulate, [fw.cons(), fnorm.cons(), facc.prod(), acc_fn]),
        Worker(dequant, [facc.cons(), fs.cons(), fc.prod(), dequant_fn]),
    ]
    rt = Runtime()
    with rt.sequence(w_ty, s_ty, x_ty, x_ty, c_ty) as (w, s, x, gamma, c):
        rt.start(*workers)
        first_tg = rt.task_group()
        rt.fill(fhidden.prod(), x, task_group=first_tg)
        rt.fill(fgamma.prod(), gamma, task_group=first_tg)
        for start in range(0, M // ROWS, 64):
            blocks = min(64, M // ROWS - start)
            elements = blocks * ROWS
            w_tap = TensorAccessPattern(
                (M, K),
                start * ROWS * K,
                [blocks, 1, 1, ROWS * K],
                [ROWS * K, 0, 0, 1],
            )
            linear_tap = TensorAccessPattern(
                (M,), start * ROWS, [1, 1, 1, elements], [0, 0, 0, 1]
            )
            tg = first_tg if start == 0 else rt.task_group()
            rt.fill(fw.prod(), w, w_tap, task_group=tg)
            rt.fill(fs.prod(), s, linear_tap, task_group=tg)
            rt.drain(fc.cons(), c, linear_tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def project_residual(
    W: In,
    Scale: In,
    X: In,
    Residual: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
):
    w_ty = np.ndarray[(M, K), np.dtype[np.int8]]
    s_ty = np.ndarray[(M,), np.dtype[np.float32]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    wb_ty = np.ndarray[(ROWS, K), np.dtype[np.int8]]
    sb_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    cb_ty = np.ndarray[(ROWS,), np.dtype[bfloat16]]
    flags = [f"-DDIM_M={ROWS}", f"-DDIM_K={K}"]
    acc_fn = ExternalFunction(
        "project_w8bf16_acc32",
        source_file=str(SRC),
        arg_types=[wb_ty, x_ty, acc_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=flags,
    )
    dequant_fn = Kernel(
        "project_dequant_bf16", acc_fn.object_file_name, [acc_ty, sb_ty, cb_ty]
    )
    add_fn = Kernel(
        "project_residual_add_bf16",
        acc_fn.object_file_name,
        [cb_ty, cb_ty, cb_ty],
    )
    fw, fs, fx, fr = (
        ObjectFifo(wb_ty, depth=1),
        ObjectFifo(sb_ty, depth=1),
        ObjectFifo(x_ty, depth=1),
        ObjectFifo(cb_ty, depth=1),
    )
    facc, fp, fc = (
        ObjectFifo(acc_ty, depth=1),
        ObjectFifo(cb_ty, depth=1),
        ObjectFifo(cb_ty, depth=1),
    )

    def accumulate(of_w, of_x, of_acc, fn):
        x = of_x.acquire(1)
        for _ in range_(M // ROWS):
            w, acc = of_w.acquire(1), of_acc.acquire(1)
            fn(w, x, acc)
            of_w.release(1)
            of_acc.release(1)
        of_x.release(1)

    def dequant(of_acc, of_s, of_p, fn):
        for _ in range_(M // ROWS):
            acc, s, p = of_acc.acquire(1), of_s.acquire(1), of_p.acquire(1)
            fn(acc, s, p)
            of_acc.release(1)
            of_s.release(1)
            of_p.release(1)

    def add(of_p, of_r, of_c, fn):
        for _ in range_(M // ROWS):
            p, r, c = of_p.acquire(1), of_r.acquire(1), of_c.acquire(1)
            fn(p, r, c)
            of_p.release(1)
            of_r.release(1)
            of_c.release(1)

    workers = [
        Worker(accumulate, [fw.cons(), fx.cons(), facc.prod(), acc_fn]),
        Worker(dequant, [facc.cons(), fs.cons(), fp.prod(), dequant_fn]),
        Worker(add, [fp.cons(), fr.cons(), fc.prod(), add_fn]),
    ]
    rt = Runtime()
    with rt.sequence(w_ty, s_ty, x_ty, c_ty, c_ty) as (w, s, x, residual, c):
        rt.start(*workers)
        first_tg = rt.task_group()
        rt.fill(fx.prod(), x, task_group=first_tg)
        rt.fill(fr.prod(), residual, task_group=first_tg)
        for start in range(0, M // ROWS, 64):
            blocks = min(64, M // ROWS - start)
            elements = blocks * ROWS
            w_tap = TensorAccessPattern(
                (M, K),
                start * ROWS * K,
                [blocks, 1, 1, ROWS * K],
                [ROWS * K, 0, 0, 1],
            )
            linear_tap = TensorAccessPattern(
                (M,), start * ROWS, [1, 1, 1, elements], [0, 0, 0, 1]
            )
            tg = first_tg if start == 0 else rt.task_group()
            rt.fill(fw.prod(), w, w_tap, task_group=tg)
            rt.fill(fs.prod(), s, linear_tap, task_group=tg)
            rt.drain(fc.cons(), c, linear_tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()
