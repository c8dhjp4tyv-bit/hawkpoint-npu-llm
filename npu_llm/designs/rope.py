"""AIE2 RoPE validation for 9 query + 3 key heads."""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorTiler2D
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


SRC = Path(__file__).resolve().parents[1] / "kernels/rope_bf16.cc"


@iron.jit
def rope(A: In, LUT: In, C: Out, *, heads: CompileTime[int]):
    tensor_ty = np.ndarray[(heads, 64), np.dtype[bfloat16]]
    row_ty = np.ndarray[(64,), np.dtype[bfloat16]]
    lut_ty = np.ndarray[(1, 64), np.dtype[bfloat16]]
    fn = ExternalFunction(
        "rope_bf16",
        source_file=str(SRC),
        arg_types=[row_ty, row_ty, row_ty],
        include_dirs=[config.cxx_header_path()],
    )
    a_fifo = ObjectFifo(row_ty, name="qk")
    lut_fifo = ObjectFifo(row_ty, name="rope_lut")
    c_fifo = ObjectFifo(row_ty, name="rotated_qk")

    def core_fn(of_a, of_lut, of_c, kernel):
        for _ in range_(heads):
            a = of_a.acquire(1)
            lut = of_lut.acquire(1)
            c = of_c.acquire(1)
            kernel(a, lut, c)
            of_a.release(1)
            of_lut.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [a_fifo.cons(), lut_fifo.cons(), c_fifo.prod(), fn])
    lut_tap = TensorTiler2D.simple_tiler(
        (1, 64), pattern_repeat=heads, prune_step=False
    )[0]
    rt = Runtime()
    with rt.sequence(tensor_ty, lut_ty, tensor_ty) as (a, lut, c):
        rt.start(worker)
        rt.fill(a_fifo.prod(), a)
        rt.fill(lut_fifo.prod(), lut, lut_tap)
        rt.drain(c_fifo.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


def _parser():
    p = argparse.ArgumentParser()
    add_compile_args(p, short_dev=None)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--position", type=int, default=7)
    p.add_argument("--emit-mlir", action="store_true")
    add_benchmark_args(p)
    return p


def _run(o):
    rng = np.random.default_rng(11)
    a_np = rng.normal(0, 0.4, (o.heads, 64)).astype(bfloat16)
    inv_freq = 1.0 / (100000.0 ** (np.arange(0, 64, 2) / 64.0))
    angle = o.position * inv_freq
    lut = np.concatenate([np.cos(angle), np.sin(angle)]).astype(np.float32)
    lut = lut.astype(bfloat16).reshape(1, 64)
    A = iron.tensor(a_np, dtype=bfloat16, device="npu")
    L = iron.tensor(lut, dtype=bfloat16, device="npu")
    C = iron.zeros((o.heads, 64), dtype=bfloat16, device="npu")
    bench = run_iters(
        rope, A, L, C, heads=o.heads, warmup=o.warmup, iters=o.iters
    )
    x = a_np.astype(np.float32)
    expected = np.empty_like(x)
    expected[:, :32] = x[:, :32] * lut[0, :32] - x[:, 32:] * lut[0, 32:]
    expected[:, 32:] = x[:, 32:] * lut[0, :32] + x[:, :32] * lut[0, 32:]
    assert_close_with_benchmark(
        C.numpy(),
        expected.astype(bfloat16),
        bench=bench,
        float_rtol=0.03,
        float_atol=0.03,
        fail_msg="RoPE mismatch",
    )


def main():
    o = _parser().parse_args()
    run_design_cli(
        rope,
        o,
        compile_kwargs=lambda x: {"heads": x.heads},
        run_and_verify=_run,
    )


if __name__ == "__main__":
    main()
