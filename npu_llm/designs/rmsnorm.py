"""AIE2 RMSNorm validation design."""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


SRC = Path(__file__).resolve().parents[1] / "kernels/rmsnorm_bf16.cc"


@iron.jit
def rmsnorm(A: In, G: In, C: Out, *, size: CompileTime[int]):
    ty = np.ndarray[(size,), np.dtype[bfloat16]]
    kernel = ExternalFunction(
        "rmsnorm_bf16",
        source_file=str(SRC),
        arg_types=[ty, ty, ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )
    a_fifo = ObjectFifo(ty, name="input")
    g_fifo = ObjectFifo(ty, name="gamma")
    c_fifo = ObjectFifo(ty, name="output")

    def core_fn(of_a, of_g, of_c, fn):
        a = of_a.acquire(1)
        g = of_g.acquire(1)
        c = of_c.acquire(1)
        fn(a, g, c, size)
        of_a.release(1)
        of_g.release(1)
        of_c.release(1)

    worker = Worker(
        core_fn, [a_fifo.cons(), g_fifo.cons(), c_fifo.prod(), kernel]
    )
    rt = Runtime()
    with rt.sequence(ty, ty, ty) as (a, g, c):
        rt.start(worker)
        rt.fill(a_fifo.prod(), a)
        rt.fill(g_fifo.prod(), g)
        rt.drain(c_fifo.cons(), c, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


def _parser():
    p = argparse.ArgumentParser()
    add_compile_args(p, short_dev=None)
    p.add_argument("--size", type=int, default=576)
    p.add_argument("--emit-mlir", action="store_true")
    add_benchmark_args(p)
    return p


def _run(o):
    rng = np.random.default_rng(9)
    a_np = rng.normal(0, 0.5, o.size).astype(bfloat16)
    g_np = rng.normal(1, 0.1, o.size).astype(bfloat16)
    A = iron.tensor(a_np, dtype=bfloat16, device="npu")
    G = iron.tensor(g_np, dtype=bfloat16, device="npu")
    C = iron.zeros(o.size, dtype=bfloat16, device="npu")
    bench = run_iters(
        rmsnorm, A, G, C, size=o.size, warmup=o.warmup, iters=o.iters
    )
    x = a_np.astype(np.float32)
    expected = (
        x / np.sqrt(np.mean(x * x) + 1e-5) * g_np.astype(np.float32)
    ).astype(bfloat16)
    assert_close_with_benchmark(
        C.numpy(),
        expected,
        bench=bench,
        float_rtol=0.03,
        float_atol=0.03,
        fail_msg="RMSNorm mismatch",
    )


def main():
    o = _parser().parse_args()
    run_design_cli(
        rmsnorm,
        o,
        compile_kwargs=lambda x: {"size": x.size},
        run_and_verify=_run,
    )


if __name__ == "__main__":
    main()
