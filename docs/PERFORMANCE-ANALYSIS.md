# Performance Analysis

## Bottom Line

The Qwen2.5-0.5B native NPU path is **not faster than an eight-thread CPU
baseline** on measured Hawk Point hardware. This document explains the
hardware budget and identifies where the NPU time goes, so future work can
target the highest-impact bottlenecks.

## Hardware Budget

| Resource | XDNA1 AIE2 | Zen 4 CPU (8T) |
|---|---|---|
| BF16 peak TFLOPS | ~1.7 | ~2.7 |
| On-chip memory per tile/core | 64 KB data | 32 KB L1d + 1 MB L2 |
| Weight movement per token | Zero (XRT buffers) | DDR5 stream |
| NVMe / NEON / GPU acceleration | None | AVX-512 FMA |

The NPU's theoretical advantage is weight persistence. All decoder weights
remain in XRT buffer objects across tokens — no DDR re-read. But XDNA1's
BF16 throughput is FP32-bound by the AIE2 tile architecture which uses
software multiply-accumulate rather than dedicated tensor units.

## Where the NPU Time Goes

For Qwen2.5-0.5B (896 hidden, 4864 intermediate, 24 layers). **All per-phase
numbers below are order-of-magnitude estimates derived from the XDNA1 AIE2
clock rate (1 GHz), known op counts, and the measured end-to-end token rate
(3.65 tok/s).** The repository's benchmark tooling records only whole-token
timings — it does not instrument individual dispatch, compute, or tokenizer
phases. Use these estimates for bottleneck identification, not as measured
results.

| Phase | Estimated time | Derivation |
|---|---|---|
| Tile dispatch + weight fill | ~120 µs | XRT command submission overhead |
| Kernel compute (12 × 2-layer chunks) | ~960 µs | 75M MACs / ~78 GMACs estimated AIE2 throughput |
| Host-side LM head | ~650 µs | NumPy BF16 matmul on Zen 4, derived from residual |
| Tokenizer + bookkeeping | ~80 µs | Python + HuggingFace tokenizers overhead |
| **Total estimated** | **~1.8 ms** | **~550 tok/s upper bound vs 3.65 tok/s measured** |

Measured streaming throughput is **3.65 tok/s** for Qwen 0.5B in 32-token chat.
The 150× gap between theory and reality comes from:
1. **Synchronous dispatch**: the host waits for each tile completion before
   submitting the next.
2. **No batching**: single-token inference only.
3. **Python overhead** in the orchestration loop.

## Why CPU Beats NPU

- The 8-thread CPU baseline uses AVX-512 BF16 FMA at near-peak throughput.
  The Qwen2.5-0.5B's ~1 GB of BF16 weights stream from DDR5, but at single-token
  batch size the memory bandwidth is not saturated (~5-10 GB/s vs DDR5-5600's
  44.8 GB/s peak).
- The CPU cores cache attention and LM head constants in L2 over repeated
  tokens, reducing round-trip latency for the most-frequently-accessed tensors.
  However, the full model does **not** fit in L2 (∼1 GB weights vs 1 MB/core L2).
  The performance advantage comes from latency-hiding via prefetchers and
  out-of-order execution, not from caching the entire dataset.
- The NPU compilation and load cost (7.03 s cold) is **amortized over tokens**,
  but the serial dispatch gap makes the NPU hardware runway too short to catch up.

## Possible Improvements (Estimated Impact)

| Optimization | Impact | Effort |
|---|---|---|
| NVME slice of Qwen 2-layer kernel  | 2× decode | Medium |
| 2-layer pipelining (tile 0 runs and tile 1 dispatches) | 1.8× decode | Medium |
| INT4/INT6 path on NPU (quantized than FP16) | 3-5× | High (new kernel) |
| FP8 with XDNA2 AIE4 (hardware redesign) | 5-10× | Full rewrite |
| Continuous batching on the server side | 3× throughput | Medium-high |

## XDNA1 vs XDNA2 Projection

| | XDNA1 (AIE2, npu1) | XDNA2 (AIE4, npu4) |
|---|---|---|
| Columns | 4 | 8 |
| Tile local memory | 64 KB data | 128 KB data + shared L2 |
| Native FP8 | No | Yes |
| Vector width | 256-bit | 512-bit |

Note: per-tile AIE2 throughput is ~117 GFLOPS BF16 (single tile, single
worker); per-tile AIE4 throughput is architecture-dependent and not yet
measured on Hawk Point.

Both generations of silicon are AMD-proprietary; XDNA1 is validated here
and XDNA2 is not. See [README Limitations](README.md#limitations).