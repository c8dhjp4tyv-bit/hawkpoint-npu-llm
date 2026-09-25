# Performance Analysis

## Scope and Evidence

This document distinguishes **measured repository evidence** from static model
arithmetic. It does not present uninstrumented phase timings, processor peak
rates, cache behavior, or projected speedups as measurements.

The controlled-comparison protocol is defined in [BENCHMARKS.md](../BENCHMARKS.md).
Its cross-placement result table is intentionally empty until evidence is
collected on the release hardware.

## Measured Native Runtime Results (chunked layer graphs)

These values were measured with the chunked two-layer graphs, before the
decoder engine described below. They remain the reference for
`HAWKPOINT_ENGINE=0` and for the opt-in W8 decoder:

| Measurement | Reported result |
|---|---:|
| Cold compile/load | 7.82 s |
| Warm `decode_token` smoke test | 13.25 token/s (0.0755 s/token) |
| 32-token streaming chat | 15.93 token/s |
| Native 16-token W8 decode, latest 3-run median | 14.42 token/s |
| Corrected-loop session medians / best individual run | 14.42–19.99 / 20.20 token/s |
| Historical decoder-step median before accounting fix | 18.42 token/s |
| Native W8 TTFT, 35-token prompt | 1.47–1.62 s session medians |
| End-to-end TTFT for acceptance prompt | 2.56 s |
| Fixed hardware context | 64 tokens |

For the 32-token streaming measurement, 15.93 token/s corresponds to about
63 ms/token. The native benchmark can now be run with
`HAWKPOINT_PROFILE=1` or `scripts/benchmark_native.py --profile`. It reports
host-side phase totals for embedding, RoPE LUT creation, NPU decoder dispatch,
CPU decoder/LM-head work, and the final normalization/head path. XRT/AIE internal cycle
attribution still requires a hardware trace, so these counters must not be
read as kernel-level timings.

The current native loop skips final normalization and the 49k-row LM head for
non-final prefill positions. It also stops without executing a discarded
successor step after `max_tokens` is reached. With the W8 accumulator reduced
by the AIE2 `reduce_add` intrinsic, the controlled 16-token run measured a
14.42–19.99 token/s medians across repeated three- and five-run sessions, with
a 20.20 token/s best individual run. The faster session measured about
2.11–2.20 seconds of total NPU decoder work, 0.08–0.11 seconds of CPU head work,
and 1.47 seconds median TTFT. The latest session measured 2.32–2.40 seconds NPU,
0.12–0.14 seconds CPU head, and 1.56 seconds median TTFT. This range is reported
explicitly because temperature/load sensitivity is material. All runs produced
the same greedy token IDs.

The older loop produced an 18.42 token/s best isolated median but executed and
timed one discarded successor step. That number remains useful as historical
decoder-step evidence, but is not directly comparable with the corrected
end-to-end loop. Results also vary with device temperature and system load.
Two-layer chunks are the largest reliable persistent graph for this design on
the validated Phoenix firmware; larger compile-time layer counts time out. The
50–75 token/s request was unmet with these graphs; the decoder engine below
meets it by changing the dataflow rather than the chunk size.
The W8 result is an accuracy/performance trade-off; the release acceptance gate
continues to use the default BF16 decoder.

## Row-Split Decoder Engine (SmolLM)

All numbers in this section were measured on the same Hawk Point system
(`RyzenAI-npu1`, NPU firmware 1.5.5.391, amdxdna from Linux 7.3.0-rc4,
MLIR-AIE `57d7494e99c`, llvm-aie `21.0.0.2026072001`). NPU times are the XRT
submit-to-completion times reported by the IRON runtime.

### Where the chunked decoder spent its time

Per decoded SmolLM2 token, the chunked layer graphs issued 15 two-layer
dispatches:

| Per token | BF16 | W8 |
|---|---:|---:|
| NPU time per two-layer dispatch | 2.72 ms | 2.43 ms |
| Host time per dispatch outside XRT | 0.24 ms | 0.24 ms |
| Decoder total | 46.4 ms | 42.1 ms |
| Everything else (CPU LM head, sampling) | 6.3 ms | 6.3 ms |
| Decode rate | 19.0 token/s | 20.7 token/s |

A BF16 layer streams about 7.1 MB of weights, so 1.36 ms per layer is about
5.2 GB/s. W8 halves the bytes but barely changes the time, so the W8 path is
limited by its dequantizing GEMV rather than by memory.

### DDR-to-core streaming bandwidth

A synthetic design streams 32 MiB of 8 KiB blocks from DDR into compute-tile
ObjectFifos that only acquire and release them:

| Parallel shim streams | Bandwidth |
|---|---:|
| 1 | 6.8 GB/s |
| 2 | 13.1 GB/s |
| 4 (one per column) | 22.4 GB/s |
| 8 (two per column) | 32.5 GB/s |

A near-empty dispatch costs about 0.14 ms of NPU time and 0.38 ms of wall
time. The chunked decoder's graphs feed each projection from one stream and
run their stages one after another, so they use roughly one stream at a time;
at 8 streams the 212 MB of BF16 weights per token would take about 7 ms.

### Engine design

`designs/engine.py` runs all 30 decoder layers and the final RMSNorm in one
dispatch. Six GEMV tiles each read their own weight stream and own a fixed
slice of the output rows of *every* projection (160 qkv, 96 o_proj, 256
gate/up, and 96 down_proj rows), so all six streams are busy in every phase
of the layer. A MemTile joins their result slices and hands them to a hub
tile, which runs RoPE and attention and broadcasts each combined vector back.
Every GEMV tile keeps its own copy of the residual stream, so the hidden
state never leaves the array between layers. The hub reads each layer's K/V
cache from DDR and returns only the new key and value rows, which the host
appends.

The projections, RMSNorms, RoPE, residual adds, and SwiGLU use the same BF16
arithmetic as the chunked graphs, and the final RMSNorm matches
`kernels/rmsnorm_bf16.cc` bit for bit. With the chunked path's attention the
engine reproduced its hidden states bit for bit at all 40 compared positions
and its logits exactly on four prompts.

### Attention was scalar soft-float

The AIE2 scalar unit has no floating-point hardware, so every scalar `float`
operation in the chunked attention kernel is a library call. With every
other stage unchanged, attention made one engine dispatch grow with the
position:

| Position | Scalar softmax | Vector softmax |
|---:|---:|---:|
| 0 | 8.24 ms | 7.59 ms |
| 20 | 10.74 ms | 7.93 ms |
| 50 | 15.64 ms | 8.74 ms |
| 63 | 17.09 ms | 8.86 ms |

The engine's softmax now runs 16 positions at a time on the vector unit: the
score offset is taken in FP32, the exponential comes from the BF16 lookup
tables in `lut_based_ops`, and the weights are normalized once instead of per
output chunk. Its results are no longer bit-identical to the chunked path.
Teacher-forced on 377 positions of CPU-reference greedy output across six
prompts, the engine agreed with the CPU BF16 reference at 338 positions and
the chunked path at 335; both missed the same seven positions whose
reference top-1 margin was at least 0.5 logits.
`npu_llm/tests/validate_engine_npu.py` repeats this check.

### A compiler hang, and how the kernels avoid it

With llvm-aie `21.0.0.2026072001`, a plain 16-lane copy loop between two
`__restrict` buffers compiles at `-O2` into a software-pipelined
zero-overhead loop. In some links it hangs the core: a single-tile design
that only copies 576 values from an ObjectFifo buffer into a local buffer
times out every time. The identical machine code at another program address
runs correctly (the chunked decoder's `layer_copy576`), and the same loop
compiled at `-O1`, without `__restrict`, with 32-lane vectors, or fully
unrolled does not hang. The engine kernels therefore fully unroll every
fixed-length copy and elementwise loop, so no such hardware loop is emitted.

### Measured result

`scripts/benchmark_native.py`, 16 tokens after a 35-token prompt, five runs:

| | Chunked graphs | Engine |
|---|---:|---:|
| Median decode rate | 17.1 token/s | 74.6 token/s |
| Individual runs | 16.2–17.3 token/s | 73.2–75.8 token/s |
| TTFT | 1.71–1.81 s | 0.28–0.29 s |
| NPU time per token | about 50 ms | 7.6–8.9 ms |

The remaining per-token cost is about 8 ms of NPU time and about 5 ms of CPU
LM head (a 49152 x 576 FP32 matrix-vector product). A quick soak of 100
completions and a 100-switch model-switch stress test through the API ran
without NPU errors.

## Static Decoder Projection Arithmetic

For Qwen2.5-0.5B, the native fused graph uses hidden size `896`, intermediate
size `4864`, QKV projection size `1152`, and `24` decoder layers. Per layer,
the four linear projections alone require:

| Projection | MAC calculation | MACs/layer |
|---|---:|---:|
| QKV | `1152 × 896` | 1,032,192 |
| Output projection | `896 × 896` | 802,816 |
| Gate + up projection | `(2 × 4864) × 896` | 8,716,288 |
| Down projection | `4864 × 896` | 4,358,144 |
| **Projection subtotal** |  | **14,909,440** |

Across 24 layers, the projection subtotal is **357,826,560 MACs/token**.
This excludes RMSNorm, RoPE, residual operations, SwiGLU, cache work, and
attention. It is therefore a lower bound on decoder arithmetic, not a timing
model.

For example, an assumed sustained rate of 78 GMAC/s would yield a projection
floor of about **4.59 ms/token** (`357.83M / 78G`) before those omitted
operations. That is arithmetic under an explicit assumption, not a hardware
measurement. It must not be compared directly with measured Qwen step latency
(the token-agreement gate records it as `npu_median_seconds` in
`qwen-token-agreement.json`) or used to assign the remaining time to any phase.
The 274 ms/token figure previously quoted here came from an older, since
removed measurement and is not current evidence.

## Candidate Areas for Measurement

The source and existing benchmark protocol identify these areas as useful
experiments, without claiming a speedup until they are benchmarked:

- Profile XRT submission and completion timing separately from whole-token
  latency.
- Measure CPU LM-head and tokenization costs with the exact validated model
  and prompt.
- Compare the default persistent W8 cache against the opt-in native Q4_K/Q6_K
  kernels. The comparison must include first-use packing, warm-token latency,
  cache hit/upload bytes, and numerical agreement on the pinned model.
- Measure prompt-prefix reuse separately from cold prefill: run
  `scripts/benchmark_native.py` with and without `--prefix-cache`. The default
  resets the cache before each timed run so TTFT stays comparable with older
  results.
- Measure continuous batching or pipelining only with fixed prompt, context,
  generation length, warm-up, and stability criteria from `BENCHMARKS.md`.
- Run any XDNA2 port as a separate hardware-validation effort; XDNA2 is not
  supported by the current XDNA1 implementation.

Until these measurements exist, benchmark data should be added only to release
evidence artifacts, following the no-invented-data policy in `BENCHMARKS.md`.
