# Performance Analysis

## Scope and Evidence

This document distinguishes **measured repository evidence** from static model
arithmetic. It does not present uninstrumented phase timings, processor peak
rates, cache behavior, or projected speedups as measurements.

The controlled-comparison protocol is defined in [BENCHMARKS.md](../BENCHMARKS.md).
Its cross-placement result table is intentionally empty until evidence is
collected on the release hardware.

## Measured Native Runtime Results

The following values are reported in the repository README for the validated
Hawk Point XDNA1 system and acceptance workload:

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
Two-layer chunks are the largest reliable persistent graph on the validated
Phoenix firmware; larger compile-time layer counts time out. The 50–75 token/s
request is therefore recorded as unmet on this hardware rather than treated as
a projection.
The W8 result is an accuracy/performance trade-off; the release acceptance gate
continues to use the default BF16 decoder.

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
measurement. It must not be compared directly with the 274 ms/token observed
end-to-end result or used to assign the remaining time to any phase.

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
- Measure continuous batching or pipelining only with fixed prompt, context,
  generation length, warm-up, and stability criteria from `BENCHMARKS.md`.
- Run any XDNA2 port as a separate hardware-validation effort; XDNA2 is not
  supported by the current XDNA1 implementation.

Until these measurements exist, benchmark data should be added only to release
evidence artifacts, following the no-invented-data policy in `BENCHMARKS.md`.
