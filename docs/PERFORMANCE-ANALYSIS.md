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
| Cold compile/load | 7.03 s |
| Warm `decode_token` smoke test | 10.72 token/s |
| 32-token streaming chat | 3.65 token/s |
| End-to-end TTFT for acceptance prompt | 8.80 s |
| Fixed hardware context | 64 tokens |

For the 32-token streaming measurement, 3.65 token/s corresponds to about
274 ms/token. The benchmark tooling does not separately instrument XRT dispatch,
AIE execution, CPU LM-head work, tokenization, or Python orchestration.
Consequently, this document makes no
per-phase attribution of that 274 ms.

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
- Compare persistent quantized-weight kernels against the current host-to-NPU
  weight-transfer path in the Ollama integration.
- Measure continuous batching or pipelining only with fixed prompt, context,
  generation length, warm-up, and stability criteria from `BENCHMARKS.md`.
- Run any XDNA2 port as a separate hardware-validation effort; XDNA2 is not
  supported by the current XDNA1 implementation.

Until these measurements exist, benchmark data should be added only to release
evidence artifacts, following the no-invented-data policy in `BENCHMARKS.md`.
