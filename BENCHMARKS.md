# Benchmark protocol

Do not interpret `npu_percent` in `ollama ps` as physical NPU utilization. It
is a model-placement estimate. Collect NPU activity separately with the
platform's XRT/driver telemetry.

Use the same prompt, model quantization, context length, generation length,
power profile, ambient conditions, and five-minute warm-up for every row.
Report median and p95 TTFT, steady-state token/s, peak resident RAM, peak GPU
VRAM, wall energy, failures, and completed requests over at least 1,000
completions.

| Placement | TTFT | token/s | RAM | VRAM | Energy | 1,000-request stability |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | Not measured under this protocol | — | — | — | — | — |
| GPU only | Not measured under this protocol | — | — | — | — | — |
| CPU + GPU | Not measured under this protocol | — | — | — | — | — |
| CPU + GPU + NPU | Not measured under this protocol | — | — | — | — | — |

The smaller component measurements already obtained during development remain
in the README, but they are not substitutes for this controlled comparison.
Empty cells are intentional: this repository does not invent benchmark data.
