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

The release runs two hardware endurance tests that upload JSON evidence. The
endurance soak keeps each model resident under sustained load, running the
1,000 completions as 250 consecutive requests per model; the model-switch
stress test does the opposite, forcing at least 100 back-to-back model switches
to exercise deterministic XRT/NPU context release. Each report contains
successful and failed request counts, per-model results, median/p95 TTFT and
token rate, start/end/peak RAM, the observed number of model switches,
available RAPL energy, fixed CPU-BF16 token references, and matching XRT/NPU
error lines. Either test fails the gate on any HTTP failure, token mismatch, or
amdxdna/XRT/NPU kernel error.
The Qwen release gate uploads its full 32-token CPU/NPU sequence separately.
That gate checks both paths against a checked-in CPU BF16 reference generated
from the pinned checkpoint and a fixed long-form prompt; EOS must not occur
before the end of the sequence.
The fused prefill benchmark also uploads top-five logits and timings for 32
positions. Exact argmax differences fail unless both BF16 paths have a
top-two candidate tie within the documented `0.05` logit margin and the same
two candidates. This near-tie exception does not weaken the independent exact
32-token generated-sequence gate. NPU/CPU speedup is evidence, not a release
correctness condition.
The same gated hardware job runs 1,000 requests in each of CPU-only, GPU-only,
CPU+GPU, and CPU+GPU+NPU placement modes with one pinned Qwen model and
the same prompt, seed, generation length, and five-minute warm-up. It measures
streaming time to the first emitted token and requires every placement to
complete all requests with zero errors and a stable response hash within its
own placement.
Cross-placement agreement is judged at the logit level, not by byte-for-byte
text equality: CPU, CUDA, and XDNA kernels diverge numerically, so identical
generated text is not a realistic requirement. A separate `llama-server` pass
teacher-forces every placement onto one fixed token prefix and compares, per
position, top-1 token agreement, top-k overlap, and the reference logit margin.
A top-1 mismatch fails the release only when the reference margin is at or above
the tolerance threshold; genuinely ambiguous low-margin near-ties are tolerated.
The pass also proves each placement really ran on its intended backend -- CUDA
placements must hold GPU memory, the NPU placement must show XDNA dispatch, and
no placement may silently fall back to CPU. Exact response hashes are still
recorded in `ollama-placement-matrix.json` for information but never fail the
release on their own. The benchmark refuses to start generation unless the
pulled Ollama manifest matches the digest in `release-pins.json`.
Release assets therefore contain the measured table for that exact release;
this source document keeps empty cells so results are never copied between
machines or releases. The evidence is also packed into a versioned tarball
whose digest is covered by the release's keyless-signed `SHA256SUMS`.
