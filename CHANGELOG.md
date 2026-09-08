# Changelog

All notable changes are documented here. This project follows semantic
versioning while its public API is experimental.

## [Unreleased]

The repository has Git tags through `v0.1.0-rc.9`; this unreleased work should
ship as the next gated candidate (`v0.1.0-rc.10`) only after the physical
Hawk Point job succeeds. Existing RC tags without GitHub Release records remain
tag history, not published artifacts.

- Port the Ollama XDNA patch and release pin from `v0.32.5` to the clean
  upstream `v0.33.3` commit `b79067b0db7417f20108363bc22adb97f35c966a`.
  Hosted CI now reads the tag, commit, and patch filename from
  `release-pins.json`, applies the patch with an explicit `--check`, and runs
  the upstream Go tests before a hardware-gated publication can proceed.
- Clarify the release boundary and placement gate: CPU/CUDA/XDNA text hashes are
  diagnostic only; release correctness uses deterministic per-placement runs,
  real backend-dispatch evidence (including an isolated XDNA-only run), and
  teacher-forced top-1/top-k logit agreement with a low-margin near-tie
  tolerance.
- Add a bounded persistent weight cache to the Ollama backend. Dense tiles and
  selected MoE experts are packed once and unchanged slots remain resident in
  XRT BOs across decode steps. Add opt-in native Q4_K/Q6_K AIE2 kernels and the
  IRON compiler needed to keep GGML blocks quantized while dequantizing inside
  the fused GEMV; generated native xclbins still require the physical gate.
- Split hardware validation into capability-based compatibility mode and
  `--strict-release` certification mode. Compatibility reports retain observed
  kernel, amdxdna, XRT, firmware, and GPU versions without exact-string
  rejection, then run a known-good XRT/AIE probe for device open, hwctx, BO,
  sync, xclbin, and kernel-submit support. Release certification applies the
  exact values from `release-pins.json` on top of those functional checks.
- Add sampling to the runtime, the terminal chat, and the OpenAI-compatible
  endpoint: `temperature`, `top_p`, `top_k`, `repetition_penalty`,
  `presence_penalty`, `frequency_penalty`, and `seed`. Penalties are scored over
  the prompt as well as the generated tokens, filters are applied in the
  llama.cpp/vLLM order, and the sampler is engaged once per emitted token rather
  than once per ingested prompt position. Greedy decoding stays the default: a
  request that sets no sampling field takes the previous code path unchanged, so
  the exact-token acceptance and release gates still hold. An unseeded sampled
  request draws a seed and reports it in `x_hawkpoint_stats.sampling` so the run
  can be replayed. Out-of-range values are rejected with `400` at the HTTP
  boundary and revalidated inside the inference worker before reaching a
  decoder. Adds hardware-free tests for the sampler and the decode loop, plus a
  hardware gate asserting `temperature: 0` reproduces the greedy sequence and a
  seeded sampled run is reproducible.
- Fix the RC3 release blocker: deterministically close the active decoder and
  release its XRT/NPU hardware context before another model loads, in worker
  finally blocks and on worker termination, instead of relying on garbage
  collection. This exhausted the driver's system-wide context pool during long
  model-switching runs (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX` -110,
  `aie2_alloc_resource failed`) and produced one HTTP 500 in the 1,000-completion
  soak.
- Split the hardware endurance coverage: keep the 1,000-completion soak but run
  it as 250 consecutive requests per model, and add a dedicated bounded
  model-switch stress test of at least 100 switches. Both preserve zero-tolerance
  reporting for HTTP failures and amdxdna/XRT/NPU kernel errors.
- Add tests proving an inference error tears down and recreates the worker
  process, and that a model switch deterministically closes the previous decoder.
- Make the XDNA ggml backend opt-in: it advertised one device unconditionally,
  so every model load tried to initialize XDNA and hard-failed with HTTP 500
  when `GGML_XDNA_XCLBIN`/`GGML_XDNA_INSTS` were unset -- breaking all non-XDNA
  (CPU/CUDA) Ollama inference. It now reports zero devices when unconfigured.
- Replace the placement benchmark's byte-for-byte cross-placement text equality
  with a teacher-forced top-1/top-k logit-agreement gate driven through
  `llama-server`. Different backends (CPU/CUDA/XDNA) diverge numerically, so a
  top-1 mismatch fails only when the reference logit margin is at or above a
  tolerance threshold; low-margin near-ties are tolerated. Each placement must
  still complete every request with zero errors, stay deterministic, prove real
  CPU/CUDA/XDNA execution (no silent fallback), and the NPU placement must show
  XDNA dispatch. Exact response hashes remain in the report for information only.

## [0.1.0-rc.3] - 2026-07-29

- Publish full fused-Qwen top-five logit and latency evidence for 32 prefill
  positions while retaining the independent exact 32-token generation gate.
- Classify only bounded BF16 top-two near-ties instead of failing on an
  unstable intermediate argmax, and fail every non-tie divergence.
- Remove the stale claim that the fused Qwen NPU path beats the controlled
  eight-thread CPU baseline.

## [0.1.0-rc.2] - 2026-07-29

- Replace the Qwen agreement prompt that reached EOS before 32 tokens with a
  long-form prompt and a checked-in 32-token CPU BF16 reference sequence.
- Persist CPU, NPU, EOS, and mismatch evidence before failing the release gate.
- Cap BLAS, OpenMP, NumExpr, and Go hardware-job parallelism at eight workers.

## [0.1.0-rc.1] - 2026-07-29

- Make release publication depend on hosted CI and a physical Hawk Point
  correctness, 1,000-completion soak, Ollama install/inference/rollback gate.
- Pin every third-party GitHub Action to a full commit SHA.
- Isolate XRT inference in a restartable worker process with hard termination
  at the request deadline and separate liveness/readiness endpoints.
- Add measured soak evidence and 32-token Qwen CPU BF16/NPU agreement reports.
- Verify the pinned MLIR-AIE/XRT runner environment before hardware work and
  produce a controlled four-placement Ollama benchmark matrix.
- Pin the Ollama source commit and Qwen manifest digest, enforce output
  equality across all four placements, and recycle the XRT worker after every
  inference error.
- Fail the release gate unless XRT, firmware, kernel, amdxdna, GPU driver, and
  CUDA UMD versions match `release-pins.json`; publish the observed stack as
  hardware evidence.

## [0.1.0-alpha.1] - 2026-07-29

### Added

- Native SmolLM and Qwen2.5 XDNA1 runtimes.
- OpenAI-compatible authenticated API and local Open WebUI launcher.
- Historical Ollama v0.32.5 XDNA1 backend patch with CPU/GPU/NPU model
  placement (the supported patch base is now v0.33.3).
- Hash-locked Python environment, pinned model revisions, atomic model
  conversion, and runtime package integrity verification.
- Hosted protocol/converter/Ollama tests and an opt-in self-hosted Hawk Point
  hardware acceptance workflow.
- Security policy, support matrix, benchmark protocol, release checksums,
  SBOM generation, and build provenance workflow.

[0.1.0-alpha.1]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/releases/tag/v0.1.0-alpha.1
[0.1.0-rc.1]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-alpha.1...v0.1.0-rc.1
[0.1.0-rc.2]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.1...v0.1.0-rc.2
[0.1.0-rc.3]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.2...v0.1.0-rc.3
[0.1.0-rc.4]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.3...v0.1.0-rc.4
[0.1.0-rc.5]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.4...v0.1.0-rc.5
[0.1.0-rc.6]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.5...v0.1.0-rc.6
[0.1.0-rc.7]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.6...v0.1.0-rc.7
[0.1.0-rc.8]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.7...v0.1.0-rc.8
[0.1.0-rc.9]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-rc.8...v0.1.0-rc.9
