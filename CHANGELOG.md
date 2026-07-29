# Changelog

All notable changes are documented here. This project follows semantic
versioning while its public API is experimental.

## [Unreleased]

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
- Ollama v0.32.5 XDNA1 backend patch with CPU/GPU/NPU model placement.
- Hash-locked Python environment, pinned model revisions, atomic model
  conversion, and runtime package integrity verification.
- Hosted protocol/converter/Ollama tests and an opt-in self-hosted Hawk Point
  hardware acceptance workflow.
- Security policy, support matrix, benchmark protocol, release checksums,
  SBOM generation, and build provenance workflow.

[0.1.0-alpha.1]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/releases/tag/v0.1.0-alpha.1
[0.1.0-rc.1]: https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/compare/v0.1.0-alpha.1...v0.1.0-rc.1
