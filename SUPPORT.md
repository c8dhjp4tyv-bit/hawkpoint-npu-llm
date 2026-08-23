# Support matrix

This is the release-candidate boundary for `v0.1.0-rc.3`. Anything outside it is
unsupported until a reproducible hardware report is added.

| Component | Validated | Status |
|---|---|---|
| AMD NPU | Hawk Point XDNA1, `RyzenAI-npu1`, AIE2, 4 columns | Native and Ollama paths tested |
| Phoenix XDNA1 | Same `npu1` architecture | Expected compatible; needs an independent report |
| XDNA2 / Strix Point | No | Unsupported |
| OS | Fedora Linux, x86-64, systemd | Validated |
| Kernel / amdxdna | `7.2.0-0.rc5.260729.fc02acf6.441.vanilla.fc45.x86_64` | Exact RC gate |
| XRT | `2.26.0`, build `8bf2fc4c090540dcf7872243ab67779ae74ef5e3` | Exact RC gate |
| NPU firmware | `1.5.5.391` | Validated |
| Ollama | `v0.32.5`, commit `eec8e0b9458b8a01be0c216a9cc53eefde24ef50` | Patch target |
| Ollama benchmark model | `qwen2.5:0.5b`, manifest `sha256:a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` | Exact RC gate |
| NVIDIA stack | RTX 5060 Laptop, driver `610.43.03`, CUDA UMD `13.3` | Exact RC gate |
| GPU backends | CPU; CUDA v12/v13, ROCm v7.2, Vulkan build options | CPU and CUDA v13 exercised locally; others need reports |
| Python | CPython 3.12 | CI target |
| Open WebUI | `v0.11.0`, OCI index `sha256:72c0ba641ba75e7aa52655cb242570906ececd09b1140fb736483038a22b3228` | Pinned tag and digest |

Exact native checkpoint revisions are recorded in
`npu_llm/model_catalog.py`. See the README for the fixed model architecture
constraints.

## Release gate

The tag-triggered `Gated release` workflow is one dependency chain. Hosted
protocol/converter and upstream Ollama tests run first. The release then waits
for a labeled physical Hawk Point runner to complete fresh pinned model
conversion, SmolLM acceptance, Qwen component and 32-token CPU BF16/NPU
agreement, a measured 1,000-completion model-switching soak, and an Ollama
build/install/inference/rollback test. The SBOM, signing, provenance, and
GitHub release job has `needs: hawk-point`; it cannot run when hardware is
absent or any hardware gate fails.

The four Ollama placements (CPU only, GPU only, CPU+GPU, CPU+GPU+NPU) are
gated as implemented in `tests/benchmark_ollama_matrix.py`:

- every placement must complete all of its requests with zero errors;
- within one placement, the response text must not change between requests;
- **cross-placement exact response digests are informational.** CPU, CUDA,
  and XDNA kernels diverge numerically, so identical text across backends is
  not a realistic requirement and a digest mismatch alone never fails the
  release. The hashes are still recorded in `ollama-placement-matrix.json`.
- cross-placement correctness is decided by the teacher-forced logit
  agreement pass: per position, top-1 agreement and top-k overlap against the
  reference placement, where a top-1 mismatch fails only when the reference
  logit margin is at or above the configured threshold
  (`--logit-margin-threshold`, default `1.0`). Genuine low-margin near-ties
  are tolerated.

That gate is independent of, and does not weaken, the native runtime's exact
checked-in 32-token Qwen sequence gate.

All third-party Actions are pinned to full commit SHAs. Repository Actions
policy also requires full-length SHA pinning. The separately dispatchable
`XDNA1 hardware acceptance` workflow is for pre-tag validation and does not
publish releases.

The self-hosted runner must keep the pinned MLIR-AIE checkout at
`$HOME/mlir-aie`, or set `HAWKPOINT_MLIR_AIE_DIR`. Before any hardware work,
`scripts/configure-hardware-runner.sh` verifies the full source commit, the
dedicated Python environment, XRT libraries, Python bindings, and access to
device 0. A mismatched or incomplete runner fails closed.
`scripts/verify_hardware_versions.py` then compares the complete observed stack
with `release-pins.json` and publishes both expected and observed values in
`hardware-versions.json`.

Because the final gate replaces and rolls back the system Ollama service, the
dedicated release runner must be provisioned for non-interactive `sudo`.
The workflow checks `sudo -n true` before downloads, conversion, or long
benchmarks so a misconfigured runner fails immediately.
