# Support matrix

This is the tested boundary for `v0.1.0-alpha.1`. Anything outside it is
unsupported until a reproducible hardware report is added.

| Component | Validated | Status |
|---|---|---|
| AMD NPU | Hawk Point XDNA1, `RyzenAI-npu1`, AIE2, 4 columns | Native and Ollama paths tested |
| Phoenix XDNA1 | Same `npu1` architecture | Expected compatible; needs an independent report |
| XDNA2 / Strix Point | No | Unsupported |
| OS | Fedora Linux, x86-64, systemd | Validated |
| NPU firmware | `1.5.5.391` | Validated |
| Ollama | `v0.32.5`, commit `eec8e0b9458b8a01be0c216a9cc53eefde24ef50` | Patch target |
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

All third-party Actions are pinned to full commit SHAs. Repository Actions
policy also requires full-length SHA pinning. The separately dispatchable
`XDNA1 hardware acceptance` workflow is for pre-tag validation and does not
publish releases.

The self-hosted runner must keep the pinned MLIR-AIE checkout at
`$HOME/mlir-aie`, or set `HAWKPOINT_MLIR_AIE_DIR`. Before any hardware work,
`scripts/configure-hardware-runner.sh` verifies the full source commit, the
dedicated Python environment, XRT libraries, Python bindings, and access to
device 0. A mismatched or incomplete runner fails closed.
