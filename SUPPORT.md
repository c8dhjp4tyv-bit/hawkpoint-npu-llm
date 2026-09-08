# Support matrix

This is the support boundary for the current `main` branch after
`v0.1.0-rc.9`. The matrix deliberately distinguishes a reproducible release
environment from a supported runtime environment. A new release candidate must
run the complete gated workflow from the commit being tagged; the historical RC
tags are not hardware evidence by themselves. This checkout already contains
tags through `v0.1.0-rc.9`, so the next candidate from this line is
`v0.1.0-rc.10`, not a retroactive `rc.4`.

| Component | Validated | Status |
|---|---|---|
| AMD NPU | Hawk Point XDNA1, `RyzenAI-npu1`, AIE2, 4 columns | Native and Ollama paths tested |
| Phoenix XDNA1 | Same `npu1` architecture | Expected compatible; needs an independent report |
| XDNA2 / Strix Point | No | Unsupported |
| OS | Fedora Linux, x86-64, systemd | Validated |
| Kernel / amdxdna | Exact release-certified value is recorded in `release-pins.json`; compatible kernels are accepted when the XDNA capability probe passes | Exact only in `--strict-release` mode |
| XRT | Exact release-certified `2.26.0`, build `8bf2fc4c090540dcf7872243ab67779ae74ef5e3`; compatible XRT is accepted when the probe passes | Exact only in `--strict-release` mode |
| NPU firmware | `1.5.5.391` release-certified; a different firmware is accepted only when it can load and execute the probe | Exact only in `--strict-release` mode |
| Ollama | `v0.33.3`, commit `b79067b0db7417f20108363bc22adb97f35c966a` | Patched target; hosted port and upstream Go tests required before publication |
| Ollama benchmark model | `qwen2.5:0.5b`, manifest `sha256:a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` | Exact RC gate |
| NVIDIA stack | RTX 5060 Laptop, driver `610.43.03`, CUDA UMD `13.3` for the release evidence | Exact in release mode; optional for XDNA-only compatibility |
| GPU backends | CPU; CUDA v12/v13, ROCm v7.2, Vulkan build options | CPU and CUDA v13 exercised locally; others need reports |
| Python | CPython 3.12 | CI target |
| Open WebUI | `v0.11.0`, OCI index `sha256:72c0ba641ba75e7aa52655cb242570906ececd09b1140fb736483038a22b3228` | Pinned tag and digest |

Exact native checkpoint revisions are recorded in
`npu_llm/model_catalog.py`. See the README for the fixed model architecture
constraints.

## Release gate

The tag-triggered `Gated release` workflow is one dependency chain. Hosted
protocol/converter and the pinned Ollama `v0.33.3` patch plus upstream Go tests
run first. The release then waits for a labeled physical Hawk Point runner to
complete fresh pinned model conversion, SmolLM acceptance, Qwen component and
32-token CPU BF16/NPU agreement, a measured 1,000-completion model-switching
soak, and an Ollama build/install/inference/rollback test.

The Ollama placement gate does not require CPU, CUDA, and XDNA to emit identical
text. Each placement must complete every request with zero errors, remain
deterministic within its own placement, and prove that its intended backend
actually executed (including XDNA dispatch for the NPU placement). A separate
teacher-forced `llama-server` pass compares CPU-only, CUDA-only, XDNA-only, and
hybrid placements using top-1 agreement, top-k overlap, and the reference
top-1/top-2 logit margin. A top-1 mismatch fails only when that margin is above
the configured threshold; low-margin near-ties are tolerated.
Response SHA-256 values remain in the evidence for diagnostics and are never a
cross-placement release condition.

The SBOM, signing, provenance, and GitHub release job has `needs: hawk-point`;
it cannot run when hardware is absent or any hardware gate fails. This is why a
Git tag without a corresponding GitHub Release is not itself a published or
validated release.

All third-party Actions are pinned to full commit SHAs. Repository Actions
policy also requires full-length SHA pinning. The separately dispatchable
`XDNA1 hardware acceptance` workflow is for pre-tag validation and does not
publish releases.

The self-hosted runner must keep the pinned MLIR-AIE checkout at
`$HOME/mlir-aie`, or set `HAWKPOINT_MLIR_AIE_DIR`. Before any hardware work,
`scripts/configure-hardware-runner.sh` verifies the full source commit, the
dedicated Python environment, XRT libraries, Python bindings, and access to
device 0. A mismatched or incomplete runner fails closed.
`scripts/verify_hardware_versions.py` then writes both expected and observed
values in `hardware-versions.json`. Its default compatibility mode records
kernel, amdxdna, XRT, firmware, and GPU versions without requiring exact
strings. It detects XDNA1 and runs a known-good probe covering device open,
hardware-context creation, BO allocation/synchronization, xclbin loading, and
kernel submission/wait. Missing capabilities fail; version drift alone does
not. The tag-triggered release workflow adds `--strict-release`, which applies
the exact values in `release-pins.json` on top of the same functional checks.

Therefore “certified on the pinned stack” does not mean “runs only on that
kernel build.” Compatibility reports retain non-fatal version differences so
new distro and mainline kernels can be qualified without changing release
pins.

Because the final gate replaces and rolls back the system Ollama service, the
dedicated release runner must be provisioned for non-interactive `sudo`.
The workflow checks `sudo -n true` before downloads, conversion, or long
benchmarks so a misconfigured runner fails immediately.
