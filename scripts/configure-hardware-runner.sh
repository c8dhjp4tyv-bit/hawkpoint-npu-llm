#!/usr/bin/env bash
set -euo pipefail

readonly expected_mlir_aie_commit="57d7494e99c214f5f53b328a0ed43a99e759e835"
readonly mlir_aie_dir="${HAWKPOINT_MLIR_AIE_DIR:-${HOME}/mlir-aie}"
readonly minimum_actions_runner_version="2.327.1"

if [[ -z "${GITHUB_ENV:-}" || -z "${GITHUB_PATH:-}" ]]; then
    printf 'This script must run inside GitHub Actions.\n' >&2
    exit 1
fi
runner_root="$(dirname "$(dirname "${RUNNER_TEMP:-/missing/_work/_temp}")")"
runner_listener="${HAWKPOINT_RUNNER_LISTENER:-${runner_root}/bin/Runner.Listener}"
if [[ ! -x "${runner_listener}" ]]; then
    printf 'GitHub Actions Runner.Listener not found at %s\n' \
        "${runner_listener}" >&2
    exit 1
fi
actual_runner_version="$("${runner_listener}" --version)"
if [[ ! "${actual_runner_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
        [[ "$(printf '%s\n' "${minimum_actions_runner_version}" \
            "${actual_runner_version}" | sort -V | head -n 1)" != \
            "${minimum_actions_runner_version}" ]]; then
    printf 'GitHub Actions runner %s is too old; version %s or newer is required for Node 24 actions.\n' \
        "${actual_runner_version}" "${minimum_actions_runner_version}" >&2
    exit 1
fi
if [[ ! -d "${mlir_aie_dir}/.git" ]]; then
    printf 'MLIR-AIE checkout not found at %s\n' "${mlir_aie_dir}" >&2
    exit 1
fi
actual_commit="$(git -C "${mlir_aie_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_mlir_aie_commit}" ]]; then
    printf 'MLIR-AIE commit mismatch: expected %s, found %s\n' \
        "${expected_mlir_aie_commit}" "${actual_commit}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${mlir_aie_dir}/ironenv/bin/activate"
# shellcheck source=/dev/null
source /opt/xilinx/xrt/setup.sh >/dev/null
# shellcheck source=/dev/null
source "${mlir_aie_dir}/utils/env_setup.sh" \
    "${mlir_aie_dir}" "${mlir_aie_dir}/peano" >/dev/null

python -c 'import aie, pyxrt; assert pyxrt.device(0)'

{
    printf 'VIRTUAL_ENV=%s\n' "${VIRTUAL_ENV}"
    printf 'XILINX_XRT=%s\n' "${XILINX_XRT}"
    printf 'MLIR_AIE_INSTALL_DIR=%s\n' "${MLIR_AIE_INSTALL_DIR}"
    printf 'PEANO_INSTALL_DIR=%s\n' "${PEANO_INSTALL_DIR}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH}"
    printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
} >>"${GITHUB_ENV}"
{
    printf '%s\n' "${VIRTUAL_ENV}/bin"
    printf '%s\n' "${MLIR_AIE_INSTALL_DIR}/bin"
    printf '%s\n' "${XILINX_XRT}/bin"
} >>"${GITHUB_PATH}"

printf 'GitHub Actions runner %s, pinned MLIR-AIE, XRT, Python, and NPU device imports are ready.\n' \
    "${actual_runner_version}"
