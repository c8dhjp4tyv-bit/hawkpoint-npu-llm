#!/usr/bin/env bash
set -euo pipefail

failures=0

pass() {
    printf 'PASS  %s\n' "$*"
}

fail() {
    printf 'FAIL  %s\n' "$*" >&2
    failures=$((failures + 1))
}

for command_name in git cmake ninja go g++ curl jq pkg-config; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        pass "${command_name}: $(command -v "${command_name}")"
    else
        fail "${command_name} is not installed"
    fi
done

if command -v go >/dev/null 2>&1; then
    go_version="$(go env GOVERSION | sed 's/^go//; s/[^0-9.].*$//')"
    if printf '%s\n%s\n' "1.26" "${go_version}" | sort -V -C; then
        pass "Go ${go_version} (minimum 1.26)"
    else
        fail "Go ${go_version} is too old; Ollama v0.33.3 requires Go 1.26"
    fi
fi

if [[ -d /sys/module/amdxdna ]]; then
    pass "amdxdna kernel driver is loaded"
else
    fail "amdxdna kernel driver is not loaded"
fi

if [[ -c /dev/accel/accel0 ]]; then
    if [[ -r /dev/accel/accel0 && -w /dev/accel/accel0 ]]; then
        pass "/dev/accel/accel0 is accessible"
    else
        fail "/dev/accel/accel0 exists but the current user cannot access it"
    fi
else
    fail "/dev/accel/accel0 is missing"
fi

if [[ -f /opt/xilinx/xrt/include/xrt/xrt_bo.h ]]; then
    pass "XRT development headers found"
else
    fail "XRT development headers missing under /opt/xilinx/xrt/include"
fi

if compgen -G '/opt/xilinx/xrt/lib64/libxrt_coreutil.so*' >/dev/null; then
    pass "XRT core runtime found"
else
    fail "XRT core runtime missing under /opt/xilinx/xrt/lib64"
fi

if command -v lspci >/dev/null 2>&1 &&
   lspci -nnk 2>/dev/null | grep -A3 -i 'Signal processing controller' |
       grep -q 'Kernel driver in use: amdxdna'; then
    pass "PCI NPU is bound to amdxdna"
else
    fail "could not confirm that the PCI NPU is bound to amdxdna"
fi

if [[ "${failures}" -ne 0 ]]; then
    printf '\n%d required check(s) failed.\n' "${failures}" >&2
    exit 1
fi

printf '\nAll required host checks passed.\n'
