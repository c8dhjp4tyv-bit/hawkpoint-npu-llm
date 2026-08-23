#!/usr/bin/env bash
# Restore the pre-XDNA Ollama install.
#
# Every destructive move is staged behind validation and covered by an error
# trap that undoes the moves already made, so a failure part-way through
# cannot leave a mixed runtime/binary/drop-in state. Nothing is deleted: the
# displaced XDNA install is kept under a .failed-xdna-<stamp> name.
set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

# Test hook: when OLLAMA_XDNA_TEST_ROOT is set every path is taken relative to
# it and systemctl/curl are stubbed, so the fault-injection tests can drive
# this script without root or a real service. Unset -- the production case --
# the script operates on the real paths and requires root, exactly as before.
prefix="${OLLAMA_XDNA_TEST_ROOT:-}"
systemctl_bin="${OLLAMA_XDNA_SYSTEMCTL:-systemctl}"
curl_bin="${OLLAMA_XDNA_CURL:-curl}"
health_tries="${OLLAMA_XDNA_HEALTH_TRIES:-30}"
if [[ -z "${prefix}" ]]; then
    [[ "$(id -u)" -eq 0 ]] || die "rollback.sh must run as root"
fi

lib_dir="${prefix}/usr/local/lib"
bin_dir="${prefix}/usr/local/bin"

latest_runtime="$(
    find "${lib_dir}" -maxdepth 1 -type d \
        -name 'ollama.pre-xdna-*' -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n1 | cut -d' ' -f2-
)"
[[ -n "${latest_runtime}" ]] || die "no Ollama-XDNA backup was found"

stamp="${latest_runtime##*-xdna-}"
backup_binary=""${bin_dir}/ollama".pre-xdna-${stamp}"
drop_in="${prefix}/etc/systemd/system/ollama.service.d/xdna.conf"
drop_in_backup="${drop_in}.pre-xdna-${stamp}"
now="$(date +%Y%m%d-%H%M%S)"
failed_runtime=""${lib_dir}/ollama".failed-xdna-${now}"
failed_binary=""${bin_dir}/ollama".failed-xdna-${now}"

# Validate every source and destination before touching anything, so the
# common failure modes abort while the system is still consistent.
[[ -x "${backup_binary}" ]] || die "matching Ollama binary backup is missing"
[[ -d "${latest_runtime}" ]] || die "backup runtime ${latest_runtime} is missing"
[[ -x "${latest_runtime}/llama-server" ]] ||
    die "backup runtime ${latest_runtime} has no llama-server"
[[ ! -e "${failed_runtime}" ]] || die "${failed_runtime} already exists"
[[ ! -e "${failed_binary}" ]] || die "${failed_binary} already exists"
command -v "${systemctl_bin}" >/dev/null || die "systemd is required for rollback"
"${systemctl_bin}" cat ollama.service >/dev/null || die "ollama.service is not installed"

runtime_moved=0
runtime_restored=0
binary_moved=0
binary_restored=0
drop_in_restored=0
drop_in_backup_consumed=0
drop_in_removed=""

# Undo whichever steps have completed, in reverse order.
undo() {
    echo "rollback failed part-way through; reverting to the XDNA install" >&2
    if [[ "${drop_in_restored}" -eq 1 ]]; then
        if [[ "${drop_in_backup_consumed}" -eq 1 && -f "${drop_in}" ]]; then
            mv -f "${drop_in}" "${drop_in_backup}" || true
            drop_in_backup_consumed=0
        fi
        if [[ -n "${drop_in_removed}" && -f "${drop_in_removed}" ]]; then
            mv -f "${drop_in_removed}" "${drop_in}" || true
            drop_in_removed=""
        fi
        drop_in_restored=0
    fi
    if [[ "${binary_restored}" -eq 1 ]]; then
        mv -f "${bin_dir}/ollama" "${backup_binary}" || true
        binary_restored=0
    fi
    if [[ "${binary_moved}" -eq 1 && -e "${failed_binary}" ]]; then
        mv -f "${failed_binary}" "${bin_dir}/ollama" || true
        binary_moved=0
    fi
    if [[ "${runtime_restored}" -eq 1 ]]; then
        mv -f "${lib_dir}/ollama" "${latest_runtime}" || true
        runtime_restored=0
    fi
    if [[ "${runtime_moved}" -eq 1 && -d "${failed_runtime}" ]]; then
        mv -f "${failed_runtime}" "${lib_dir}/ollama" || true
        runtime_moved=0
    fi
    "${systemctl_bin}" daemon-reload || true
    "${systemctl_bin}" start ollama || true
    echo "error: rollback aborted; the XDNA install was put back" >&2
    exit 1
}

"${systemctl_bin}" stop ollama
trap undo ERR

if [[ -d "${lib_dir}/ollama" ]]; then
    mv "${lib_dir}/ollama" "${failed_runtime}"
    runtime_moved=1
fi
mv "${latest_runtime}" "${lib_dir}/ollama"
runtime_restored=1

if [[ -e "${bin_dir}/ollama" ]]; then
    mv "${bin_dir}/ollama" "${failed_binary}"
    binary_moved=1
fi
mv "${backup_binary}" "${bin_dir}/ollama"
binary_restored=1

# The XDNA drop-in is moved aside, never deleted, so this step is reversible
# and the unit override can be inspected after a rollback.
if [[ -f "${drop_in}" ]]; then
    drop_in_removed="${drop_in}.failed-xdna-${now}"
    mv "${drop_in}" "${drop_in_removed}"
fi
if [[ -f "${drop_in_backup}" ]]; then
    mv "${drop_in_backup}" "${drop_in}"
    drop_in_backup_consumed=1
fi
drop_in_restored=1

"${systemctl_bin}" daemon-reload
"${systemctl_bin}" start ollama

healthy=0
for _ in $(seq 1 "${health_tries}"); do
    if "${curl_bin}" -fsS http://127.0.0.1:11434/api/version >/dev/null; then
        healthy=1
        break
    fi
    sleep 1
done
if [[ "${healthy}" -ne 1 ]]; then
    # An unhealthy restore is a failed rollback: undo it rather than leaving
    # the machine with a service that will not start.
    echo "error: restored Ollama did not become healthy" >&2
    undo
fi
trap - ERR

echo "Restored the pre-XDNA Ollama install."
if [[ "${runtime_moved}" -eq 1 ]]; then
    echo "Displaced runtime: ${failed_runtime}"
fi
if [[ "${binary_moved}" -eq 1 ]]; then
    echo "Displaced binary:  ${failed_binary}"
fi
if [[ -n "${drop_in_removed}" ]]; then
    echo "Displaced drop-in: ${drop_in_removed}"
fi
"${systemctl_bin}" --no-pager --full status ollama
