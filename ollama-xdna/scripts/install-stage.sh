#!/usr/bin/env bash
set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

[[ "$(id -u)" -eq 0 ]] || die "install-stage.sh must run as root"
[[ "$#" -eq 4 ]] || die "usage: $0 STAGE VERSION BACKEND GPU_LAYERS"

stage="$(realpath "$1")"
version="$2"
backend="$3"
gpu_layers="$4"
stamp="$(date +%Y%m%d-%H%M%S)"
runtime_backup="/usr/local/lib/ollama.pre-xdna-${stamp}"
binary_backup="/usr/local/bin/ollama.pre-xdna-${stamp}"
failed_runtime="/usr/local/lib/ollama.failed-xdna-${stamp}"
drop_in="/etc/systemd/system/ollama.service.d/xdna.conf"
drop_in_backup="${drop_in}.pre-xdna-${stamp}"

[[ -x "${stage}/bin/ollama" ]] || die "staged Ollama binary is missing"
[[ -x "${stage}/lib/ollama/llama-server" ]] ||
    die "staged llama-server is missing"
[[ -f "${stage}/lib/ollama/xdna/libggml-xdna.so" ]] ||
    die "staged XDNA backend is missing"
[[ -f "${stage}/lib/ollama/xdna/experts.xclbin" ]] ||
    die "staged AIE program is missing"
command -v systemctl >/dev/null || die "systemd is required for system install"
systemctl cat ollama.service >/dev/null || die "ollama.service is not installed"
[[ -x /usr/local/bin/ollama ]] || die "/usr/local/bin/ollama is missing"
[[ -d /usr/local/lib/ollama ]] || die "/usr/local/lib/ollama is missing"

systemctl stop ollama
if [[ -f "${drop_in}" ]]; then
    cp -a "${drop_in}" "${drop_in_backup}"
fi
mv /usr/local/bin/ollama "${binary_backup}"
mv /usr/local/lib/ollama "${runtime_backup}"
install -d -m 0755 /usr/local/lib/ollama

restore_previous() {
    systemctl stop ollama 2>/dev/null || true
    if [[ -d /usr/local/lib/ollama ]]; then
        mv /usr/local/lib/ollama "${failed_runtime}"
    fi
    mv "${runtime_backup}" /usr/local/lib/ollama
    if [[ -e /usr/local/bin/ollama ]]; then
        mv /usr/local/bin/ollama \
            "/usr/local/bin/ollama.failed-xdna-${stamp}"
    fi
    mv "${binary_backup}" /usr/local/bin/ollama
    if [[ -f "${drop_in_backup}" ]]; then
        mv "${drop_in_backup}" "${drop_in}"
    else
        rm -f "${drop_in}"
    fi
    systemctl daemon-reload
    systemctl start ollama
}
trap restore_previous ERR

cp -a "${stage}/lib/ollama/." /usr/local/lib/ollama/
install -m 0755 "${stage}/bin/ollama" /usr/local/bin/ollama
install -d -m 0755 /etc/systemd/system/ollama.service.d

{
    printf '%s\n' '[Service]'
    printf '%s\n' 'LimitMEMLOCK=infinity'
    printf '%s\n' \
        'Environment="GGML_BACKEND_PATH=/usr/local/lib/ollama/xdna/libggml-xdna.so"'
    printf '%s\n' \
        'Environment="GGML_XDNA_XCLBIN=/usr/local/lib/ollama/xdna/experts.xclbin"'
    printf '%s\n' \
        'Environment="GGML_XDNA_INSTS=/usr/local/lib/ollama/xdna/insts.bin"'
    printf 'Environment="OLLAMA_XDNA_GPU_LAYERS=%s"\n' "${gpu_layers}"
    if [[ "${backend}" != "cpu" ]]; then
        printf 'Environment="OLLAMA_LLM_LIBRARY=%s"\n' "${backend}"
    fi
} > "${drop_in}"

install -m 0755 "$(dirname "$0")/rollback.sh" \
    /usr/local/lib/ollama/xdna/rollback.sh
systemctl daemon-reload
systemctl start ollama

healthy=0
for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/version >/dev/null; then
        healthy=1
        break
    fi
    sleep 1
done
[[ "${healthy}" -eq 1 ]] || die "health check failed"
trap - ERR

echo "Installed Ollama ${version}-xdna with ${backend} + XDNA."
echo "Runtime backup: ${runtime_backup}"
echo "Binary backup:  ${binary_backup}"
