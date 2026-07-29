#!/usr/bin/env bash
set -euo pipefail

[[ "$(id -u)" -eq 0 ]] || {
    echo "error: rollback.sh must run as root" >&2
    exit 1
}

latest_runtime="$(
    find /usr/local/lib -maxdepth 1 -type d \
        -name 'ollama.pre-xdna-*' -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n1 | cut -d' ' -f2-
)"
[[ -n "${latest_runtime}" ]] || {
    echo "error: no Ollama-XDNA backup was found" >&2
    exit 1
}

stamp="${latest_runtime##*-xdna-}"
backup_binary="/usr/local/bin/ollama.pre-xdna-${stamp}"
drop_in="/etc/systemd/system/ollama.service.d/xdna.conf"
drop_in_backup="${drop_in}.pre-xdna-${stamp}"
[[ -x "${backup_binary}" ]] || {
    echo "error: matching Ollama binary backup is missing" >&2
    exit 1
}

systemctl stop ollama
mv /usr/local/lib/ollama \
    "/usr/local/lib/ollama.failed-xdna-$(date +%Y%m%d-%H%M%S)"
mv "${latest_runtime}" /usr/local/lib/ollama
mv /usr/local/bin/ollama \
    "/usr/local/bin/ollama.failed-xdna-$(date +%Y%m%d-%H%M%S)"
mv "${backup_binary}" /usr/local/bin/ollama
if [[ -f "${drop_in_backup}" ]]; then
    mv "${drop_in_backup}" "${drop_in}"
else
    rm -f "${drop_in}"
fi
systemctl daemon-reload
systemctl start ollama
systemctl --no-pager --full status ollama
