# API operations guide

This runbook covers the native OpenAI-compatible API. The NPU kernels remain
experimental; only a release that passes the gated Hawk Point workflow carries
hardware acceptance evidence.

## Deployment boundary

- Keep the default `127.0.0.1` bind unless a trusted TLS reverse proxy protects
  the service. Never expose an unauthenticated plaintext listener.
- Generate a dedicated high-entropy `HAWKPOINT_API_KEY`, store it in the
  service manager's secret facility, and rotate it after suspected disclosure.
- Run as an unprivileged account with access only to the required XDNA/XRT
  device nodes and model directory. Do not run the API as root.
- Keep model files read-only for the service account after checksum-verified
  conversion. Back up only configuration and manifests; weights are
  reproducible from pinned upstream revisions.
- Put request and connection limits on the reverse proxy as an outer layer.
  The application also enforces body size, admission, rate, and time limits.

## Start and verify

Run compatibility checks before starting a newly provisioned host:

```bash
python scripts/verify_hardware_versions.py
```

Start with an explicit key and conservative admission settings:

```bash
export HAWKPOINT_API_KEY="<service-manager-secret>"
python launcher.py api
```

Use liveness only to detect a running HTTP process:

```bash
curl --fail http://127.0.0.1:8000/health
```

Use readiness for traffic admission. It returns `503` before prewarm completes,
after a worker failure, and while graceful shutdown drains active work:

```bash
curl --fail http://127.0.0.1:8000/ready
```

Finally, make an authenticated model-list and small completion request. A
successful health check alone does not prove that the NPU can execute.

## Ubuntu 24.04 with kernel 7.0

These observations come from one host and are independent compatibility
evidence, not part of the certified stack in `release-pins.json`: Ubuntu
24.04.5, kernel `7.0.0-38-generic` with its in-tree amdxdna driver, Ryzen 9
PRO 8945HS. They do not establish an affected or fixed kernel-version range;
qualify a host with the compatibility check above.

- **Legacy firmware.** The host's installed `linux-firmware-amd-misc`
  (`20240318.git3b128b60-0ubuntu3.1`) provided only the legacy
  `amdnpu/1502_00/npu.sbin` (1.5.2.380) and no `npu_7.sbin`. `xrt-smi examine`
  still worked, but the tested NPU execution attempts failed with
  `Message callback ret -22` in `dmesg`. Upstream linux-firmware maps
  `npu_7.sbin` to `npu.sbin.1.5.5.391`; installing that file as
  `/lib/firmware/updates/amdnpu/1502_00/npu_7.sbin` and reloading the module
  (`sudo rmmod amdxdna && sudo modprobe amdxdna`) fixed it.
- **Command-buffer size.** XRT `8bf2fc4` allocates 4 MiB execution buffers;
  this driver rejects command buffers above 32 KiB, so submissions using those
  buffers failed with `Cmd bo is too big 4194304` in `dmesg`. An XRT build
  with upstream commit `4f7258f0ac82` (which removes the 1024x size
  multiplier) fixed it. The patched build still reports the `8bf2fc4` hash.

## Capacity and timeouts

One physical NPU context serializes inference. Tune these controls together:

- `--queue-capacity`: waiting requests beyond the active request; excess work
  receives `429` with `Retry-After`.
- `--rate-limit-per-minute`: per-client request admission; `0` disables it only
  when an upstream limiter is deliberately responsible.
- `--request-timeout`: idle socket and inference deadline.
- `--graceful-shutdown-timeout`: maximum drain window. Keep it slightly above
  the request timeout so admitted work normally finishes before worker close.
- `--max-body-bytes`: reject unexpectedly large JSON before parsing.

Load-test with the intended model, prompt distribution, proxy, and client
timeouts. Do not infer capacity from the component smoke benchmark.

## Monitoring and Observability

The API server exposes standard OpenMetrics at `GET /metrics` and health endpoints compatible with Kubernetes liveness and readiness probes (`/health/live`, `/health/ready`).

### Prometheus Scrape Configuration

Add the scrape target to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'hawkpoint-npu'
    scrape_interval: 10s
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/secrets/hawkpoint_api_key
    static_configs:
      - targets: ['127.0.0.1:8000']
```

Place the same `HAWKPOINT_API_KEY` used by the server in the credentials file.
Keep it readable only by the Prometheus service account, mount it read-only in
containers, and do not commit it. For remote targets, configure HTTPS at the
server or reverse proxy and scrape that HTTPS endpoint.

### Metrics Catalog

| Metric | Type | Description |
|:-------|:-----|:------------|
| `hawkpoint_api_requests_total` | Counter | Total HTTP requests categorized by `endpoint` and `status` |
| `hawkpoint_api_active_requests` | Gauge | Current number of concurrently running or queued requests |
| `hawkpoint_worker_restarts_total` | Counter | Total worker process restarts triggered by timeouts or faults |
| `hawkpoint_tokens_generated_total` | Counter | Cumulative completion tokens generated per `model` |
| `hawkpoint_prompt_tokens_total` | Counter | Cumulative prompt tokens processed per `model` |
| `hawkpoint_inference_duration_seconds` | Summary | Inference wall time duration summary (`_count` and `_sum`) |

### Recommended Prometheus Alerting Rules

```yaml
groups:
  - name: hawkpoint-npu.rules
    rules:
      - alert: NpuWorkerCrashLooping
        expr: increase(hawkpoint_worker_restarts_total[5m]) > 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "HawkPoint NPU worker is crash-looping or timing out"
          description: "NPU inference worker restarted {{ $value }} times in 5 minutes."

      - alert: NpuHighRequestQueue
        expr: hawkpoint_api_active_requests >= 3
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "HawkPoint NPU request queue saturation"
          description: "Active requests ({{ $value }}) exceed optimal pipeline capacity."
```

### Kubernetes Probe Specs

When running inside Kubernetes or K3s containers with access to `/dev/accel/accel*`:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

## API Benchmarking and Capacity Testing

Use the dedicated benchmark tool to measure latency percentiles (min, p50, p95, p99), Time To First Token (TTFT), and decode tokens/second against an active API server:

```bash
# Measure streaming throughput with 10 requests:
python scripts/benchmark_api.py --url http://127.0.0.1:8000 -n 10 -c 1

# Export a Markdown report with SLA enforcement:
python scripts/benchmark_api.py --format markdown -o benchmark-report.md --min-tps 45.0
```

## Shutdown and recovery

Send SIGTERM once. The server immediately becomes unready, rejects new
completions with `503`, drains admitted work, closes HTTP sockets, and releases
the worker's XRT context. Let the configured graceful timeout expire before a
service manager sends SIGKILL. After the drain deadline, active inference is
cancelled independently of its generation lock. Worker teardown can take up
to two additional five-second terminate/kill waits, so allow that cleanup
margin in the supervisor timeout.

After a timeout or inference error, the failed worker is discarded. The next
request starts a fresh worker; readiness stays false until a request or prewarm
succeeds. If restarts repeat, stop traffic and preserve logs plus the output of
`scripts/verify_hardware_versions.py` before restarting the host service.

## Release checklist

A public release should not be described as validated unless all are true:

1. Hosted CI passes from the exact candidate commit.
2. The tag-triggered physical Hawk Point gate passes fresh model conversion,
   correctness, switching, quick acceptance, and Ollama rollback checks.
   The quick loops use 100 native plus 4 × 25 Ollama requests. Long endurance
   is optional via a manual `Gated release` run with `profile: endurance`;
   do not claim long-duration stability from quick evidence.
3. The publish job produces its SBOM, signatures, provenance, and GitHub
   Release from that same dependency chain.
4. Compatibility changes, known limitations, and measured results are updated
   in `SUPPORT.md`, `CHANGELOG.md`, and `BENCHMARKS.md`.
5. A rollback candidate and the previous known-good artifacts remain available.

See `SECURITY.md` for disclosure and network-boundary requirements.

### Container device permissions

Before starting the API Compose profile, set the numeric supplementary groups
from the actual host device nodes (adjust the render node for your host):

```bash
export HAWKPOINT_ACCEL_GID=$(stat -c '%g' /dev/accel/accel0)
export HAWKPOINT_DRI_GID=$(stat -c '%g' /dev/dri/renderD128)
docker compose --profile full up -d
```

The container keeps its unprivileged `app` user. Host device GIDs must match
`group_add`; image-local `render` and `video` names need not match the host.
