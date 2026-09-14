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

## Monitoring

- Alert when `/ready` remains `503`, `worker_restarts` rises repeatedly, or
  completion latency approaches the request timeout.
- Correlate client errors with access logs using `X-Request-ID`.
- Treat an SSE response without `[DONE]` as incomplete. Inspect sanitized SSE
  error events even when HTTP status is already `200`.
- Track host RAM, NPU driver errors, request `429`/`5xx` rates, and disk space
  for converted models outside the process.

## Shutdown and recovery

Send SIGTERM once. The server immediately becomes unready, rejects new
completions with `503`, drains admitted work, closes HTTP sockets, and releases
the worker's XRT context. Let the configured graceful timeout expire before a
service manager sends SIGKILL.

After a timeout or inference error, the failed worker is discarded. The next
request starts a fresh worker; readiness stays false until a request or prewarm
succeeds. If restarts repeat, stop traffic and preserve logs plus the output of
`scripts/verify_hardware_versions.py` before restarting the host service.

## Release checklist

A public release should not be described as validated unless all are true:

1. Hosted CI passes from the exact candidate commit.
2. The tag-triggered physical Hawk Point gate passes fresh model conversion,
   correctness, switching, endurance, and Ollama rollback checks.
3. The publish job produces its SBOM, signatures, provenance, and GitHub
   Release from that same dependency chain.
4. Compatibility changes, known limitations, and measured results are updated
   in `SUPPORT.md`, `CHANGELOG.md`, and `BENCHMARKS.md`.
5. A rollback candidate and the previous known-good artifacts remain available.

See `SECURITY.md` for disclosure and network-boundary requirements.
