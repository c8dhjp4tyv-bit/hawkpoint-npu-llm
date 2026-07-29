# Security policy

## Supported versions

Only the latest tagged alpha release receives security fixes. Experimental
hardware support and performance limitations are not security vulnerabilities.

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub's
private vulnerability reporting page:

<https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm/security/advisories/new>

Include the affected version, configuration, reproduction steps, impact, and
any suggested mitigation. Expect an acknowledgement within seven days. No
bounty or fixed remediation deadline is promised.

## Deployment boundary

The API binds to localhost by default, requires a bearer token, limits request
size and request admission, and restricts browser origins. TLS is available
with `--tls-cert` and `--tls-key`; otherwise keep the service on localhost or
put it behind a trusted TLS reverse proxy. Open WebUI uses Linux host
networking but both services remain bound to `127.0.0.1`.
