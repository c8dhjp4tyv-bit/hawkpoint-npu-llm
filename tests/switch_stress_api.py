#!/usr/bin/env python3
"""Bounded model-switch stress test for a separately started API.

Where ``soak_api.py`` measures sustained per-model endurance, this test does the
opposite: it switches the active model on *every* request, exercising the
deterministic close/release of the previous model's XRT/NPU hardware context
before the next one loads. This is the exact path that leaked driver contexts in
the RC3 release soak (``DRM_IOCTL_AMDXDNA_CREATE_HWCTX`` -110 /
``aie2_alloc_resource failed``).

Zero tolerance: any HTTP failure, any token mismatch, any amdxdna/XRT/NPU kernel
error, or fewer than the required number of switches fails the gate.
"""

import os
from pathlib import Path
import time

from hardware_soak_common import (
    API_PID,
    REQUEST_ERRORS,
    ResourceMonitor,
    complete_once,
    defaultdict,
    energy_uj,
    installed_models,
    kernel_errors,
    load_expected,
    summarize,
    write_report,
)


REQUIRED_SWITCHES = int(os.environ.get("HAWKPOINT_SWITCH_COUNT", "100"))
# One extra request so that REQUIRED_SWITCHES transitions actually occur
# (the first request is never a switch). Callers may raise the iteration
# count, but never below what is needed to reach the required switches.
ITERATIONS = max(
    REQUIRED_SWITCHES + 1,
    int(os.environ.get("HAWKPOINT_SWITCH_ITERATIONS", "0")),
)
REPORT = Path(
    os.environ.get("HAWKPOINT_SWITCH_REPORT", "switch-stress-report.json")
)
API_LOG = Path(os.environ.get("HAWKPOINT_API_LOG", "hawkpoint-api.log"))


def main():
    expected = load_expected()
    models = installed_models(expected)
    if len(models) < 2:
        raise SystemExit(
            "model-switch stress needs at least 2 installed models; "
            f"found {len(models)}"
        )

    started_epoch = int(time.time())
    start_energy = energy_uj()

    latencies = []
    ttfts = []
    token_rates = []
    per_model = defaultdict(lambda: {"success": 0, "failure": 0})
    failures = []
    switches = 0
    switch_failures = 0
    previous_model = None

    with ResourceMonitor(API_PID) as monitor:
        for index in range(ITERATIONS):
            # Alternate on every request to force a model switch each time.
            model = models[index % len(models)]
            switched = previous_model is not None and previous_model != model
            if switched:
                switches += 1
            try:
                elapsed, ttft, token_rate = complete_once(model, expected[model])
                latencies.append(elapsed)
                ttfts.append(ttft)
                token_rates.append(token_rate)
                per_model[model]["success"] += 1
            except REQUEST_ERRORS as exc:
                per_model[model]["failure"] += 1
                if switched:
                    switch_failures += 1
                failures.append(
                    {"index": index, "model": model, "error": str(exc)[:500]}
                )
            previous_model = model
            if index and index % 25 == 0:
                print(f"switched {switches} (request {index}/{ITERATIONS})", flush=True)

    end_energy = energy_uj()
    xrt_errors = kernel_errors(started_epoch, API_LOG)
    insufficient = switches < REQUIRED_SWITCHES
    report = {
        "schema_version": 2,
        "test": "model_switch_stress",
        "started_epoch": started_epoch,
        "completed_epoch": int(time.time()),
        "iterations": ITERATIONS,
        "required_switches": REQUIRED_SWITCHES,
        "model_switches": switches,
        "successful_requests": len(latencies),
        "failed_requests": len(failures),
        "switch_failures": switch_failures,
        "models": dict(per_model),
        "latency_seconds": summarize(latencies),
        "ttft_seconds": summarize(ttfts),
        "decode_tokens_per_second": summarize(token_rates),
        "ram_mib": {
            "start": monitor.start_ram,
            "end": monitor.end_ram,
            "peak": monitor.peak_ram,
            "growth": monitor.end_ram - monitor.start_ram,
        },
        "energy_joules": (
            (end_energy - start_energy) / 1_000_000
            if start_energy is not None
            and end_energy is not None
            and end_energy >= start_energy
            else None
        ),
        "xrt_npu_error_count": len(xrt_errors),
        "xrt_npu_errors": xrt_errors,
        "expected_token_ids": expected,
        "failures": failures[-200:],
    }
    write_report(REPORT, report)

    if failures or xrt_errors or insufficient:
        raise SystemExit(
            f"FAIL failures={len(failures)} xrt_npu_errors={len(xrt_errors)} "
            f"switches={switches}/{REQUIRED_SWITCHES}"
        )
    print(
        f"PASS {switches} model switches across {len(models)} models "
        f"with no HTTP or XRT/NPU errors"
    )


if __name__ == "__main__":
    main()
