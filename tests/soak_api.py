#!/usr/bin/env python3
"""Sustained per-model endurance test for a separately started API.

Runs a fixed number of completions (100 by default) as consecutive per-model
blocks of ``HAWKPOINT_SOAK_RUN_LENGTH`` requests (25 by default). This keeps
each model resident under sustained load for a long stretch, which is what a
real serving session looks like, while the dedicated ``switch_stress_api.py``
covers rapid model switching.

Zero tolerance: any HTTP failure, any token mismatch, or any amdxdna/XRT/NPU
kernel error fails the gate.
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


COMPLETIONS = int(os.environ.get("HAWKPOINT_SOAK_COMPLETIONS", "100"))
RUN_LENGTH = int(os.environ.get("HAWKPOINT_SOAK_RUN_LENGTH", "25"))
REPORT = Path(os.environ.get("HAWKPOINT_SOAK_REPORT", "soak-report.json"))
API_LOG = Path(os.environ.get("HAWKPOINT_API_LOG", "hawkpoint-api.log"))

if RUN_LENGTH < 1:
    raise SystemExit("HAWKPOINT_SOAK_RUN_LENGTH must be at least 1")
if COMPLETIONS < 1:
    raise SystemExit("HAWKPOINT_SOAK_COMPLETIONS must be at least 1")


def model_for(index, models):
    """Consecutive blocks of RUN_LENGTH requests per model, round-robin."""
    return models[(index // RUN_LENGTH) % len(models)]


def main():
    """Run the selected per-model budget and fail on request or kernel errors."""
    expected = load_expected()
    models = installed_models(expected)

    started_epoch = int(time.time())
    start_energy = energy_uj()

    latencies = []
    ttfts = []
    token_rates = []
    per_model = defaultdict(lambda: {"success": 0, "failure": 0})
    failures = []
    switches = 0
    switch_errors = 0
    previous_model = None

    with ResourceMonitor(API_PID) as monitor:
        for index in range(COMPLETIONS):
            model = model_for(index, models)
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
                    switch_errors += 1
                failures.append(
                    {"index": index, "model": model, "error": str(exc)[:500]}
                )
            previous_model = model
            if index and index % 100 == 0:
                print(f"completed {index}/{COMPLETIONS}", flush=True)

    end_energy = energy_uj()
    xrt_errors = kernel_errors(started_epoch, API_LOG)
    report = {
        "schema_version": 2,
        "test": "endurance",
        "test_profile": (
            "quick" if (COMPLETIONS, RUN_LENGTH) == (100, 25)
            else "endurance" if (COMPLETIONS, RUN_LENGTH) == (1000, 250)
            else "custom"
        ),
        "started_epoch": started_epoch,
        "completed_epoch": int(time.time()),
        "requested_completions": COMPLETIONS,
        "run_length_per_model": RUN_LENGTH,
        "model_switches": switches,
        "successful_requests": len(latencies),
        "failed_requests": len(failures),
        "model_switch_errors": switch_errors,
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

    if failures or xrt_errors:
        raise SystemExit(
            f"FAIL failures={len(failures)} xrt_npu_errors={len(xrt_errors)}"
        )
    print(
        f"PASS {COMPLETIONS} measured completions "
        f"({RUN_LENGTH} consecutive per model) across {len(models)} model(s)"
    )


if __name__ == "__main__":
    main()
