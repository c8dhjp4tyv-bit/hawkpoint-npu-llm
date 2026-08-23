#!/usr/bin/env python3
"""Shared RAPL/powercap energy accounting for the hardware evidence runs.

``/sys/class/powercap`` links every zone as a sibling, subzones
(``intel-rapl:0:0``) included, so summing every ``energy_uj`` counts the same
joules two or three times over. Both the endurance soak and the Ollama
placement matrix report energy into release evidence, so the discovery rule
lives here once instead of drifting between them.
"""

from pathlib import Path


POWERCAP_ROOT = "/sys/class/powercap"


def energy_zones(root=POWERCAP_ROOT):
    """Return the top-level zone directories, parents only, sorted by name.

    A zone is skipped when its own parent directory also exposes
    ``energy_uj``: that means it is a subdomain of a zone already counted.
    """
    zones = []
    for path in sorted(Path(root).glob("*/energy_uj")):
        try:
            resolved = path.resolve()
            if (resolved.parent.parent / "energy_uj").exists():
                continue
        except OSError:
            continue
        zones.append(path)
    return zones


def energy_uj(root=POWERCAP_ROOT):
    """Total microjoules across the top-level zones, or None if unreadable."""
    total, _ = energy_uj_with_zones(root)
    return total


def energy_uj_with_zones(root=POWERCAP_ROOT):
    """Return ``(total_uj, zone_names)`` so evidence can record what was read."""
    readings = []
    names = []
    for path in energy_zones(root):
        try:
            readings.append(int(path.read_text()))
        except (OSError, ValueError):
            continue
        names.append(path.parent.name)
    if not readings:
        return None, []
    return sum(readings), names
