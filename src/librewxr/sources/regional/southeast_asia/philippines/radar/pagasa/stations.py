# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA Philippines radar station inventory.

Nine-station radar network feeding the PANAHON national mosaic.  May 2026
coordinates come from the PANAHON JS bundle (centres of each station's
rendering rectangle in the web app, which closely match the documented
siting at airports and PAGASA weather offices).  Davao is new since May
2026 — the app's station list now carries nine entries (Baguio, Baler,
Echague, Daet, Guiuan, Iloilo, Davao, Kabacan, Panabo); a 2026-08-19
probe of ``https://panahon.gov.ph/js/beta.js`` confirmed the list but
ships only ``{name, slug}`` pairs, so Davao's coordinates are the
approximate PAGASA Davao radar site rather than an exact bundle value.
"""
from __future__ import annotations


PHCOMP_STATIONS: list[tuple[float, float]] = [
    (15.744, 121.632),   # Baler, Aurora
    (16.351, 120.559),   # Baguio, Benguet
    (16.716, 121.684),   # Echague, Isabela
    (14.124, 122.983),   # Daet, Camarines Norte
    (11.044, 125.754),   # Guiuan, Eastern Samar
    (10.770, 122.580),   # Iloilo, Iloilo
    (7.130, 125.650),    # Davao, Davao City — approximate (PAGASA Davao radar site; not in the JS bundle)
    (7.100, 124.834),    # Kabacan, North Cotabato
    (7.316, 125.634),    # Panabo, Davao del Norte
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "PHCOMP": PHCOMP_STATIONS,
}

# Range overrides: none.  A 2026-08-19 probe of the PANAHON API found a
# uniform ~240 km cap on every station that could be tested (Echague,
# Baler, Baguio, Daet, Iloilo), refuting the May 2026 claim of 80 km
# products for Echague / Kabacan / Panabo.  Kabacan / Panabo / Davao
# were untestable (no rain over Mindanao) — if they ever prove
# short-range, add overrides here.  The default 240 km applies to all.
RANGE_OVERRIDES: dict[str, float] = {}