# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA MSM — self-contained NWP source package.

5 km Mesoscale Model from the Japan Meteorological Agency, distributed
via Open-Meteo's anonymous AWS S3 mirror (``s3://openmeteo`` in
``us-west-2``).  Covers 22.4-47.6°N, 120-150°E (Japan + Korean
Peninsula + Taiwan + Yellow Sea + adjacent western Pacific).  8 daily
runs (00/03/06/09/12/15/18/21 UTC); 78 h horizon from the main 00Z/12Z
runs, 39 h from the others; hourly steps.

Priority 20 in the chain — Japan's flagship regional NWP, same
authority tier as HRDPS for Canada.  Sits ahead of the AROME-OM
variants (26-29) and ahead of the lower-priority European LAMs.

Note: this is the publicly-redistributable Open-Meteo mirror; JMA's
direct JMBSC feed is paid-contract-only.  The Open-Meteo AWS Open
Data Sponsorship arrangement is the only viable anonymous access path.
LFM (2 km local model) and MEPS ensemble are NOT mirrored — MSM is the
only JMA NWP we can ingest today.

Data attribution: Japan Meteorological Agency (JMA), distributed via
Open-Meteo's AWS Open Data mirror.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import JMAMSMGrid

__all__ = ["JMAMSMGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return a JMA MSM contribution when ``settings.jma_msm_enabled`` is set."""
    if not getattr(settings, "jma_msm_enabled", True):
        return None
    return NWPContribution(
        instance=JMAMSMGrid(cache_dir=cache_dir),
        priority=20,
        name="JMA MSM",
    )
