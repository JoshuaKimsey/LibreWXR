# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""RRQPE radar source — serves the GLB-5 scan store as radar frames.

Implements the RadarSource fetch protocol over the scan store in
``grid.RRQPEGrid``.  Frame → scan matching is the constant-shift relative
matching described in ``grid.py`` (reused wholesale via
``RRQPEGrid.match_timestamp``): every past frame is served the scan
exactly ``RRQPE_LAG_SECONDS`` (30 min) its senior, so consecutive frames
map to consecutive scans (1:1 distinct, deterministic) and client
animations step smoothly.  ``None`` is returned when no constant-shift
scan matches — the region is simply absent from that frame, so the
fetcher's carry-forward covers up to two intervals and tiles then fall
through to NWP fill, the correct degradation for a stale store.

The scan store is refreshed lazily on the newest-slot request of each
fetch cycle: the fetch protocol is per (region, ts), but the scan store
is a bulk S3 download that must only run once per cycle (the grid's own
throttle collapses the startup double-call into one pass).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np

from librewxr.config import settings
from librewxr.data.regions import RegionDef

from .grid import RRQPEGrid, SCAN_INTERVAL_SECONDS

logger = logging.getLogger(__name__)


class RRQPESource:
    """NOAA Enterprise Rain Rate GLB-5 blend as a radar source."""

    name = "rrqpe"

    def __init__(self, grid: RRQPEGrid):
        self._grid = grid

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        """Serve the constant-shift scan for the frame ``minutes_ago`` back.

        The newest-slot request (``minutes_ago == 0``) triggers the
        scan-store refresh for this cycle first, so newly published
        scans become available before any slot is matched.
        """
        if minutes_ago == 0:
            await self._refresh_scans()
        now_aligned = (
            int(time.time()) // settings.fetch_interval
        ) * settings.fetch_interval
        target_ts = now_aligned - minutes_ago * 60
        return self._serve(region, target_ts)

    async def fetch_archive_frame(
        self, region: RegionDef, when: datetime,
    ) -> np.ndarray | None:
        """Serve the constant-shift scan for an archive frame timestamp."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return self._serve(region, int(when.timestamp()))

    async def _refresh_scans(self) -> None:
        """Pull any newly published scans into the store.

        The history window covers every slot the frame timestamps in one
        cycle can reference after the constant shift (``max_frames``
        intervals back), plus the grid's tolerance and one extra scan
        interval so an older-neighbor fallback for the oldest frame's
        target is never evicted.
        """
        history = (
            settings.max_frames * settings.fetch_interval
            + settings.rrqpe_match_tolerance_seconds
            + SCAN_INTERVAL_SECONDS
        )
        try:
            await self._grid.fetch(history_seconds=history)
        except Exception:
            logger.exception("RRQPE scan-store refresh failed")

    def _serve(self, region: RegionDef, ts: int) -> np.ndarray | None:
        """The uint8 dBZ-encoded frame for the constant-shift match, or None."""
        match = self._grid.match_timestamp(ts)
        if match is None:
            return None
        frame = self._grid.frame_at(match)
        if frame is None:
            return None
        return frame

    async def close(self) -> None:
        """Release the scan store (memmaps + HTTP client)."""
        await self._grid.close()
