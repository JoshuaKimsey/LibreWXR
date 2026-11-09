# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Read-only tile-request history endpoint.

Serves the bounded in-memory ring of cumulative tile-counter snapshots
captured by the render lifespan's sampling loop.  Every snapshot records
cumulative totals at capture time; consumers compute intervals from
consecutive records.  The router is only included in the app when
``settings.tile_history_enabled`` is true, so a disabled endpoint has no
route at all (a plain 404).
"""

from fastapi import APIRouter

from librewxr.api import routes
from librewxr.config import settings

router = APIRouter()


@router.get(settings.tile_history_path)
async def tile_history() -> dict:
    """Return the per-worker tile-request history ring."""
    tracker = routes.tile_request_tracker
    if tracker is None:
        return {
            "enabled": True,
            "tracking_enabled": False,
            "interval_s": settings.tile_history_interval_s,
            "max_entries": settings.tile_history_max_entries,
            "snapshots": [],
        }
    return {
        "enabled": True,
        "tracking_enabled": True,
        "interval_s": settings.tile_history_interval_s,
        "max_entries": settings.tile_history_max_entries,
        "snapshots": tracker.history(),
    }
