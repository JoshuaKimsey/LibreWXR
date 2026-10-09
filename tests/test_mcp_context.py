# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Stdio-mode MCP context: storm-cell store hydration from state.json.

Regression coverage for the gap where ``build_stdio_lifespan`` never
built a ``StormCellStore`` or assigned ``routes.storm_cell_store``, so
``get_storm_cells`` always returned ``[]`` over the stdio transport even
when the pipeline had detected cells.  The lifespan now mirrors the
render-only worker: construct the store (``cleanup_tmp=False``; the
pipeline owns the shared dir), add it to the ``stores`` dict so
``apply_state`` reloads it, and expose it on ``routes``.
"""

import numpy as np
import pytest

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.master_state import dump_state
from librewxr.data.storm_cells import _CELL_DTYPE, StormCellStore
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.mcp.context import build_stdio_lifespan
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH

pytestmark = pytest.mark.mcp

# routes.* singletons the stdio lifespan writes; saved and restored so a
# context boot in one test can't leak into the rest of the suite.
_ROUTES_ATTRS = (
    "frame_store",
    "tile_cache",
    "nwp_grids",
    "ecmwf_grid",
    "nwp_chain",
    "satellite_grids",
    "nowcast_store",
    "tile_request_tracker",
    "start_time",
    "enabled_regions",
    "radar_cache",
    "radar_fetcher",
    "alerts_store",
    "alerts_fetcher",
    "alerts_enabled",
    "storm_cell_store",
)


@pytest.fixture(autouse=True)
def _restore_routes_state():
    """Snapshot/restore the routes module-level singletons per test."""
    saved = {name: getattr(routes, name) for name in _ROUTES_ATTRS}
    yield
    for name, value in saved.items():
        setattr(routes, name, value)


def _configure_settings(monkeypatch, cache_dir) -> None:
    """Point the stdio boot at the tmp snapshot with optional stores off.

    Mirrors ``test_data_pipeline.py``'s render-only setup: every optional
    store except storm cells is disabled so the boot stays light.  Storm
    cells are toggled per-test by the caller.
    """
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)


async def _seed_state(cache_dir, *, with_storm_cells: bool) -> None:
    """Write a real state.json: a FrameStore plus an optional cell store."""
    frame_store = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await frame_store.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))

    stores: dict[str, object] = {"frame_store": frame_store}
    if with_storm_cells:
        cell_store = StormCellStore(cache_dir=cache_dir)
        cells = np.array([
            (10.0, 20.0, 50.0, 61.6, 45.0,
             0.0, 0.0, float("nan"), float("nan")),
        ], dtype=_CELL_DTYPE)
        await cell_store.replace_cells({"USCOMP": cells})
        stores["storm_cell_store"] = cell_store

    dump_state(stores, cache_dir)


async def test_stdio_context_restores_storm_cells(tmp_path, monkeypatch):
    """Snapshot carries cells -> the booted context exposes a live store."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _configure_settings(monkeypatch, cache_dir)
    monkeypatch.setattr(settings, "storm_cells_enabled", True)
    await _seed_state(cache_dir, with_storm_cells=True)

    async with build_stdio_lifespan(None):
        store = routes.storm_cell_store
        assert store is not None
        counts = await store.get_counts()
        assert sum(counts.values()) > 0


async def test_stdio_context_without_storm_cell_key_is_none(
    tmp_path, monkeypatch,
):
    """Snapshot omits the key (enabled here) -> degraded None, no raise."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _configure_settings(monkeypatch, cache_dir)
    monkeypatch.setattr(settings, "storm_cells_enabled", True)
    await _seed_state(cache_dir, with_storm_cells=False)

    async with build_stdio_lifespan(None):
        assert routes.storm_cell_store is None


async def test_stdio_context_storm_cells_disabled_is_none(tmp_path, monkeypatch):
    """``storm_cells_enabled=False`` -> None even when the snapshot has it."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _configure_settings(monkeypatch, cache_dir)
    monkeypatch.setattr(settings, "storm_cells_enabled", False)
    await _seed_state(cache_dir, with_storm_cells=True)

    async with build_stdio_lifespan(None):
        assert routes.storm_cell_store is None
