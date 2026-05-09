# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Smoke tests for the standalone data-pipeline entry point.

We don't exercise the full fetch loop here — that touches real network
endpoints and is covered by the per-source integration tests.  The
goals are:

1. ``run_pipeline`` errors out loudly when ``LIBREWXR_CACHE_DIR`` is
   unset (the multi-worker split is meaningless without a shared dir).
2. The module imports without dragging in FastAPI / uvicorn.
3. A minimal pipeline can be wired up far enough to dump a state.json
   snapshot and shut down cleanly when SIGTERM arrives.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.store


def test_module_does_not_import_fastapi():
    # The whole point of the split is that the pipeline doesn't need
    # FastAPI / uvicorn / starlette / librewxr.api pulled in.  We can't
    # check sys.modules in the full suite (other tests get there first),
    # but a static scan of the source file catches the regression we care
    # about: someone copy-pasting an import out of main.py.
    mod = importlib.import_module("librewxr.data_pipeline")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = ("fastapi", "uvicorn", "starlette", "librewxr.api")
    for name in forbidden:
        assert name not in src, f"data_pipeline must not import {name}"
    assert hasattr(mod, "run_pipeline")
    assert hasattr(mod, "main")


def test_run_pipeline_requires_cache_dir(monkeypatch):
    # No cache_dir → SystemExit with a clear message.  Without this the
    # pipeline would silently start with no shared snapshot and render
    # workers would idle forever waiting for state.json.
    from librewxr.config import settings

    monkeypatch.setattr(settings, "cache_dir", "")
    from librewxr import data_pipeline

    with pytest.raises(SystemExit, match="LIBREWXR_CACHE_DIR"):
        asyncio.run(data_pipeline.run_pipeline())


def test_pipeline_writes_state_json_via_hook(tmp_path, monkeypatch):
    # End-to-end: bypass the heavy bits (fetcher, nwp_chain, alerts)
    # and verify that on_cycle_complete dumps state.json with the
    # frame_store entry populated.  This is the contract render-only
    # workers depend on.
    from librewxr.data.master_state import (
        STATE_FILENAME,
        STATE_VERSION,
        load_state,
    )
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    store = FrameStore(max_frames=4, cache_dir=cache_dir)

    async def _exercise():
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        await store.add_frame(RadarFrame(timestamp=12345, regions={"USCOMP": data}))

        # Build the same "stores" dict data_pipeline.py builds and run
        # the dump path the on_cycle_complete hook calls.
        from librewxr.data.master_state import dump_state

        stores = {
            "frame_store": store,
            "ecmwf_grid": None,
            "hrrr_grid": None,
            "hrrr_alaska_grid": None,
            "hrdps_grid": None,
            "arome_antilles_grid": None,
            "wrf_smn_grid": None,
            "icon_eu_grid": None,
            "dmi_dini_grid": None,
            "cloud_grid": None,
            "nowcast_store": None,
        }
        dump_state(stores, cache_dir)

    try:
        asyncio.run(_exercise())
    finally:
        store.cleanup()

    state_path = cache_dir / STATE_FILENAME
    assert state_path.exists()
    payload = load_state(cache_dir)
    assert payload["version"] == STATE_VERSION
    assert "frame_store" in payload["stores"]
    # Disabled stores should be absent — render workers shouldn't try to
    # __setstate__ on a None entry.
    assert "ecmwf_grid" not in payload["stores"]


@pytest.mark.asyncio
async def test_render_only_lifespan_picks_up_snapshot(tmp_path, monkeypatch):
    # Pipeline-side: write a state.json with one frame.  Render-only-side:
    # spin up _render_only_lifespan and confirm the FrameStore came back
    # populated and routes were wired.
    from librewxr.config import settings
    from librewxr.data.master_state import dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Producer side
    producer = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await producer.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))
    dump_state({"frame_store": producer}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    # Disable optional stores that don't appear in the snapshot — keeps
    # the render-only path from spinning up Cloud / Nowcast plumbing
    # this smoke test doesn't care about.
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)

    from librewxr import main as main_module
    from librewxr.api import routes

    # FastAPI app stub — _render_only_lifespan only takes app for symmetry.
    class _StubApp:
        pass

    async with main_module._render_only_lifespan(_StubApp()):
        # Frame store should have been populated from the snapshot.
        assert routes.frame_store is not None
        timestamps = await routes.frame_store.get_timestamps()
        assert timestamps == [42]
        # Render-only workers must not have a fetcher or radar_cache.
        assert routes.radar_fetcher is None
        assert routes.radar_cache is None
        # Disabled stores should be absent (snapshot didn't include them).
        assert routes.ecmwf_grid is None

    # cleanup happens inside the lifespan __aexit__; nothing to assert.


@pytest.mark.asyncio
async def test_render_only_requires_cache_dir(monkeypatch):
    from librewxr.config import settings

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", "")

    from librewxr import main as main_module

    class _StubApp:
        pass

    with pytest.raises(RuntimeError, match="LIBREWXR_CACHE_DIR"):
        async with main_module._render_only_lifespan(_StubApp()):
            pass
