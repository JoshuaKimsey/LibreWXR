# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Smoke tests for the standalone data-pipeline entry point.

We don't exercise the full fetch loop here — that touches real network
endpoints and is covered by the per-source integration tests.  The
goals are:

1. An unset ``LIBREWXR_CACHE_DIR`` falls back to a stable tempdir via
   ``resolve_cache_dir`` (covered in test_config.py); a render-only
   worker still fails loudly when no pipeline ever writes state.json.
2. The module imports without dragging in FastAPI / uvicorn.
3. A minimal pipeline can be wired up far enough to dump a state.json
   snapshot and shut down cleanly when SIGTERM arrives.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import threading
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

        # All per-source grid entries are None here — dump_state should
        # skip them, mirroring what the pipeline does when those sources
        # are disabled by config.  The slug shape is still arbitrary;
        # what matters is that the test treats them all as opt-out.
        stores = {
            "frame_store": store,
            "ecmwf_grid": None,
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
    # this smoke test doesn't care about.  Both ``nowcast_enabled`` and
    # ``arrow_flow_enabled`` must be off: the latter would otherwise
    # construct NowcastStore via the widened build condition added when
    # decoupling optical flow from nowcast.
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)
    # Keeps the warm+jitter wiring covered without a multi-GB zoom-6 store
    # publish in a unit test.
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)

    from librewxr import main as main_module
    from librewxr.api import routes

    # The warm-pass jitter (uniform 0..15 s) would stall this smoke test;
    # the constant is module-level precisely so tests can neutralize it.
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

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
async def test_render_only_lifespan_nowcast_store_skips_tmp_sweep(
    tmp_path, monkeypatch,
):
    """Render workers construct their NowcastStore with ``cleanup_tmp=False``.

    The pipeline owns the shared nowcast dir and may be mid-write when a
    worker boots; a default ``True`` sweep in ``__init__`` would unlink
    its in-flight ``*.tmp`` files (the sweep stays the pipeline's job at
    its own boot).  Capture the constructor kwargs on the boot path.
    """
    from librewxr.config import settings
    from librewxr.data.master_state import dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
    from librewxr import main as main_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    producer = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await producer.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))
    dump_state({"frame_store": producer}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    # Nowcast on so the boot path actually constructs the store; other
    # optional stores stay off (the snapshot doesn't include them).
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", True)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

    captured: list[dict] = []
    real_nowcast_store = main_module.NowcastStore

    def _capturing_nowcast_store(*args, **kwargs):
        captured.append(kwargs)
        return real_nowcast_store(*args, **kwargs)

    monkeypatch.setattr(main_module, "NowcastStore", _capturing_nowcast_store)

    class _StubApp:
        pass

    async with main_module._render_only_lifespan(_StubApp()):
        # The boot path constructed the store with the sweep opt-out.
        assert captured, "render-only boot did not construct NowcastStore"
        assert all(
            kwargs.get("cleanup_tmp") is False for kwargs in captured
        )


@pytest.mark.asyncio
async def test_render_only_lifespan_storm_cell_store_skips_tmp_sweep(
    tmp_path, monkeypatch,
):
    """Render workers construct their StormCellStore with ``cleanup_tmp=False``.

    The pipeline owns the shared storm-cells dir and may be mid-write
    when a worker boots; a default ``True`` sweep in ``__init__`` would
    unlink its in-flight ``*.tmp`` files (the sweep stays the pipeline's
    job at its own boot).  Capture the constructor kwargs on the boot
    path.
    """
    from librewxr.config import settings
    from librewxr.data.master_state import dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
    from librewxr import main as main_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    producer = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await producer.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))
    dump_state({"frame_store": producer}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    # Storm cells on so the boot path actually constructs the store;
    # other optional stores stay off (the snapshot doesn't include them).
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "storm_cells_enabled", True)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

    captured: list[dict] = []
    real_storm_cell_store = main_module.StormCellStore

    def _capturing_storm_cell_store(*args, **kwargs):
        captured.append(kwargs)
        return real_storm_cell_store(*args, **kwargs)

    monkeypatch.setattr(
        main_module, "StormCellStore", _capturing_storm_cell_store,
    )

    class _StubApp:
        pass

    async with main_module._render_only_lifespan(_StubApp()):
        # The boot path constructed the store with the sweep opt-out.
        assert captured, "render-only boot did not construct StormCellStore"
        assert all(
            kwargs.get("cleanup_tmp") is False for kwargs in captured
        )


@pytest.mark.asyncio
async def test_render_only_lifespan_frame_store_skips_tmp_sweep(
    tmp_path, monkeypatch,
):
    """Render workers construct their FrameStore with ``cleanup_tmp=False``.

    The pipeline writes radar memmaps into ``<cache_dir>/radar`` and may
    be mid-write when a worker boots; a default ``True`` sweep in
    ``__init__`` would unlink its in-flight ``*.tmp`` files, making the
    pipeline's ``os.replace`` raise and abort its fetch cycle.  Capture
    the constructor kwargs on the boot path.
    """
    from librewxr.config import settings
    from librewxr.data.master_state import dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
    from librewxr import main as main_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    producer = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await producer.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))
    dump_state({"frame_store": producer}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

    captured: list[dict] = []
    real_frame_store = main_module.FrameStore

    def _capturing_frame_store(*args, **kwargs):
        captured.append(kwargs)
        return real_frame_store(*args, **kwargs)

    monkeypatch.setattr(main_module, "FrameStore", _capturing_frame_store)

    class _StubApp:
        pass

    async with main_module._render_only_lifespan(_StubApp()):
        assert captured, "render-only boot did not construct FrameStore"
        assert all(
            kwargs.get("cleanup_tmp") is False for kwargs in captured
        )


def test_pipeline_stores_use_frame_deletion_grace(tmp_path, monkeypatch):
    """The pipeline builds a lingering FrameStore + matching cache grace.

    Both layers are required together: a snapshot published this cycle
    still names a timestamp the pipeline evicts next cycle, and the
    ``RadarFrameCache`` cleanup walks the same directory.  Capture the
    construction kwargs and abort the pipeline right there (everything
    after that point touches the network / real fetch loop).
    """
    from librewxr.config import settings
    from librewxr import data_pipeline

    class _Stop(Exception):
        """Abort run_pipeline once both construction sites were captured."""

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "volatile_cache_dir", "")
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "storm_cells_enabled", False)

    # Stub the source walkers / coverage builders the pipeline runs before
    # the two construction sites this test cares about.
    monkeypatch.setattr(data_pipeline, "collect_nwp_contributions", lambda *a, **k: [])
    monkeypatch.setattr(
        data_pipeline, "collect_satellite_contributions", lambda *a, **k: [],
    )
    monkeypatch.setattr(
        data_pipeline, "collect_radar_coverage_metadata",
        lambda *a, **k: ({}, {}, []),
    )
    monkeypatch.setattr(data_pipeline, "build_coverage_masks", lambda *a, **k: None)
    monkeypatch.setattr(data_pipeline, "build_feather_masks", lambda *a, **k: None)
    monkeypatch.setattr(
        data_pipeline, "persist_masks_in_background", lambda *a, **k: None,
    )

    captured: dict[str, dict] = {}
    real_frame_store = data_pipeline.FrameStore

    def _capturing_frame_store(*args, **kwargs):
        captured["frame_store"] = kwargs
        return real_frame_store(*args, **kwargs)

    def _capturing_radar_cache(*args, **kwargs):
        captured["radar_cache"] = kwargs
        raise _Stop

    monkeypatch.setattr(data_pipeline, "FrameStore", _capturing_frame_store)
    monkeypatch.setattr(data_pipeline, "RadarFrameCache", _capturing_radar_cache)

    with pytest.raises(_Stop):
        asyncio.run(data_pipeline.run_pipeline())

    expected = max(60.0, float(settings.fetch_interval))
    assert captured["frame_store"]["evict_grace_s"] == expected
    assert captured["frame_store"]["evict_grace_s"] > 0
    # The radar-cache cleanup walks the same directory and must use the
    # same window, or it deletes exactly what eviction deliberately kept.
    assert captured["radar_cache"]["grace_s"] == expected
    assert captured["radar_cache"]["grace_s"] > 0


@pytest.mark.asyncio
async def test_render_only_lifespan_yields_before_coord_warm(tmp_path, monkeypatch):
    """The coordinate warm must never block a render worker's readiness.

    Regression for the slow-storage boot incident: with a cold shared
    coord store the eager warm held the lifespan before ``yield`` for
    14-26 minutes, so the worker served no tiles.  The warm now runs as
    a background task after the lifespan yields; this test stubs
    ``warm_coordinate_caches`` with a blocker and asserts the lifespan
    enters (yields) while the warm is still blocked.
    """
    from librewxr.config import settings
    from librewxr.data.master_state import dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
    from librewxr import main as main_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    producer = FrameStore(max_frames=4, cache_dir=cache_dir)
    arr = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await producer.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": arr}))
    dump_state({"frame_store": producer}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    # Disable optional stores that don't appear in the snapshot — same
    # rationale as test_render_only_lifespan_picks_up_snapshot.
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", False)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)
    # Force the warm ON so this test proves it no longer blocks entry.
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

    warm_started = threading.Event()
    release_warm = threading.Event()

    def _blocking_warm(regions, max_zoom, tile_size=256):
        warm_started.set()
        release_warm.wait(30)
        return 0

    # Patch the module-level reference main.py resolves at call time.
    monkeypatch.setattr(main_module, "warm_coordinate_caches", _blocking_warm)

    class _StubApp:
        pass

    entered = asyncio.Event()

    async def _hold():
        async with main_module._render_only_lifespan(_StubApp()):
            entered.set()
            await asyncio.sleep(0.5)

    task = asyncio.create_task(_hold())
    try:
        # Wait for the warm to be *running* (blocked in the stub), then
        # assert the lifespan already yielded.  If the warm were still
        # eager/synchronous, entered would never be set while the warm is
        # blocked and this assertion fails.
        assert await asyncio.to_thread(warm_started.wait, 5), (
            "coordinate warm never started"
        )
        assert entered.is_set(), (
            "render-only lifespan did not yield while the coord warm was "
            "still running"
        )
    finally:
        release_warm.set()
        await task


@pytest.mark.asyncio
async def test_render_only_requires_cache_dir(monkeypatch):
    # A missing LIBREWXR_CACHE_DIR no longer hard-fails: resolve_cache_dir
    # falls back to a stable per-host tempdir.  With no pipeline writing
    # state.json there, the worker fails on the state wait timeout, and
    # the error names the fallback path so the operator can see where it
    # looked.
    import tempfile
    from pathlib import Path

    from librewxr.config import settings

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", "")
    monkeypatch.setattr(settings, "state_wait_timeout", 1)
    monkeypatch.setattr(settings, "state_poll_interval", 0.1)

    from librewxr import main as main_module

    fallback = Path(tempfile.gettempdir()) / "librewxr-cache"

    class _StubApp:
        pass

    with pytest.raises(RuntimeError) as excinfo:
        async with main_module._render_only_lifespan(_StubApp()):
            pass
    message = str(excinfo.value)
    assert "Is the data pipeline running?" in message
    assert str(fallback) in message


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="needs /proc/self/fd"
)
async def test_restored_frames_release_files_after_eviction(tmp_path):
    """Boot-restored frames must not pin their files once evicted.

    The pipeline used to keep the restored ``RadarFrame`` list alive in
    ``run_pipeline``'s scope, so every boot-time frame's memmap stayed
    open after the store evicted and unlinked it -- the deleted files'
    disk blocks were never released for the life of the process.
    """
    import gc

    from librewxr.data.radar_cache import RadarFrameCache
    from librewxr.data.regions import RegionDef
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.data_pipeline import _restore_cached_frames

    region = RegionDef(
        name="TESTCOMP", west=0.0, east=1.0, south=0.0, north=1.0,
        pixel_size=0.125, group="TEST",
    )
    assert (region.height, region.width) == (8, 8)
    regions = {"TESTCOMP": region}
    radar_dir = tmp_path / "radar"

    def _frame(ts: int) -> RadarFrame:
        return RadarFrame(
            timestamp=ts,
            regions={"TESTCOMP": np.full((8, 8), ts % 200, dtype=np.uint8)},
        )

    # A prior session leaves three frames on disk.
    prior = FrameStore(max_frames=3, cache_dir=tmp_path)
    for ts in (600, 1200, 1800):
        await prior.add_frame(_frame(ts))
    RadarFrameCache(tmp_path).save_metadata(regions, [600, 1200, 1800])
    del prior
    gc.collect()

    store = FrameStore(max_frames=3, cache_dir=tmp_path)
    await _restore_cached_frames(RadarFrameCache(tmp_path), store, regions)
    # Newer frames evict (and unlink) every restored one.
    for ts in (2400, 3000, 3600):
        await store.add_frame(_frame(ts))
    gc.collect()

    pinned = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(str(radar_dir)) and target.endswith("(deleted)"):
            pinned.append(target)
    assert pinned == []


# ---------------------------------------------------------------------------
# Alerts artifact: the pipeline writes <cache_dir>/alerts/current.db after
# every alerts store update instead of serializing the store into state.json.
# ---------------------------------------------------------------------------

def _alert_entry(source_id: str = "test-1"):
    """Minimal AlertEntry (polygon-less keeps the fixture shapely-free)."""
    from librewxr.data.alerts_store import AlertEntry

    return AlertEntry(
        source_id=source_id,
        event="Severe Thunderstorm Warning",
        description="A description",
        severity="Severe",
        effective="2026-05-07T00:00:00+00:00",
        expires="2099-05-07T06:00:00+00:00",
        area_desc="Test area",
        url=f"https://example.test/{source_id}",
        polygon=None,
    )


def _stub_pipeline_boot(monkeypatch, cache_dir: Path) -> None:
    """Disable / stub everything run_pipeline touches before its fetch loop.

    Same set of stubs ``test_pipeline_stores_use_frame_deletion_grace``
    uses: the source walkers, PP mask builders and network plumbing are all
    irrelevant here, but they run before the ``stores`` dict is built.
    """
    from librewxr.config import settings
    from librewxr import data_pipeline

    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "volatile_cache_dir", "")
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "storm_cells_enabled", False)
    monkeypatch.setattr(data_pipeline, "collect_nwp_contributions", lambda *a, **k: [])
    monkeypatch.setattr(
        data_pipeline, "collect_satellite_contributions", lambda *a, **k: [],
    )
    monkeypatch.setattr(
        data_pipeline, "collect_radar_coverage_metadata", lambda *a, **k: ({}, {}, []),
    )
    monkeypatch.setattr(data_pipeline, "build_coverage_masks", lambda *a, **k: None)
    monkeypatch.setattr(data_pipeline, "build_feather_masks", lambda *a, **k: None)
    monkeypatch.setattr(
        data_pipeline, "persist_masks_in_background", lambda *a, **k: None,
    )


def test_pipeline_stores_dict_omits_alerts_store(tmp_path, monkeypatch):
    """Alerts ride the SQLite artifact, never the state.json snapshot.

    Capture the mapping the pipeline hands to ``snapshot_state`` (reached
    through the ``on_cycle_complete`` callback it builds) and assert the
    ``alerts_store`` entry is gone while the other stores remain.  Alerts
    stay ENABLED: the producer store is still constructed and wired to the
    fetcher, it just must not enter the snapshot.
    """
    from librewxr.config import settings
    from librewxr import data_pipeline

    class _Stop(Exception):
        """Abort run_pipeline once the stores dict has been observed."""

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    _stub_pipeline_boot(monkeypatch, cache_dir)
    monkeypatch.setattr(settings, "alerts_enabled", True)

    captured: dict[str, object] = {}

    def _capturing_snapshot_state(stores):
        captured.update(stores)
        raise _Stop

    monkeypatch.setattr(data_pipeline, "snapshot_state", _capturing_snapshot_state)

    held: dict[str, object] = {}

    class _StubFetcher:
        def __init__(self, *args, **kwargs):
            held["on_cycle_complete"] = kwargs["on_cycle_complete"]

        async def start(self):
            raise _Stop

        async def stop(self):  # pragma: no cover - the boot was aborted
            raise AssertionError("stop must not run after an aborted start")

    monkeypatch.setattr(data_pipeline, "RadarFetcher", _StubFetcher)

    with pytest.raises(_Stop):
        asyncio.run(data_pipeline.run_pipeline())

    assert "on_cycle_complete" in held
    with pytest.raises(_Stop):
        asyncio.run(held["on_cycle_complete"]())

    assert "frame_store" in captured
    assert "alerts_store" not in captured


async def test_alerts_artifact_hook_writes_and_publishes(tmp_path):
    """The post-update hook writes the artifact and publishes it when configured."""
    from librewxr.data.alerts_artifact import ARTIFACT_NAME, AlertsArtifactStore
    from librewxr.data.alerts_store import AlertsStore
    from librewxr import data_pipeline

    store = AlertsStore()
    store.replace_all([_alert_entry("test-1")])

    published: list[int] = []

    class _Publisher:
        def publish_alerts(self):
            published.append(1)
            return True

    await data_pipeline._publish_alerts_artifact(store, tmp_path, _Publisher())

    assert (tmp_path / ARTIFACT_NAME).exists()
    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    assert artifact.count == store.count == 1
    assert artifact.last_updated == pytest.approx(store.last_updated)
    assert artifact.fetch_success is store.fetch_success is True
    assert [a.source_id for a in artifact.alerts()] == ["test-1"]
    assert len(published) == 1

    # No publisher (snapshot sync off): the artifact is still written, and
    # nothing is uploaded.
    await data_pipeline._publish_alerts_artifact(store, tmp_path, None)
    assert len(published) == 1


async def test_alerts_artifact_hook_survives_write_and_publish_failures(
    tmp_path, caplog,
):
    """A failed artifact write is logged and never escapes the hook.

    A regular file where the ``alerts/`` directory belongs makes every write
    fail deterministically (a chmod would be bypassed by a root test run), so
    the writer returns False instead of raising -- and a publisher that blows
    up must not escape either, or the alerts loop would die with it.
    """
    from librewxr.data.alerts_store import AlertsStore
    from librewxr import data_pipeline

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "alerts").write_text("not a directory")

    store = AlertsStore()
    store.replace_all([_alert_entry("degraded-1")])
    store.replace_all([], fetch_success=False)

    class _BrokenPublisher:
        def publish_alerts(self):
            raise OSError("store unavailable")

    with caplog.at_level(logging.WARNING):
        await data_pipeline._publish_alerts_artifact(
            store, cache_dir, _BrokenPublisher()
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Failed to write alerts artifact" in m for m in messages), messages
    assert any("Alerts artifact publish failed" in m for m in messages), messages


async def _drive_alerts_fetch(tmp_path, monkeypatch, retry_get, seed=None):
    """Run one real ``_fetch_once`` with the HTTP layer faked out.

    Returns ``(store, shared_cache_dir)``; the hook is the real pipeline
    hook, pointed at a cache dir separate from the fetcher's own CAP/zone
    cache so the artifact's location is observable.
    """
    from librewxr.data import alerts_fetcher as alerts_fetcher_module
    from librewxr.data.alerts_fetcher import WMOAlertsFetcher
    from librewxr.data.alerts_store import AlertsStore
    from librewxr import data_pipeline

    shared_cache = tmp_path / "cache"
    shared_cache.mkdir()
    store = AlertsStore()
    if seed:
        store.replace_all(seed)

    monkeypatch.setattr(alerts_fetcher_module, "retry_get", retry_get)

    fetcher = WMOAlertsFetcher(
        store=store,
        cache_dir=str(tmp_path / "alerts-cache"),
        on_store_update=lambda: data_pipeline._publish_alerts_artifact(
            store, shared_cache, None
        ),
    )

    async def _no_nws():
        return []

    fetcher._fetch_nws_alerts = _no_nws  # type: ignore[method-assign]
    try:
        await fetcher._fetch_once()
    finally:
        await fetcher.close()
    return store, shared_cache


async def test_alerts_fetcher_writes_artifact_on_degraded_update(
    tmp_path, monkeypatch,
):
    """A failed WMO fetch still lands in the artifact (never goes stale)."""
    from librewxr.data.alerts_artifact import ARTIFACT_NAME, AlertsArtifactStore

    async def _all_down(client, url, **kwargs):
        return None

    store, cache_dir = await _drive_alerts_fetch(tmp_path, monkeypatch, _all_down)

    assert store.fetch_success is False
    assert (cache_dir / ARTIFACT_NAME).exists()
    # The artifact belongs to the SHARED cache dir, not the fetcher's own.
    assert not (tmp_path / "alerts-cache" / ARTIFACT_NAME).exists()

    artifact = AlertsArtifactStore(cache_dir)
    assert artifact.maybe_reload() is True
    assert artifact.count == store.count == 0
    assert artifact.fetch_success is False
    assert artifact.last_updated == pytest.approx(store.last_updated)


async def test_alerts_fetcher_writes_artifact_on_successful_update(
    tmp_path, monkeypatch,
):
    """A healthy fetch replaces the artifact, including an emptied store."""
    from librewxr.data import alerts_fetcher as alerts_fetcher_module
    from librewxr.data.alerts_artifact import AlertsArtifactStore

    class _Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    # No operating sources and no hub items: a healthy cycle with no alerts,
    # which must still replace a previously populated store.
    responses = {
        alerts_fetcher_module._SOURCES_URL: _Response({"sources": []}),
        alerts_fetcher_module._WMO_ALL_URL: _Response({"items": []}),
    }

    async def _ok(client, url, **kwargs):
        return responses[url]

    # Seed a populated store so "count 0 afterwards" proves the artifact
    # tracks the update rather than the pre-fetch state.
    store, cache_dir = await _drive_alerts_fetch(
        tmp_path, monkeypatch, _ok, seed=[_alert_entry("stale-1")],
    )
    assert store.fetch_success is True
    assert store.count == 0

    artifact = AlertsArtifactStore(cache_dir)
    assert artifact.maybe_reload() is True
    assert artifact.count == 0
    assert artifact.fetch_success is True
    assert artifact.last_updated == pytest.approx(store.last_updated)
