# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for ``main._compute_cache_invalidation`` (diff-based tile-cache
invalidation for multi-mode render workers).

The helper is a pure function over two ``state.json`` payloads; these tests
exercise it directly with synthetic snapshots.  The integration behaviour
(the mtime poller wiring in ``_render_only_lifespan``) is covered by the
existing snapshot tests in ``tests/test_data_pipeline.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import numpy as np
import pytest

from librewxr.api import routes
from librewxr.colors import schemes
from librewxr.data.nowcast import NowcastFrame, NowcastStore
from librewxr.data.palette_fetcher import palette_artifact_path
from librewxr.main import (
    _apply_palette_from_payload,
    _compute_cache_invalidation,
    _drop_absent_stores,
    _maybe_resurrect_nowcast_store,
    _maybe_resurrect_precip_mask,
)
from librewxr.tiles.cache import TileCache

pytestmark = pytest.mark.store


def _payload(**stores) -> dict:
    return {"stores": stores}


def test_first_poll_clears_full_cache():
    # No previous payload to diff against — full clear (matches the
    # pre-diff unconditional cache.clear() on the first refresh).
    ts_set, full_clear = _compute_cache_invalidation(None, _payload())
    assert full_clear is True
    assert ts_set is None


def test_unchanged_signature_no_invalidations():
    # Identical snapshots, no nowcast entries: nothing to invalidate.
    payload = _payload(frame_store={"frame_versions": {"100": 1, "200": 1}})
    ts_set, full_clear = _compute_cache_invalidation(payload, payload)
    assert full_clear is False
    assert ts_set == set()


def test_radar_timestamp_evicted_is_invalidated():
    prev = _payload(frame_store={"frame_versions": {"100": 1, "200": 1}})
    cur = _payload(frame_store={"frame_versions": {"200": 1}})
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == {100}


def test_radar_version_bumped_by_merge_is_invalidated():
    prev = _payload(frame_store={"frame_versions": {"100": 1, "200": 1}})
    cur = _payload(frame_store={"frame_versions": {"100": 2, "200": 1}})
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == {100}


def test_nowcast_overlap_invalidates_all_timestamps():
    # The sliding forecast window persists 2 of 3 timestamps across the
    # cycle, but content is regenerated — every timestamp on either side
    # must be invalidated.
    prev = _payload(nowcast_store={
        "frames": [
            {"timestamp": "200"},
            {"timestamp": "300"},
            {"timestamp": "400"},
        ],
    })
    cur = _payload(nowcast_store={
        "frames": [
            {"timestamp": "300"},
            {"timestamp": "400"},
            {"timestamp": "500"},
        ],
    })
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == {200, 300, 400, 500}


def test_ifs_reference_time_change_triggers_full_clear():
    prev = _payload(ecmwf_grid={
        "reference_time": "2026-08-03T00:00:00Z",
        "timesteps": {"100": [1]},
    })
    cur = _payload(ecmwf_grid={
        "reference_time": "2026-08-03T06:00:00Z",
        "timesteps": {"100": [1]},
    })
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is True
    assert ts_set is None


def test_ifs_timestep_set_change_triggers_full_clear():
    # Same reference_time, but the hourly timestep set slid (one removed,
    # one added) — the same unix timestamps now sample different content.
    prev = _payload(ecmwf_grid={
        "reference_time": "2026-08-03T00:00:00Z",
        "timesteps": {"100": [1], "200": [2]},
    })
    cur = _payload(ecmwf_grid={
        "reference_time": "2026-08-03T00:00:00Z",
        "timesteps": {"200": [2], "300": [3]},
    })
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is True
    assert ts_set is None


def test_other_nwp_store_reference_time_change_triggers_full_clear():
    # Regional NWP overlays (HRRR / HRDPS / ICON-EU / ...) expose
    # reference_time too — a change there must clear the whole cache.
    prev = _payload(hrrr_grid={"reference_time": "00Z"})
    cur = _payload(hrrr_grid={"reference_time": "06Z"})
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is True
    assert ts_set is None


def test_store_without_reference_time_does_not_clear():
    # A store that exposes no reference_time contributes nothing to the
    # signature, so the targeted path is taken (here: a radar merge).
    prev = _payload(
        arbitrary_store={"something": "stable"},
        frame_store={"frame_versions": {"100": 1}},
    )
    cur = _payload(
        arbitrary_store={"something": "stable"},
        frame_store={"frame_versions": {"100": 2}},
    )
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == {100}


def test_combined_radar_eviction_merge_and_nowcast():
    prev = _payload(
        frame_store={"frame_versions": {"100": 1, "200": 1}},
        nowcast_store={"frames": [{"timestamp": "300"}, {"timestamp": "400"}]},
    )
    cur = _payload(
        frame_store={"frame_versions": {"200": 2}},
        nowcast_store={"frames": [{"timestamp": "400"}, {"timestamp": "500"}]},
    )
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    # 100 evicted, 200 bumped by merge, nowcast {300, 400, 500}.
    assert ts_set == {100, 200, 300, 400, 500}


def test_prev_payload_without_frame_versions_is_backward_compatible():
    # A state.json written by pre-fix code has no frame_versions entry.
    # Nothing in prev to diff against, so none of cur's timestamps are
    # invalidated (they're new — no cached geometry exists for them yet).
    prev = _payload(frame_store={"frames": [{"timestamp": 100}]})
    cur = _payload(
        frame_store={"frame_versions": {"100": 1, "200": 1}},
    )
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == set()


def test_prev_payload_without_frame_store_is_backward_compatible():
    # Even more conservative: prev has no frame_store entry at all.
    prev = _payload(nowcast_store={"frames": []})
    cur = _payload(
        frame_store={"frame_versions": {"100": 1, "200": 1}},
    )
    ts_set, full_clear = _compute_cache_invalidation(prev, cur)
    assert full_clear is False
    assert ts_set == set()


def test_drop_absent_stores_keeps_precip_mask_against_stale_snapshot():
    """A legacy first snapshot lacking precip_mask must not permanently
    null the mask store - it would be unrecoverable because apply_state
    skips None stores.  frame_store and precip_mask are both exempt.
    """
    frame_store = object()
    precip_mask = object()
    icon_eu = object()
    stores = {
        "frame_store": frame_store,
        "precip_mask": precip_mask,
        "icon_eu_grid": icon_eu,
    }
    # Stale snapshot: frame_store refreshed, precip_mask + icon_eu absent.
    refreshed = ["frame_store"]
    _drop_absent_stores(stores, refreshed)
    assert stores["frame_store"] is frame_store     # exempt (always shipped)
    assert stores["precip_mask"] is precip_mask     # exempt - THE FIX
    assert stores["icon_eu_grid"] is None          # genuinely-absent grid dropped


def test_maybe_resurrect_precip_mask_heals_nulled_store(tmp_path):
    """A worker whose precip mask was nulled at boot self-heals on the
    first poll whose snapshot carries a precip_mask entry.  Idempotent.
    """
    stores = {"precip_mask": None, "frame_store": object()}
    payload = {"stores": {"precip_mask": {"version": 1, "masks": {}}}}
    assert _maybe_resurrect_precip_mask(stores, payload, tmp_path) is True
    assert stores["precip_mask"] is not None
    # Idempotent: already live -> no-op.
    assert _maybe_resurrect_precip_mask(stores, payload, tmp_path) is False
    # The recovered store is re-readable: __setstate__ tolerates missing
    # mask files (empty masks -> conservative True on query, not a crash).
    assert stores["precip_mask"].has_precip_in_bbox(12345, (-10, -5, -8, 0))


def test_maybe_resurrect_precip_mask_noop_when_snapshot_lacks_it(tmp_path):
    """While the snapshot still has no precip_mask, the resurrection is a
    no-op and leaves the store None (conservative fallback to Tier 1)."""
    stores = {"precip_mask": None}
    payload = {"stores": {}}
    assert _maybe_resurrect_precip_mask(stores, payload, tmp_path) is False
    assert stores["precip_mask"] is None


def test_drop_absent_stores_keeps_nowcast_store_against_stale_snapshot():
    """A legacy first snapshot lacking nowcast_store must not permanently
    null the store - it would be unrecoverable because apply_state skips
    None stores.  frame_store, precip_mask, and nowcast_store are exempt.
    """
    frame_store = object()
    precip_mask = object()
    nowcast_store = object()
    icon_eu = object()
    stores = {
        "frame_store": frame_store,
        "precip_mask": precip_mask,
        "nowcast_store": nowcast_store,
        "icon_eu_grid": icon_eu,
    }
    # Stale snapshot: frame_store refreshed, nowcast_store + icon_eu absent.
    refreshed = ["frame_store"]
    _drop_absent_stores(stores, refreshed)
    assert stores["frame_store"] is frame_store      # exempt (always shipped)
    assert stores["precip_mask"] is precip_mask      # exempt
    assert stores["nowcast_store"] is nowcast_store  # exempt - THE FIX
    assert stores["icon_eu_grid"] is None            # genuinely-absent grid dropped


async def test_maybe_resurrect_nowcast_store_heals_nulled_store(tmp_path):
    """A worker whose nowcast store was nulled at boot self-heals on the
    first poll whose snapshot carries a nowcast_store entry.  Idempotent.
    """
    # Build a real snapshot from a persistent producer store, JSON
    # round-tripped exactly like the pipeline's dump_state.
    producer = NowcastStore(cache_dir=tmp_path)
    frame = NowcastFrame(
        timestamp=1700000600,
        blend_weight=0.6,
        regions={"R1": np.ones((4, 6), dtype=np.uint8)},
    )
    await producer.replace_all([frame])
    await producer.replace_flows({"R1": np.zeros((4, 6, 2), dtype=np.float32)})
    snapshot = json.loads(json.dumps(producer.__getstate__()))

    stores = {"nowcast_store": None, "frame_store": object()}
    payload = {"stores": {"nowcast_store": snapshot}}
    assert _maybe_resurrect_nowcast_store(stores, payload, tmp_path) is True
    assert stores["nowcast_store"] is not None
    # __setstate__ was applied: the frames re-open from the snapshot.
    timestamps = await stores["nowcast_store"].get_timestamps()
    assert timestamps == [1700000600]
    frame, weight = await stores["nowcast_store"].get_frame(1700000600)
    assert frame is not None
    assert weight == pytest.approx(0.6)
    np.testing.assert_array_equal(
        frame.regions["R1"], np.ones((4, 6), dtype=np.uint8),
    )
    # Idempotent: already live -> no-op.
    assert _maybe_resurrect_nowcast_store(stores, payload, tmp_path) is False


def test_maybe_resurrect_nowcast_store_noop_when_snapshot_lacks_it(tmp_path):
    """While the snapshot still has no nowcast_store, the resurrection is
    a no-op and leaves the store None (routes then serve an empty nowcast
    list, same as a live-but-empty store)."""
    stores = {"nowcast_store": None}
    payload = {"stores": {}}
    assert _maybe_resurrect_nowcast_store(stores, payload, tmp_path) is False
    assert stores["nowcast_store"] is None


# ---------------------------------------------------------------------------
# Per-tick external-palette apply
# ---------------------------------------------------------------------------

def _table_bytes(names=("Tick", "Palette")) -> bytes:
    header = ",".join(["dBZ / RGBA", *names])
    colors = tuple("#01020304" for _ in names)
    rows = [",".join([str(i - 32), *colors]) for i in range(256)]
    return "\n".join([header, *rows]).encode()


def _version(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:12]


def _write_artifact(cache_dir, raw: bytes) -> None:
    path = palette_artifact_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _palette_section(raw: bytes) -> dict:
    return {
        "version": _version(raw),
        "source": "path",
        "fetched_at": "2026-11-01T00:00:00+00:00",
        "last_status": "ok",
    }


@pytest.fixture
def packaged_palette():
    """Restore the packaged table after a test mutates the scheme globals."""
    yield
    schemes.apply_external_table(schemes.packaged_table_bytes())


def test_apply_palette_from_payload_applies_and_purges(tmp_path, monkeypatch, packaged_palette):
    raw = _table_bytes()
    _write_artifact(tmp_path, raw)
    old = schemes.current_version()

    cache = TileCache(max_mb=10)
    cache.put((100, f"p{old}"), b"stale-old-palette")
    cache.put((100, "pdeadbeef0000"), b"unrelated")
    calls: list[str] = []
    real_purge = cache.purge_token

    def spy(token: str) -> int:
        calls.append(token)
        return real_purge(token)

    monkeypatch.setattr(cache, "purge_token", spy)
    monkeypatch.setattr(routes, "tile_cache", cache)
    monkeypatch.setattr(routes, "palette_info", None)

    section = _palette_section(raw)
    assert _apply_palette_from_payload({"palette": section}, tmp_path) is True

    assert schemes.current_version() == _version(raw)
    assert routes.palette_info is section
    assert calls == [f"p{old}"]
    # Old-version entries swept; unrelated entries kept; accounting intact.
    assert cache.entries() == [((100, "pdeadbeef0000"), len(b"unrelated"))]
    assert cache.total_bytes == len(b"unrelated")
    # The timestamp index was kept consistent (no KeyError on sweep).
    cache.invalidate_timestamp(100)
    assert cache.size == 0


def test_apply_palette_missing_artifact_warns_and_keeps_current(
    tmp_path, monkeypatch, caplog, packaged_palette,
):
    before = schemes.current_version()
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "palette_info", None)

    with caplog.at_level(logging.WARNING, logger="librewxr.main"):
        applied = _apply_palette_from_payload(
            {"palette": {"version": "0123456789ab"}}, tmp_path,
        )

    assert applied is False
    assert schemes.current_version() == before
    assert routes.palette_info is None
    assert any("no artifact" in record.message for record in caplog.records)


def test_apply_palette_advertised_equals_applied_is_noop(tmp_path, monkeypatch, packaged_palette):
    # Artifact exists with different bytes: if the helper applied it, the
    # version would change.  Advertised == applied must short-circuit, but
    # still publish the snapshot's palette section for /health (builtin).
    _write_artifact(tmp_path, _table_bytes(names=("Ignored", "Table")))
    before = schemes.current_version()
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "palette_info", "sentinel")

    section = {"version": before}
    assert _apply_palette_from_payload({"palette": section}, tmp_path) is False
    assert schemes.current_version() == before
    assert routes.palette_info is section


def test_apply_palette_artifact_hash_mismatch_is_rejected(
    tmp_path, monkeypatch, caplog, packaged_palette,
):
    # Crash window: the pipeline wrote a new artifact but died before
    # state.json.  Re-applying (and re-purging current-version keys) every
    # poll tick would thrash the cache, so the helper rejects the artifact.
    raw = _table_bytes()
    _write_artifact(tmp_path, raw)
    advertised = "0123456789ab"
    assert _version(raw) != advertised
    before = schemes.current_version()

    cache = TileCache(max_mb=10)
    calls: list[str] = []

    def spy(token: str) -> int:
        calls.append(token)
        return 0

    monkeypatch.setattr(cache, "purge_token", spy)
    monkeypatch.setattr(routes, "tile_cache", cache)
    monkeypatch.setattr(routes, "palette_info", "sentinel")

    with caplog.at_level(logging.WARNING, logger="librewxr.main"):
        applied = _apply_palette_from_payload(
            {"palette": {"version": advertised}}, tmp_path,
        )

    assert applied is False
    assert schemes.current_version() == before
    assert calls == []
    assert routes.palette_info == "sentinel"
    assert any(
        advertised in record.message and _version(raw) in record.message
        for record in caplog.records
    )


def test_apply_palette_absent_section_is_noop(tmp_path, monkeypatch, packaged_palette):
    _write_artifact(tmp_path, _table_bytes(names=("Unused", "Table")))
    before = schemes.current_version()
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "palette_info", None)

    assert _apply_palette_from_payload({}, tmp_path) is False
    assert _apply_palette_from_payload(None, tmp_path) is False
    assert _apply_palette_from_payload({"palette": {}}, tmp_path) is False
    assert schemes.current_version() == before
    assert routes.palette_info is None


def test_apply_palette_invalid_artifact_keeps_current(tmp_path, monkeypatch, packaged_palette):
    _write_artifact(tmp_path, b"not a color table")
    before = schemes.current_version()
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "palette_info", None)

    assert _apply_palette_from_payload(
        {"palette": {"version": "0123456789ab"}}, tmp_path,
    ) is False
    assert schemes.current_version() == before
    assert routes.palette_info is None


# ---------------------------------------------------------------------------
# Per-tick alerts artifact refresh
# ---------------------------------------------------------------------------

#: Every ``routes`` attribute the render lifespan writes; snapshotted so an
#: in-test lifespan run cannot leak its stores (or its teardown Nones) into
#: the rest of the suite.
_LIFESPAN_ROUTES_ATTRS = (
    "frame_store", "tile_cache", "shared_tile_store", "nwp_grids",
    "ecmwf_grid", "nwp_chain", "satellite_grids", "nowcast_store",
    "storm_cell_store", "lightning_store", "lightning_enabled",
    "tile_request_tracker", "start_time", "enabled_regions", "radar_cache",
    "radar_fetcher", "alerts_store", "alerts_fetcher", "alerts_enabled",
    "precip_mask", "memory_monitor", "ws_hub", "palette_info",
    "present_executor", "io_executor",
)


@pytest.fixture
def restore_routes_state(monkeypatch):
    for name in _LIFESPAN_ROUTES_ATTRS:
        monkeypatch.setattr(routes, name, getattr(routes, name))
    yield


async def test_poll_tick_reloads_alerts_artifact_without_state_write(
    tmp_path, monkeypatch, restore_routes_state,
):
    """An alerts artifact rewrite is picked up with state.json untouched.

    The pipeline rewrites ``<cache_dir>/alerts/current.db`` on the alerts
    fetch cadence and never touches state.json for it, so the poller checks
    the artifact every tick -- before, and independently of, the snapshot
    mtime early-out.  Observable through the ``routes.alerts_store`` the
    lifespan wires (its scalars only leave the cold state via reload).
    """
    from librewxr.config import settings
    from librewxr.data.alerts_artifact import write_alerts_artifacts
    from librewxr.data.alerts_store import AlertEntry, AlertsStore
    from librewxr.data.master_state import STATE_FILENAME, dump_state
    from librewxr.data.store import FrameStore, RadarFrame
    from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
    from shapely.geometry import Polygon

    from librewxr import main as main_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Pipeline side: one snapshot, written once and never again.
    frame_store = FrameStore(max_frames=4, cache_dir=cache_dir)
    grid = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    await frame_store.add_frame(RadarFrame(timestamp=42, regions={"USCOMP": grid}))
    dump_state({"frame_store": frame_store}, cache_dir)

    monkeypatch.setattr(settings, "render_only", True)
    monkeypatch.setattr(settings, "cache_dir", str(cache_dir))
    # Optional stores off: this test only cares about the alerts tick.
    monkeypatch.setattr(settings, "satellite_enabled", False)
    monkeypatch.setattr(settings, "nowcast_enabled", False)
    monkeypatch.setattr(settings, "arrow_flow_enabled", False)
    monkeypatch.setattr(settings, "storm_cells_enabled", False)
    monkeypatch.setattr(settings, "lightning_enabled", False)
    monkeypatch.setattr(settings, "websocket_enabled", False)
    monkeypatch.setattr(settings, "alerts_enabled", True)
    monkeypatch.setattr(settings, "state_wait_timeout", 5.0)
    monkeypatch.setattr(settings, "state_poll_interval", 0.05)
    # Keeps the coord warm pass negligible; the jitter constant is
    # module-level so the test can neutralize the 0-15 s stall.
    monkeypatch.setattr(settings, "warm_coord_zoom", 2)
    monkeypatch.setattr(main_module, "_WARM_JITTER_MAX_S", 0.0)

    class _StubApp:
        pass

    try:
        async with main_module._render_only_lifespan(_StubApp()):
            store = routes.alerts_store
            assert store is not None
            # Cold: the pipeline has not written an artifact yet.
            assert store.last_updated == 0.0
            state_path = cache_dir / STATE_FILENAME
            state_mtime = state_path.stat().st_mtime_ns

            # Pipeline side: write the artifact only -- state.json untouched.
            writer = AlertsStore()
            writer.replace_all([
                AlertEntry(
                    source_id="test-1",
                    event="High Wind Warning",
                    description="Strong winds expected",
                    severity="Severe",
                    effective="2026-05-07T01:00:00-08:00",
                    expires="2099-05-07T18:00:00-08:00",
                    area_desc="San Francisco, CA",
                    url="https://example.com/alert2",
                    polygon=Polygon([
                        (-122.5, 37.0), (-121.5, 37.0), (-121.5, 38.0),
                        (-122.5, 38.0), (-122.5, 37.0),
                    ]),
                ),
            ])
            assert write_alerts_artifacts(writer, cache_dir) is True
            assert state_path.stat().st_mtime_ns == state_mtime

            # Next tick (0.05 s, jittered 0.5-1.5x) reloads it.
            deadline = asyncio.get_running_loop().time() + 5.0
            while store.last_updated == 0.0:
                assert asyncio.get_running_loop().time() < deadline, (
                    "poller never reloaded the alerts artifact"
                )
                await asyncio.sleep(0.02)

            assert store.last_updated == pytest.approx(writer.last_updated)
            assert store.count == 1
            assert store.fetch_success is True
            # And the refresh was driven by the artifact, not the snapshot.
            assert state_path.stat().st_mtime_ns == state_mtime
    finally:
        frame_store.cleanup()
