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

import pytest

from librewxr.main import (
    _compute_cache_invalidation,
    _drop_absent_stores,
    _maybe_resurrect_precip_mask,
)

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
