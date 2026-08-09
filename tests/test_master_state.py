# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for ``master_state.dump_state`` / ``load_state`` / ``apply_state``.

These cover the multi-worker hand-off: the data pipeline writes a single
``state.json`` snapshot, a render-only worker reads it, and every store
that opts into the snapshot has its ``__setstate__`` called in place.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from librewxr.data.master_state import (
    STATE_FILENAME,
    STATE_VERSION,
    apply_state,
    dump_state,
    load_state,
    state_mtime,
)
from librewxr.data.store import FrameStore, RadarFrame

pytestmark = pytest.mark.store


# ──────────────────────────────────────────────────────────────────────────
# dump_state / load_state basics
# ──────────────────────────────────────────────────────────────────────────


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    assert load_state(tmp_path) is None
    assert state_mtime(tmp_path) is None


def test_dump_state_writes_json_with_version_and_timestamp(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    before = int(time.time())
    path = dump_state({}, cache)
    after = int(time.time())

    assert path == cache / STATE_FILENAME
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload["version"] == STATE_VERSION
    assert before <= payload["written_at"] <= after
    assert payload["stores"] == {}


def test_dump_state_skips_none_stores(tmp_path: Path) -> None:
    dump_state({"a": None, "b": None}, tmp_path)
    payload = load_state(tmp_path)
    assert payload is not None
    assert payload["stores"] == {}


def test_dump_state_skips_objects_without_getstate(tmp_path: Path) -> None:
    class NoState:  # noqa: D401
        pass

    dump_state({"weird": NoState()}, tmp_path)
    payload = load_state(tmp_path)
    assert payload is not None
    assert payload["stores"] == {}


def test_dump_state_atomic_replaces_existing(tmp_path: Path) -> None:
    """Writing twice must leave only the new content — no partial files."""
    dump_state({}, tmp_path)
    dump_state({}, tmp_path)

    files = sorted(p.name for p in tmp_path.iterdir())
    # Only the final state.json — never a stray .tmp afterwards.
    assert files == [STATE_FILENAME]


# ──────────────────────────────────────────────────────────────────────────
# Round-trip with a real store (FrameStore)
# ──────────────────────────────────────────────────────────────────────────


class TestRoundTripWithFrameStore:
    @pytest.mark.asyncio
    async def test_pipeline_to_tile_server_handoff(self, tmp_path: Path) -> None:
        """Simulate the multi-worker hand-off end to end."""
        cache = tmp_path / "cache"

        # Pipeline-side: produce data and dump state.
        producer = FrameStore(max_frames=4, cache_dir=cache)
        arr = np.full((8, 8), 99, dtype=np.uint8)
        await producer.add_frame(RadarFrame(timestamp=1700000000, regions={"R": arr}))
        dump_state({"frame_store": producer}, cache)

        # Tile-server-side: load state into a fresh store and verify data.
        consumer = FrameStore(max_frames=4)
        payload = load_state(cache)
        assert payload is not None
        refreshed = apply_state(payload, {"frame_store": consumer})
        assert refreshed == ["frame_store"]

        timestamps = await consumer.get_timestamps()
        assert timestamps == [1700000000]
        frame = await consumer.get_frame(1700000000)
        assert frame is not None
        np.testing.assert_array_equal(frame.regions["R"], arr)

    @pytest.mark.asyncio
    async def test_apply_state_in_place_updates(self, tmp_path: Path) -> None:
        """A second snapshot updates the consumer in place."""
        cache = tmp_path / "cache"

        producer = FrameStore(max_frames=4, cache_dir=cache)
        await producer.add_frame(
            RadarFrame(timestamp=1700000000, regions={"R": np.zeros((4, 4), np.uint8)}),
        )
        dump_state({"frame_store": producer}, cache)

        # First load
        consumer = FrameStore()
        apply_state(load_state(cache), {"frame_store": consumer})
        assert await consumer.get_timestamps() == [1700000000]

        # Pipeline adds a new frame, dumps again.
        await producer.add_frame(
            RadarFrame(timestamp=1700000600, regions={"R": np.zeros((4, 4), np.uint8)}),
        )
        dump_state({"frame_store": producer}, cache)

        # Same consumer object — state must update in place.
        apply_state(load_state(cache), {"frame_store": consumer})
        assert await consumer.get_timestamps() == [1700000000, 1700000600]


# ──────────────────────────────────────────────────────────────────────────
# Robustness: partial / mismatched store dicts
# ──────────────────────────────────────────────────────────────────────────


class TestApplyStateRobustness:
    def test_missing_consumer_store_silently_skipped(self, tmp_path: Path) -> None:
        """A snapshot that mentions stores the consumer doesn't have must not crash."""
        cache = tmp_path / "cache"
        producer = FrameStore(cache_dir=cache)
        dump_state({"frame_store": producer}, cache)
        payload = load_state(cache)

        # Consumer enabled a different subset — only nowcast_store, no frame_store.
        refreshed = apply_state(payload, {"nowcast_store": None})
        assert refreshed == []

    def test_consumer_store_missing_from_snapshot_silently_skipped(
        self, tmp_path: Path,
    ) -> None:
        """A consumer-side store that the producer didn't snapshot is left alone."""
        cache = tmp_path / "cache"
        producer = FrameStore(cache_dir=cache)
        dump_state({"frame_store": producer}, cache)
        payload = load_state(cache)

        # Consumer also has 'extra_store' (not in snapshot).
        consumer = FrameStore()
        refreshed = apply_state(
            payload,
            {"frame_store": consumer, "extra_store": FrameStore()},
        )
        assert refreshed == ["frame_store"]


# ──────────────────────────────────────────────────────────────────────────
# dump_state with an empty vs populated NowcastStore
# ──────────────────────────────────────────────────────────────────────────


class TestDumpStateNowcastStore:
    def test_dump_state_omits_empty_nowcast_store(self, tmp_path: Path) -> None:
        """An all-empty NowcastStore serializes to ``None`` — the entry
        must be omitted from the snapshot so serving render workers keep
        their current frames until the first real generation lands."""
        from librewxr.data.nowcast import NowcastStore

        cache = tmp_path / "cache"
        store = NowcastStore(cache_dir=cache)
        try:
            dump_state({"nowcast_store": store}, cache)
        finally:
            store.cleanup()
        payload = load_state(cache)
        assert payload is not None
        assert "nowcast_store" not in payload["stores"]

    @pytest.mark.asyncio
    async def test_dump_state_includes_populated_nowcast_store(
        self, tmp_path: Path,
    ) -> None:
        from librewxr.data.nowcast import NowcastFrame, NowcastStore

        cache = tmp_path / "cache"
        store = NowcastStore(cache_dir=cache)
        try:
            await store.replace_all([
                NowcastFrame(
                    timestamp=1700000600,
                    blend_weight=0.6,
                    regions={"R1": np.zeros((4, 6), dtype=np.uint8)},
                ),
            ])
            dump_state({"nowcast_store": store}, cache)
        finally:
            store.cleanup()
        payload = load_state(cache)
        assert payload is not None
        snapshot = payload["stores"].get("nowcast_store")
        assert snapshot is not None
        assert [int(f["timestamp"]) for f in snapshot["frames"]] == [1700000600]


# ──────────────────────────────────────────────────────────────────────────
# Mtime polling helper
# ──────────────────────────────────────────────────────────────────────────


def test_state_mtime_advances_after_each_dump(tmp_path: Path) -> None:
    dump_state({}, tmp_path)
    first = state_mtime(tmp_path)
    assert first is not None

    # mtime resolution on most Linux filesystems is sub-second; force a tiny
    # bump to guarantee the next mtime is strictly greater.
    time.sleep(0.01)
    dump_state({}, tmp_path)
    second = state_mtime(tmp_path)
    assert second is not None
    assert second >= first


# ──────────────────────────────────────────────────────────────────────────
# Incremental reload: per-store payload diffing
# ──────────────────────────────────────────────────────────────────────────


class _FakeStore:
    """Minimal store that counts how often ``__setstate__`` runs."""

    def __init__(self) -> None:
        self.loads = 0

    def __setstate__(self, state: dict) -> None:
        self.loads += 1
        self.state = state


def _payload(**store_states) -> dict:
    return {"version": STATE_VERSION, "written_at": 0, "stores": dict(store_states)}


def test_apply_state_loads_every_store_on_first_apply() -> None:
    """Boot path: no previous payload -> every store's __setstate__ runs."""
    stores = {
        "frame_store": _FakeStore(),
        "hrrr_grid": _FakeStore(),
    }
    payload = _payload(frame_store={"reference_time": 1}, hrrr_grid={"run_ts": 2})

    refreshed = apply_state(payload, stores)

    assert sorted(refreshed) == ["frame_store", "hrrr_grid"]
    assert stores["frame_store"].loads == 1
    assert stores["hrrr_grid"].loads == 1


def test_apply_state_skips_unchanged_stores() -> None:
    """A second payload where only one store changed reloads only that store."""
    stores = {
        "frame_store": _FakeStore(),
        "hrrr_grid": _FakeStore(),
    }
    payload1 = _payload(frame_store={"reference_time": 1}, hrrr_grid={"run_ts": 2})
    apply_state(payload1, stores)

    # Next cycle: frame_store moved, hrrr_grid identical.
    payload2 = _payload(frame_store={"reference_time": 2}, hrrr_grid={"run_ts": 2})
    refreshed = apply_state(payload2, stores, prev_payload=payload1)

    assert refreshed == ["frame_store"]
    assert stores["frame_store"].loads == 2
    assert stores["hrrr_grid"].loads == 1


def test_apply_state_without_prev_payload_reloads_everything() -> None:
    """Omitting prev_payload (no apply history) always reloads every store."""
    stores = {"frame_store": _FakeStore()}
    payload = _payload(frame_store={"reference_time": 1})

    apply_state(payload, stores)
    refreshed = apply_state(payload, stores)

    assert refreshed == ["frame_store"]
    assert stores["frame_store"].loads == 2


def test_apply_state_reloads_new_store_not_in_prev() -> None:
    """A store absent from the previous payload must load even if another
    store's sub-dict is identical (e.g. a store enabled mid-run)."""
    stores = {
        "frame_store": _FakeStore(),
        "nowcast_store": _FakeStore(),
    }
    payload1 = _payload(frame_store={"reference_time": 1})
    apply_state(payload1, stores)

    payload2 = _payload(
        frame_store={"reference_time": 1},
        nowcast_store={"frames": [{"timestamp": 100}]},
    )
    refreshed = apply_state(payload2, stores, prev_payload=payload1)

    assert refreshed == ["nowcast_store"]
    assert stores["frame_store"].loads == 1
    assert stores["nowcast_store"].loads == 1
