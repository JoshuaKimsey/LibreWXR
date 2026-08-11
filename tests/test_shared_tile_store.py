# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the shared on-disk encoded-tile store.

Covers publish/get round-trips (identical bytes, miss → None), timestamp
sharding + per-timestamp invalidation (both across shards and within one
shared shard, e.g. ts and ts+64), clear (empty store, publish works after),
sweep_final_files (published files gone, live tmp kept, counters reset),
budget pruning (oldest-mtime-first with deterministic mtimes, counters
resynced from the scan), the unshardable-key fallback shard, and the
age-based stale-``*.tmp`` sweep in the constructor (old tmps removed,
fresh tmps left alone).
"""
from __future__ import annotations

import os
import time

import pytest

from librewxr.tiles.shared_tile_store import SharedTileStore

pytestmark = pytest.mark.store


def _store(tmp_path, max_mb: int = 1) -> SharedTileStore:
    return SharedTileStore(tmp_path, max_mb=max_mb)


def _shard_of(ts: int) -> str:
    return f"{ts % 64:02d}"


def _path_of(tmp_path, key: str):
    """On-disk path for a key, derived from the key's leading timestamp."""
    ts = int(key.split("-", 1)[0])
    return tmp_path / "tiles_shared" / _shard_of(ts) / f"{key}.tile"


def test_publish_get_roundtrip(tmp_path):
    """Publish → get returns identical bytes; missing key → None."""
    store = _store(tmp_path)
    ts = 1_712_000_000
    key = f"{ts}-v1-7-70-63-512"
    data = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
    store.publish(key, data)

    assert store.get(key) == data
    assert store.get(f"{ts}-v1-7-70-64-512") is None
    assert _path_of(tmp_path, key).is_file()
    # No stray tmp files left behind by the publish.
    assert not list((tmp_path / "tiles_shared").rglob("*.tmp"))


def test_invalidate_removes_only_that_timestamp(tmp_path):
    """Three timestamps across ≥2 shards; invalidating one leaves the rest."""
    ts_a = 1_712_000_000
    ts_b = 1_712_000_100
    ts_c = 1_712_000_200
    keys = {
        ts_a: f"{ts_a}-v1-7-70-63",
        ts_b: f"{ts_b}-v1-7-70-63",
        ts_c: f"{ts_c}-v1-7-70-63",
    }
    # Sanity: the test needs at least two distinct shards.
    assert len({_shard_of(t) for t in (ts_a, ts_b, ts_c)}) >= 2
    store = _store(tmp_path)
    for ts, key in keys.items():
        store.publish(key, f"data-{ts}".encode())

    store.invalidate_timestamp(ts_b)

    assert store.get(keys[ts_a]) == f"data-{ts_a}".encode()
    assert store.get(keys[ts_b]) is None
    assert store.get(keys[ts_c]) == f"data-{ts_c}".encode()
    # Nothing with the invalidated prefix remains in its shard dir.
    shard_dir = tmp_path / "tiles_shared" / _shard_of(ts_b)
    assert [
        p.name for p in shard_dir.iterdir() if p.name.startswith(f"{ts_b}-")
    ] == []


def test_invalidate_same_shard_leaves_other_timestamp(tmp_path):
    """ts and ts+64 share a shard; invalidating one keeps the other."""
    store = _store(tmp_path)
    ts_a = 1_712_000_000
    ts_b = ts_a + 64
    assert _shard_of(ts_a) == _shard_of(ts_b)
    store.publish(f"{ts_a}-v1-7-70-63", b"a" * 10)
    store.publish(f"{ts_b}-v1-7-70-63", b"b" * 20)

    store.invalidate_timestamp(ts_a)

    assert store.get(f"{ts_a}-v1-7-70-63") is None
    assert store.get(f"{ts_b}-v1-7-70-63") == b"b" * 20


def test_unshardable_key_uses_fallback_shard(tmp_path):
    """A key with no leading timestamp stores under shard '00' and is
    skipped by timestamp invalidation (documented fallback)."""
    store = _store(tmp_path)
    store.publish("noversion-key-1", b"x")

    assert store.get("noversion-key-1") == b"x"
    assert (tmp_path / "tiles_shared" / "00" / "noversion-key-1.tile").is_file()

    # Invalidation is keyed by the leading integer, so the unshardable
    # entry is invisible to it (only prune/clear reclaim it).
    store.invalidate_timestamp(0)
    assert store.get("noversion-key-1") == b"x"

    with pytest.raises(ValueError):
        SharedTileStore._ts_of("no-leading-int")
    with pytest.raises(ValueError):
        SharedTileStore._ts_of("-starts-with-dash")


def test_clear_empties_store_and_publish_works_after(tmp_path):
    """clear drops every entry, resets counters, and the store stays usable."""
    store = _store(tmp_path)
    store.publish("1712345600-v1-7-70-63", b"x" * 10)
    store.publish("1712345601-v1-7-70-63", b"y" * 20)

    store.clear()

    assert store.get("1712345600-v1-7-70-63") is None
    assert store.get("1712345601-v1-7-70-63") is None
    assert store.size == 0
    assert store.total_bytes == 0
    root = tmp_path / "tiles_shared"
    assert root.is_dir()
    assert list(root.iterdir()) == []

    store.publish("1712345602-v1-7-70-63", b"z")
    assert store.get("1712345602-v1-7-70-63") == b"z"
    assert store.size == 1
    assert store.total_bytes == 1


def test_prune_keeps_newest_within_budget(tmp_path):
    """Over-budget store evicts oldest-mtime-first and counters resync."""
    store = _store(tmp_path)
    store._max_bytes = 100
    keys = [f"{1_712_000_000 + i}-v1-7-70-{i}" for i in range(4)]
    for key in keys:
        store.publish(key, b"d" * 40)  # 40 B each → 160 B total > 100 B budget
    # Deterministic mtimes: keys[0] oldest, keys[3] newest.
    base = time.time() - 1000
    for i, key in enumerate(keys):
        os.utime(_path_of(tmp_path, key), (base + i * 100, base + i * 100))
    assert store.total_bytes == 160

    store.prune()

    # 160 > 100: unlink oldest until the remainder fits → newest 2 survive.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is None
    assert store.get(keys[2]) == b"d" * 40
    assert store.get(keys[3]) == b"d" * 40
    assert store.size == 2
    assert store.total_bytes == 80


def test_prune_noop_when_within_budget(tmp_path):
    """Within budget → everything survives and counters match the scan."""
    store = _store(tmp_path)
    store._max_bytes = 1 << 20
    store.publish("1712345600-v1-7-70-63", b"d" * 40)

    store.prune()

    assert store.get("1712345600-v1-7-70-63") == b"d" * 40
    assert store.size == 1
    assert store.total_bytes == 40


def test_constructor_sweeps_stale_tmp(tmp_path):
    """Old tmps (crash leftovers) are removed at construction; a fresh tmp
    from a live publisher survives the sweep."""
    root = tmp_path / "tiles_shared"
    shard_a = root / "00"
    shard_a.mkdir(parents=True)
    stale_a = shard_a / ".1712345600-v1-7-70-63.tile.tmp"
    stale_a.write_bytes(b"partial")
    shard_b = root / "07"
    shard_b.mkdir(parents=True)
    stale_b = shard_b / ".1712345607-v1-7-70-63.tile.tmp"
    stale_b.write_bytes(b"partial2")
    # Fresh tmp — a live publisher mid-publish, seconds old.
    fresh = shard_b / ".1712345608-v1-7-70-63.tile.tmp"
    fresh.write_bytes(b"in-flight")

    # Age the crash leftovers well past the 60 s threshold; leave the
    # fresh tmp at "now" (mtime survives the sweep intact).
    past = time.time() - 120
    os.utime(stale_a, (past, past))
    os.utime(stale_b, (past, past))

    store = _store(tmp_path)

    assert not stale_a.exists()
    assert not stale_b.exists()
    assert fresh.exists()
    assert list(root.rglob("*.tmp")) == [fresh]
    assert store.size == 0
    assert store.total_bytes == 0


def test_sweep_final_files_removes_tiles_keeps_tmps(tmp_path):
    """sweep_final_files drops every published .tile, keeps a live in-flight
    tmp, resets the counters, and the store stays usable."""
    store = _store(tmp_path)
    store.publish("1712345600-v1-7-70-63", b"x" * 10)
    store.publish("1712345601-v1-7-70-63", b"y" * 20)
    shard_dir = tmp_path / "tiles_shared" / _shard_of(1_712_000_000)
    in_flight = shard_dir / ".1712345600-v1-7-70-63.tile.tmp"
    in_flight.write_bytes(b"partial")
    assert list(shard_dir.iterdir())

    store.sweep_final_files()

    assert store.get("1712345600-v1-7-70-63") is None
    assert store.get("1712345601-v1-7-70-63") is None
    assert in_flight.exists()
    assert not list((tmp_path / "tiles_shared").rglob("*.tile"))
    assert store.size == 0
    assert store.total_bytes == 0

    store.publish("1712345602-v1-7-70-63", b"z")
    assert store.get("1712345602-v1-7-70-63") == b"z"
    assert store.size == 1
    assert store.total_bytes == 1
