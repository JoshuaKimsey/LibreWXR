# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the content-addressed CoordStore coordinate-array cache.

Covers the publish/open round trip (read-only memmaps), content addressing
(skip-if-exists, signature namespacing), corruption self-heal (garbage /
truncated files unlinked, transient OSErrors preserved), concurrent
publishers converging on one file, budget pruning (oldest-mtime-first, 90%
drain target, stale-tmp sweep), stats counters, and the one-shot manifest.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from librewxr.data.coord_store import (
    ALGO_VERSION,
    FORMAT_VERSION,
    KIND_FRACTIONAL,
    KIND_INDICES,
    KIND_LATLON,
    CoordStore,
)
from librewxr.data.regions import REGIONS

pytestmark = pytest.mark.store

# Real regions guaranteed present after the discovery walker merges the
# US radar package (see tests/test_coverage.py for the same pattern).
_ENABLED = ["USCOMP", "AKCOMP"]


def _data(seed: int = 7, tile_size: int = 16, dtype: np.dtype = np.int32) -> np.ndarray:
    """Deterministic (2, tile_size, tile_size) stacked array."""
    return np.full(
        (2, tile_size, tile_size), seed, dtype=dtype,
    ) + np.arange(2 * tile_size * tile_size, dtype=dtype).reshape(
        2, tile_size, tile_size,
    )


def _store(tmp_path, enabled: list[str] | None = None) -> CoordStore:
    return CoordStore(tmp_path, enabled if enabled is not None else _ENABLED)


def test_roundtrip_int32_and_float32_stacked(tmp_path):
    """Publish/open round trip for both dtypes; values identical, map RO."""
    for dtype, kind in [(np.int32, KIND_INDICES), (np.float32, KIND_LATLON)]:
        store = _store(tmp_path)
        data = _data(tile_size=16, dtype=dtype)
        assert store.publish(kind, "USCOMP", 3, 2, 1, 16, 0, data) is True
        out = store.open(kind, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype)
        assert out is not None
        np.testing.assert_array_equal(out, data)
        assert not out.flags.writeable


def test_open_miss_returns_none(tmp_path):
    """Missing entry -> None, and nothing is created on disk."""
    store = _store(tmp_path)
    data = _data()
    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is None
    assert not store.root.exists()


def test_publish_skip_if_exists_preserves_bytes(tmp_path):
    """Second publish of an existing key is a no-op keeping original bytes."""
    store = _store(tmp_path)
    data = _data(seed=7)
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    path = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0)
    original = path.read_bytes()

    altered = data.copy()
    altered[0, 0, 0] = 999
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, altered) is False
    assert path.read_bytes() == original

    out = store.open(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype)
    np.testing.assert_array_equal(out, data)


def test_concurrent_publish_same_key(tmp_path):
    """8 threads publishing one key -> exactly one valid file, right bytes."""
    store = _store(tmp_path)
    data = _data(seed=11)

    def _publish(_i: int) -> bool:
        return store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_publish, range(8)))

    entries = list(store.root.rglob("*.npy"))
    assert len(entries) == 1
    assert not list(store.root.rglob("*.tmp"))
    assert results.count(True) >= 1

    out = store.open(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype)
    assert out is not None
    np.testing.assert_array_equal(out, data)


def test_garbage_bytes_unlinked_on_open(tmp_path):
    """A non-npy file at the entry path -> open returns None AND unlinks."""
    store = _store(tmp_path)
    data = _data()
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    path = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0)
    path.write_bytes(b"\x00\x01\x02 this is not a numpy file at all")

    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is None
    assert not path.exists()


def test_truncated_entry_unlinked_on_open(tmp_path):
    """A truncated .npy (valid header, short data) -> None + unlinked."""
    store = _store(tmp_path)
    data = _data()
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    path = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0)
    # Keep the header (~128 B) but cut the data tail so the on-disk size
    # falls below the declared array extent.
    path.write_bytes(path.read_bytes()[: data.nbytes - 64])

    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is None
    assert not path.exists()


def test_other_oserror_does_not_unlink(tmp_path, monkeypatch):
    """A transient OSError (EIO) -> None returned, file left in place."""
    store = _store(tmp_path)
    data = _data()
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    path = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0)

    def _raise_eio(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("numpy.lib.format.open_memmap", _raise_eio)
    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is None
    assert path.is_file()


def test_signature_and_paths_depend_on_enabled_regions(tmp_path):
    """Different enabled-region sets -> different signature and entry path."""
    assert "USCOMP" in REGIONS and "AKCOMP" in REGIONS
    a = _store(tmp_path, ["USCOMP"])
    b = _store(tmp_path, ["USCOMP", "AKCOMP"])
    assert a.signature != b.signature
    assert len(a.signature) == 64
    assert set(a.signature) <= set("0123456789abcdef")
    assert a.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 256, 0) != b.entry_path(
        KIND_INDICES, "USCOMP", 3, 2, 1, 256, 0,
    )


def test_prune_evicts_oldest_and_sweeps_stale_tmp(tmp_path):
    """Budget enforcement is oldest-mtime-first to the 90% drain target;
    stale *.tmp swept, fresh *.tmp kept, counts reflect exactly what was
    removed."""
    store = _store(tmp_path)
    data = _data(tile_size=16, dtype=np.float64)
    keys = [
        (KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0),
        (KIND_INDICES, "USCOMP", 3, 2, 2, 16, 0),
        (KIND_INDICES, "USCOMP", 3, 3, 1, 16, 0),
    ]
    for key in keys:
        assert store.publish(*key, data) is True
    paths = [store.entry_path(*key) for key in keys]
    sizes = [p.stat().st_size for p in paths]

    # Control mtimes: paths[0] oldest, paths[2] newest.
    base = time.time() - 1000
    for i, p in enumerate(paths):
        os.utime(p, (base + i * 100, base + i * 100))

    # Plant a stale (>1h) and a fresh tmp under the root.
    stale = store.root / "stale.tmp"
    fresh = store.root / "fresh.tmp"
    stale.write_bytes(b"x" * 100)
    fresh.write_bytes(b"y" * 100)
    os.utime(stale, (time.time() - 4000, time.time() - 4000))
    os.utime(fresh, (time.time(), time.time()))

    manifest_size = (store.root / "manifest.json").stat().st_size
    # Entry sum as prune computes it (tmps excluded, manifest included).
    total = sum(sizes) + manifest_size
    # Budget that evicts exactly the two oldest entries: after removing
    # them the remainder fits 0.9*budget, before that it does not.
    after_two = sizes[2] + manifest_size
    budget = math.ceil(after_two / 0.9)
    assert total > budget  # prune must actually evict

    removed_bytes, removed_entries = store.prune(budget)

    assert removed_entries == 3  # two entries + stale tmp
    assert removed_bytes == sizes[0] + sizes[1] + 100
    assert not paths[0].exists()
    assert not paths[1].exists()
    assert paths[2].exists()
    assert not stale.exists()
    assert fresh.exists()
    assert (store.root / "manifest.json").exists()
    # 90% drain target respected for the remaining entries.
    remaining = paths[2].stat().st_size + manifest_size
    assert remaining <= 0.9 * budget


def test_prune_noop_when_within_budget(tmp_path):
    """Within budget -> (0, 0) and every entry survives."""
    store = _store(tmp_path)
    data = _data()
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    assert store.prune(1 << 40) == (0, 0)
    path = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0)
    assert path.exists()


def test_stats_counters(tmp_path):
    """hits/misses/publishes tracked per instance; entries/bytes scanned."""
    store = _store(tmp_path)
    s = store.stats()
    assert s["hits"] == 0 and s["misses"] == 0 and s["publishes"] == 0
    assert s["entries"] == 0 and s["bytes"] == 0

    data = _data()
    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is None
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True
    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is not None
    assert store.open(
        KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data.shape, data.dtype,
    ) is not None

    s = store.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["publishes"] == 1
    assert s["entries"] >= 1
    assert s["bytes"] >= data.nbytes
    # Scan result is cached and stable within the TTL.
    assert store.stats()["entries"] == s["entries"]


def test_manifest_written_once(tmp_path):
    """First publish writes the manifest; later publishes never rewrite it."""
    store = _store(tmp_path)
    data = _data()
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 16, 0, data) is True

    manifest_path = store.root / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["algo_version"] == ALGO_VERSION
    assert manifest["signature"] == store.signature
    assert manifest["pid"] == os.getpid()

    mtime_before = manifest_path.stat().st_mtime_ns
    content_before = manifest_path.read_bytes()
    time.sleep(0.01)
    # A different key (different tile_size) still publishes, but must not
    # rewrite the manifest.
    assert store.publish(KIND_INDICES, "USCOMP", 3, 2, 1, 17, 0, data) is True
    assert manifest_path.read_bytes() == content_before
    assert manifest_path.stat().st_mtime_ns == mtime_before


def test_entry_path_layout(tmp_path):
    """entry_path is stable, lowercase-hex, sharded by the first 2 chars."""
    store = _store(tmp_path)
    p = store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 256, 8)

    assert p.parent.parent == store.root
    assert store.root == tmp_path / "coord"
    assert p.suffix == ".npy"
    assert len(p.stem) == 40
    assert set(p.stem) <= set("0123456789abcdef")
    assert p.parent.name == p.stem[:2]
    assert p.name == f"{p.stem}.npy"

    # Stable across calls; sensitive to every key component.
    assert p == store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 1, 256, 0)
    assert p != store.entry_path(KIND_LATLON, "USCOMP", 3, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_FRACTIONAL, "USCOMP", 3, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, None, 3, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, "AKCOMP", 3, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, "USCOMP", 4, 2, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, "USCOMP", 3, 3, 1, 256, 8)
    assert p != store.entry_path(KIND_INDICES, "USCOMP", 3, 2, 2, 256, 8)
