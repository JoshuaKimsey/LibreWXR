# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the artifact-backed LightningStore (Phase 1, pipeline side)."""

import logging
import os
import time

import numpy as np
import pytest

from librewxr.data import lightning_store as lightning_store_module
from librewxr.data.lightning_store import LightningStore, _POINT_DTYPE

pytestmark = pytest.mark.lightning

_LOGGER = "librewxr.data.lightning_store"


def _make_points(rows) -> np.ndarray:
    """Build a structured point array from ``(time_s, lat, lon, energy, sat)``."""
    arr = np.zeros(len(rows), dtype=_POINT_DTYPE)
    for i, (t, lat, lon, energy, sat) in enumerate(rows):
        arr[i]["time_s"] = t
        arr[i]["lat"] = lat
        arr[i]["lon"] = lon
        arr[i]["energy"] = energy
        arr[i]["satellite"] = sat
    return arr


# ---------------------------------------------------------------------------
# Artifact round-trip + versioning
# ---------------------------------------------------------------------------

async def test_round_trip(tmp_path):
    meta = {"units": "J", "source": "glm"}
    points = _make_points([
        (1000, 10.0, -80.0, 1.0, 18),
        (1010, 11.0, -81.0, 2.0, 19),
        (1020, 12.0, -82.0, 3.0, 18),
    ])
    writer = LightningStore(tmp_path)
    await writer.replace_points(
        points, last_seen_s=1020, max_age_s=1800, meta=meta,
    )
    assert await writer.save_snapshot() is True

    reader = LightningStore(tmp_path)
    assert await reader.maybe_reload() is True
    assert np.array_equal(reader.points, writer.points)
    assert reader.meta == writer.meta
    assert reader.meta["units"] == "J"
    assert reader.meta["source"] == "glm"
    assert reader.version == writer.version
    assert reader.total_count == 3


async def test_version_monotonic(tmp_path):
    store = LightningStore(tmp_path)
    await store.replace_points(
        _make_points([(1000, 1.0, 1.0, 1.0, 18)]),
        last_seen_s=1000, max_age_s=1800,
    )
    first = store.version
    await store.replace_points(
        _make_points([(1010, 1.0, 1.0, 1.0, 18)]),
        last_seen_s=1010, max_age_s=1800,
    )
    assert store.version > first


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

async def test_max_age_trim(tmp_path):
    store = LightningStore(tmp_path)
    points = _make_points([
        (0, 1.0, 1.0, 1.0, 18),
        (1800, 1.0, 1.0, 1.0, 18),
        (5400, 1.0, 1.0, 1.0, 18),
        (5401, 1.0, 1.0, 1.0, 18),
        (7200, 1.0, 1.0, 1.0, 18),
    ])
    await store.replace_points(points, last_seen_s=7200, max_age_s=1800)
    # window_end = 7200, cutoff = 7200 - 1800 = 5400 -> keep time_s >= 5400.
    assert list(store.points["time_s"]) == [5400, 5401, 7200]
    assert store.window_start_s == 5400
    assert store.window_end_s == 7200


async def test_ceiling_warn_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lightning_store_module, "MAX_POINTS", 4)
    store = LightningStore(tmp_path)

    def _overflow() -> np.ndarray:
        return _make_points([(i, 1.0, 1.0, 1.0, 18) for i in range(10)])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await store.replace_points(_overflow(), last_seen_s=9, max_age_s=10**9)
        assert store.total_count == 4
        # Sorted ascending, oldest dropped -> keep times 6,7,8,9.
        assert list(store.points["time_s"]) == [6, 7, 8, 9]
        assert store.ceiling_trimmed_total == 6
        first_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == _LOGGER
        ]
        assert len(first_warnings) == 1

        caplog.clear()
        await store.replace_points(_overflow(), last_seen_s=9, max_age_s=10**9)
        second_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == _LOGGER
        ]
        assert second_warnings == []

    assert store.ceiling_trimmed_total == 12


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

async def test_points_in_filter(tmp_path):
    store = LightningStore(tmp_path)
    points = _make_points([
        (100, 10.0, -80.0, 1.0, 18),   # in box
        (200, 11.5, -79.0, 2.0, 19),   # in box
        (300, 20.0, -70.0, 3.0, 18),   # outside box
    ])
    await store.replace_points(points, last_seen_s=300, max_age_s=1800)

    sel = store.points_in(10.0, 12.0, -81.0, -78.0)
    assert list(sel["time_s"]) == [100, 200]

    sel_since = store.points_in(10.0, 12.0, -81.0, -78.0, since_s=150)
    assert list(sel_since["time_s"]) == [200]

    # Inclusive bounds on all four edges.
    sel_edge = store.points_in(10.0, 10.0, -80.0, -80.0)
    assert list(sel_edge["time_s"]) == [100]

    empty = LightningStore(tmp_path / "empty")
    out = empty.points_in(-90.0, 90.0, -180.0, 180.0)
    assert out.shape[0] == 0
    assert out.dtype == _POINT_DTYPE


# ---------------------------------------------------------------------------
# Reader reload
# ---------------------------------------------------------------------------

async def test_maybe_reload_absent(tmp_path):
    store = LightningStore(tmp_path)
    assert await store.maybe_reload() is False
    assert store.total_count == 0
    assert store.window_start_s is None
    assert store.window_end_s is None


async def test_maybe_reload_appears_later(tmp_path):
    reader = LightningStore(tmp_path)
    assert await reader.maybe_reload() is False
    assert reader.total_count == 0

    writer = LightningStore(tmp_path)
    await writer.replace_points(
        _make_points([(5000, 5.0, -75.0, 4.0, 19)]),
        last_seen_s=5000, max_age_s=1800,
    )
    assert await writer.save_snapshot() is True
    assert await reader.maybe_reload() is True
    assert reader.total_count == 1
    assert list(reader.points["time_s"]) == [5000]


async def test_maybe_reload_corrupt_keeps_old(tmp_path, caplog):
    writer = LightningStore(tmp_path)
    await writer.replace_points(
        _make_points([(5000, 5.0, -75.0, 4.0, 19)]),
        last_seen_s=5000, max_age_s=1800,
    )
    assert await writer.save_snapshot() is True

    reader = LightningStore(tmp_path)
    assert await reader.maybe_reload() is True
    good = reader.points.copy()

    artifact = tmp_path / "lightning" / "current.npz"
    artifact.write_bytes(b"not a valid npz")
    future = time.time() + 100.0
    os.utime(artifact, (future, future))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await reader.maybe_reload() is False
    assert np.array_equal(reader.points, good)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Write failure handling
# ---------------------------------------------------------------------------

async def test_save_snapshot_oserror(tmp_path, monkeypatch, caplog):
    store = LightningStore(tmp_path)
    await store.replace_points(
        _make_points([(5000, 5.0, -75.0, 4.0, 19)]),
        last_seen_s=5000, max_age_s=1800,
    )

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(lightning_store_module.os, "replace", _boom)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = await store.save_snapshot()
    assert result is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_save_snapshot_without_cache_dir_is_false():
    store = LightningStore(None)
    assert await store.save_snapshot() is False
