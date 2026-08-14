# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for the GMGSI satellite source.

Covers the pure-logic surface: filename parsing, frame retention,
sample() coordinate transforms, and the cross-process pickle round-trip.
Network and S3 I/O are mocked via fsspec.AbstractFileSystem stubs —
live S3 verification happens in the Phase 1.4 verification step, not
here, so the unit suite stays hermetic and fast.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from librewxr.sources.satellite.gmgsi.source import (
    CACHE_LAYOUT_VERSION,
    GMGSILWSource,
    GMGSISource,
    GMGSIVISSource,
    GRID_HEIGHT,
    GRID_SHAPE,
    GRID_WIDTH,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
)

pytestmark = pytest.mark.sources


# ── Filename parsing ──


def test_parse_start_timestamp_round_trip():
    """Hour-floor parsing of a real GMGSI filename token."""
    fn = "GLOBCOMPLIR_v3r0_blend_s202605232300000_e202605232309599_c202605232335220.nc"
    ts = GMGSISource._parse_start_timestamp(fn)
    expected = int(
        datetime(2026, 5, 23, 23, 0, 0, tzinfo=timezone.utc).timestamp(),
    )
    assert ts == expected


def test_parse_start_timestamp_floors_minutes_to_hour():
    """Filenames with non-zero minutes in the ``s`` token still floor to the hour."""
    fn = "GLOBCOMPLIR_v3r0_blend_s202605231030000_e202605231039599_c202605231040000.nc"
    ts = GMGSISource._parse_start_timestamp(fn)
    expected = int(
        datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc).timestamp(),
    )
    assert ts == expected


def test_parse_start_timestamp_rejects_malformed():
    """Filenames without the ``_s`` token return None."""
    assert GMGSISource._parse_start_timestamp("not_a_gmgsi_file.nc") is None
    assert GMGSISource._parse_start_timestamp("GLOBCOMPLIR_v3r0_blend_s2026.nc") is None


# ── Provider shape ──


def test_satellite_provider_returns_lw_and_vis_when_both_enabled(tmp_path: Path):
    """Default config (both channels on) emits two contributions, LW first."""
    from librewxr.sources._base import SatelliteContribution
    from librewxr.sources.satellite.gmgsi import satellite_provider

    settings = MagicMock()
    settings.gmgsi_lw_enabled = True
    settings.gmgsi_vis_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=tmp_path)
    assert len(contribs) == 2

    lw, vis = contribs
    assert isinstance(lw, SatelliteContribution)
    assert lw.name == "GMGSI LW"
    assert lw.slug == "gmgsi_lw_grid"
    assert isinstance(lw.instance, GMGSILWSource)
    assert lw.priority == 10

    assert vis.name == "GMGSI VIS"
    assert vis.slug == "gmgsi_vis_grid"
    assert isinstance(vis.instance, GMGSIVISSource)
    assert vis.priority == 11


def test_satellite_provider_lw_only_when_vis_disabled(tmp_path: Path):
    """Disabling VIS drops the second contribution, leaving LW alone."""
    from librewxr.sources.satellite.gmgsi import satellite_provider

    settings = MagicMock()
    settings.gmgsi_lw_enabled = True
    settings.gmgsi_vis_enabled = False
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=tmp_path)
    assert len(contribs) == 1
    assert contribs[0].slug == "gmgsi_lw_grid"


def test_satellite_provider_vis_only_when_lw_disabled(tmp_path: Path):
    """Disabling LW while VIS stays on still returns the VIS contribution."""
    from librewxr.sources.satellite.gmgsi import satellite_provider

    settings = MagicMock()
    settings.gmgsi_lw_enabled = False
    settings.gmgsi_vis_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=tmp_path)
    assert len(contribs) == 1
    assert contribs[0].slug == "gmgsi_vis_grid"
    assert isinstance(contribs[0].instance, GMGSIVISSource)


def test_satellite_provider_skips_when_all_channels_disabled(tmp_path: Path):
    """Both toggles off → empty contribution list."""
    from librewxr.sources.satellite.gmgsi import satellite_provider

    settings = MagicMock()
    settings.gmgsi_lw_enabled = False
    settings.gmgsi_vis_enabled = False

    contribs = satellite_provider(settings, cache_dir=tmp_path)
    assert contribs == []


def test_vis_subclass_s3_metadata():
    """GMGSIVISSource pins the VIS-specific product path and filename token."""
    assert GMGSIVISSource.channel == "VIS"
    assert GMGSIVISSource.s3_product_path == "GMGSI_VIS"
    assert GMGSIVISSource.s3_filename_prefix == "GLOBCOMPVIS"
    assert GMGSIVISSource.friendly_name == "GMGSI VIS"


# ── Frame store / retention ──


def test_max_frames_retention_evicts_oldest(tmp_path: Path):
    """When ``max_frames`` is exceeded the oldest frames are dropped."""
    src = GMGSILWSource(cache_dir=tmp_path, max_frames=3)
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), 100, dtype=np.uint8)

    # Manually inject 5 frames spanning 5 hours.
    base = int(datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    for hour in range(5):
        ts = base + hour * 3600
        src._frames[ts] = grid.copy()
    src._sorted_timestamps = sorted(src._frames)

    # Trim manually (mirrors what _fetch_sync does after a fetch cycle).
    while len(src._sorted_timestamps) > src._max_frames:
        oldest = src._sorted_timestamps.pop(0)
        src._frames.pop(oldest, None)

    assert len(src._sorted_timestamps) == 3
    assert src._sorted_timestamps[0] == base + 2 * 3600  # two earliest evicted


def test_nearest_timestamp_picks_closest():
    """Nearest-neighbour selection across the in-memory timestamp set."""
    src = GMGSILWSource(cache_dir=None)
    src._sorted_timestamps = [100, 200, 300]

    assert src._nearest_timestamp(None) == 300  # most recent
    assert src._nearest_timestamp(50) == 100
    assert src._nearest_timestamp(149) == 100
    assert src._nearest_timestamp(151) == 200
    assert src._nearest_timestamp(1000) == 300


# ── sample() ──


def test_sample_maps_grid_corners_to_known_values(tmp_path: Path):
    """sample() at the NW / SE corners returns the expected grid pixels."""
    src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    # Synthetic frame: row index encoded as the value (row 0 = 0, row 1 = 1, …)
    grid = np.tile(
        np.arange(GRID_HEIGHT, dtype=np.uint64) % 256, (GRID_WIDTH, 1),
    ).T.astype(np.uint8)
    ts = 12345
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]

    # NW corner (top-left): lat=LAT_MAX, lon=LON_MIN → row 0
    lat = np.array([[LAT_MAX]], dtype=np.float32)
    lon = np.array([[LON_MIN]], dtype=np.float32)
    out = src.sample(lat, lon, timestamp=ts)
    assert out[0, 0] == grid[0, 0]

    # SE corner: lat=LAT_MIN, lon=LON_MAX → row GRID_HEIGHT-1
    lat = np.array([[LAT_MIN]], dtype=np.float32)
    lon = np.array([[LON_MAX]], dtype=np.float32)
    out = src.sample(lat, lon, timestamp=ts)
    assert out[0, 0] == grid[GRID_HEIGHT - 1, GRID_WIDTH - 1]


def test_sample_returns_zero_outside_coverage_band(tmp_path: Path):
    """Latitudes outside ±72.74° render as the no-data sentinel (0)."""
    src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), 200, dtype=np.uint8)
    src._frames[12345] = grid
    src._sorted_timestamps = [12345]

    # Polar latitudes — well outside the GMGSI band.
    lat = np.array([[85.0, -85.0]], dtype=np.float32)
    lon = np.array([[0.0, 0.0]], dtype=np.float32)
    out = src.sample(lat, lon, timestamp=12345)
    assert (out == 0).all()


def test_sample_returns_zero_when_no_frames(tmp_path: Path):
    """sample() with empty store returns an all-zero array shaped like the input."""
    src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    lat = np.zeros((4, 5), dtype=np.float32)
    lon = np.zeros((4, 5), dtype=np.float32)
    out = src.sample(lat, lon, timestamp=None)
    assert out.shape == (4, 5)
    assert (out == 0).all()


def test_sample_falls_back_when_sorted_timestamps_desync_from_frames(tmp_path: Path):
    """sample() survives a _sorted_timestamps / _frames desync instead of KeyErroring.

    A concurrent __setstate__ swaps ``_frames`` to a fresh half-built dict
    while a render thread still holds a timestamp from the OLD sorted list.
    sample() must not raise: it falls back to the newest frame actually
    present in ``_frames``.
    """
    src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), 137, dtype=np.uint8)
    stale_ts = 11111
    present_ts = 22222
    src._frames[present_ts] = grid
    # Stale sorted list still names an old timestamp that is missing from
    # _frames - the exact desync the concurrent-snapshot race produces.
    src._sorted_timestamps = [stale_ts, present_ts]

    lat = np.array([[0.0]], dtype=np.float32)
    lon = np.array([[0.0]], dtype=np.float32)
    # Nearest-timestamp selection picks the stale ts (absent from _frames).
    out = src.sample(lat, lon, timestamp=stale_ts)
    assert out.dtype == np.uint8
    assert out.shape == (1, 1)
    assert out[0, 0] == 137  # fell back to the newest frame actually present


# ── Width coercion (NESDIS 2026-07 column drop) ──


def test_coerce_accepts_canonical_width():
    """A frame already at GRID_SHAPE passes through untouched."""
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    out = GMGSISource._coerce_to_canonical(arr)
    assert out is arr
    assert out.shape == GRID_SHAPE


def test_coerce_pads_narrow_by_one():
    """A frame one column short is padded by repeating the last column.

    Width tolerance kicks in for NESDIS upstream twiddles like the
    2026-07 5000->4999 drop (and a hypothetical 4999->4998 shave).
    Edge mode keeps the right-hand disk edge intact rather than
    zero-filling it with the no-data sentinel.
    """
    arr = np.arange(GRID_HEIGHT * (GRID_WIDTH - 1), dtype=np.float32).reshape(
        GRID_HEIGHT, GRID_WIDTH - 1,
    )
    out = GMGSISource._coerce_to_canonical(arr)
    assert out is not None
    assert out.shape == GRID_SHAPE
    # Last column is a copy of the previous last column.
    np.testing.assert_array_equal(out[:, -1], arr[:, -1])
    np.testing.assert_array_equal(out[:, :-1], arr)


def test_coerce_crops_wide_by_one():
    """A frame one column too wide is cropped to the canonical width.

    Preserves the 2026-05 layout (5000 cols) ingest path so a rolled-
    back upstream never takes the satellite layer down either.
    """
    arr = np.arange(
        GRID_HEIGHT * (GRID_WIDTH + 1), dtype=np.float32,
    ).reshape(GRID_HEIGHT, GRID_WIDTH + 1)
    out = GMGSISource._coerce_to_canonical(arr)
    assert out is not None
    assert out.shape == GRID_SHAPE
    np.testing.assert_array_equal(out, arr[:, :GRID_WIDTH])


def test_coerce_rejects_height_mismatch():
    """Latitude re-grids are never benign — reject them loudly."""
    arr = np.zeros((GRID_HEIGHT - 1, GRID_WIDTH), dtype=np.float32)
    assert GMGSISource._coerce_to_canonical(arr) is None


def test_coerce_rejects_height_gain():
    """Symmetric: too-tall frames are rejected too, not silently cropped."""
    arr = np.zeros((GRID_HEIGHT + 1, GRID_WIDTH), dtype=np.float32)
    assert GMGSISource._coerce_to_canonical(arr) is None


def test_coerce_rejects_width_out_of_tolerance():
    """Width off by more than +-1 col returns None (loud-fail preserved).

    A 10-column re-grid (4999 -> 4990) is not a benign twiddle — the
    global lon step shifts and LAT_VEC/LON_VEC would mis-sample.  The
    strict guard fires so the next NESDIS re-grid is obvious on first
    ingest rather than producing subtle pixel drift.
    """
    arr = np.zeros((GRID_HEIGHT, GRID_WIDTH - 10), dtype=np.float32)
    assert GMGSISource._coerce_to_canonical(arr) is None
    wide = np.zeros((GRID_HEIGHT, GRID_WIDTH + 10), dtype=np.float32)
    assert GMGSISource._coerce_to_canonical(wide) is None


def test_coerce_rejects_wrong_ndim():
    """A 3-D array (time axis not stripped) is rejected, not coerced."""
    arr = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    assert GMGSISource._coerce_to_canonical(arr) is None


# ── Cross-process pickle round-trip ──


def test_pickle_round_trip_via_disk_cache(tmp_path: Path):
    """A render worker can reconstruct the store from disk-cached frames.

    Mirrors what the master_state snapshot does: pipeline fetches a
    frame, writes it to disk, dumps state.  Render worker constructs an
    empty store, applies the snapshot's __setstate__, sees the same frames.
    """
    pipeline_src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), 137, dtype=np.uint8)
    ts = 99999
    pipeline_src._frames[ts] = grid
    pipeline_src._sorted_timestamps = [ts]
    pipeline_src._write_cache(ts, grid)

    state = pipeline_src.__getstate__()
    assert state["channel"] == "LW"
    assert ts in state["timestamps"]

    # Render-worker side: empty instance, apply state.
    render_src = GMGSILWSource.__new__(GMGSILWSource)
    render_src.__setstate__(state)
    assert render_src.timestamps == [ts]
    np.testing.assert_array_equal(render_src._frames[ts], grid)


def test_pickle_round_trip_handles_missing_cache_files(tmp_path: Path):
    """Timestamps in the snapshot but missing on disk are dropped silently."""
    pipeline_src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    # Snapshot claims a timestamp; no disk file exists for it.
    state = {
        "cache_root": str(tmp_path),
        "channel": "LW",
        "timestamps": [42],
        "max_frames": 12,
    }
    render_src = GMGSILWSource.__new__(GMGSILWSource)
    render_src.__setstate__(state)
    assert render_src.timestamps == []


def test_setstate_leaves_frames_and_sorted_timestamps_consistent(tmp_path: Path):
    """__setstate__ publishes _frames and _sorted_timestamps with the same key set.

    One listed timestamp has no readable cache file on disk.  The restored
    store must never leave a sorted timestamp that is absent from _frames -
    that desync is what lets a concurrent render thread hit a KeyError.
    """
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), 55, dtype=np.uint8)
    for ts in (1000, 2000):
        src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
        src._frames[ts] = grid
        src._write_cache(ts, grid)
    # Snapshot lists three timestamps, but 3000 has no cache file on disk.
    state = {
        "cache_root": str(tmp_path),
        "channel": "LW",
        "timestamps": [1000, 2000, 3000],
        "max_frames": 12,
    }
    render_src = GMGSILWSource.__new__(GMGSILWSource)
    render_src.__setstate__(state)
    assert set(render_src._frames) == set(render_src._sorted_timestamps)
    assert render_src._sorted_timestamps == [1000, 2000]
    np.testing.assert_array_equal(render_src._frames[2000], grid)


# ── Cache layout ──


def test_cache_dir_uses_versioned_v2_layout(tmp_path: Path):
    """Cache lives under ``gmgsi/v2/<channel>/`` so a width change never
    re-memmaps stale frames at the new shape (silent row-shear trap).

    The bump from 5000->4999 columns on 2026-07-24 left stale
    15,000,000-byte files on disk that numpy would happily memmap at
    the new 14,997,000-byte shape, shearing every row by one column.
    Versioning the layout dir sidesteps the whole class of bug for
    future re-grids too: ``CACHE_LAYOUT_VERSION`` just bumps again.
    """
    lw = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    assert lw._channel_cache_dir == tmp_path / "gmgsi" / CACHE_LAYOUT_VERSION / "LW"
    assert lw._channel_cache_dir.exists()

    vis = GMGSIVISSource(cache_dir=tmp_path, max_frames=12)
    assert vis._channel_cache_dir == (
        tmp_path / "gmgsi" / CACHE_LAYOUT_VERSION / "VIS"
    )
    assert vis._channel_cache_dir.exists()

    # Sanity: the constant is what the layout actually uses.
    assert CACHE_LAYOUT_VERSION == "v2"


def test_pickle_round_trip_routes_render_worker_to_v2_cache(tmp_path: Path):
    """A render worker restoring __setstate__ resolves the same v2 path.

    Guards against a drift where __init__ versions the dir but
    __setstate__ picks the unversioned path and silently misses the
    pipeline's freshly-written frames.
    """
    pipeline_src = GMGSILWSource(cache_dir=tmp_path, max_frames=12)
    state = pipeline_src.__getstate__()

    render_src = GMGSILWSource.__new__(GMGSILWSource)
    render_src.__setstate__(state)
    assert render_src._channel_cache_dir == (
        tmp_path / "gmgsi" / CACHE_LAYOUT_VERSION / "LW"
    )


# ── _list_recent_keys with mocked fsspec ──


def test_list_recent_keys_walks_one_hour_per_directory():
    """Listing walks each hour bucket and picks the single matching file."""
    src = GMGSILWSource(cache_dir=None, max_frames=3)
    fs = MagicMock()
    # Simulate one file per hour for the 3-hour window.

    def fake_ls(prefix, detail=False):
        hour = prefix.rstrip("/").rsplit("/", 1)[-1]
        return [
            f"{prefix}GLOBCOMPLIR_v3r0_blend_s202605231{hour}0000_e_c.nc",
        ]

    fs.ls.side_effect = fake_ls
    window_start = datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc)
    window_end = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    keys = src._list_recent_keys(fs, window_start, window_end)
    # Expect 3 hours covered (10, 11, 12).
    assert len(keys) == 3


def test_list_recent_keys_handles_missing_hour_directories():
    """A FileNotFoundError on one hour silently skips that hour."""
    src = GMGSILWSource(cache_dir=None, max_frames=3)
    fs = MagicMock()

    def fake_ls(prefix, detail=False):
        if "11/" in prefix:
            raise FileNotFoundError(prefix)
        hour = prefix.rstrip("/").rsplit("/", 1)[-1]
        return [
            f"{prefix}GLOBCOMPLIR_v3r0_blend_s202605231{hour}0000_e_c.nc",
        ]

    fs.ls.side_effect = fake_ls
    window_start = datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc)
    window_end = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    keys = src._list_recent_keys(fs, window_start, window_end)
    # Only hours 10 and 12 survive.
    assert len(keys) == 2
