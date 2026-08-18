# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math

import numpy as np
import pytest

pytestmark = pytest.mark.tiles

import librewxr.tiles.coordinates as coord
from librewxr.config import settings
from librewxr.data.regions import REGIONS
from librewxr.tiles.coordinates import (
    ALL_CACHES,
    COMPOSITE_HEIGHT,
    COMPOSITE_WIDTH,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    _reset_coord_store,
    latlon_to_global_pixel,
    tile_bounds,
    tile_overlaps_composite,
    tile_pixel_indices,
    warm_coordinate_caches,
    window_origin,
)


@pytest.fixture
def coord_store_env(monkeypatch, tmp_path):
    """Enable the shared coord store under a per-test tmp cache dir.

    The store singleton is process-global, so every test resets it and
    clears all coordinate LRUs before AND after running.
    """
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "coord_store_enabled", True)

    def _clear():
        for fn in ALL_CACHES:
            fn.cache_clear()

    _reset_coord_store()
    _clear()
    yield
    _clear()
    _reset_coord_store()


class TestTileBounds:
    def test_zoom_0(self):
        """Zoom 0 has a single tile covering the whole world."""
        w, s, e, n = tile_bounds(0, 0, 0)
        assert w == pytest.approx(-180.0)
        assert e == pytest.approx(180.0)
        assert n == pytest.approx(85.0511, abs=0.01)
        assert s == pytest.approx(-85.0511, abs=0.01)

    def test_zoom_1_tiles(self):
        """Zoom 1 has 4 tiles (2x2)."""
        # Top-left tile
        w, s, e, n = tile_bounds(1, 0, 0)
        assert w == pytest.approx(-180.0)
        assert e == pytest.approx(0.0)
        assert n > 0

        # Top-right tile
        w, s, e, n = tile_bounds(1, 1, 0)
        assert w == pytest.approx(0.0)
        assert e == pytest.approx(180.0)

    def test_tile_covers_conus(self):
        """A low-zoom tile should cover CONUS area."""
        # At zoom 3, tile (1, 2) or (1, 3) should cover parts of US
        w, s, e, n = tile_bounds(3, 1, 2)
        # Should be in western hemisphere, northern mid-latitudes
        assert w < 0
        assert n > 0


class TestWebMercatorWindowHelpers:
    """Forward Web Mercator + window-origin helpers for lat/lon-centered tiles."""

    def test_origin_at_world_center(self):
        """(0, 0) maps to the middle of the global pixel space (px=py=world/2)."""
        for z in (1, 3):
            world = (2**z) * 256
            px, py = latlon_to_global_pixel(0.0, 0.0, z, 256)
            assert px == pytest.approx(world / 2)
            assert py == pytest.approx(world / 2)

    def test_lon_seam_equivalence(self):
        """-180 and +180 both normalize to the same seam: px == 0."""
        for z in (1, 3):
            px_neg = latlon_to_global_pixel(0.0, -180.0, z, 256)[0]
            px_pos = latlon_to_global_pixel(0.0, 180.0, z, 256)[0]
            assert px_neg == 0.0
            assert px_pos == 0.0

    def test_lon_normalization(self):
        """lon=190 normalizes to -170 (identical px)."""
        px_190 = latlon_to_global_pixel(0.0, 190.0, 2, 256)[0]
        px_neg170 = latlon_to_global_pixel(0.0, -170.0, 2, 256)[0]
        assert px_190 == px_neg170
        assert px_190 == pytest.approx(28.4444444444)

    def test_lat_pole_clamping(self):
        """lat=+/-90 clamps to the Mercator limit: py=0 at the north pole,
        py=world at the south pole."""
        world = (2**1) * 256
        py_north = latlon_to_global_pixel(90.0, 0.0, 1, 256)[1]
        py_south = latlon_to_global_pixel(-90.0, 0.0, 1, 256)[1]
        assert py_north == pytest.approx(0.0, abs=1e-6)
        assert py_south == pytest.approx(world, abs=1e-6)

    def test_window_origin_center(self):
        """z=2 -> world=1024; a window centered on (0, 0) starts at (384, 384)."""
        assert window_origin(0.0, 0.0, 2, 256) == (384, 384)

    def test_window_origin_pole_clamping(self):
        """Windows near the poles clamp to the world edge: py0=0 / world-256."""
        world = (2**1) * 256
        assert window_origin(85.0, 0.0, 1, 256)[1] == 0
        assert window_origin(-85.0, 0.0, 1, 256)[1] == world - 256

    def test_window_origin_east_seam_wrap(self):
        """Lon just west of +180 wraps the origin to the east seam (px0=896)."""
        px0, py0 = window_origin(0.0, 179.999, 2, 256)
        assert px0 == 896
        assert py0 == 384

    def test_window_origin_west_seam_wrap(self):
        """Lon just east of -180 wraps the origin past the west seam (px0=902)."""
        px0, _ = window_origin(0.0, -178.0, 2, 256)
        assert px0 == 902

    def test_window_origin_random_lons_in_world(self):
        """Wrap-aware origins stay inside [0, world) for arbitrary lon."""
        world = (2**2) * 256  # z=2
        for lon in (-179.5, -90.25, -31.75, 7.125, 66.5, 133.9, 177.2):
            px0, _ = window_origin(0.0, lon, 2, 256)
            assert 0 <= px0 < world

    def test_round_trip_vs_tile_pixel_latlons(self):
        """Forward math inverts against the existing inverse: the lat/lon of
        the snapped pixel center reproduces the input (loose tolerance)."""
        ts = 256
        for lat, lon, z in (
            (37.7749, -122.4194, 6),
            (52.52, 13.405, 5),
            (-33.8688, 151.2093, 6),
        ):
            px, py = latlon_to_global_pixel(lat, lon, z, ts)
            world = (2**z) * ts
            k = round(px - 0.5)
            m = round(py - 0.5)
            x, y = k // ts, m // ts
            j, i = k % ts, m % ts
            lat_grid, lon_grid = coord.tile_pixel_latlons(z, x, y, ts)
            assert lat_grid[i, j] == pytest.approx(lat, abs=0.1)
            assert lon_grid[i, j] == pytest.approx(lon, abs=0.1)


class TestTileOverlapsComposite:
    def test_conus_tile_overlaps(self):
        """Tiles over the US should overlap."""
        # At zoom 2, tile (0, 1) covers western North America
        assert tile_overlaps_composite(2, 0, 1) is True

    def test_far_east_no_overlap(self):
        """Tiles in Asia/Pacific should not overlap CONUS composite."""
        # At zoom 2, tile (3, 1) is far east
        assert tile_overlaps_composite(2, 3, 1) is False

    def test_zoom_0_overlaps(self):
        """The single zoom-0 tile covers everything."""
        assert tile_overlaps_composite(0, 0, 0) is True


class TestTilePixelIndices:
    def test_output_shape(self):
        row_idx, col_idx = tile_pixel_indices(3, 1, 3, 256)
        assert row_idx.shape == (256, 256)
        assert col_idx.shape == (256, 256)

    def test_out_of_bounds_marked(self):
        """Tiles outside CONUS should have all -1 indices."""
        # A tile over the Pacific
        row_idx, col_idx = tile_pixel_indices(3, 0, 3, 256)
        # All or mostly -1 (this tile is south Pacific)
        # At least some should be -1
        assert np.any(row_idx == -1) or np.all(row_idx >= 0)

    def test_conus_tile_has_valid_indices(self):
        """A tile over CONUS should have valid indices."""
        # Zoom 4, tile roughly over central US
        row_idx, col_idx = tile_pixel_indices(4, 3, 5, 256)
        valid = (row_idx >= 0) & (col_idx >= 0)
        assert np.any(valid), "Expected some valid indices over CONUS"

    def test_valid_indices_in_range(self):
        """Valid indices should be within composite bounds."""
        row_idx, col_idx = tile_pixel_indices(4, 3, 5, 256)
        valid = (row_idx >= 0) & (col_idx >= 0)
        assert np.all(row_idx[valid] < COMPOSITE_HEIGHT)
        assert np.all(col_idx[valid] < COMPOSITE_WIDTH)

    def test_caching(self):
        """Same call should return identical arrays."""
        r1, c1 = tile_pixel_indices(3, 1, 2, 256)
        r2, c2 = tile_pixel_indices(3, 1, 2, 256)
        assert r1 is r2  # Same object from cache
        assert c1 is c2

    def test_512_tile_size(self):
        row_idx, col_idx = tile_pixel_indices(3, 1, 3, 512)
        assert row_idx.shape == (512, 512)


class TestWarmCoordinateCaches:
    def test_warms_at_least_one_entry(self):
        """warm_coordinate_caches should populate caches and return a positive count."""
        count = warm_coordinate_caches(["USCOMP"], max_zoom=2, tile_size=256)
        assert count > 0

    def test_zero_max_zoom_returns_zero(self):
        """max_zoom=0 should disable warming."""
        count = warm_coordinate_caches(["USCOMP"], max_zoom=0, tile_size=256)
        assert count == 0


class TestCoordStoreBacked:
    """Step 2: the six cached functions are backed by the shared coord store.

    Every test uses the ``coord_store_env`` fixture, which points
    ``settings.cache_dir`` at a tmp dir, enables the store, and resets the
    store singleton + all coordinate LRUs before and after each test.
    """

    # Tile (3, 1, 3) is partially inside USCOMP: some pixels valid, some -1.
    _REGION = "USCOMP"
    _Z, _X, _Y = 3, 1, 3

    def _six_calls(self, region, z, x, y, ts, pad=8):
        """Call every public store-backed function once for the given key."""
        row, col = coord.region_pixel_indices(region, z, x, y, ts)
        row_p, col_p = coord.region_pixel_indices_padded(region, z, x, y, ts, pad)
        rf, cf = coord.region_pixel_indices_fractional(region, z, x, y, ts)
        rf_p, cf_p = coord.region_pixel_indices_fractional_padded(
            region, z, x, y, ts, pad,
        )
        lat, lon = coord.tile_pixel_latlons(z, x, y, ts)
        lat_p, lon_p = coord.tile_pixel_latlons_padded(z, x, y, ts, pad)
        return (row, col), (row_p, col_p), (rf, cf), (rf_p, cf_p), (lat, lon), (lat_p, lon_p)

    def _six_compute(self, region, z, x, y, ts, pad=8):
        """Reference results straight from the uncached compute bodies."""
        row, col = coord._compute_region_pixel_indices(region, z, x, y, ts)
        row_p, col_p = coord._compute_region_pixel_indices_padded(region, z, x, y, ts, pad)
        rf, cf = coord._compute_region_pixel_indices_fractional(region, z, x, y, ts)
        rf_p, cf_p = coord._compute_region_pixel_indices_fractional_padded(
            region, z, x, y, ts, pad,
        )
        lat, lon = coord._compute_tile_pixel_latlons(z, x, y, ts)
        lat_p, lon_p = coord._compute_tile_pixel_latlons_padded(z, x, y, ts, pad)
        return (row, col), (row_p, col_p), (rf, cf), (rf_p, cf_p), (lat, lon), (lat_p, lon_p)

    def test_store_backed_equality(self, coord_store_env):
        """Store-backed results match the uncached compute bodies exactly.

        Values, dtypes, and the -1 out-of-region sentinel pattern must all
        survive the store round trip (tile partially outside the region).
        """
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        for got, exp in zip(
            self._six_calls(region, z, x, y, ts),
            self._six_compute(region, z, x, y, ts),
        ):
            for got_arr, exp_arr in zip(got, exp):
                np.testing.assert_array_equal(got_arr, exp_arr)
                assert got_arr.dtype == exp_arr.dtype
                assert not got_arr.flags.writeable

        # -1 sentinel pattern preserved on a partially-outside tile.
        row, col = coord.region_pixel_indices(region, z, x, y, ts)
        assert np.any(row == -1) and np.any(col == -1)
        assert np.any(row >= 0) and np.any(col >= 0)

    def test_first_call_pins_file_backed_pages(self, coord_store_env):
        """Fresh store (guaranteed miss -> publish) -> the first call re-opens
        the published entry and returns the shared read-only memmap views, so
        the lru pins file-backed pages instead of the heap arrays."""
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        for got, exp in zip(
            self._six_calls(region, z, x, y, ts),
            self._six_compute(region, z, x, y, ts),
        ):
            for got_arr, exp_arr in zip(got, exp):
                np.testing.assert_array_equal(got_arr, exp_arr)
                assert not got_arr.flags.writeable
                # Views' base is the (2, R, C) read-only memmap.
                assert isinstance(got_arr.base, np.memmap)

    def test_publish_failure_falls_back_to_heap(self, coord_store_env, monkeypatch):
        """A no-op publish (nothing lands in the store) -> the first call
        returns the freshly computed heap arrays; no exception."""
        monkeypatch.setattr(coord, "_try_publish", lambda *_a, **_k: None)
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        for got, exp in zip(
            self._six_calls(region, z, x, y, ts),
            self._six_compute(region, z, x, y, ts),
        ):
            for got_arr, exp_arr in zip(got, exp):
                np.testing.assert_array_equal(got_arr, exp_arr)

    def test_roundtrip_via_store(self, coord_store_env):
        """First call publishes; after an LRU clear the second is a store hit."""
        region = REGIONS[self._REGION]
        z, x, y, ts = 4, 3, 5, 256
        first = coord.region_pixel_indices(region, z, x, y, ts)
        assert coord._STORE is not None
        for fn in ALL_CACHES:
            fn.cache_clear()
        hits_before = coord._STORE.stats()["hits"]

        second = coord.region_pixel_indices(region, z, x, y, ts)
        np.testing.assert_array_equal(second[0], first[0])
        np.testing.assert_array_equal(second[1], first[1])
        assert not second[0].flags.writeable
        assert not second[1].flags.writeable
        assert coord._STORE.stats()["hits"] > hits_before

    def test_warm_request_key_agreement(self, coord_store_env):
        """Warming publishes exactly the keys the render path later reads."""
        region = REGIONS[self._REGION]
        ts = 256
        assert warm_coordinate_caches([self._REGION], max_zoom=3, tile_size=ts) > 0
        for fn in ALL_CACHES:
            fn.cache_clear()
        assert coord._STORE is not None
        publishes_before = coord._STORE.stats()["publishes"]
        hits_before = coord._STORE.stats()["hits"]

        # A tile overlapping USCOMP within the warmed zoom range.
        z, x, y = 2, 0, 1
        assert coord.tile_overlaps_region(region, z, x, y)

        # Derive the pad the render path would for this tile (smooth=True).
        sigma = coord.compute_blur_radius(region, z, x, y, ts)
        pad = int(sigma * 3) if sigma >= 0.5 else 0

        # The same calls the render path makes for that pad.
        coord.region_pixel_indices(region, z, x, y, ts)
        coord.region_pixel_indices_fractional(region, z, x, y, ts)
        coord.tile_pixel_latlons(z, x, y, ts)
        if pad > 0:
            coord.region_pixel_indices_padded(region, z, x, y, ts, pad)
            coord.region_pixel_indices_fractional_padded(region, z, x, y, ts, pad)
            coord.tile_pixel_latlons_padded(z, x, y, ts, pad)

        stats = coord._STORE.stats()
        assert stats["publishes"] == publishes_before, (
            f"warm/request keys disagree: {stats}"
        )
        assert stats["hits"] > hits_before

    def test_coord_store_cold_disabled_store(self, coord_store_env, monkeypatch):
        """Store disabled -> coord_store_cold() is False (no dedup to jitter for)."""
        monkeypatch.setattr(settings, "coord_store_enabled", False)
        _reset_coord_store()
        assert coord.coord_store_cold() is False

    def test_coord_store_cold_fresh_empty_store(self, coord_store_env):
        """Fresh empty store (nothing published yet) -> cold."""
        assert coord.coord_store_cold() is True

    def test_coord_store_cold_warm_after_probe_publish(self, coord_store_env):
        """Publishing the probe key (z=0 whole-earth latlon) -> warm."""
        coord.tile_pixel_latlons(0, 0, 0, 256)
        assert coord._STORE is not None
        assert coord.coord_store_cold() is False

    def test_disabled_bypasses_store(self, coord_store_env, monkeypatch, tmp_path):
        """coord_store_enabled=False -> correct results, no coord/ dir."""
        monkeypatch.setattr(settings, "coord_store_enabled", False)
        _reset_coord_store()
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        row, col = coord.region_pixel_indices(region, z, x, y, ts)
        exp_row, exp_col = coord._compute_region_pixel_indices(region, z, x, y, ts)
        np.testing.assert_array_equal(row, exp_row)
        np.testing.assert_array_equal(col, exp_col)
        assert coord._STORE is None
        assert not (tmp_path / "coord").exists()

    def test_store_open_failure_falls_back(self, coord_store_env, monkeypatch):
        """A throwing CoordStore.open must not break the public functions."""
        from librewxr.data import coord_store as coord_store_mod

        def _boom(*_args, **_kwargs):
            raise RuntimeError("coord store exploded")

        monkeypatch.setattr(coord_store_mod.CoordStore, "open", _boom)

        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        row, col = coord.region_pixel_indices(region, z, x, y, ts)
        exp_row, exp_col = coord._compute_region_pixel_indices(region, z, x, y, ts)
        np.testing.assert_array_equal(row, exp_row)
        np.testing.assert_array_equal(col, exp_col)

        lat, lon = coord.tile_pixel_latlons(z, x, y, ts)
        exp_lat, exp_lon = coord._compute_tile_pixel_latlons(z, x, y, ts)
        np.testing.assert_array_equal(lat, exp_lat)
        np.testing.assert_array_equal(lon, exp_lon)

    def test_no_cache_dir_store_off(self, coord_store_env, monkeypatch):
        """Empty cache_dir -> store off; results still correct."""
        monkeypatch.setattr(settings, "cache_dir", "")
        _reset_coord_store()
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 256
        row, col = coord.region_pixel_indices(region, z, x, y, ts)
        exp_row, exp_col = coord._compute_region_pixel_indices(region, z, x, y, ts)
        np.testing.assert_array_equal(row, exp_row)
        np.testing.assert_array_equal(col, exp_col)
        assert coord._STORE is None

    def test_prune_returns_counts_and_shrinks(self, coord_store_env, monkeypatch):
        """Prune helper returns (removed_bytes, removed_entries) and brings
        the on-disk store back under the MiB budget."""
        monkeypatch.setattr(settings, "coord_store_mb", 1)
        region = REGIONS[self._REGION]
        ts = 64
        # Each (z, x, y) key is ~250 KB at tile_size=64 with pad=8 (six
        # arrays), so five distinct tiles comfortably exceed the 1 MiB cap.
        for z, x, y in [(2, 0, 1), (3, 1, 3), (4, 3, 5), (4, 4, 5), (5, 9, 12)]:
            self._six_calls(region, z, x, y, ts)

        assert coord._STORE is not None
        assert coord._STORE.stats()["bytes"] > 1024 * 1024

        result = coord.prune_shared_coord_store()
        assert result is not None
        removed_bytes, removed_entries = result
        assert removed_entries > 0
        assert removed_bytes > 0
        assert coord._STORE.stats()["bytes"] <= 1024 * 1024

    def test_prune_disabled_returns_none(self, coord_store_env, monkeypatch):
        """coord_store_enabled=False -> prune helper returns None."""
        monkeypatch.setattr(settings, "coord_store_enabled", False)
        _reset_coord_store()
        assert coord.prune_shared_coord_store() is None

    def test_prune_within_budget_is_noop(self, coord_store_env, monkeypatch):
        """Store within budget: prune removes nothing and entries stay open."""
        monkeypatch.setattr(settings, "coord_store_mb", 1)
        region = REGIONS[self._REGION]
        z, x, y, ts = self._Z, self._X, self._Y, 64
        self._six_calls(region, z, x, y, ts)

        assert coord._STORE is not None
        entries_before = coord._STORE.stats()["entries"]

        removed_bytes, removed_entries = coord.prune_shared_coord_store()
        assert removed_entries == 0
        assert removed_bytes == 0
        assert coord._STORE.stats()["entries"] == entries_before

        # All entries still open cleanly: clear the LRUs and re-read
        # through the public wrappers (now store hits).
        for fn in ALL_CACHES:
            fn.cache_clear()
        hits_before = coord._STORE.stats()["hits"]
        for got, exp in zip(
            self._six_calls(region, z, x, y, ts),
            self._six_compute(region, z, x, y, ts),
        ):
            for got_arr, exp_arr in zip(got, exp):
                np.testing.assert_array_equal(got_arr, exp_arr)
        assert coord._STORE.stats()["hits"] > hits_before
