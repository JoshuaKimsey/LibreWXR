# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np
import pytest

pytestmark = pytest.mark.tiles

from librewxr.tiles.renderer import TileGeometry
from librewxr.tiles.window import (
    TileSpan,
    compute_window_geometry,
    covered_tiles,
    stitch_coverage,
    stitch_geometries,
)


def _geom(
    value: int,
    tile_size: int = 256,
    pad: int = 0,
    blur_radius: float = 0.0,
    snow_mask: np.ndarray | None = None,
) -> TileGeometry:
    values = np.full(
        (tile_size + 2 * pad, tile_size + 2 * pad), value, dtype=np.uint8
    )
    return TileGeometry(
        values=values,
        snow_mask=snow_mask,
        tile_size=tile_size,
        pad=pad,
        blur_radius=blur_radius,
    )


class TestCoveredTiles:
    def test_single_tile_window(self):
        spans = covered_tiles(z=2, px0=0, py0=0, extent=256, tile_size=256)
        assert spans == [
            TileSpan(tx=0, ty=0, win_x=0, win_y=0, tile_x=0, tile_y=0, w=256, h=256)
        ]

    def test_2x2_window_at_tile_boundary(self):
        spans = covered_tiles(z=2, px0=256, py0=256, extent=512, tile_size=256)
        assert spans == [
            TileSpan(1, 1, 0, 0, 0, 0, 256, 256),
            TileSpan(2, 1, 256, 0, 0, 0, 256, 256),
            TileSpan(1, 2, 0, 256, 0, 0, 256, 256),
            TileSpan(2, 2, 256, 256, 0, 0, 256, 256),
        ]

    def test_seam_wrap_at_z1(self):
        # world = 512, tile 256: px0=384 wraps from tile 1 back to tile 0.
        spans = covered_tiles(z=1, px0=384, py0=0, extent=256, tile_size=256)
        assert spans == [
            TileSpan(1, 0, 0, 0, 128, 0, 128, 256),
            TileSpan(0, 0, 128, 0, 0, 0, 128, 256),
        ]

    def test_defensive_y_clip(self):
        # z=3: world = 512 with tile_size 64.
        # Negative py0: off-world rows at the top are skipped, window
        # offsets preserved.
        spans = covered_tiles(z=3, px0=0, py0=-10, extent=64, tile_size=64)
        assert spans == [TileSpan(0, 0, 0, 10, 0, 0, 64, 54)]
        # py0 + extent beyond world end: trailing rows clipped.
        spans = covered_tiles(z=3, px0=0, py0=500, extent=64, tile_size=64)
        assert spans == [TileSpan(0, 7, 0, 0, 0, 52, 64, 12)]
        # Run split across two tile rows (and columns).
        spans = covered_tiles(z=1, px0=0, py0=200, extent=300, tile_size=256)
        assert spans == [
            TileSpan(0, 0, 0, 0, 0, 200, 256, 56),
            TileSpan(1, 0, 256, 0, 0, 200, 44, 56),
            TileSpan(0, 1, 0, 56, 0, 0, 256, 244),
            TileSpan(1, 1, 256, 56, 0, 0, 44, 244),
        ]


class TestStitchGeometries:
    def test_exact_canvas_mapping_with_wrap(self):
        # z=1: tiles 1 and 0 split the window at the wrap seam.
        components = {(1, 0): _geom(100), (0, 0): _geom(200)}
        stitched = stitch_geometries(
            components, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert not stitched.is_transparent
        assert stitched.values.shape == (256, 256)
        # Left half comes from tile (1,0) core at offset 128.
        assert np.all(stitched.values[:, :128] == 100)
        # Right half comes from tile (0,0) core at offset 0.
        assert np.all(stitched.values[:, 128:] == 200)
        assert stitched.snow_mask is None
        assert stitched.tile_size == 256
        assert stitched.pad == 0

    def test_padded_component_core_slice(self):
        # pad=2: the pad border must be excluded by the core slice.
        components = {(0, 0): _geom(7, pad=2)}
        geom = components[(0, 0)]
        geom.values[:, :2] = 9
        geom.values[:, -2:] = 9
        geom.values[:2, :] = 9
        geom.values[-2:, :] = 9
        stitched = stitch_geometries(
            components, z=0, px0=0, py0=0, extent=256, tile_size=256
        )
        assert np.all(stitched.values == 7)

    def test_missing_and_transparent_components_are_zeros(self):
        components = {(1, 0): _geom(100), (0, 0): TileGeometry.transparent(256)}
        stitched = stitch_geometries(
            components, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert np.all(stitched.values[:, :128] == 100)
        assert np.all(stitched.values[:, 128:] == 0)
        # Missing component (tile (1,0) absent) -> zeros too.
        stitched = stitch_geometries(
            {(0, 0): _geom(200)}, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert np.all(stitched.values[:, :128] == 0)
        assert np.all(stitched.values[:, 128:] == 200)

    def test_snow_mask_lazy_allocation(self):
        snow = np.zeros((256, 256), dtype=bool)
        snow[:, :64] = True
        components = {(1, 0): _geom(100), (0, 0): _geom(200, snow_mask=snow)}
        stitched = stitch_geometries(
            components, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert stitched.snow_mask is not None
        assert stitched.snow_mask.shape == (256, 256)
        # Tile (1,0) had no mask -> its canvas half stays False.
        assert not stitched.snow_mask[:, :128].any()
        # Tile (0,0) mask cols 0:64 land at canvas cols 128:192.
        assert np.all(stitched.snow_mask[:, 128:192])
        assert not stitched.snow_mask[:, 192:].any()

    def test_snow_mask_none_when_all_components_none(self):
        stitched = stitch_geometries(
            {(0, 0): _geom(5)}, z=0, px0=0, py0=0, extent=256, tile_size=256
        )
        assert stitched.snow_mask is None

    def test_snow_transparent_component_contributes_false(self):
        snow = np.ones((256, 256), dtype=bool)
        components = {(1, 0): TileGeometry.transparent(256), (0, 0): _geom(5, snow_mask=snow)}
        stitched = stitch_geometries(
            components, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert stitched.snow_mask is not None
        assert not stitched.snow_mask[:, :128].any()
        assert np.all(stitched.snow_mask[:, 128:])

    def test_blur_radius_is_max(self):
        components = {(1, 0): _geom(5, blur_radius=0.5), (0, 0): _geom(5, blur_radius=2.0)}
        stitched = stitch_geometries(
            components, z=1, px0=384, py0=0, extent=256, tile_size=256
        )
        assert stitched.blur_radius == 2.0
        assert stitched.pad == 0
        assert stitched.tile_size == 256

    def test_all_transparent_result(self):
        stitched = stitch_geometries(
            {(0, 0): TileGeometry.transparent(256)},
            z=0, px0=0, py0=0, extent=256, tile_size=256,
        )
        assert stitched.is_transparent
        assert stitched.values.shape == (0, 0)
        assert stitched.tile_size == 256

    def test_tile_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            stitch_geometries(
                {(0, 0): _geom(5, tile_size=128)},
                z=0, px0=0, py0=0, extent=256, tile_size=256,
            )


class _FakeProvider:
    """Async (tx, ty) -> TileGeometry with a call log."""

    def __init__(self, geoms: dict[tuple[int, int], TileGeometry]):
        self.geoms = geoms
        self.calls: list[tuple[int, int]] = []

    async def __call__(self, tx: int, ty: int) -> TileGeometry:
        self.calls.append((tx, ty))
        return self.geoms[(tx, ty)]


class TestComputeWindowGeometry:
    async def test_no_blur_returns_base_window(self):
        provider = _FakeProvider({(0, 0): _geom(5)})
        geom = await compute_window_geometry(provider, z=0, px0=0, py0=0, tile_size=256)
        assert not geom.is_transparent
        assert geom.tile_size == 256
        assert geom.pad == 0
        assert geom.blur_radius == 0.0
        assert geom.values.shape == (256, 256)
        assert np.all(geom.values == 5)
        assert provider.calls == [(0, 0)]

    async def test_blur_expands_window_and_fetches_extra_tiles(self):
        # z=1 (world 512), tile 256, one component blur 1.0 -> pad 3 ->
        # expanded extent 262; the three wrap-neighbor tiles get fetched.
        provider = _FakeProvider({
            (0, 0): _geom(5, blur_radius=1.0),
            (1, 0): _geom(6),
            (0, 1): _geom(7),
            (1, 1): _geom(8),
        })
        geom = await compute_window_geometry(provider, z=1, px0=0, py0=0, tile_size=256)
        assert not geom.is_transparent
        assert geom.tile_size == 256
        assert geom.pad == int(1.0 * 3) == 3
        assert geom.blur_radius == 1.0
        assert geom.values.shape == (256 + 2 * 3, 256 + 2 * 3)
        assert sorted(provider.calls) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    async def test_all_transparent_short_circuit(self):
        provider = _FakeProvider({(1, 0): TileGeometry.transparent(256)})
        geom = await compute_window_geometry(provider, z=1, px0=256, py0=0, tile_size=256)
        assert geom.is_transparent
        assert geom.tile_size == 256
        # Phase 2 must never run.
        assert provider.calls == [(1, 0)]

    async def test_expansion_skipped_when_world_too_small(self):
        # z=0: world == tile_size == 256, so 256 + 2*3 > 256 -> no phase 2.
        provider = _FakeProvider({(0, 0): _geom(5, blur_radius=1.0)})
        geom = await compute_window_geometry(provider, z=0, px0=0, py0=0, tile_size=256)
        assert not geom.is_transparent
        assert geom.tile_size == 256
        assert geom.pad == 0
        assert geom.blur_radius == 1.0
        assert geom.values.shape == (256, 256)
        assert provider.calls == [(0, 0)]


class TestStitchCoverage:
    def test_mapping_and_offsets(self):
        a = np.zeros((256, 256, 4), dtype=np.uint8)
        a[...] = [1, 2, 3, 4]
        b = np.zeros((256, 256, 4), dtype=np.uint8)
        b[...] = [5, 6, 7, 8]
        canvas = stitch_coverage({(1, 0): a, (0, 0): b}, z=1, px0=384, py0=0, tile_size=256)
        assert canvas.shape == (256, 256, 4)
        assert np.all(canvas[:, :128] == [1, 2, 3, 4])
        assert np.all(canvas[:, 128:] == [5, 6, 7, 8])

    def test_none_component_yields_zeros(self):
        a = np.zeros((256, 256, 4), dtype=np.uint8)
        a[...] = [1, 2, 3, 4]
        canvas = stitch_coverage({(1, 0): a}, z=1, px0=384, py0=0, tile_size=256)
        assert np.all(canvas[:, :128] == [1, 2, 3, 4])
        assert np.all(canvas[:, 128:] == 0)


class TestWindowLegacyBlurDonor:
    """Window mode must keep the legacy finest-bbox blur donor.

    Tile mode now sizes its blur from the finest region whose coverage
    mask actually claims the tile core, while window-mode components keep
    the old finest-bbox rule (the stitch applies the max component radius
    to the whole canvas, so a per-component re-donor would shift that
    rule).  Because the two modes disagree, window geometries must also
    live under a cache key distinct from tile mode's — a shared key would
    let whichever mode ran first serve its mis-sized geometry to the
    other.

    There is no route-level harness in this file, so this test drives the
    public radar route through a TestClient (the pattern
    tests/test_window_routes.py uses for window mode) with
    ``routes.compute_tile_geometry`` and ``routes.tile_cache`` replaced by
    recording stubs.
    """

    def test_components_pass_legacy_donor_and_use_distinct_geom_key(
        self, monkeypatch,
    ):
        import time

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from librewxr.api import routes
        from librewxr.data.store import RadarFrame

        ts = int(time.time() // 300) * 300
        # Content is irrelevant: compute_tile_geometry is stubbed.  Only
        # the region name matters (it gates the transparent fast path).
        frame = RadarFrame(
            timestamp=ts,
            regions={"USCOMP": np.zeros((2, 2), dtype=np.uint8)},
        )

        computes: list[dict] = []
        puts: list[tuple] = []

        def _fake_compute(**kwargs):
            computes.append(kwargs)
            return TileGeometry.transparent(kwargs["tile_size"])

        class _RecordingCache:
            def get(self, key):
                return None

            def put(self, key, value):
                puts.append(key)

        class _FrameStore:
            async def get_frame(self, timestamp):
                return frame if timestamp == ts else None

            async def get_timestamps(self):
                return [ts]

        monkeypatch.setattr(routes, "compute_tile_geometry", _fake_compute)
        monkeypatch.setattr(routes, "tile_cache", _RecordingCache())
        monkeypatch.setattr(routes, "frame_store", _FrameStore())
        monkeypatch.setattr(routes, "nowcast_store", None)
        monkeypatch.setattr(routes, "nwp_chain", None)
        monkeypatch.setattr(routes, "precip_mask", None)
        monkeypatch.setattr(routes, "enabled_regions", ["USCOMP"])
        monkeypatch.setattr(routes, "tile_request_tracker", None)
        monkeypatch.setattr(routes, "_latest_ts_cache", None)
        monkeypatch.setattr(routes, "_TRANSPARENT_RENDER_MEMO", {})

        app = FastAPI()
        app.include_router(routes.router)
        with TestClient(app) as client:
            resp = client.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200

        # (a) every window component asks for the legacy donor policy.
        assert computes, "window path never reached compute_tile_geometry"
        assert all(call["legacy_blur_donor"] is True for call in computes)

        # (b) the geometry key is tile mode's tuple plus a "win" marker.
        tile_mode_keys = set()
        window_keys = []
        for key in puts:
            if len(key) == 8 and key[-1] == "win":
                window_keys.append(key)
                tile_mode_keys.add(key[:-1])
        assert window_keys, "window geometry was not cached under a marked key"
        for key in puts:
            assert key not in tile_mode_keys, (
                f"window geometry must not share a key with tile mode: {key}"
            )
        # The marked keys carry the same viewport fields as tile mode's.
        ts0, z0, tx0, ty0, size0, smooth0, snow0, _marker = window_keys[0]
        assert (ts0, z0, size0, smooth0, snow0) == (ts, 5, 256, False, False)
        assert isinstance(tx0, int) and isinstance(ty0, int)
