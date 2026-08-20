# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.nowcast import NowcastFrame, NowcastStore
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


class _StubSatelliteSource:
    """Duck-typed GMGSI grid for the satellite tile route tests.

    Mirrors the ``GMGSISource`` surface the route + renderers touch: a
    ``timestamps`` property and a ``sample(lat, lon, timestamp)`` method
    returning a constant uint8 grid (same trick as the ``_ConstantSource``
    in ``test_gmgsi_composite_renderer.py``).
    """

    def __init__(self, value: int, timestamps: list[int]) -> None:
        self._timestamps = sorted(timestamps)
        self.value = value

    @property
    def timestamps(self) -> list[int]:
        return list(self._timestamps)

    @property
    def data_bytes(self) -> int:
        # GMGSISource exposes this for the /health memory breakdown; the
        # stub keeps no backing array, so report zero footprint.
        return 0

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        return np.full(lat.shape, self.value, dtype=np.uint8)


class _StubStormCellStore:
    """Duck-typed StormCellStore holding one cell centred on a given tile.

    The centroid is derived from ``region_pixel_indices_fractional`` so it
    is guaranteed to land inside the tile's coverage of the region, for
    any projection — the same trick ``test_storm_cell_render.py`` uses.
    """

    def __init__(self, detected_at: int, region: str, z: int, x: int, y: int) -> None:
        from librewxr.data.regions import REGIONS
        from librewxr.data.storm_cells import _CELL_DTYPE
        from librewxr.tiles.coordinates import (
            region_pixel_indices,
            region_pixel_indices_fractional,
        )

        region_def = REGIONS[region]
        row_f, col_f = region_pixel_indices_fractional(region_def, z, x, y, 256)
        row_i, _ = region_pixel_indices(region_def, z, x, y, 256)
        valid = row_i >= 0
        assert valid.any(), f"tile {z}/{x}/{y} does not cover {region}"

        cell = np.zeros((1,), dtype=_CELL_DTYPE)
        cell["centroid_row"][0] = float(row_f[valid].mean())
        cell["centroid_col"][0] = float(col_f[valid].mean())
        cell["area_km2"][0] = 500.0
        cell["max_dbz"][0] = 55.0
        cell["motion_speed_kmh"][0] = np.nan

        self._cells = {region: cell}
        self._counts = {region: 1}
        self.detected_at_timestamp = detected_at
        # Content version for the detected cells (mirrors the real
        # StormCellStore's read-only property surface used by the overlay
        # cache keying).
        self.cells_version = 0

    async def get_cells(self) -> dict[str, np.ndarray]:
        return dict(self._cells)

    async def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _StubNowcastStore:
    """Duck-typed NowcastStore holding a constant radar flow field.

    Enough of the real surface for the arrow overlay path: per-region
    flows (constant eastward motion at reduced resolution, so arrows are
    actually drawn on the fixture tile), no composite NWP flow, and a
    mutable ``flow_version`` so tests can simulate a pipeline
    regeneration re-keying the overlay cache.
    """

    def __init__(self, flow_version: int = 1) -> None:
        self.flow_version = flow_version
        flow = np.zeros((2, 2, 2), dtype=np.float32)
        flow[..., 0] = 3.0  # constant eastward motion
        self._flows = {"USCOMP": flow}
        self._nwp_flow = None

    async def get_flows(self) -> dict[str, np.ndarray]:
        return dict(self._flows)

    async def get_nwp_flow(self) -> np.ndarray | None:
        return self._nwp_flow

    async def get_frame(self, timestamp: int):
        return None, 0.0


def _make_test_app() -> tuple[FastAPI, FrameStore, TileCache, int, int]:
    """Create a minimal FastAPI app with just the router — no lifespan."""
    store = FrameStore(max_frames=12)
    cache = TileCache(max_mb=10)
    ts = int(time.time() // 300) * 300
    ts_prev = ts - 600

    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    data[2500:2700, 6000:6200] = 128

    import asyncio
    frame = RadarFrame(timestamp=ts, regions={"USCOMP": data})
    asyncio.run(store.add_frame(frame))
    prev_frame = RadarFrame(timestamp=ts_prev, regions={"USCOMP": data})
    asyncio.run(store.add_frame(prev_frame))

    # Wire shared state directly — same as main.py does after lifespan init
    routes.frame_store = store
    routes.tile_cache = cache
    routes.ecmwf_grid = None
    routes.tile_warmer = None
    routes.nowcast_store = None
    routes.start_time = time.time()
    routes.enabled_regions = ["USCOMP"]

    # Duck-typed GMGSI stubs so the satellite route renders (composite:
    # cold LW cloud over VIS=0 night side) instead of returning 503.
    routes.satellite_grids = {
        "gmgsi_lw_grid": _StubSatelliteSource(180, [ts_prev, ts]),
        "gmgsi_vis_grid": _StubSatelliteSource(0, [ts_prev, ts]),
    }

    test_app = FastAPI()
    test_app.include_router(routes.router)
    return test_app, store, cache, ts, ts_prev


# Module-scoped: built once, shared across all tests in this file
_app, _store, _cache, _ts, _ts_prev = _make_test_app()


@pytest.fixture(scope="module")
def client():
    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c, _ts, _ts_prev


class TestWeatherMapsEndpoint:
    def test_returns_valid_json(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0"
        assert "generated" in data
        assert "host" in data
        assert "radar" in data
        assert "past" in data["radar"]
        assert "nowcast" in data["radar"]
        assert "satellite" in data

    def test_past_contains_timestamps(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        past = data["radar"]["past"]
        assert len(past) >= 1
        # past is sorted oldest-first; ts_prev was added first (earlier)
        assert past[0]["time"] == ts_prev
        assert past[0]["path"] == f"/v2/radar/{ts_prev}"

    def test_nowcast_excludes_overlapping_past_timestamp(self, client, monkeypatch):
        """Regression: during the nowcast-regeneration window the store is
        still anchored to the previous cycle, so a nowcast slot equal to the
        newest past frame must not be advertised as both past and nowcast."""
        c, ts, _ = client
        store = NowcastStore()
        asyncio.run(store.replace_all([
            NowcastFrame(timestamp=ts),
            NowcastFrame(timestamp=ts + 600),
            NowcastFrame(timestamp=ts + 1200),
        ]))
        monkeypatch.setattr(routes, "nowcast_store", store)
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        nowcast = data["radar"]["nowcast"]
        assert all(entry["time"] > ts for entry in nowcast)
        assert nowcast[0]["time"] == ts + 600
        assert len(nowcast) == 2
        assert [entry["path"] for entry in nowcast] == [
            f"/v2/radar/{ts + 600}",
            f"/v2/radar/{ts + 1200}",
        ]

    def test_nowcast_future_timestamps_untouched(self, client, monkeypatch):
        """Nowcast slots strictly newer than the newest past frame survive
        unchanged and in order."""
        c, ts, _ = client
        store = NowcastStore()
        asyncio.run(store.replace_all([
            NowcastFrame(timestamp=ts + 600),
            NowcastFrame(timestamp=ts + 1200),
        ]))
        monkeypatch.setattr(routes, "nowcast_store", store)
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        nowcast = data["radar"]["nowcast"]
        assert [entry["time"] for entry in nowcast] == [ts + 600, ts + 1200]
        assert [entry["path"] for entry in nowcast] == [
            f"/v2/radar/{ts + 600}",
            f"/v2/radar/{ts + 1200}",
        ]


class TestRadarTileEndpoint:
    def test_valid_tile_request(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_format(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.webp")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_missing_timestamp(self, client):
        c, _, _ = client
        resp = c.get("/v2/radar/9999999999/256/4/3/5/2/0_0.png")
        assert resp.status_code == 404

    def test_latest_frame_cache_header(self, client):
        """Latest frame gets short cache lifetime."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert "cache-control" in resp.headers
        assert "max-age=300" in resp.headers["cache-control"]

    def test_historical_frame_cache_header(self, client):
        """Historical frames get long cache lifetime since they are immutable."""
        c, _, ts_prev = client
        resp = c.get(f"/v2/radar/{ts_prev}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert "cache-control" in resp.headers
        assert "max-age=7200" in resp.headers["cache-control"]

    def test_timestamp_zero_serves_latest(self, client):
        c, ts, ts_prev = client
        resp = c.get("/v2/radar/0/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["x-frame-timestamp"] == str(ts)
        assert resp.content == c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png").content

    def test_timestamp_zero_cache_header(self, client):
        """The resolved latest frame must NOT get the 7200 historical bucket."""
        c, ts, ts_prev = client
        resp = c.get("/v2/radar/0/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert "max-age=300" in resp.headers["cache-control"]

    def test_canonical_url_has_frame_timestamp_header(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["x-frame-timestamp"] == str(ts)

    def test_timestamp_zero_304_keeps_header(self, client):
        c, ts, ts_prev = client
        url = "/v2/radar/0/256/4/3/5/2/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.headers["x-frame-timestamp"] == str(ts)

    def test_radar_tile_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_radar_tile_304_on_match(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/4/3/5/2/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control", "").startswith("public, max-age=")
        # httpx/TestClient omits Content-Length on 304 (see test_conditional.py)
        assert resp.headers.get("content-length") is None

    def test_radar_tile_304_on_star(self, client):
        c, ts, _ = client
        resp = c.get(
            f"/v2/radar/{ts}/256/4/3/5/2/0_0.png",
            headers={"If-None-Match": "*"},
        )
        assert resp.status_code == 304
        assert resp.content == b""

    def test_radar_tile_304_on_mismatch(self, client):
        c, ts, _ = client
        resp = c.get(
            f"/v2/radar/{ts}/256/4/3/5/2/0_0.png",
            headers={"If-None-Match": '"deadbeefdeadbeef"'},
        )
        assert resp.status_code == 200
        assert resp.content != b""

    def test_radar_tile_etag_stable_across_requests(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/4/3/5/2/0_0.png"
        first = c.get(url)
        second = c.get(url)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.headers["etag"] == second.headers["etag"]

    def test_radar_tile_overlay_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png?arrows=1")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_cells_overlay_renders_on_warm_geometry_cache(self, client, monkeypatch):
        """?cells= must draw even when the geometry cache already holds the tile.

        Regression: ``need_frame`` only accounted for ``arrow_style``, so a
        cells-only request that hit the geometry cache left ``frame`` unset.
        ``present_tile`` then received ``frame_regions=None``, and
        ``_draw_storm_cells`` fell back to an empty region list — the tile
        came back byte-identical to the plain one, with no error.  In
        production the cache is warm essentially always, so ?cells= looked
        inert while ?arrows= (which forced the fetch) worked.
        """
        c, ts, _ = client
        # 4/3/6 is the z=4 tile that actually covers the fixture's radar
        # block — a transparent tile short-circuits before any overlay.
        url = f"/v2/radar/{ts}/256/4/3/6/2/0_0.png"

        # Warm the geometry cache first — that's the condition that used to
        # silently disable the overlay.
        plain = c.get(url)
        assert plain.status_code == 200

        monkeypatch.setattr(
            routes, "storm_cell_store", _StubStormCellStore(ts, "USCOMP", 4, 3, 6),
        )
        with_cells = c.get(f"{url}?cells=1")
        assert with_cells.status_code == 200
        assert with_cells.content != plain.content

    def test_cells_overlay_skipped_on_other_frames(self, client, monkeypatch):
        """Cells only render on the frame detection actually ran on."""
        c, ts, ts_prev = client
        monkeypatch.setattr(
            routes, "storm_cell_store", _StubStormCellStore(ts, "USCOMP", 4, 3, 6),
        )
        url = f"/v2/radar/{ts_prev}/256/4/3/6/2/0_0.png"
        assert c.get(f"{url}?cells=1").content == c.get(url).content

    def test_past_frame_arrows_cached_in_worker(self, client, monkeypatch):
        """Past-frame overlay requests now cache: the first renders, the
        second is served from the in-worker cache without re-presenting
        (previously past-frame overlays re-rendered per request)."""
        c, _, ts_prev = client
        url = f"/v2/radar/{ts_prev}/256/4/3/6/2/0_0.png?arrows=1"
        monkeypatch.setattr(routes, "nowcast_store", _StubNowcastStore())
        routes.tile_cache.clear()

        first = c.get(url)
        assert first.status_code == 200

        # Second request must not run the present stage - in-worker cache hit.
        def _present_must_not_run(*_args, **_kwargs):
            raise AssertionError("present ran on a cached overlay hit")

        monkeypatch.setattr(routes, "_present_tile_async", _present_must_not_run)
        second = c.get(url)
        assert second.status_code == 200
        assert second.content == first.content

    def test_flow_version_bump_rekeys_overlay(self, client, monkeypatch):
        """Bumping ``flow_version`` re-keys the overlay so the next request
        re-renders instead of serving the previous flow's cached bytes."""
        c, _, ts_prev = client
        url = f"/v2/radar/{ts_prev}/256/4/3/6/2/0_0.png?arrows=1"
        stub = _StubNowcastStore(flow_version=1)
        monkeypatch.setattr(routes, "nowcast_store", stub)
        routes.tile_cache.clear()

        orig_present = routes._present_tile_async
        calls = {"present": 0}

        async def counting_present(*args, **kwargs):
            calls["present"] += 1
            return await orig_present(*args, **kwargs)

        monkeypatch.setattr(routes, "_present_tile_async", counting_present)

        first = c.get(url)
        assert first.status_code == 200
        assert calls["present"] == 1

        # Repeat under the same flow version: in-worker cache hit.
        repeat = c.get(url)
        assert repeat.status_code == 200
        assert repeat.content == first.content
        assert calls["present"] == 1

        # Pipeline regenerates flows: new version -> new key -> re-render.
        stub.flow_version = 2
        again = c.get(url)
        assert again.status_code == 200
        assert again.content == first.content
        assert calls["present"] == 2


class TestCoverageTileEndpoint:
    def test_valid_coverage_request(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_coverage_tile_has_etag(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_coverage_tile_304_on_match(self, client):
        c, _, _ = client
        url = "/v2/coverage/0/256/4/3/5/0/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("cache-control") == "public, max-age=300"


class TestSatelliteTileEndpoint:
    def test_satellite_tile_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/satellite/{ts}/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_satellite_tile_304_on_match(self, client):
        c, ts, _ = client
        url = f"/v2/satellite/{ts}/256/4/3/5/0/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control", "").startswith("public, max-age=")

    def test_satellite_timestamp_zero_serves_latest(self, client):
        c, ts, _ = client
        resp = c.get("/v2/satellite/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["x-frame-timestamp"] == str(ts)
        assert "max-age=300" in resp.headers["cache-control"]
        assert resp.content == c.get(f"/v2/satellite/{ts}/256/4/3/5/0/0_0.png").content


class TestHealthEndpoint:
    def test_health_memory_breakdown_without_store(self, client, monkeypatch):
        """Shared store inactive -> breakdown reports the per-worker lru
        byte estimate under ``coord_caches_mb`` and zero store fields.

        The store singleton is process-global and lazily constructed from
        ``settings.cache_dir``, so the default case must pin cache_dir off
        and reset the singleton (mirrors the ``coord_store_env`` fixture in
        test_coordinates.py) instead of relying on ambient settings.
        """
        from librewxr.tiles.coordinates import _reset_coord_store

        monkeypatch.setattr(settings, "cache_dir", "")
        _reset_coord_store()
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        breakdown = data["memory"]["breakdown"]
        assert breakdown["coord_store_mb"] == 0.0
        assert breakdown["coord_store_entries"] == 0
        assert breakdown["coord_caches_mb"] >= 0.0
        # The "coord_caches" block serializes the new ``store`` sub-dict
        # automatically; it is None while the store is inactive.
        assert data["coord_caches"]["store"] is None

    def test_health_memory_breakdown_store_active(self, client, monkeypatch):
        """Store-backed: entries are shared read-only memmap pages, not
        private heap - ``coord_caches_mb`` drops to 0 and the on-disk
        footprint is reported separately."""
        c, _, _ = client
        store_stats = {
            "hits": 1,
            "misses": 0,
            "publishes": 1,
            "entries": 3,
            "bytes": 3 * 1024 * 1024,
        }
        monkeypatch.setattr(
            routes,
            "coord_cache_stats",
            lambda: {"max_size": 2048, "caches": {}, "store": store_stats},
        )
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        breakdown = data["memory"]["breakdown"]
        assert breakdown["coord_store_mb"] == 3.0
        assert breakdown["coord_store_entries"] == 3
        assert breakdown["coord_caches_mb"] == 0.0
        assert data["coord_caches"]["store"] == store_stats


class TestHealthCluster:
    """The /health ``cluster`` section aggregating the worker pulses."""

    def test_cluster_section_present_with_this_worker(self, client):
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        cluster = data["cluster"]
        # No cache_dir in the test harness -> only the live worker's own
        # payload is aggregated; it always reports at least itself.
        assert cluster["workers_reporting"] >= 1

        memory = cluster["memory"]
        # Container split is only meaningful inside a cgroup; when present
        # it must carry the full key set.
        container = memory["container"]
        if container is not None:
            assert set(container) == {"anon_mb", "file_mb", "shmem_mb", "limit_mb"}
        workers_rss = memory["workers_rss_mb"]
        assert set(workers_rss) == {"sum", "min", "max"}
        assert workers_rss["min"] <= workers_rss["max"]
        assert workers_rss["sum"] >= 0.0

        tile_cache = cluster["tile_cache"]
        assert set(tile_cache) == {"entries", "used_mb"}
        assert tile_cache["entries"] >= 0
        assert tile_cache["used_mb"] >= 0.0

        coord = cluster["coord"]
        assert isinstance(coord["caches"], dict)
        for name, info in coord["caches"].items():
            assert set(info) == {"entries", "hits", "misses", "hit_ratio"}
            assert info["hits"] >= 0 and info["misses"] >= 0
        # Shared store is off in the test harness -> None; when present it
        # carries the summed per-process counters plus global entries/bytes.
        if coord["store"] is not None:
            assert set(coord["store"]) == {
                "hits", "misses", "publishes", "entries", "bytes",
            }

        requests = cluster["requests"]
        assert requests["total_requests"] >= 0
        assert requests["cache_hits"] >= 0
        assert requests["cache_misses"] >= 0
        assert requests["fast_path_total"] >= 0
        assert requests["hot_tiles"] >= 0
        assert "hit_rate" in requests

    def test_cluster_survives_reader_failure(self, client, monkeypatch):
        """A failing pulse scan degrades the section to None — /health
        itself must never raise."""
        c, _, _ = client

        def boom():
            raise OSError("pulse scan failed")

        monkeypatch.setattr(
            "librewxr.api.routes.read_worker_pulses", boom,
        )
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["cluster"] is None

    def test_cluster_sums_tile_cache_and_requests_across_pulses(self, client, monkeypatch):
        """Aggregation folds a synthetic second worker's pulse into the
        summed sections, with the coord-store block taking entries/bytes
        from this worker's live stats rather than a cross-worker sum."""
        import tempfile
        from pathlib import Path

        from librewxr.data.worker_pulse import write_worker_pulse

        c, _, _ = client
        # Live coord-store stats are global on-disk values; the fake gives
        # this worker a 9-entry / 9 MB store so the block's entries/bytes
        # come from here, NOT from summing the synthetic pulse (which
        # carries only per-process counters).
        monkeypatch.setattr(
            routes,
            "coord_cache_stats",
            lambda: {
                "max_size": 2048,
                "caches": {},
                "store": {
                    "hits": 0, "misses": 0, "publishes": 0,
                    "entries": 9, "bytes": 9 * 1024 * 1024,
                },
            },
        )
        with tempfile.TemporaryDirectory(prefix="librewxr_cluster_test_") as td:
            cache_dir = Path(td)
            # A synthetic "other" worker.
            write_worker_pulse(cache_dir, {
                "pid": 999999,
                "written_at": int(time.time()),
                "rss_bytes": 100 * 1024 * 1024,
                "tile_cache": {"entries": 7, "total_bytes": 2 * 1024 * 1024,
                               "max_bytes": 128 * 1024 * 1024},
                "coord": {
                    "caches": {
                        "region_pixel_indices": {"entries": 4, "hits": 10, "misses": 2},
                    },
                    "store": {"hits": 5, "misses": 1, "publishes": 3},
                },
                "requests": {
                    "enabled": True,
                    "total_requests": 50,
                    "hot_tiles": 3,
                    "fast_path_total": 4,
                    "cache_hits": 40,
                    "cache_misses": 10,
                },
            })
            monkeypatch.setattr(settings, "cache_dir", td)
            resp = c.get("/health")
            assert resp.status_code == 200
            cluster = resp.json()["cluster"]
            # This worker + the synthetic one.
            assert cluster["workers_reporting"] == 2

            # RSS aggregation over both workers.
            rss = cluster["memory"]["workers_rss_mb"]
            own_rss_mb = psutil_rss_mb()
            assert rss["min"] == min(100.0, own_rss_mb)
            assert rss["max"] == max(100.0, own_rss_mb)

            # Tile cache: synthetic worker's 2 MB + this worker's cache.
            assert cluster["tile_cache"]["entries"] == 7 + _cache.size
            assert cluster["tile_cache"]["used_mb"] >= 2.0

            # Per-cache sums + hit ratio recomputed from the sums.  The
            # synthetic worker contributes 10 hits / 2 misses; this
            # worker's live pulse has empty caches (fake above).
            idx = cluster["coord"]["caches"]["region_pixel_indices"]
            assert idx["hits"] == 10
            assert idx["misses"] == 2
            assert idx["entries"] == 4
            assert idx["hit_ratio"] == round(10 / 12, 3)

            # Store: per-process counters summed (synthetic 5/1/3 + live
            # 0/0/0), entries/bytes from this worker's live store stats.
            store = cluster["coord"]["store"]
            assert store["hits"] == 5
            assert store["misses"] == 1
            assert store["publishes"] == 3
            assert store["entries"] == 9
            assert store["bytes"] == 9 * 1024 * 1024

            # Requests: synthetic worker's counters summed in.
            req = cluster["requests"]
            assert req["total_requests"] == 50
            assert req["cache_hits"] == 40
            assert req["cache_misses"] == 10
            assert req["fast_path_total"] == 4
            assert req["hot_tiles"] == 3
            assert req["hit_rate"] == 0.8

    def test_cluster_preserves_existing_health_fields(self, client):
        """The pre-existing /health keys are untouched by the cluster work."""
        c, _, _ = client
        resp = c.get("/health")
        data = resp.json()
        for key in (
            "status", "uptime_seconds", "memory", "frames", "tile_cache",
            "nwp_chain", "nowcast", "satellite", "enabled_regions", "sources",
            "radar_cache", "coord_caches", "tile_requests", "alerts", "mcp",
            "storm_cells", "cluster",
        ):
            assert key in data, f"missing pre-existing health key {key}"


class TestSharedTileStoreIntegration:
    """End-to-end shared encoded-tile store path through the radar route.

    Exercises the exact multi-mode wiring: a plain past-frame tile renders,
    its fresh encode is published to the shared store, and a subsequent
    request serves the same bytes from the store with geometry computation
    completely bypassed.
    """

    def test_shared_store_serves_without_recompute(self, client, tmp_path, monkeypatch):
        """First request publishes; second request is served from the store.

        The compute monkeypatch hard-fails, so the only way the second
        request can return 200 with byte-identical content is a shared-store
        hit — proving the encode (not a recompute) was reused.
        """
        from librewxr.tiles.shared_tile_store import SharedTileStore

        c, ts, _ = client
        # 4/3/6 is the z=4 tile that actually covers the fixture's radar
        # block (see test_cells_overlay_renders_on_warm_geometry_cache), so
        # the encode is real content, not a transparent fast-path tile.
        url = f"/v2/radar/{ts}/256/4/3/6/2/0_0.png"

        store = SharedTileStore(tmp_path, max_mb=64)
        monkeypatch.setattr(routes, "shared_tile_store", store)

        # Cold in-memory caches force a full compute + present, which
        # publishes the fresh encode to the shared store (fire-and-forget
        # background task in the handler).
        routes.tile_cache.clear()
        first = c.get(url)
        assert first.status_code == 200

        # Wait for the background publish to land (it runs on the
        # TestClient's event loop, not this thread).
        deadline = time.monotonic() + 5.0
        while store.size < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.size == 1, "plain past-frame tile was not published"

        # "Second worker": in-memory caches cleared and geometry compute
        # hard-failing — success can only come from the shared store.
        def _compute_must_not_run(*_args, **_kwargs):
            raise AssertionError("compute_tile_geometry ran on a shared-store hit")

        monkeypatch.setattr(routes, "compute_tile_geometry", _compute_must_not_run)
        routes.tile_cache.clear()
        second = c.get(url)
        assert second.status_code == 200
        assert second.content == first.content

    def test_shared_store_serves_overlay_without_recompute(self, client, tmp_path, monkeypatch):
        """A past-frame overlay render publishes to the shared store under
        a flow/cells-versioned key and a second request serves from it.

        The present monkeypatch hard-fails, so the only way the second
        request can return 200 with byte-identical content is a
        shared-store hit - proving the overlay encode (not a re-render)
        was reused.
        """
        from librewxr.tiles.shared_tile_store import SharedTileStore

        c, _, ts_prev = client
        url = f"/v2/radar/{ts_prev}/256/4/3/6/2/0_0.png?arrows=1"
        store = SharedTileStore(tmp_path, max_mb=64)
        monkeypatch.setattr(routes, "shared_tile_store", store)
        stub = _StubNowcastStore(flow_version=1)
        monkeypatch.setattr(routes, "nowcast_store", stub)
        routes.tile_cache.clear()

        # Cold in-memory caches force a full compute + present, which
        # publishes the fresh overlay encode to the shared store
        # (fire-and-forget background task in the handler).
        first = c.get(url)
        assert first.status_code == 200

        # Wait for the background publish to land (it runs on the
        # TestClient's event loop, not this thread).
        deadline = time.monotonic() + 5.0
        while store.size < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.size == 1, "past-frame overlay tile was not published"

        # Key shape: ts prefix + frame version + flow/cells version
        # segments (arrow_style/cell_style tail included).
        entries = list(store.root.rglob("*.tile"))
        assert len(entries) == 1
        key = entries[0].stem
        assert key.startswith(f"{ts_prev}-")
        assert "-f1-" in key
        assert "-c0-" in key
        assert "-alight-k" in key

        # "Second worker": in-memory caches cleared and present
        # hard-failing - success can only come from the shared store.
        def _present_must_not_run(*_args, **_kwargs):
            raise AssertionError("present ran on a shared-store overlay hit")

        monkeypatch.setattr(routes, "_present_tile_async", _present_must_not_run)
        routes.tile_cache.clear()
        second = c.get(url)
        assert second.status_code == 200
        assert second.content == first.content


def psutil_rss_mb() -> float:
    """This process's RSS in MB (floored like the /health rounding)."""
    import psutil

    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
