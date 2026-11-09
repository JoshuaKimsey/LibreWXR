# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.api.metrics import render_prometheus
from librewxr.api.metrics import router as metrics_router
from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache


def _payload() -> dict:
    """A realistic /health payload with every section populated.

    ``alerts`` stays disabled on purpose so the disabled-section rule is
    exercised alongside the populated families.
    """
    return {
        "status": "ok",
        "uptime_seconds": 1234,
        "version": "0.9.0",
        "color_schemes": {
            "version": "0123456789ab",
            "source": "path",
            "fetched_at": "2026-11-01T00:00:00+00:00",
            "last_status": "ok",
        },
        "cluster": {
            "workers_reporting": 2,
            "memory": {
                "container": {
                    "anon_mb": 100.0,
                    "file_mb": 50.0,
                    "shmem_mb": 5.0,
                    "current_mb": 160.0,
                    "limit_mb": 4096.0,
                },
                "workers_rss_mb": {"sum": 900.0, "min": 400.0, "max": 500.0},
            },
            "tile_cache": {"entries": 42, "used_mb": 12.5},
            "coord": {
                "caches": {
                    "tile_bounds": {
                        "entries": 10,
                        "hits": 90,
                        "misses": 10,
                        "hit_ratio": 0.9,
                    },
                    "empty_cache": {
                        "entries": 0,
                        "hits": 0,
                        "misses": 0,
                        "hit_ratio": None,
                    },
                },
                "store": {
                    "hits": 7,
                    "misses": 3,
                    "publishes": 2,
                    "entries": 5,
                    "bytes": 1048576,
                },
            },
            "requests": {
                "total_requests": 1000,
                "cache_hits": 900,
                "cache_misses": 100,
                "fast_path_total": 500,
                "hot_tiles": 3,
                "hit_rate": 0.9,
            },
            "tile_latency": {
                "avg_request_ms": 12.5,
                "avg_compute_ms": 8.0,
                "avg_present_ms": 4.5,
            },
        },
        "memory": {
            "resident_mb": 512.0,
            "limit_mb": 2048.0,
            "usage_pct": 25.0,
            "breakdown": {
                "radar_frames_mb": 300.0,
                "tile_cache_mb": 12.5,
                "hrrr_grid_mb": 40.0,
                # Not a MB value -> must not become a breakdown sample.
                "coord_store_entries": 5,
                "other_mb": 100.0,
            },
        },
        "frames": {
            "count": 12,
            "max": 12,
            "latest": 1700000000,
            "oldest": 1699993400,
            "latest_age_seconds": 120,
            "per_region": {"USCOMP": 12, "CACOMP": 6},
        },
        "tile_cache": {
            "entries": 42,
            "used_mb": 12.5,
            "max_mb": 128,
            "geometry_entries": 10,
            "geometry_bytes": 1024,
            "present_entries": 20,
            "present_bytes": 2048,
            "overlay_entries": 5,
            "overlay_bytes": 512,
            "window_entries": 5,
            "window_bytes": 512,
            "satellite_entries": 2,
            "satellite_bytes": 256,
        },
        # IFS-style grid block (reference_time + timesteps, no enabled).
        "ecmwf_grid": {
            "loaded": True,
            "reference_time": "2026-11-08T00:00:00+00:00",
            "timesteps": 24,
        },
        # Chain-source grid block (enabled + latest_run + frames).
        "hrrr_grid": {
            "enabled": True,
            "loaded": True,
            "latest_run": "2026-11-08T06:00:00+00:00",
            "frames": 12,
        },
        # Unloaded grid block.
        "arome_polyn_grid": {"enabled": False, "loaded": False},
        "nwp_chain": {"sources": ["hrrr", "ecmwf"]},
        "nowcast": {
            "enabled": True,
            "arrow_flow_enabled": True,
            "flows": 3,
            "nwp_flow": True,
            "frames": [1700000600, 1700001200],
            "count": 2,
        },
        "satellite": {
            "enabled": True,
            "channels": {
                "gmgsi_lw_grid": {"loaded": True, "frames": 12, "latest": 1},
                "gmgsi_vis_grid": {"loaded": False, "frames": 0, "latest": None},
            },
        },
        "enabled_regions": ["USCOMP", "CACOMP"],
        "sources": {
            "na_source": "mrms_fallback",
            "ca_source": "msc",
            "cacomp_msc_blending": True,
        },
        "radar_cache": {"enabled": True, "files": 10, "used_mb": 250.0},
        "coord_caches": {
            "max_size": 512,
            "caches": {
                "tile_bounds": {
                    "entries": 10,
                    "hits": 90,
                    "misses": 10,
                    "hit_ratio": 0.9,
                },
            },
            "store": {
                "hits": 7,
                "misses": 3,
                "publishes": 2,
                "entries": 5,
                "bytes": 1048576,
            },
        },
        "tile_requests": {
            "enabled": True,
            "tracked_tiles": 30,
            "total_requests": 1000,
            "hot_tiles": 3,
            "by_zoom": {
                7: {"tiles": 20, "requests": 800},
                8: {"tiles": 10, "requests": 200},
            },
            "top": [{"z": 7, "x": 12, "y": 34, "count": 99}],
            "fast_path": {
                "total": 500,
                "by_reason": {"transparent": 400, "precip_mask": 100},
            },
            "cache": {"hits": 900, "misses": 100, "hit_rate": 0.9},
            "latency": {
                "avg_request_ms": 12.5,
                "avg_compute_ms": 8.0,
                "avg_present_ms": 4.5,
            },
        },
        "alerts": {"enabled": False},
        "mcp": {"enabled": True, "mounted": True},
        "storm_cells": {
            "enabled": True,
            "count": 4,
            "last_updated": 1700000000,
            "per_region": {"USCOMP": 4},
        },
        "lightning": {
            "enabled": True,
            "points": 500,
            "version": 1700000001,
            "ceiling_trimmed_total": 0,
        },
        "websocket": {
            "enabled": True,
            "connections": 2,
            "topics": {"weather-maps": 2},
        },
    }


class TestRenderPrometheus:
    def test_populated_payload(self):
        text = render_prometheus(_payload())

        # Status / uptime / version.
        assert "librewxr_health_status 1" in text
        assert "# TYPE librewxr_uptime_seconds gauge" in text
        assert 'librewxr_version_info{version="0.9.0"} 1' in text

        # Cluster aggregation.
        assert "librewxr_cluster_workers_reporting 2" in text
        assert 'librewxr_cluster_workers_rss_mb{stat="sum"} 900.0' in text
        assert 'librewxr_cluster_container_mb{kind="anon"} 100.0' in text
        assert 'librewxr_cluster_container_mb{kind="current"} 160.0' in text
        assert "librewxr_cluster_container_limit_mb 4096.0" in text
        assert "librewxr_cluster_cache_hits 900" in text

        # NWP grids: IFS shape, chain shape, unloaded shape.
        assert 'librewxr_nwp_grid_loaded{grid="ecmwf_grid"} 1' in text
        assert 'librewxr_nwp_grid_timesteps{grid="ecmwf_grid"} 24' in text
        assert 'librewxr_nwp_grid_enabled{grid="ecmwf_grid"}' not in text
        assert 'librewxr_nwp_grid_enabled{grid="hrrr_grid"} 1' in text
        assert 'librewxr_nwp_grid_frames{grid="hrrr_grid"} 12' in text
        assert 'librewxr_nwp_grid_timesteps{grid="hrrr_grid"}' not in text
        assert 'librewxr_nwp_grid_enabled{grid="arome_polyn_grid"} 0' in text
        assert 'librewxr_nwp_grid_loaded{grid="arome_polyn_grid"} 0' in text

        # Memory breakdown skips non-MB keys.
        assert (
            'librewxr_memory_breakdown_mb{component="radar_frames"} 300.0'
            in text
        )
        assert (
            'librewxr_memory_breakdown_mb{component="coord_store_entries"}'
            not in text
        )

        # Tile-request labeled families.
        assert 'librewxr_tile_requests_zoom_tiles{zoom="7"} 20' in text
        assert 'librewxr_tile_requests_zoom_requests{zoom="8"} 200' in text
        assert 'librewxr_tile_requests_top{z="7",x="12",y="34"} 99' in text
        assert (
            'librewxr_tile_requests_fast_path{reason="transparent"} 400'
            in text
        )
        assert "librewxr_tile_requests_fast_path_total 500" in text

        # Disabled section emits ONLY the _enabled 0 gauge.
        assert "librewxr_alerts_enabled 0" in text
        assert "librewxr_alerts_count" not in text
        assert "librewxr_alerts_ingest_ok" not in text

        # Null hit ratios are omitted, non-null ones present.
        assert (
            'librewxr_coord_cache_hit_ratio{cache="empty_cache"}' not in text
        )
        assert (
            'librewxr_coord_cache_hit_ratio{cache="tile_bounds"} 0.9'
            in text
        )

    def test_cluster_memory_pss_and_pipeline_series(self):
        payload = _payload()
        payload["cluster"]["workers"] = [
            {"pid": 101, "role": "render", "rss_mb": 300.0, "pss_mb": 200.0,
             "tile_cache_mb": 8.0, "age_s": 3.0},
            {"pid": 102, "role": "render", "rss_mb": 340.0, "pss_mb": None,
             "tile_cache_mb": 9.0, "age_s": 4.0},
        ]
        payload["cluster"]["workers_pss_mb"] = {
            "sum": 200.0, "min": 150.0, "max": 200.0,
        }
        payload["cluster"]["pipeline"] = {
            "reporting": True,
            "pid": 4242,
            "rss_mb": 250.0,
            "pss_mb": 180.0,
            "age_s": 2.0,
        }
        text = render_prometheus(payload)
        assert 'librewxr_cluster_workers_pss_mb{stat="sum"} 200.0' in text
        assert 'librewxr_cluster_workers_pss_mb{stat="min"} 150.0' in text
        assert 'librewxr_cluster_workers_pss_mb{stat="max"} 200.0' in text
        assert "librewxr_cluster_pipeline_reporting 1" in text
        assert "librewxr_cluster_pipeline_rss_mb 250.0" in text
        assert "librewxr_cluster_pipeline_pss_mb 180.0" in text
        # The per-worker list is deliberately not exported (pid churn).
        assert "librewxr_cluster_workers{" not in text
        assert 'pid="' not in text

    def test_cluster_container_current_omitted_when_absent(self):
        payload = _payload()
        del payload["cluster"]["memory"]["container"]["current_mb"]
        text = render_prometheus(payload)
        assert 'librewxr_cluster_container_mb{kind="current"}' not in text
        # The existing kinds and limit are unchanged.
        assert 'librewxr_cluster_container_mb{kind="anon"} 100.0' in text
        assert 'librewxr_cluster_container_mb{kind="file"} 50.0' in text
        assert 'librewxr_cluster_container_mb{kind="shmem"} 5.0' in text
        assert "librewxr_cluster_container_limit_mb 4096.0" in text

    def test_cluster_pss_absent_when_null(self):
        payload = _payload()
        payload["cluster"]["workers_pss_mb"] = None
        payload["cluster"]["pipeline"] = {"reporting": False}
        text = render_prometheus(payload)
        assert "librewxr_cluster_workers_pss_mb" not in text
        assert "librewxr_cluster_pipeline_reporting 0" in text
        assert "librewxr_cluster_pipeline_rss_mb" not in text
        assert "librewxr_cluster_pipeline_pss_mb" not in text

    def test_cluster_pipeline_pss_omitted_when_null(self):
        payload = _payload()
        payload["cluster"]["pipeline"] = {
            "reporting": True,
            "pid": 4242,
            "rss_mb": 250.0,
            "pss_mb": None,
            "age_s": 2.0,
        }
        text = render_prometheus(payload)
        assert "librewxr_cluster_pipeline_reporting 1" in text
        assert "librewxr_cluster_pipeline_rss_mb 250.0" in text
        assert "librewxr_cluster_pipeline_pss_mb" not in text

    def test_cluster_null_omits_cluster_metrics(self):
        payload = _payload()
        payload["cluster"] = None
        text = render_prometheus(payload)
        assert "librewxr_cluster_" not in text
        # The rest of the payload still renders.
        assert "librewxr_health_status 1" in text

    def test_degraded_status(self):
        payload = _payload()
        payload["status"] = "degraded"
        text = render_prometheus(payload)
        assert "librewxr_health_status 0" in text

    def test_unknown_top_level_key_is_nwp_grid(self):
        text = render_prometheus({
            "status": "ok",
            "mystery_grid": {"enabled": True, "loaded": True, "frames": 5},
        })
        assert 'librewxr_nwp_grid_loaded{grid="mystery_grid"} 1' in text
        assert 'librewxr_nwp_grid_enabled{grid="mystery_grid"} 1' in text
        assert 'librewxr_nwp_grid_frames{grid="mystery_grid"} 5' in text

    def test_label_values_are_escaped(self):
        text = render_prometheus({
            "status": "ok",
            "frames": {"per_region": {'a"b\\c\n': 3}},
        })
        assert 'librewxr_frames_per_region{region="a\\"b\\\\c\\n"} 3' in text

    def test_empty_and_malformed_payloads_never_raise(self):
        # An empty payload still emits the always-present status gauge.
        text = render_prometheus({})
        assert "librewxr_health_status 0" in text
        # Wrong-shaped sections are skipped, not crashed on.
        text = render_prometheus({
            "status": "ok",
            "frames": "nope",
            "cluster": 42,
            "memory": None,
            "tile_requests": [],
        })
        assert "librewxr_health_status 1" in text


def _wire_routes_globals(monkeypatch) -> None:
    """Wire minimal read-side globals, restored after the test.

    ``monkeypatch`` is required: ``tests/test_api.py`` builds its app from
    these same module globals at import time, so restoring them keeps that
    module's harness intact when both files run in one session.
    """
    monkeypatch.setattr(routes, "frame_store", FrameStore(max_frames=12))
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "nwp_grids", {})
    monkeypatch.setattr(routes, "satellite_grids", {})
    monkeypatch.setattr(routes, "nowcast_store", None)
    monkeypatch.setattr(routes, "start_time", time.time())
    monkeypatch.setattr(routes, "enabled_regions", ["USCOMP"])


def test_metrics_endpoint_serves_prometheus(monkeypatch):
    _wire_routes_globals(monkeypatch)

    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(metrics_router)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert (
            resp.headers["content-type"]
            == "text/plain; version=0.0.4; charset=utf-8"
        )
        assert "librewxr_health_status" in resp.text
        # /health coexists unchanged next to the exporter.
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] in {"ok", "degraded"}
