# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PSS / per-worker-list / pipeline coverage for the /health cluster block.

Extends the pulse-fixture style of ``test_api.py`` ``TestHealthCluster``:
fake worker pulses and a pipeline pulse are written into a temp cache dir,
the route globals are wired, and the additive cluster keys are asserted
against the frozen dashboard-facing shapes.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.data.worker_pulse import (
    PULSE_MAX_AGE_S,
    write_pipeline_pulse,
)
from librewxr.tiles.cache import TileCache

_MB = 1024 * 1024


@pytest.fixture
def cluster_client(monkeypatch, tmp_path):
    """Minimal app + route globals + a temp cache dir for pulse scans."""
    monkeypatch.setattr(routes, "frame_store", FrameStore(max_frames=12))
    monkeypatch.setattr(routes, "tile_cache", TileCache(max_mb=10))
    monkeypatch.setattr(routes, "nwp_grids", {})
    monkeypatch.setattr(routes, "satellite_grids", {})
    monkeypatch.setattr(routes, "nowcast_store", None)
    monkeypatch.setattr(routes, "start_time", time.time())
    monkeypatch.setattr(routes, "enabled_regions", ["USCOMP"])
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path


def _live_pulse(pss_mb: float | None) -> dict:
    """Deterministic stand-in for this process's live collect_worker_pulse."""
    return {
        "pid": os.getpid(),
        "role": "render",
        "written_at": int(time.time()),
        "rss_bytes": 150 * _MB,
        "pss_mb": pss_mb,
        "tile_cache": {
            "entries": 2,
            "total_bytes": 5 * _MB,
            "max_bytes": 128 * _MB,
        },
    }


def _write_worker_pulse(
    cache_dir: Path,
    pid: int,
    payload: dict,
    age_s: float = 0.0,
) -> None:
    workers = cache_dir / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    path = workers / f"worker_{pid}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_s:
        stat = path.stat()
        os.utime(path, (stat.st_atime, time.time() - age_s))


def _render_payload(
    pid: int,
    *,
    rss_mb: float,
    pss_mb: float | None = None,
    tile_cache_mb: float = 0.0,
    role: str | None = "render",
) -> dict:
    payload = {
        "pid": pid,
        "written_at": int(time.time()),
        "rss_bytes": int(rss_mb * _MB),
        "tile_cache": {
            "entries": 1,
            "total_bytes": int(tile_cache_mb * _MB),
            "max_bytes": 128 * _MB,
        },
    }
    if role is not None:
        payload["role"] = role
    if pss_mb is not None:
        payload["pss_mb"] = pss_mb
    return payload


class TestClusterPerWorkerMemory:
    """cluster.workers / workers_pss_mb / cluster.pipeline frozen shapes."""

    def test_workers_list_pss_and_pipeline(self, cluster_client, monkeypatch):
        client, cache_dir = cluster_client
        monkeypatch.setattr(
            routes, "collect_worker_pulse", lambda: _live_pulse(pss_mb=None)
        )

        _write_worker_pulse(
            cache_dir,
            200001,
            _render_payload(200001, rss_mb=200.0, pss_mb=100.0, tile_cache_mb=10.0),
            age_s=5.0,
        )
        _write_worker_pulse(
            cache_dir,
            200002,
            _render_payload(200002, rss_mb=300.0, tile_cache_mb=20.0),
            age_s=7.0,
        )
        write_pipeline_pulse(cache_dir, {
            "pid": 200100,
            "role": "pipeline",
            "written_at": int(time.time()),
            "rss_mb": 50.0,
            "pss_mb": 40.0,
        })

        resp = client.get("/health")
        assert resp.status_code == 200
        cluster = resp.json()["cluster"]

        workers = cluster["workers"]
        # Live serving worker + the two fake render workers; the pipeline
        # pulse is NOT a render worker.
        assert len(workers) == 3
        assert cluster["workers_reporting"] == 3
        assert [entry["pid"] for entry in workers] == sorted(
            entry["pid"] for entry in workers
        )
        for entry in workers:
            assert set(entry) == {
                "pid", "role", "rss_mb", "pss_mb", "tile_cache_mb", "age_s",
            }
            assert entry["role"] == "render"

        by_pid = {entry["pid"]: entry for entry in workers}
        assert by_pid[os.getpid()]["pss_mb"] is None
        assert by_pid[os.getpid()]["tile_cache_mb"] == 5.0
        assert by_pid[os.getpid()]["age_s"] == 0.0
        assert by_pid[200001]["pss_mb"] == 100.0
        assert by_pid[200001]["tile_cache_mb"] == 10.0
        assert by_pid[200001]["rss_mb"] == 200.0
        assert 4.0 <= by_pid[200001]["age_s"] <= PULSE_MAX_AGE_S
        assert by_pid[200002]["pss_mb"] is None
        assert by_pid[200002]["tile_cache_mb"] == 20.0
        assert 6.0 <= by_pid[200002]["age_s"] <= PULSE_MAX_AGE_S

        # Only the non-null PSS values aggregate (the live worker and
        # 200002 report null).
        assert cluster["workers_pss_mb"] == {
            "sum": 100.0, "min": 100.0, "max": 100.0,
        }

        pipeline = cluster["pipeline"]
        assert pipeline["reporting"] is True
        assert pipeline["pid"] == 200100
        assert pipeline["rss_mb"] == 50.0
        assert pipeline["pss_mb"] == 40.0
        assert 0.0 <= pipeline["age_s"] <= PULSE_MAX_AGE_S

    def test_all_null_pss_yields_null_aggregate(self, cluster_client, monkeypatch):
        client, cache_dir = cluster_client
        monkeypatch.setattr(
            routes, "collect_worker_pulse", lambda: _live_pulse(pss_mb=None)
        )
        # An older pulse with neither role nor pss_mb: defaults to render
        # with a null PSS and a zero tile-cache footprint.
        _write_worker_pulse(cache_dir, 200003, {
            "pid": 200003,
            "written_at": int(time.time()),
            "rss_bytes": 64 * _MB,
        })

        cluster = client.get("/health").json()["cluster"]
        assert cluster["workers_pss_mb"] is None
        by_pid = {entry["pid"]: entry for entry in cluster["workers"]}
        assert by_pid[200003]["role"] == "render"
        assert by_pid[200003]["pss_mb"] is None
        assert by_pid[200003]["tile_cache_mb"] == 0.0

    def test_no_pipeline_pulse_reports_false(self, cluster_client, monkeypatch):
        client, _ = cluster_client
        monkeypatch.setattr(
            routes, "collect_worker_pulse", lambda: _live_pulse(pss_mb=10.0)
        )
        cluster = client.get("/health").json()["cluster"]
        assert cluster["pipeline"] == {"reporting": False}
        assert cluster["workers_pss_mb"] == {
            "sum": 10.0, "min": 10.0, "max": 10.0,
        }

    def test_stale_pipeline_pulse_reports_false(self, cluster_client, monkeypatch):
        client, cache_dir = cluster_client
        monkeypatch.setattr(
            routes, "collect_worker_pulse", lambda: _live_pulse(pss_mb=None)
        )
        write_pipeline_pulse(cache_dir, {
            "pid": 200100,
            "role": "pipeline",
            "written_at": int(time.time()),
            "rss_mb": 50.0,
            "pss_mb": 40.0,
        })
        path = cache_dir / "workers" / "pipeline.json"
        stat = path.stat()
        os.utime(path, (stat.st_atime, time.time() - PULSE_MAX_AGE_S - 30))

        cluster = client.get("/health").json()["cluster"]
        assert cluster["pipeline"] == {"reporting": False}

    def test_cluster_null_path_unchanged(self, cluster_client, monkeypatch):
        client, _ = cluster_client

        def boom():
            raise OSError("pulse scan failed")

        monkeypatch.setattr(routes, "read_worker_pulses", boom)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["cluster"] is None


class TestClusterContainerMemory:
    """cluster.memory.container forwards the monitor's MB split verbatim,
    including the additive kernel-usage ``current_mb``."""

    def test_container_current_mb_flows_to_health(self, cluster_client, monkeypatch):
        client, _ = cluster_client
        container = {
            "anon_mb": 400,
            "file_mb": 900,
            "shmem_mb": 120,
            "current_mb": 1200.5,
            "limit_mb": 1318,
        }
        monkeypatch.setattr(
            routes,
            "memory_monitor",
            SimpleNamespace(cgroup_memory_mb=container),
        )
        cluster = client.get("/health").json()["cluster"]
        assert cluster["memory"]["container"] == container

    def test_container_absent_stays_null(self, cluster_client, monkeypatch):
        client, _ = cluster_client
        monkeypatch.setattr(routes, "memory_monitor", None)
        cluster = client.get("/health").json()["cluster"]
        assert cluster["memory"]["container"] is None
