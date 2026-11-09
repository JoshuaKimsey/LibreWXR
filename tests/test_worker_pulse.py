# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the cluster worker-pulse file helpers.

Covers the atomic write / mtime-filtered read round-trip, the freshness
filter, the dead-worker sweep, corrupt-file tolerance, and the missing-
directory case.  The pulse loop wiring in ``main.py`` and the /health
``cluster`` aggregation in ``routes.py`` are covered by ``test_api.py``.
"""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from librewxr.data.worker_pulse import (
    PULSE_MAX_AGE_S,
    PULSE_STALE_S,
    parse_pss_mb,
    read_pipeline_pulse,
    read_pss_mb,
    read_worker_pulses,
    write_pipeline_pulse,
    write_worker_pulse,
)

pytestmark = pytest.mark.store


def _workers_dir(cache_dir: Path) -> Path:
    return cache_dir / "workers"


def test_write_read_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    payload = {"pid": 1234, "written_at": int(time.time()), "rss_bytes": 1_000_000}
    write_worker_pulse(cache_dir, payload)
    pulses = read_worker_pulses(cache_dir)
    assert pulses == [payload]
    # The file is published at the pid-suffixed path (the writer's own
    # pid, not the payload's) and the tmp file is gone.
    assert (_workers_dir(cache_dir) / f"worker_{os.getpid()}.json").exists()
    assert list(_workers_dir(cache_dir).glob("*.tmp")) == []


def test_read_returns_empty_when_no_pulses(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    assert read_worker_pulses(cache_dir) == []


def test_read_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert read_worker_pulses(tmp_path / "does-not-exist") == []


def test_freshness_filter_excludes_old_mtime(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_worker_pulse(cache_dir, {"pid": os.getpid(), "rss_bytes": 1})
    fake_other = _workers_dir(cache_dir) / "worker_999999.json"
    fake_other.write_text(json.dumps({"pid": 999999, "rss_bytes": 2}))

    # Age the real pulse beyond max_age (but below the stale sweep) — it
    # must be excluded from the read while staying on disk.
    real = _workers_dir(cache_dir) / f"worker_{os.getpid()}.json"
    st = real.stat()
    old_mtime = time.time() - PULSE_MAX_AGE_S - 30
    os.utime(real, (st.st_atime, old_mtime))

    pulses = read_worker_pulses(cache_dir)
    assert [p["pid"] for p in pulses] == [999999]
    assert real.exists()


def test_freshness_keeps_recent_mtime(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_worker_pulse(cache_dir, {"pid": os.getpid(), "rss_bytes": 1})
    pulses = read_worker_pulses(cache_dir)
    assert [p["pid"] for p in pulses] == [os.getpid()]


def test_stale_files_unlinked_during_read(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_worker_pulse(cache_dir, {"pid": os.getpid(), "rss_bytes": 1})
    workers_dir = _workers_dir(cache_dir)
    stale = workers_dir / "worker_999999.json"
    stale.write_text(json.dumps({"pid": 999999, "rss_bytes": 2}))
    st = stale.stat()
    os.utime(stale, (st.st_atime, time.time() - PULSE_STALE_S - 60))

    pulses = read_worker_pulses(cache_dir)
    assert all(p["pid"] != 999999 for p in pulses)
    # Dead worker's file is swept from disk; the live one stays.
    assert not stale.exists()
    assert (workers_dir / f"worker_{os.getpid()}.json").exists()


def test_corrupt_json_skipped_without_failing_others(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_worker_pulse(cache_dir, {"pid": os.getpid(), "rss_bytes": 1})
    workers_dir = _workers_dir(cache_dir)
    (workers_dir / "worker_999998.json").write_text("{not valid json")
    (workers_dir / "worker_999999.json").write_text(
        json.dumps({"pid": 999999, "rss_bytes": 2}),
    )

    pulses = read_worker_pulses(cache_dir)
    assert len(pulses) == 2
    assert any(p["pid"] == os.getpid() for p in pulses)
    assert any(p["pid"] == 999999 for p in pulses)


def test_non_dict_json_is_skipped(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    workers_dir = _workers_dir(cache_dir)
    workers_dir.mkdir(parents=True)
    (workers_dir / "worker_999999.json").write_text("[1, 2, 3]")
    assert read_worker_pulses(cache_dir) == []


def test_write_never_raises_on_unusable_cache_dir(tmp_path: Path) -> None:
    # A file where the workers dir would go — mkdir fails, write must
    # swallow it and return normally.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    write_worker_pulse(blocker, {"pid": 1, "rss_bytes": 1})
    assert not (blocker / "workers").exists()


class TestPssParsing:
    """The smaps_rollup PSS parser feeding the pulse payloads."""

    def test_parses_pss_line_into_mb(self) -> None:
        text = (
            "55e0: 00000000-00000000 ---p 00000000 00:00 0\n"
            "Rss:               40960 kB\n"
            "Pss:               20480 kB\n"
            "Pss_Anon:          10240 kB\n"
        )
        assert parse_pss_mb(text) == 20.0

    def test_missing_pss_line_returns_none(self) -> None:
        assert parse_pss_mb("Rss: 4096 kB\n") is None

    def test_garbage_pss_line_returns_none(self) -> None:
        assert parse_pss_mb("Pss: not-a-number kB\n") is None
        assert parse_pss_mb("Pss:\n") is None

    def test_read_pss_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_pss_mb(tmp_path / "does-not-exist") is None

    def test_read_pss_garbage_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "smaps_rollup"
        path.write_text("complete nonsense")
        assert read_pss_mb(path) is None


class TestPipelinePulse:
    """The single pipeline.json pulse and its reader."""

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        payload = {
            "pid": 4321,
            "role": "pipeline",
            "written_at": int(time.time()),
            "rss_mb": 123.4,
            "pss_mb": 100.0,
        }
        write_pipeline_pulse(tmp_path, payload)
        record = read_pipeline_pulse(tmp_path)
        assert record is not None
        data, age = record
        assert data == payload
        assert 0.0 <= age < PULSE_MAX_AGE_S
        # It is not picked up as a render-worker pulse.
        assert read_worker_pulses(tmp_path) == []

    def test_missing_pipeline_pulse_returns_none(self, tmp_path: Path) -> None:
        assert read_pipeline_pulse(tmp_path) is None

    def test_stale_pipeline_pulse_returns_none(self, tmp_path: Path) -> None:
        write_pipeline_pulse(tmp_path, {"pid": 4321, "role": "pipeline"})
        path = _workers_dir(tmp_path) / "pipeline.json"
        st = path.stat()
        os.utime(path, (st.st_atime, time.time() - PULSE_MAX_AGE_S - 30))
        assert read_pipeline_pulse(tmp_path) is None

    async def test_pulse_loop_publishes_live_pulse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The loop resolves PULSE_INTERVAL_S off the data_pipeline module at
        # call time, so patching there keeps this test fast.
        from librewxr.data_pipeline import _pipeline_pulse_loop

        monkeypatch.setattr("librewxr.data_pipeline.PULSE_INTERVAL_S", 0.2)
        cache_dir = tmp_path / "cache"
        stop = asyncio.Event()
        task = asyncio.create_task(_pipeline_pulse_loop(stop, cache_dir))
        pulse_path = _workers_dir(cache_dir) / "pipeline.json"
        try:
            deadline = time.monotonic() + 5.0
            while not pulse_path.exists():
                assert time.monotonic() < deadline, (
                    "pipeline pulse was never written"
                )
                await asyncio.sleep(0.02)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)

        payload = json.loads(pulse_path.read_text(encoding="utf-8"))
        assert payload["role"] == "pipeline"
        assert isinstance(payload["pid"], int)
        assert isinstance(payload["rss_mb"], float)
        assert payload["pss_mb"] is None or isinstance(payload["pss_mb"], float)
        assert isinstance(payload["written_at"], int)
        assert not task.cancelled()
        assert task.exception() is None
