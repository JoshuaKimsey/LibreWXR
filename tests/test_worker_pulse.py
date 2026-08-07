# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the cluster worker-pulse file helpers.

Covers the atomic write / mtime-filtered read round-trip, the freshness
filter, the dead-worker sweep, corrupt-file tolerance, and the missing-
directory case.  The pulse loop wiring in ``main.py`` and the /health
``cluster`` aggregation in ``routes.py`` are covered by ``test_api.py``.
"""
import json
import os
import time
from pathlib import Path

import pytest

from librewxr.data.worker_pulse import (
    PULSE_MAX_AGE_S,
    PULSE_STALE_S,
    read_worker_pulses,
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
