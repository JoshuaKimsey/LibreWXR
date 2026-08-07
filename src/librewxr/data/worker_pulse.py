# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Cluster-wide worker pulse files for the /health aggregation.

Multi-worker deployments run N uvicorn worker processes in one renderer
container that share a listen socket, so individual workers can't be
polled over HTTP.  They DO share the cache volume, so each worker
periodically writes a tiny pid-unique JSON pulse under
``<cache_dir>/workers/worker_<pid>.json``; any worker's ``/health``
handler scans those files (mtime-filtered) to aggregate the whole
cluster.

The idiom mirrors the rest of the project: pid-unique tmp files +
``os.replace`` for atomic publication, mtime filtering instead of
locks, best-effort dead-worker sweep.  No locks, no readers blocking
writers, no writers blocking readers — a scan either sees a file or
doesn't, and a torn write is impossible because the rename is atomic.
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# How often each worker writes its pulse (jittered at the loop level so
# N workers don't write in lockstep).
PULSE_INTERVAL_S = 15.0
# Pulses whose mtime is older than this are excluded from aggregation
# (a worker that crashed or stalled just stops refreshing its file).
PULSE_MAX_AGE_S = 60.0
# Pulses older than this are swept off disk during scans — the worker is
# presumed dead and its file is just litter.
PULSE_STALE_S = 600.0

_WORKERS_SUBDIR = "workers"


def write_worker_pulse(cache_dir: Path, payload: dict) -> None:
    """Atomically publish ``payload`` as this process's pulse file.

    Writes ``<cache_dir>/workers/worker_<pid>.json`` via a pid+uuid tmp
    file and ``os.replace`` (concurrent readers see either the previous
    file or the new one — never a partial write).  Best-effort and never
    raises: a broken cache dir just costs this worker its pulse.
    """
    try:
        workers_dir = Path(cache_dir) / _WORKERS_SUBDIR
        workers_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        final = workers_dir / f"worker_{pid}.json"
        tmp = workers_dir / f"worker_{pid}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, final)
    except Exception:
        logger.exception("Failed to write worker pulse under %s", cache_dir)


def read_worker_pulses(
    cache_dir: Path, max_age_s: float = PULSE_MAX_AGE_S
) -> list[dict]:
    """Read the live worker pulses under ``<cache_dir>/workers/``.

    Returns the parsed payloads of every ``*.json`` file whose mtime is
    within ``max_age_s`` seconds, newest-file-order sorted (files are
    sorted by name so the ordering is stable, not meaningful).  Files
    older than :data:`PULSE_STALE_S` are unlinked during the scan (the
    dead-worker sweep; best-effort).  Unparseable or non-dict files are
    skipped and logged.  Returns ``[]`` on any directory-level failure.
    """
    workers_dir = Path(cache_dir) / _WORKERS_SUBDIR
    try:
        paths = sorted(workers_dir.glob("*.json"))
    except OSError:
        return []
    now = time.time()
    pulses: list[dict] = []
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        if age > PULSE_STALE_S:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        if age > max_age_s:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("Skipping unparseable worker pulse %s", path)
            continue
        if isinstance(data, dict):
            pulses.append(data)
        else:
            logger.debug("Skipping non-dict worker pulse %s", path)
    return pulses
