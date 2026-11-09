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

The data pipeline writes a single ``<cache_dir>/workers/pipeline.json``
on the same cadence so the cluster block also covers the fetch process
(when it shares this host's cache dir).  Worker scans deliberately skip
that file: it is not a render worker pulse.

Pulse payloads carry ``role`` (``"render"`` / ``"pipeline"``) and
``pss_mb`` (proportional set size, the true-unique-memory view from
``/proc/self/smaps_rollup``; ``null`` where unavailable).  Older pulses
without those keys are still readable — callers default the role to
``"render"`` and treat a missing PSS as ``null``.

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
_PIPELINE_PULSE_NAME = "pipeline.json"

# /proc/self/smaps_rollup is the single-read proportional-set-size source.
_SMAPS_ROLLUP_PATH = Path("/proc/self/smaps_rollup")


class WorkerPulse(dict):
    """A parsed pulse payload plus the age (seconds) of its file.

    A ``dict`` subclass so callers that only care about the payload keep
    working unchanged (equality with the plain parsed JSON is preserved);
    the scanner sets :attr:`age_s` from the file mtime for the /health
    per-worker list.  The live in-process payload is a plain dict and
    defaults to age 0.0 via ``getattr``.
    """

    age_s: float = 0.0


def parse_pss_mb(smaps_rollup_text: str) -> float | None:
    """Parse the ``Pss:`` line of an smaps_rollup dump into MB.

    ``Pss: 20480 kB`` -> ``20.0``.  Returns ``None`` when the line is
    absent or malformed.  Never raises.
    """
    for line in smaps_rollup_text.splitlines():
        if not line.startswith("Pss:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return float(parts[1]) / 1024.0
        except ValueError:
            return None
    return None


def read_pss_mb(path: Path = _SMAPS_ROLLUP_PATH) -> float | None:
    """This process's proportional set size in MB, or None.

    Proportional set size (PSS) de-double-counts shared pages: a page
    mapped by N processes contributes 1/N to each, so summing PSS across
    worker processes approximates the true unique memory footprint (unlike
    RSS, which counts every shared memmap page in full in every worker).
    Best-effort: a missing ``/proc`` (non-Linux), permissions failure, or
    garbage content yields ``None`` and never raises.
    """
    try:
        return parse_pss_mb(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _write_pulse_file(cache_dir: Path, filename: str, payload: dict) -> None:
    """Atomically publish ``payload`` as ``<cache_dir>/workers/<filename>``.

    Writes via a pid+uuid tmp file and ``os.replace`` (concurrent readers
    see either the previous file or the new one — never a partial write).
    Best-effort and never raises: a broken cache dir just costs this
    process its pulse.
    """
    try:
        workers_dir = Path(cache_dir) / _WORKERS_SUBDIR
        workers_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        final = workers_dir / filename
        tmp = workers_dir / f"{filename}.{pid}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, final)
    except Exception:
        logger.exception("Failed to write pulse %s under %s", filename, cache_dir)


def write_worker_pulse(cache_dir: Path, payload: dict) -> None:
    """Atomically publish ``payload`` as this process's pulse file.

    Writes ``<cache_dir>/workers/worker_<pid>.json``.  Best-effort and
    never raises: a broken cache dir just costs this worker its pulse.
    """
    _write_pulse_file(cache_dir, f"worker_{os.getpid()}.json", payload)


def write_pipeline_pulse(cache_dir: Path, payload: dict) -> None:
    """Atomically publish the data pipeline's pulse file.

    Writes ``<cache_dir>/workers/pipeline.json`` (a single file: there is
    exactly one pipeline process per cache dir).  Same atomic
    tmp+``os.replace`` machinery as the worker pulses; never raises.
    """
    _write_pulse_file(cache_dir, _PIPELINE_PULSE_NAME, payload)


def read_worker_pulses(
    cache_dir: Path, max_age_s: float = PULSE_MAX_AGE_S
) -> list[dict]:
    """Read the live worker pulses under ``<cache_dir>/workers/``.

    Returns the parsed payloads (as :class:`WorkerPulse` dicts carrying
    ``age_s``) of every ``worker_*.json`` file whose mtime is within
    ``max_age_s`` seconds, in stable filename order.  The pipeline pulse
    (``pipeline.json``) is skipped — it is not a render worker.  Files
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
        if path.name == _PIPELINE_PULSE_NAME:
            continue
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
            # age_s is a scan-time attribute derived from the file mtime, never
            # a payload key; drop a writer-supplied one so it can't leak through.
            data.pop("age_s", None)
            pulse = WorkerPulse(data)
            pulse.age_s = age
            pulses.append(pulse)
        else:
            logger.debug("Skipping non-dict worker pulse %s", path)
    return pulses


def read_pipeline_pulse(
    cache_dir: Path, max_age_s: float = PULSE_MAX_AGE_S
) -> tuple[dict, float] | None:
    """Read the pipeline pulse if it exists and is fresh.

    Returns ``(payload, age_s)`` for ``<cache_dir>/workers/pipeline.json``
    when its mtime is within ``max_age_s``; ``None`` when it is missing,
    stale, unparseable, or non-dict.  Older files past
    :data:`PULSE_STALE_S` are unlinked (same dead-process sweep as the
    worker pulses).  Never raises.
    """
    path = Path(cache_dir) / _WORKERS_SUBDIR / _PIPELINE_PULSE_NAME
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    age = time.time() - mtime
    if age > PULSE_STALE_S:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if age > max_age_s:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug("Skipping unparseable pipeline pulse %s", path)
        return None
    if not isinstance(data, dict):
        logger.debug("Skipping non-dict pipeline pulse %s", path)
        return None
    return data, age

