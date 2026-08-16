# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Prime freshly written memmap files into the host page cache.

In the multi-worker split, render workers memory-map the same frame
files the pipeline writes and page-fault them in while holding the GIL;
on a slow backing disk those cold faults can stall a worker past
uvicorn's healthcheck timeout and get it SIGKILLed.  After each fetch
cycle the pipeline issues posix_fadvise(WILLNEED) for every memmap file
render workers will open, so the host page cache (shared between the
pipeline and renderer containers) already holds the frames by the time
any worker touches them.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_RECORD_NAME = "primed_memmaps.json"


def _collect_memmap_files(store_payload: dict) -> set[Path]:
    """All .dat paths referenced by one store's snapshot payload.

    Snapshot payloads list memmap arrays as [basename, dtype, shape]
    triples nested at varying depths, under one or more "memmap_dir"
    keys.  Pair each basename with every memmap_dir found inside the
    same store subtree and dedupe.  One deviation: satellite (GMGSI)
    stores snapshot cache_root + channel + plain timestamps instead of a
    memmap_dir — their frame files live under
    <cache_root>/gmgsi/<layout>/<channel>/ and are resolved separately.
    """
    found: set[Path] = set()

    def dat_basenames(node) -> list[str]:
        names: list[str] = []
        if isinstance(node, dict):
            for value in node.values():
                names.extend(dat_basenames(value))
        elif isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and isinstance(node[0], str)
                and node[0].endswith(".dat")
            ):
                names.append(node[0])
            else:
                for value in node:
                    names.extend(dat_basenames(value))
        return names

    def find_dirs(node) -> None:
        if isinstance(node, dict):
            dir_value = node.get("memmap_dir")
            if dir_value:
                base = Path(dir_value)
                for basename in dat_basenames(node):
                    found.add(base / basename)
            # Satellite (GMGSI) stores deviate: the snapshot carries
            # cache_root + channel + plain timestamps (no memmap_dir), and
            # the memmap files sit at
            # <cache_root>/gmgsi/<layout_version>/<channel>/frame_<ts>.dat.
            # Glob the layout dirs so a layout-version bump can't silently
            # stop priming the frames.
            cache_root = node.get("cache_root")
            channel = node.get("channel")
            timestamps = node.get("timestamps")
            if cache_root and channel and isinstance(timestamps, list):
                for layout_dir in Path(cache_root).glob("gmgsi/*"):
                    channel_dir = layout_dir / str(channel)
                    if channel_dir.is_dir():
                        for ts in timestamps:
                            if isinstance(ts, int):
                                found.add(channel_dir / f"frame_{ts}.dat")
            for value in node.values():
                find_dirs(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                find_dirs(value)

    find_dirs(store_payload)
    return found


def prime_fresh_memmaps(snapshot_payload: dict, cache_dir: str | Path) -> int:
    """Advise the kernel to preload freshly written memmap files.

    Walks the master-state snapshot, finds every .dat memmap file render
    workers will open, and issues posix_fadvise(WILLNEED) for any file not
    yet primed at its current mtime. Priming state persists in
    <cache_dir>/primed_memmaps.json. Never raises; returns bytes primed.
    """
    record_path = Path(cache_dir) / _RECORD_NAME
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {}
    if not hasattr(os, "posix_fadvise"):
        return 0
    total = 0
    for store_payload in snapshot_payload.get("stores", {}).values():
        if not isinstance(store_payload, dict):
            continue
        for path in _collect_memmap_files(store_payload):
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            if record.get(key) == st.st_mtime_ns:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                finally:
                    os.close(fd)
                record[key] = st.st_mtime_ns
                total += st.st_size
            except OSError:
                continue
    try:
        tmp = record_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, record_path)
    except OSError:
        pass
    return total
