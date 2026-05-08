# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Cross-process state snapshot for the multi-worker tile-server split.

The data pipeline writes a single ``state.json`` file under the shared
cache volume after each fetch cycle.  Render-only tile-server workers
poll its mtime, re-read it on change, and call ``apply_state`` to
refresh every store in place — existing references stay valid so
in-flight renders are never interrupted.

Format::

    {
      "version": 1,
      "written_at": 1712345600,
      "stores": {
        "frame_store":  { ... __getstate__ output ... },
        "ecmwf_grid":   { ... },
        "hrrr_grid":    { ... },
        ...
      }
    }

Stores whose value is ``None`` (disabled by config) are skipped.  Stores
present in the snapshot but absent from the consumer's ``stores`` dict
are silently ignored, so a tile-server worker can be started with a
subset of stores enabled.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"
STATE_VERSION = 1


def dump_state(stores: dict[str, Any], cache_dir: Path) -> Path:
    """Atomically write a snapshot of every store's ``__getstate__`` to disk.

    The write goes to ``<cache_dir>/.state.json.tmp`` first and is then
    atomically renamed to ``state.json``.  Concurrent readers either see
    the old file (if they read before the rename) or the new file
    (after) — never a partial write.

    Args:
        stores: mapping of store name → store object.  Values that are
            ``None`` are skipped.  Each non-None value must implement
            ``__getstate__()`` returning a JSON-serialisable dict.
        cache_dir: directory shared with the tile-server workers.
            ``state.json`` is written at the top level of this directory.

    Returns:
        The path to the newly-written ``state.json``.
    """
    payload: dict[str, Any] = {
        "version": STATE_VERSION,
        "written_at": int(time.time()),
        "stores": {},
    }
    for name, obj in stores.items():
        if obj is None:
            continue
        # Python 3.11+ adds a default ``object.__getstate__`` to every
        # class.  ``hasattr`` therefore can't distinguish stores that
        # explicitly opt in from arbitrary objects — but the default
        # returns ``None`` for objects with no instance state, so we
        # filter on the result instead.
        try:
            store_state = obj.__getstate__()
        except Exception:
            logger.exception("Failed to serialise store %r, skipping", name)
            continue
        if store_state is None:
            logger.warning(
                "Store %r returned no state, skipping in state.json", name,
            )
            continue
        payload["stores"][name] = store_state

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / STATE_FILENAME
    tmp = cache_dir / f".{STATE_FILENAME}.tmp"
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(tmp, final)
    logger.debug(
        "Wrote state.json: %d store(s) → %s", len(payload["stores"]), final,
    )
    return final


def load_state(cache_dir: Path) -> dict[str, Any] | None:
    """Read ``state.json`` from ``cache_dir``.

    Returns the parsed payload (with ``version`` / ``written_at`` /
    ``stores`` keys) or ``None`` if the file is absent.  Raises if the
    file exists but is malformed — callers should let those propagate
    so the worker fails loudly on corruption.
    """
    path = Path(cache_dir) / STATE_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STATE_VERSION:
        logger.warning(
            "state.json version mismatch: file has %r, expected %d",
            version, STATE_VERSION,
        )
    return payload


def apply_state(payload: dict[str, Any], stores: dict[str, Any]) -> list[str]:
    """Call ``__setstate__`` on every matching store in ``stores``.

    Args:
        payload: parsed result of :func:`load_state`.
        stores: mapping of store name → store object.  Stores whose
            value is ``None`` are skipped (e.g. disabled by config).
            Stores present in ``payload["stores"]`` but absent (or
            ``None``) here are silently ignored.

    Returns:
        List of store names that were successfully refreshed.
    """
    refreshed: list[str] = []
    snapshot = payload.get("stores", {})
    for name, obj in stores.items():
        if obj is None:
            continue
        store_state = snapshot.get(name)
        if store_state is None:
            continue
        try:
            obj.__setstate__(store_state)
            refreshed.append(name)
        except Exception:
            logger.exception("Failed to apply state for store %r", name)
    return refreshed


def state_mtime(cache_dir: Path) -> float | None:
    """Return the modification time of ``state.json`` (or ``None``).

    Used by tile-server workers to poll for changes without re-reading
    and re-parsing the file every tick.
    """
    path = Path(cache_dir) / STATE_FILENAME
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None
