# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Snapshot puller for a render node without a data pipeline.

The counterpart of the pipeline's snapshot publisher (``LIBREWXR_SNAPSHOT_SYNC_URL``):
polls the artifact store, brings each new snapshot and the files it
names into ``LIBREWXR_CACHE_DIR``, and writes ``state.json`` last, so
render-only workers sharing that directory serve it exactly as they would
a local pipeline's.  It also does the per-node maintenance the pipeline
would otherwise own: page-cache priming of fresh frames and pruning of
the shared coordinate and tile stores.

Run with::

    python -m librewxr.snapshot_pull

next to ``python -m librewxr.main`` with ``LIBREWXR_RENDER_ONLY=1`` on the
same host (see docs/network-split.md).
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

from librewxr.config import resolve_cache_dir, settings
from librewxr.data.master_state import load_state
from librewxr.data.pagecache import prime_fresh_memmaps
from librewxr.data.snapshot_sync import SnapshotPuller
from librewxr.logging_setup import setup_logging
from librewxr.tiles.coordinates import prune_shared_coord_store
from librewxr.tiles.shared_tile_store import SharedTileStore

logger = logging.getLogger(__name__)


async def run_puller() -> None:
    if not settings.snapshot_sync_url:
        raise SystemExit("LIBREWXR_SNAPSHOT_SYNC_URL is not set; nothing to pull")
    cache_dir = resolve_cache_dir(settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    puller = SnapshotPuller(
        cache_dir,
        settings.snapshot_sync_url,
        endpoint_url=settings.snapshot_sync_endpoint_url,
        workers=settings.snapshot_sync_workers,
    )
    # Same budget resolution as the render workers / pipeline (main.py)
    mb = settings.shared_tile_store_mb
    mb = 2048 if mb is None else mb
    shared_tiles = SharedTileStore(cache_dir, max_mb=mb) if mb > 0 else None
    logger.info(
        "Snapshot puller starting (store=%s, cache_dir=%s, interval=%.0fs)",
        settings.snapshot_sync_url, cache_dir, settings.snapshot_pull_interval,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    while not stop_event.is_set():
        try:
            applied = await asyncio.to_thread(puller.pull)
        except Exception:
            # A store or transfer error (the client libraries raise their
            # own types) leaves the local cache as it was; the next poll
            # retries from the manifest.
            logger.exception("Snapshot pull failed")
            applied = False
        if applied:
            # The housekeeping a pipeline would do on this host each cycle
            if settings.pagecache_prime_enabled:
                payload = load_state(cache_dir)
                if payload is not None:
                    await asyncio.to_thread(prime_fresh_memmaps, payload, cache_dir)
                # This coroutine lives for the whole process, so keeping the
                # parsed snapshot here would hold hundreds of MB (it embeds every
                # alert polygon) through the next pull, which parses it again.
                del payload
            await asyncio.to_thread(prune_shared_coord_store)
            if shared_tiles is not None:
                await asyncio.to_thread(shared_tiles.prune)
        try:
            await asyncio.wait_for(stop_event.wait(), settings.snapshot_pull_interval)
        except asyncio.TimeoutError:
            pass
    logger.info("Snapshot puller stopped")


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run_puller())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
