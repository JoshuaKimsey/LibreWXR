# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    timestamp: int  # Unix timestamp
    regions: dict[str, np.ndarray] = field(default_factory=dict)


class FrameStore:
    """Frame store backed by memory-mapped files.

    Region arrays are written to disk and accessed via np.memmap,
    allowing the OS to manage physical RAM through the page cache.
    This reduces RSS and lets the kernel reclaim frame memory under
    pressure instead of triggering OOM kills.

    When ``cache_dir`` is given, memmap files are stored under
    ``<cache_dir>/radar/`` and survive process restarts.  Multiple
    processes can map the same files read-only to share the OS page
    cache — the foundation of the multi-worker tile-server split.
    """

    def __init__(self, max_frames: int = 12, cache_dir: Path | None = None):
        self._max_frames = max_frames
        self._frames: list[RadarFrame] = []
        # O(1) timestamp -> frame index, kept in sync with ``_frames`` on
        # every append / merge / eviction / reload / cleanup.
        self._by_ts: dict[int, RadarFrame] = {}
        # Per-timestamp content version: set to 1 when a new frame is
        # appended, bumped on every merge into an existing timestamp, and
        # removed on eviction.  Render workers diff this against the
        # previous state.json snapshot to decide which cached tiles stay
        # valid across fetch cycles.
        self._frame_versions: dict[int, int] = {}
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "radar"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_frames_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale .tmp files from a crash mid-write so subsequent
        # atomic writes don't trip on existing files.
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.debug(
            "Frame memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, timestamp: int, region_name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memory-mapped view.

        Atomic write (.tmp → os.replace) ensures readers in other processes
        never see a half-written file — required for multi-worker safety.
        """
        final = self._memmap_dir / f"{timestamp}_{region_name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    def _cleanup_timestamp(self, timestamp: int) -> None:
        """Delete memmap files for an evicted timestamp."""
        for path in self._memmap_dir.glob(f"{timestamp}_*.dat"):
            try:
                path.unlink()
            except OSError:
                pass

    async def add_frame(self, frame: RadarFrame) -> tuple[int | None, bool]:
        """Add a frame, evicting the oldest if at capacity.

        If a frame with the same timestamp exists, merge the region data.
        Returns (evicted_timestamp | None, was_merged).
        """
        async with self._lock:
            # Convert regions to memory-mapped files.  Each region's
            # write + flush runs in its own worker thread (USCOMP is
            # ~63 MB) so all regions of one frame write concurrently;
            # the async lock still serialises concurrent add_frame calls.
            region_items = list(frame.regions.items())
            memmaps = await asyncio.gather(*[
                asyncio.to_thread(self._to_memmap, frame.timestamp, name, data)
                for name, data in region_items
            ])
            for (name, _data), memmap in zip(region_items, memmaps):
                frame.regions[name] = memmap

            # Merge into existing frame if same timestamp.
            # Copy-on-write: build a NEW regions dict and swap the reference instead of
            # mutating in place. The tile warmer/renderer captures `frame.regions` (a live
            # dict ref) and renders many tiles over several seconds; an in-place `.update()`
            # mid-warm makes tiles rendered before the swap read the old arrays and tiles
            # after read the new ones -> a visible seam ("tearing") on any frame whose data
            # updates mid-warm. Swapping the reference leaves in-flight renders on a
            # consistent snapshot (region arrays are never mutated in place, so sharing the
            # old arrays with an in-flight reader is safe).
            for existing in self._frames:
                if existing.timestamp == frame.timestamp:
                    existing.regions = {**existing.regions, **frame.regions}
                    # Content changed for an unchanged timestamp (e.g. a
                    # transient region-fetch failure carried forward then
                    # filled in) — bump the version so render workers know
                    # the cached geometry for this timestamp is stale.
                    self._frame_versions[frame.timestamp] = (
                        self._frame_versions.get(frame.timestamp, 0) + 1
                    )
                    # Merge keeps the same object, so the ``_by_ts`` entry
                    # stays valid — the dict lookup returns the live frame.
                    return None, True

            evicted_ts = None
            if len(self._frames) >= self._max_frames:
                evicted = self._frames.pop(0)
                evicted_ts = evicted.timestamp
                self._cleanup_timestamp(evicted_ts)
                self._frame_versions.pop(evicted_ts, None)
                self._by_ts.pop(evicted_ts, None)

            self._frames.append(frame)
            self._frames.sort(key=lambda f: f.timestamp)
            self._frame_versions[frame.timestamp] = 1
            self._by_ts[frame.timestamp] = frame
            return evicted_ts, False

    async def get_frame(self, timestamp: int) -> RadarFrame | None:
        async with self._lock:
            return self._by_ts.get(timestamp)

    def frame_version(self, timestamp: int) -> int | None:
        """Content version for a timestamp, or None when unknown.

        Used by the shared tile store to key encoded tiles by frame content
        identity (versions bump on merges, vanish on eviction).  Synchronous
        and lock-free on purpose: ``_frame_versions`` is replaced atomically
        by ``__setstate__`` and only mutated under the async lock by the
        pipeline, so a plain dict.get under the GIL always sees a consistent
        snapshot.  Nowcast timestamps never appear here — callers use that
        to exclude nowcast tiles from the shared store.
        """
        return self._frame_versions.get(timestamp)

    async def get_latest_frame(self) -> RadarFrame | None:
        async with self._lock:
            return self._frames[-1] if self._frames else None

    async def get_timestamps(self) -> list[int]:
        async with self._lock:
            return [f.timestamp for f in self._frames]

    async def get_region_keys(self) -> dict[int, set[str]]:
        """Return a mapping of timestamp -> set of region names present."""
        async with self._lock:
            return {f.timestamp: set(f.regions.keys()) for f in self._frames}

    async def frame_count(self) -> int:
        async with self._lock:
            return len(self._frames)

    @property
    def data_bytes(self) -> int:
        """Total bytes across all region arrays in all frames."""
        total = 0
        for frame in self._frames:
            for arr in frame.regions.values():
                total += arr.nbytes
        return total

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload.

        Returns a JSON-serializable dict describing the on-disk layout.
        Only meaningful in persistent mode (``cache_dir`` configured) —
        a tile-server worker re-opens the same memmaps via __setstate__.
        File paths are stored as basenames (relative to ``memmap_dir``)
        so the snapshot is portable across processes that share the
        cache volume even if it's mounted at different absolute paths.
        """
        return {
            "max_frames": self._max_frames,
            "memmap_dir": str(self._memmap_dir),
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "regions": {
                        name: [
                            os.path.basename(str(arr.filename)),
                            arr.dtype.str,
                            list(arr.shape),
                        ]
                        for name, arr in f.regions.items()
                    },
                }
                for f in self._frames
            ],
            # String keys so the dict survives JSON round-trips (int keys
            # would be coerced to strings anyway); __setstate__ converts
            # back.  The version counter is monotonic and stays correct
            # across dumps — do NOT clear it here on dump.
            "frame_versions": {
                str(ts): v for ts, v in self._frame_versions.items()
            },
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``.

        Re-opens memmaps read-only from the recorded basenames under
        ``memmap_dir``.  Used by the tile-server worker on startup and
        on every state.json refresh — replaces the in-memory frame list
        in place so existing references to ``FrameStore`` stay valid for
        ongoing renders (Linux holds the old memmap inodes alive until
        all readers release them).

        Region arrays are reused in place instead of re-opened whenever
        the timestamp's content version is unchanged between the previous
        state and this one.  The reuse is version-guarded because merges
        rewrite the same basename via ``os.replace`` (see ``add_frame``
        and ``_to_memmap``): an unchanged filename does not imply
        unchanged content, so the per-timestamp version is bumped on every
        merge and an unchanged version is the only safe signal that the
        live memmap still matches the payload.  The payload descriptor
        (basename, dtype, shape) is also verified against the live array
        before reuse; anything that fails the check is re-opened fresh.
        """
        max_frames = state["max_frames"]
        memmap_dir = Path(state["memmap_dir"])
        # JSON coerces int keys to strings; convert back.  The .get(..., {})
        # default handles snapshots written before this field existed.
        new_versions = {
            int(k): v for k, v in state.get("frame_versions", {}).items()
        }
        old_versions = getattr(self, "_frame_versions", {}) or {}
        old_by_ts = getattr(self, "_by_ts", {}) or {}
        new_frames: list[RadarFrame] = []
        reused = 0
        reopened = 0
        for f_info in state["frames"]:
            ts = f_info["timestamp"]
            frame = RadarFrame(timestamp=ts)
            old_frame = old_by_ts.get(ts)
            version_ok = (
                old_frame is not None
                and old_versions.get(ts) is not None
                and new_versions.get(ts) is not None
                and old_versions.get(ts) == new_versions.get(ts)
            )
            for name, (basename, dtype_str, shape) in f_info["regions"].items():
                old_arr = old_frame.regions.get(name) if old_frame is not None else None
                if (
                    version_ok
                    and old_arr is not None
                    and os.path.basename(str(old_arr.filename)) == basename
                    and old_arr.dtype.str == dtype_str
                    and list(old_arr.shape) == list(shape)
                ):
                    frame.regions[name] = old_arr
                    reused += 1
                else:
                    frame.regions[name] = np.memmap(
                        memmap_dir / basename,
                        dtype=np.dtype(dtype_str), mode="r",
                        shape=tuple(shape),
                    )
                    reopened += 1
            new_frames.append(frame)
        new_frames.sort(key=lambda f: f.timestamp)

        # Apply atomically — if this object is being updated in place,
        # readers see either the old list or the new list, never partial.
        self._max_frames = max_frames
        self._memmap_dir = memmap_dir
        self._frames = new_frames
        # Rebuild the O(1) timestamp index alongside the frame list so
        # the two never drift apart across snapshot refreshes.
        self._by_ts = {f.timestamp: f for f in new_frames}
        self._frame_versions = new_versions
        self._persistent = True
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        logger.debug(
            'FrameStore refresh: %d regions reused, %d reopened across %d frames',
            reused, reopened, len(new_frames),
        )

    def cleanup(self) -> None:
        """Release in-memory frame references; remove temp dir if non-persistent.

        In persistent mode, the on-disk memmap files are intentionally
        kept so a fresh process can pick them up via the constructor's
        warm-restart logic or via __setstate__.
        """
        self._frames = []
        self._by_ts.clear()
        if self._persistent:
            logger.info("Frame memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.debug("Frame memmap directory cleaned up")
