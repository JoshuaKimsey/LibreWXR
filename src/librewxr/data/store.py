# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
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
    """

    def __init__(self, max_frames: int = 12):
        self._max_frames = max_frames
        self._frames: list[RadarFrame] = []
        self._lock = asyncio.Lock()
        self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_frames_"))
        logger.info("Frame memmap directory: %s", self._memmap_dir)

    def _to_memmap(self, timestamp: int, region_name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk and return a read-only memory-mapped view."""
        path = self._memmap_dir / f"{timestamp}_{region_name}.dat"
        mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        return np.memmap(path, dtype=data.dtype, mode="r", shape=data.shape)

    def _cleanup_timestamp(self, timestamp: int) -> None:
        """Delete memmap files for an evicted timestamp."""
        for path in self._memmap_dir.glob(f"{timestamp}_*.dat"):
            try:
                path.unlink()
            except OSError:
                pass

    async def add_frame(self, frame: RadarFrame) -> int | None:
        """Add a frame, evicting the oldest if at capacity.

        If a frame with the same timestamp exists, merge the region data.
        Returns the timestamp of the evicted frame, or None.
        """
        async with self._lock:
            # Convert regions to memory-mapped files
            for name, data in list(frame.regions.items()):
                frame.regions[name] = self._to_memmap(frame.timestamp, name, data)

            # Merge into existing frame if same timestamp
            for existing in self._frames:
                if existing.timestamp == frame.timestamp:
                    existing.regions.update(frame.regions)
                    return None

            evicted_ts = None
            if len(self._frames) >= self._max_frames:
                evicted = self._frames.pop(0)
                evicted_ts = evicted.timestamp
                self._cleanup_timestamp(evicted_ts)

            self._frames.append(frame)
            self._frames.sort(key=lambda f: f.timestamp)
            return evicted_ts

    async def get_frame(self, timestamp: int) -> RadarFrame | None:
        async with self._lock:
            for f in self._frames:
                if f.timestamp == timestamp:
                    return f
        return None

    async def get_latest_frame(self) -> RadarFrame | None:
        async with self._lock:
            return self._frames[-1] if self._frames else None

    async def get_timestamps(self) -> list[int]:
        async with self._lock:
            return [f.timestamp for f in self._frames]

    async def frame_count(self) -> int:
        async with self._lock:
            return len(self._frames)

    def cleanup(self) -> None:
        """Remove all memmap files and the temporary directory."""
        shutil.rmtree(self._memmap_dir, ignore_errors=True)
        logger.info("Frame memmap directory cleaned up")
