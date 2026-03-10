# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RadarFrame:
    timestamp: int  # Unix timestamp
    regions: dict[str, np.ndarray] = field(default_factory=dict)


class FrameStore:
    """In-memory store for radar composite frames."""

    def __init__(self, max_frames: int = 12):
        self._max_frames = max_frames
        self._frames: list[RadarFrame] = []
        self._lock = asyncio.Lock()

    async def add_frame(self, frame: RadarFrame) -> int | None:
        """Add a frame, evicting the oldest if at capacity.

        If a frame with the same timestamp exists, merge the region data.
        Returns the timestamp of the evicted frame, or None.
        """
        async with self._lock:
            # Merge into existing frame if same timestamp
            for existing in self._frames:
                if existing.timestamp == frame.timestamp:
                    existing.regions.update(frame.regions)
                    return None

            evicted_ts = None
            if len(self._frames) >= self._max_frames:
                evicted = self._frames.pop(0)
                evicted_ts = evicted.timestamp

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
