# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GRID_SHAPE = (3, 1801, 3600)  # (layers, height, width)
GRID_DTYPE = np.uint8
MAX_AGE_HOURS = 24


class CloudGridCache:
    """Persistent disk cache for processed cloud cover grids.

    Stores each timestep as a single raw binary file (shape 3x1801x3600,
    uint8) that can be memory-mapped for serving.  Files are written
    atomically (write-to-tmp, then os.replace) so a crash mid-write
    cannot corrupt the cache.

    Cache key is the valid_time Unix timestamp — if a newer model run
    produces data for the same hour, it simply overwrites the file.
    """

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir / "satellite"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._dir / "metadata.json"

    def _file_path(self, unix_ts: int) -> Path:
        return self._dir / f"cloud_{unix_ts}.dat"

    def has(self, unix_ts: int) -> bool:
        return self._file_path(unix_ts).exists()

    def write(
        self,
        unix_ts: int,
        high: np.ndarray,
        mid: np.ndarray,
        low: np.ndarray,
    ) -> None:
        """Write a (high, mid, low) triplet to disk atomically."""
        final = self._file_path(unix_ts)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=GRID_DTYPE, mode="w+", shape=GRID_SHAPE)
        mm[0] = high
        mm[1] = mid
        mm[2] = low
        mm.flush()
        del mm
        os.replace(tmp, final)

    def read(
        self, unix_ts: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return memmap-backed (high, mid, low) views, or None if missing."""
        path = self._file_path(unix_ts)
        if not path.exists():
            return None
        try:
            mm = np.memmap(path, dtype=GRID_DTYPE, mode="r", shape=GRID_SHAPE)
            return mm[0], mm[1], mm[2]
        except Exception:
            logger.warning("Failed to memmap %s, removing", path)
            path.unlink(missing_ok=True)
            return None

    def load_all(self) -> tuple[str | None, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Load metadata and memmap all cached timesteps.

        Returns (reference_time, {unix_ts: (high, mid, low)}).
        """
        ref_time, ts_list = self._load_metadata()
        result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        if not ts_list:
            # Fall back to scanning .dat files if metadata is missing
            ts_list = sorted(self.get_cached_timestamps())

        for ts in ts_list:
            data = self.read(ts)
            if data is not None:
                result[ts] = data

        return ref_time, result

    def save_metadata(self, reference_time: str, timestamps: list[int]) -> None:
        """Atomically write metadata JSON."""
        tmp = self._metadata_path.with_suffix(".json.tmp")
        payload = {"reference_time": reference_time, "timestamps": sorted(timestamps)}
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._metadata_path)

    def _load_metadata(self) -> tuple[str | None, list[int]]:
        """Read metadata JSON. Returns (None, []) on any error."""
        if not self._metadata_path.exists():
            return None, []
        try:
            data = json.loads(self._metadata_path.read_text())
            return data.get("reference_time"), data.get("timestamps", [])
        except Exception:
            logger.warning("Corrupt metadata.json, ignoring")
            return None, []

    def get_cached_timestamps(self) -> set[int]:
        """Scan .dat filenames and return the set of cached Unix timestamps."""
        result = set()
        for path in self._dir.glob("cloud_*.dat"):
            try:
                ts = int(path.stem.removeprefix("cloud_"))
                result.add(ts)
            except ValueError:
                pass
        return result

    def cleanup(self, active_timestamps: list[int]) -> None:
        """Remove .dat files that are not active and older than 24 hours."""
        if not active_timestamps:
            return
        active_set = set(active_timestamps)
        newest = max(active_timestamps)
        cutoff = newest - MAX_AGE_HOURS * 3600

        removed = 0
        for path in self._dir.glob("cloud_*.dat"):
            try:
                ts = int(path.stem.removeprefix("cloud_"))
            except ValueError:
                continue
            if ts not in active_set and ts < cutoff:
                path.unlink(missing_ok=True)
                removed += 1

        # Clean up any stale .tmp files
        for path in self._dir.glob("*.tmp"):
            path.unlink(missing_ok=True)

        if removed:
            logger.info("Cloud cache cleanup: removed %d old files", removed)
