# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import json
import logging
import os
from pathlib import Path

import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.store import RadarFrame

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DTYPE = np.uint8


class RadarFrameCache:
    """Persistent disk cache for radar frame region arrays.

    Each (timestamp, region) is stored as its own raw uint8 file written
    atomically (write-to-tmp, then os.replace). A ``metadata.json`` file
    records the schema version and per-region (height, width) — on load
    any region whose shape no longer matches its current ``RegionDef``
    is silently dropped, so a code-side region resize can't restore
    broken data.

    The cache directory is shared with the ``FrameStore``'s persistent
    memmaps (``<ts>_<region>.dat``), which are the canonical on-disk
    radar frames whenever the store runs in persistent mode — the fetcher
    then skips ``write_frame`` and this class reads the FrameStore files
    back on restart instead.  The legacy ``radar_<ts>_<region>.dat``
    files written by ``write_frame`` are still produced when the store is
    non-persistent (single mode), so the load path understands both
    layouts and prefers the FrameStore memmap when both exist.

    Cleanup is driven by the in-memory ``FrameStore`` ring buffer:
    anything that's been evicted from memory is also evicted from disk.
    """

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir / "radar"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._dir / "metadata.json"

    def _file_path(self, unix_ts: int, region_name: str) -> Path:
        return self._dir / f"radar_{unix_ts}_{region_name}.dat"

    def _framestore_file_path(self, unix_ts: int, region_name: str) -> Path:
        """Path of the FrameStore's memmap for a (timestamp, region)."""
        return self._dir / f"{unix_ts}_{region_name}.dat"

    def _resolve_region_file(
        self, unix_ts: int, region_name: str,
    ) -> Path | None:
        """Resolve the on-disk file for a (timestamp, region).

        Prefers the FrameStore memmap (the canonical persistent copy when
        the store runs in persistent mode) over the legacy ``radar_*``
        file.  Returns ``None`` when neither exists.
        """
        framestore_path = self._framestore_file_path(unix_ts, region_name)
        if framestore_path.exists():
            return framestore_path
        legacy = self._file_path(unix_ts, region_name)
        return legacy if legacy.exists() else None

    def has(self, unix_ts: int, region_name: str) -> bool:
        return self._resolve_region_file(unix_ts, region_name) is not None

    def write_frame(self, frame: RadarFrame) -> None:
        """Write every region in a frame to disk atomically."""
        for region_name, data in frame.regions.items():
            self._write_region(frame.timestamp, region_name, np.asarray(data))

    def _write_region(
        self, unix_ts: int, region_name: str, data: np.ndarray
    ) -> None:
        if data.dtype != DTYPE:
            data = data.astype(DTYPE, copy=False)
        final = self._file_path(unix_ts, region_name)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=DTYPE, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)

    def load_frames(
        self, regions: dict[str, RegionDef]
    ) -> list[RadarFrame]:
        """Load all cached frames, validated against current RegionDef shapes.

        Only regions listed in ``regions`` are restored, and only if the
        cached file's shape matches the current ``RegionDef``. Any
        timestamp with at least one valid region is returned as a frame
        with whatever subset of regions survived validation.

        Self-healing: timestamps discovered by scanning the on-disk frame
        files (FrameStore memmaps ``<ts>_<region>.dat`` and legacy
        ``radar_*_*.dat`` files) are unioned with whatever
        ``metadata.json`` declares, so a crash mid-backfill that leaves
        the metadata stale doesn't orphan valid frame files. If a
        per-region shape isn't recorded in metadata, ``np.memmap``
        validates by file size at read time.
        """
        meta = self._load_metadata() or {}

        # An explicit schema version mismatch is fatal; missing metadata
        # entirely is fine — disk scan + per-file size check still works.
        declared_version = meta.get("schema_version")
        if declared_version is not None and declared_version != SCHEMA_VERSION:
            logger.info(
                "Radar cache schema_version mismatch (have=%s, expect=%d); "
                "ignoring cache",
                declared_version, SCHEMA_VERSION,
            )
            return []

        cached_shapes = meta.get("regions", {})
        metadata_timestamps = set(meta.get("timestamps", []))
        disk_timestamps = self._scan_timestamps()
        orphans = disk_timestamps - metadata_timestamps
        if orphans:
            logger.info(
                "Radar cache: %d timestamp(s) on disk missing from metadata; "
                "including them",
                len(orphans),
            )
        timestamps = sorted(metadata_timestamps | disk_timestamps)

        frames: list[RadarFrame] = []
        for ts in timestamps:
            regions_data: dict[str, np.ndarray] = {}
            for name, region in regions.items():
                expected_shape = (region.height, region.width)
                cached_meta = cached_shapes.get(name)
                if cached_meta is not None:
                    cached_shape = tuple(cached_meta.get("shape", []))
                    if cached_shape != expected_shape:
                        # Region was reshaped in code — drop any stale
                        # copies in either on-disk layout.
                        self._file_path(ts, name).unlink(missing_ok=True)
                        self._framestore_file_path(ts, name).unlink(missing_ok=True)
                        continue
                # Either the region's shape is recorded and matches, or
                # metadata doesn't know about it — fall through and let
                # _read_region's memmap fail by file size if it's wrong.
                arr = self._read_region(ts, name, expected_shape)
                if arr is not None:
                    regions_data[name] = arr
            if regions_data:
                frames.append(RadarFrame(timestamp=ts, regions=regions_data))
        return frames

    def _scan_timestamps(self) -> set[int]:
        """Enumerate timestamps from both on-disk radar frame layouts.

        ``radar_<ts>_<region>.dat`` (legacy cache files) and
        ``<ts>_<region>.dat`` (FrameStore memmaps) both live in this
        directory.
        """
        result: set[int] = set()
        for path in self._dir.glob("*.dat"):
            stem_parts = path.stem.split("_")
            # radar_<ts>_<region>.dat -> ts at index 1; <ts>_<region>.dat
            # -> ts at index 0.
            idx = 1 if stem_parts and stem_parts[0] == "radar" else 0
            if len(stem_parts) > idx:
                try:
                    result.add(int(stem_parts[idx]))
                except ValueError:
                    pass
        return result

    def _read_region(
        self, unix_ts: int, region_name: str, shape: tuple[int, int]
    ) -> np.ndarray | None:
        path = self._resolve_region_file(unix_ts, region_name)
        if path is None:
            return None
        try:
            return np.memmap(path, dtype=DTYPE, mode="r", shape=shape)
        except Exception:
            logger.warning("Failed to memmap %s, removing", path)
            path.unlink(missing_ok=True)
            return None

    def save_metadata(
        self, regions: dict[str, RegionDef], timestamps: list[int]
    ) -> None:
        """Atomically write metadata JSON with current shapes and timestamps."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "regions": {
                name: {"shape": [r.height, r.width], "dtype": "uint8"}
                for name, r in regions.items()
            },
            "timestamps": sorted(timestamps),
        }
        tmp = self._metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self._metadata_path)

    def _load_metadata(self) -> dict | None:
        if not self._metadata_path.exists():
            return None
        try:
            return json.loads(self._metadata_path.read_text())
        except Exception:
            logger.warning("Corrupt radar metadata.json, ignoring")
            return None

    def stats(self) -> dict:
        """Return a snapshot of on-disk radar frame state for /health.

        Counts both layouts that share this directory: the legacy
        ``radar_*`` files and the FrameStore memmaps that are the
        canonical persistent copy.
        """
        files = list(self._dir.glob("*.dat"))
        total_bytes = 0
        timestamps: set[int] = set()
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
            stem_parts = path.stem.split("_")
            idx = 1 if stem_parts and stem_parts[0] == "radar" else 0
            if len(stem_parts) > idx:
                try:
                    timestamps.add(int(stem_parts[idx]))
                except ValueError:
                    pass
        return {
            "files": len(files),
            "used_mb": round(total_bytes / (1024 * 1024), 1),
            "oldest_ts": min(timestamps) if timestamps else None,
            "newest_ts": max(timestamps) if timestamps else None,
        }

    def cleanup(self, active_timestamps: list[int]) -> None:
        """Remove radar frame files for timestamps no longer in the active set.

        Covers both on-disk layouts that share this directory: the legacy
        ``radar_<ts>_<region>.dat`` files (written when the FrameStore is
        non-persistent) and the FrameStore's own ``<ts>_<region>.dat``
        memmaps (the canonical persistent copy, normally removed by the
        store's own eviction — this is the safety net for crash leftovers).
        A legacy file that duplicates an existing FrameStore file is also
        removed: it is identical data left behind by a pre-dedupe session,
        superseded by the FrameStore copy.
        """
        active = set(active_timestamps)
        removed = 0
        for path in self._dir.glob("*.dat"):
            stem_parts = path.stem.split("_")
            idx = 1 if stem_parts and stem_parts[0] == "radar" else 0
            if len(stem_parts) <= idx:
                continue
            try:
                ts = int(stem_parts[idx])
            except ValueError:
                continue
            if ts in active:
                if (
                    stem_parts[0] == "radar"
                    and len(stem_parts) > 2
                ):
                    # Duplicate of an existing FrameStore memmap — drop the
                    # legacy copy (identical bytes, same directory).
                    region_name = "_".join(stem_parts[2:])
                    if self._framestore_file_path(ts, region_name).exists():
                        path.unlink(missing_ok=True)
                        removed += 1
                continue
            path.unlink(missing_ok=True)
            removed += 1

        for path in self._dir.glob("*.tmp"):
            path.unlink(missing_ok=True)

        if removed:
            logger.debug("Radar cache cleanup: removed %d old files", removed)
