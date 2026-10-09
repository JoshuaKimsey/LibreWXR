# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Lightning flash-point store backed by a self-contained on-disk artifact.

Pipeline-owned writer: ``replace_points`` swaps a fresh structured array of
GOES GLM flash points into memory and ``save_snapshot`` writes it to
``<cache_dir>/lightning/current.npz``.  Render workers and the stdio MCP
transport are readers: ``maybe_reload`` stats the artifact and reloads when
its mtime changes.

Deliberately no state.json section -- unlike the radar/NWP/alerts stores,
lightning points cross the process boundary through this artifact rather
than the master snapshot, so enabling or disabling lightning never changes
state.json.  This follows the shared-tile-store / precip-mask precedent.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# RAM circuit-breaker ceiling.  Every flash within MAX_AGE is held (there is
# no data-side cap -- MCP/REST must always see the full picture), but a
# runaway feed is bounded here to protect the pipeline and every render
# worker that loads the artifact.  Oldest-first trim with a warn-once log;
# normal 30-minute volume is tens of thousands of points.  Not config.
MAX_POINTS = 500_000

# Structured dtype for one flash point.  Field access via ``arr["lat"]`` or
# ``arr[i]`` (numpy void scalar).  ``satellite`` is the GOES bird number
# (18 / 19) used as an internal identity; consumers decode it to a human
# label.
_POINT_DTYPE = np.dtype([
    ("time_s", "int64"),    # UTC epoch seconds
    ("lat", "float32"),
    ("lon", "float32"),
    ("energy", "float32"),  # joules (decoded upstream)
    ("satellite", "int8"),  # 18 / 19 (GOES bird number, internal identity)
])


# ---------------------------------------------------------------------------
# LightningStore
# ---------------------------------------------------------------------------

class LightningStore:
    """Sparse structured flash-point store with an on-disk artifact.

    The pipeline is the only writer; render workers and the stdio MCP
    transport call ``maybe_reload`` (stat-guarded) to pick up changes.
    Intentionally no ``__getstate__`` / ``__setstate__``: this store is
    absent from the state.json snapshot because the artifact IS the
    cross-process handoff.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self._artifact_path: Path | None = None
        if cache_dir is not None:
            self._artifact_path = Path(cache_dir) / "lightning" / "current.npz"
        self._points: np.ndarray = np.empty(0, dtype=_POINT_DTYPE)
        self._meta: dict = {}
        self._version: int = 0
        self._last_updated: float = 0.0
        self._artifact_mtime: float | None = None
        self._ceiling_warned: bool = False
        self._ceiling_trimmed_total: int = 0
        # Guards the load path so concurrent ``maybe_reload`` calls do not
        # race two ``np.load`` reads (and two state swaps) at once.
        self._reload_lock = asyncio.Lock()

    # -- pipeline-side write -------------------------------------------------

    async def replace_points(
        self,
        points: np.ndarray,
        *,
        last_seen_s: float,
        max_age_s: int,
        meta: dict | None = None,
    ) -> None:
        """Pipeline-side atomic swap of the held flash points.

        ``points`` is coerced to ``_POINT_DTYPE`` and sorted ascending by
        time, then trimmed to ``[window_end - max_age_s .. window_end]``
        where ``window_end`` is the later of ``last_seen_s`` and the newest
        point's ``time_s``.  A final ceiling trim drops the oldest points
        above ``MAX_POINTS`` (RAM brake; a single warning per store
        instance).
        """
        arr = np.asarray(points, dtype=_POINT_DTYPE)
        # ``np.sort`` on a structured array needs the ``order`` name.
        arr = np.sort(arr, order="time_s")

        if arr.shape[0] > 0:
            window_end = max(float(last_seen_s), float(arr["time_s"].max()))
        else:
            window_end = float(last_seen_s)
        cutoff = window_end - max_age_s
        arr = arr[arr["time_s"] >= cutoff]

        if arr.shape[0] > MAX_POINTS:
            dropped = arr.shape[0] - MAX_POINTS
            # Already sorted ascending -- drop the oldest (front) rows.
            arr = arr[dropped:]
            self._ceiling_trimmed_total += dropped
            if not self._ceiling_warned:
                logger.warning(
                    "LightningStore point ceiling (%d) exceeded; dropped %d "
                    "oldest point(s).  Further ceiling trims are silent.",
                    MAX_POINTS, dropped,
                )
                self._ceiling_warned = True

        # Single attribute assignment: readers on the event loop always see
        # either the old or the new array, never a partial swap.
        self._points = arr
        self._meta = dict(meta or {})
        self._version = max(int(time.time()), self._version + 1)
        # Persist the version inside meta so a reader reloading the artifact
        # recovers the same version (used to key render-side caches).
        self._meta["version"] = self._version
        self._last_updated = time.time()

    # -- artifact I/O --------------------------------------------------------

    async def save_snapshot(self) -> bool:
        """Write the current points to the on-disk artifact (best-effort).

        Returns True on success; False when there is no cache_dir, no
        ``replace_points`` call has happened yet, or the write fails.  Never
        raises -- an unwritable artifact degrades to the in-memory store.
        """
        if self._artifact_path is None:
            return False
        if self._version == 0:
            # replace_points was never called -- nothing to persist.
            return False

        final = self._artifact_path
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            # pid+uuid tmp name so a concurrent writer cannot steal the file
            # out from under this writer's os.replace (the storm-cells hazard).
            tmp = final.with_name(
                f"{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    points=self._points,
                    meta=np.array(json.dumps(self._meta)),
                )
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
            return True
        except OSError as exc:
            logger.warning("LightningStore.save_snapshot failed: %s", exc)
            return False

    async def maybe_reload(self) -> bool:
        """Reader-side refresh from the on-disk artifact.

        Cheap ``os.stat`` check; reloads only when the artifact's mtime
        changed.  Returns True when new data was loaded; False otherwise.
        Never raises to the caller -- a missing or corrupt artifact leaves
        the previously loaded data in place.
        """
        if self._artifact_path is None:
            return False
        try:
            st = os.stat(self._artifact_path)
        except FileNotFoundError:
            # No artifact yet -- keep the current (empty) state.
            return False
        except OSError as exc:
            logger.warning("LightningStore.maybe_reload stat failed: %s", exc)
            return False
        if self._artifact_mtime is not None and st.st_mtime == self._artifact_mtime:
            return False

        async with self._reload_lock:
            # Re-check under the lock: a concurrent caller may have loaded
            # the same version while we waited.
            try:
                st = os.stat(self._artifact_path)
            except OSError:
                return False
            if (
                self._artifact_mtime is not None
                and st.st_mtime == self._artifact_mtime
            ):
                return False
            try:
                await asyncio.to_thread(self._load_artifact)
            except Exception as exc:  # noqa: BLE001 - best-effort reader path
                logger.warning("LightningStore.maybe_reload failed: %s", exc)
                return False
            self._artifact_mtime = st.st_mtime
            return True

    def _load_artifact(self) -> None:
        """Load the artifact synchronously (runs in a worker thread).

        Raises on a missing ``points`` array or a dtype mismatch; the caller
        (``maybe_reload``) catches and logs so corrupt files never propagate.
        """
        if self._artifact_path is None:
            return
        with np.load(self._artifact_path, allow_pickle=False) as loaded:
            if "points" not in loaded:
                raise ValueError("lightning artifact missing 'points' array")
            points = loaded["points"]
            if points.dtype != _POINT_DTYPE:
                raise ValueError(
                    f"lightning artifact point dtype {points.dtype} does not "
                    f"match {_POINT_DTYPE}"
                )
            meta: dict = {}
            if "meta" in loaded:
                raw = str(loaded["meta"])
                if raw:
                    meta = json.loads(raw)
            self._points = np.asarray(points, dtype=_POINT_DTYPE)

        self._meta = meta
        if "version" in meta:
            self._version = int(meta["version"])
            self._last_updated = float(meta["version"])

    # -- queries -------------------------------------------------------------

    def points_in(
        self,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
        *,
        since_s: int | None = None,
        until_s: int | None = None,
    ) -> np.ndarray:
        """Return points inside the inclusive lat/lon box.

        Optional ``since_s`` keeps only points with ``time_s >= since_s``;
        optional ``until_s`` keeps only points with ``time_s <= until_s``.
        Both bounds are inclusive, so a frame's slot query uses
        ``since_s = T - slot + 1`` (exclusive lower edge) and
        ``until_s = T`` (inclusive upper edge) to yield the half-open
        window ``(T - slot, T]``.  Returns a fresh array (an empty
        structured array when nothing matches); never raises on an empty
        store.
        """
        points = self._points
        if points.shape[0] == 0:
            return np.empty(0, dtype=_POINT_DTYPE)
        mask = (
            (points["lat"] >= lat0)
            & (points["lat"] <= lat1)
            & (points["lon"] >= lon0)
            & (points["lon"] <= lon1)
        )
        if since_s is not None:
            mask &= points["time_s"] >= since_s
        if until_s is not None:
            mask &= points["time_s"] <= until_s
        return points[mask]

    # -- read-only properties ------------------------------------------------

    @property
    def version(self) -> int:
        """Content version, restored from the artifact on a reader reload."""
        return self._version

    @property
    def points(self) -> np.ndarray:
        """The internal point array; callers must treat it as read-only."""
        return self._points

    @property
    def total_count(self) -> int:
        """Number of held flash points."""
        return int(self._points.shape[0])

    @property
    def meta(self) -> dict:
        """Shallow copy of the artifact metadata."""
        return dict(self._meta)

    @property
    def last_updated(self) -> float:
        """Unix timestamp of the last ``replace_points`` / artifact load."""
        return self._last_updated

    @property
    def window_start_s(self) -> int | None:
        """Oldest held point time in UTC epoch seconds; None when empty."""
        if self._points.shape[0] == 0:
            return None
        return int(self._points["time_s"].min())

    @property
    def window_end_s(self) -> int | None:
        """Newest held point time in UTC epoch seconds; None when empty."""
        if self._points.shape[0] == 0:
            return None
        return int(self._points["time_s"].max())

    @property
    def ceiling_trimmed_total(self) -> int:
        """Cumulative points dropped by the MAX_POINTS ceiling."""
        return self._ceiling_trimmed_total
