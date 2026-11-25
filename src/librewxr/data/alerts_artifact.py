# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Single-file SQLite artifact for weather alerts.

The data pipeline writes ``<cache_dir>/alerts/current.db`` once per alerts fetch
cycle; render workers open it read-only on demand.  The artifact is the
cross-process handoff in place of a ``stores.alerts_store`` section in
``state.json`` (same pattern as the shared tile store, the precip mask and the
lightning artifact): the pipeline owns writing, renderers only read.

Layout:

* ``meta(key TEXT PRIMARY KEY, value BLOB NOT NULL)`` -
  ``last_updated`` (float), ``fetch_success`` (int 0/1), ``count`` (int) and
  ``response`` (zlib-compressed UTF-8 bytes of the packaged
  ``AlertsResponse`` JSON, so serving the whole collection never re-runs the
  shapely union / simplify pass).
* ``entries(seq INTEGER PRIMARY KEY, minx/miny/maxx/maxy REAL, payload BLOB)`` -
  one row per alert in original list order; the bbox columns hold
  ``a.polygon.bounds`` (all NULL for a polygon-less alert) so spatial queries
  are a cheap SQL prefilter, and ``payload`` is the zlib-compressed JSON of the
  per-entry snapshot shape (``alerts_store.entry_to_dict``).

All reads tolerate a missing or corrupt artifact: they log and keep the last
good state instead of raising.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import zlib
from pathlib import Path

from shapely.errors import GEOSException
from shapely.geometry import Point, box, shape

from librewxr.data.alerts_response import build_alerts_response
from librewxr.data.alerts_store import (
    AlertEntry,
    AlertsStore,
    ensure_valid_polygon,
    entry_to_dict,
)

log = logging.getLogger(__name__)

#: Artifact path, relative to ``cache_dir``.
ARTIFACT_NAME = "alerts/current.db"

#: zlib level used for both the packaged response and the entry payloads.
_ZLIB_LEVEL = 6

#: Padding applied to the equirectangular envelope in ``alerts_near_candidates``
#: so the SQL prefilter can only ever over-select (the caller re-filters).
_NEAR_PAD = 1.15

#: Kilometres per degree of latitude (and of longitude at the equator).
_KM_PER_DEG = 111.32


def _compress(data: bytes) -> bytes:
    return zlib.compress(data, _ZLIB_LEVEL)


def _decode_entry(payload: bytes) -> AlertEntry | None:
    """Rebuild one ``AlertEntry`` from an artifact payload row.

    Mirrors ``AlertsStore.__setstate__``: fields from the dict, polygon via
    ``shape(...)`` + ``ensure_valid_polygon(...)``, ``None`` when the dict's
    polygon is falsy.  Returns ``None`` (with a warning) for an undecodable
    row so a single bad entry cannot fail the whole read.
    """
    try:
        raw = json.loads(zlib.decompress(payload).decode("utf-8"))
        polygon = raw.get("polygon")
        return AlertEntry(
            source_id=raw["source_id"],
            event=raw["event"],
            description=raw["description"],
            severity=raw["severity"],
            effective=raw["effective"],
            expires=raw["expires"],
            area_desc=raw["area_desc"],
            url=raw["url"],
            polygon=ensure_valid_polygon(shape(polygon) if polygon else None),
        )
    except (zlib.error, ValueError, KeyError, TypeError, GEOSException) as exc:
        log.warning("Skipping undecodable alerts artifact row: %s", exc)
        return None


def write_alerts_artifacts(store: AlertsStore, cache_dir: str | Path) -> bool:
    """Write the alerts artifact for ``store`` into ``cache_dir``.

    Builds a temporary SQLite database next to the artifact and ``os.replace``s
    it into place, so a reader either sees the previous complete artifact or the
    new one, never a partial file.  Always writes -- including an empty store
    and ``fetch_success=False`` (scalars plus zero rows plus the response built
    from the empty list), so a degraded cycle is representable instead of stale.

    Never raises: any error is logged and reported as ``False``, which leaves
    the last good artifact in place.  The tmp file is removed on failure.

    Single writer in practice (the pipeline's own loop).  The entry list and the
    scalar snapshot are read one after the other, so a ``replace_all`` landing
    between them can pair this cycle's entries with the next cycle's scalars;
    that benign race is accepted (both are rewritten atomically every cycle).

    Returns True when the artifact was replaced, False otherwise.
    """
    alerts = store.alerts
    last_updated = float(store.last_updated)
    fetch_success = bool(store.fetch_success)

    alerts_dir = Path(cache_dir) / "alerts"
    tmp_path = alerts_dir / f".current.db.{os.getpid()}.{threading.get_ident()}.tmp"
    conn: sqlite3.Connection | None = None
    try:
        alerts_dir.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            tmp_path.unlink()
        conn = sqlite3.connect(str(tmp_path))
        conn.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE entries ("
            "seq INTEGER PRIMARY KEY, "
            "minx REAL, miny REAL, maxx REAL, maxy REAL, "
            "payload BLOB NOT NULL)"
        )
        rows = []
        for seq, alert in enumerate(alerts):
            polygon = alert.polygon
            if polygon is None:
                minx = miny = maxx = maxy = None
            else:
                minx, miny, maxx, maxy = polygon.bounds
            rows.append(
                (
                    seq,
                    minx,
                    miny,
                    maxx,
                    maxy,
                    _compress(
                        json.dumps(
                            entry_to_dict(alert), ensure_ascii=False
                        ).encode("utf-8")
                    ),
                )
            )
        if rows:
            conn.executemany(
                "INSERT INTO entries "
                "(seq, minx, miny, maxx, maxy, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        response_json = build_alerts_response(alerts).model_dump_json()
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("last_updated", last_updated),
                ("fetch_success", 1 if fetch_success else 0),
                ("count", len(alerts)),
                ("response", _compress(response_json.encode("utf-8"))),
            ],
        )
        conn.commit()
        conn.close()
        conn = None
        os.replace(tmp_path, alerts_dir / "current.db")
        return True
    except Exception as exc:  # noqa: BLE001 - writer is best-effort by contract
        log.warning("Failed to write alerts artifact: %s", exc)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return False


class AlertsArtifactStore:
    """Read-only view of the alerts artifact, safe to share across threads.

    Starts in the empty/degraded state (count 0, ``last_updated`` 0.0,
    ``fetch_success`` False) and only leaves it through ``maybe_reload``, so an
    absent or corrupt artifact degrades to "no alerts" rather than raising.
    Every operation opens its own short-lived read-only connection, so no
    connection is ever shared between threads; only the scalar swap in
    ``maybe_reload`` takes the lock.
    """

    def __init__(self, cache_dir: str | Path):
        self._path = Path(
            os.path.abspath(os.fspath(Path(cache_dir) / ARTIFACT_NAME))
        )
        self._lock = threading.Lock()
        # (st_mtime_ns, st_size) of the artifact the cached scalars came from;
        # None until the first successful reload.
        self._stamp: tuple[int, int] | None = None
        self._count = 0
        self._last_updated = 0.0
        self._fetch_success = False

    # -- scalars ----------------------------------------------------------

    @property
    def count(self) -> int:
        return self._count

    @property
    def last_updated(self) -> float:
        return self._last_updated

    @property
    def fetch_success(self) -> bool:
        return self._fetch_success

    # -- artifact access --------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived read-only connection to the artifact."""
        return sqlite3.connect(
            f"{self._path.as_uri()}?mode=ro", uri=True, timeout=1.0
        )

    def maybe_reload(self) -> bool:
        """Refresh the cached scalars when the artifact changed on disk.

        Returns True only for the call that observed and applied a new
        ``(st_mtime_ns, st_size)`` stamp; unchanged stamps return False without
        opening the file.  A read error keeps the last good scalars *and* the
        old stamp, so the next call retries.  Never raises.
        """
        try:
            stat = os.stat(self._path)
        except OSError:
            return False
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._stamp == stamp:
                return False

        try:
            conn = self._connect()
            try:
                rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            finally:
                conn.close()
            last_updated = float(rows["last_updated"])
            fetch_success = bool(int(rows["fetch_success"]))
            count = int(rows["count"])
            if rows.get("response") is None:
                raise ValueError("alerts artifact meta is missing 'response'")
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError) as exc:
            log.warning("Failed to read alerts artifact %s: %s", self._path, exc)
            return False

        with self._lock:
            if self._stamp == stamp:
                # Another thread applied this stamp while we were reading.
                return False
            self._stamp = stamp
            self._count = count
            self._last_updated = last_updated
            self._fetch_success = fetch_success
            return True

    def _load_entries(
        self, where: str = "", params: tuple = ()
    ) -> list[AlertEntry]:
        """Read entries matching ``where`` (optional) in ``seq`` order."""
        sql = "SELECT payload FROM entries"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY seq"
        try:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            log.warning("Failed to read alerts artifact %s: %s", self._path, exc)
            return []
        entries: list[AlertEntry] = []
        for (payload,) in rows:
            entry = _decode_entry(payload)
            if entry is not None:
                entries.append(entry)
        return entries

    # -- queries ----------------------------------------------------------

    def alerts(self) -> list[AlertEntry]:
        """Every alert in the artifact, in original order."""
        return self._load_entries()

    def alerts_in_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    ) -> list[AlertEntry]:
        """Alerts whose polygon intersects the lon/lat box.

        SQL bbox prefilter on the stored bounds, then an exact
        ``box(...).intersects(polygon)`` per surviving entry.  Polygon-less
        alerts are never returned (they have NULL bounds).
        """
        entries = self._load_entries(
            "minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ?",
            (max_lon, min_lon, max_lat, min_lat),
        )
        bbox = box(min_lon, min_lat, max_lon, max_lat)
        out: list[AlertEntry] = []
        for entry in entries:
            polygon = entry.polygon
            if polygon is None:
                continue
            try:
                if bbox.intersects(polygon):
                    out.append(entry)
            except GEOSException as exc:
                log.warning("Skipping alert with invalid geometry: %s", exc)
        return out

    def alerts_at_point(self, lon: float, lat: float) -> list[AlertEntry]:
        """Alerts whose polygon contains/intersects the point."""
        entries = self._load_entries(
            "minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ?",
            (lon, lon, lat, lat),
        )
        point = Point(lon, lat)
        out: list[AlertEntry] = []
        for entry in entries:
            polygon = entry.polygon
            if polygon is None:
                continue
            try:
                if point.intersects(polygon):
                    out.append(entry)
            except GEOSException as exc:
                log.warning("Skipping alert with invalid geometry: %s", exc)
        return out

    def alerts_near_candidates(
        self, lat: float, lon: float, radius_km: float,
    ) -> list[AlertEntry]:
        """Conservative candidates for a radius query around (lat, lon).

        Returns a superset of the alerts any correct radius implementation could
        match: the padded equirectangular envelope is prefiltered in SQL and the
        surviving entries are returned unrepaired, in original order, with no
        geometric filtering.  The caller (``mcp.alerts_query``) remains the
        authority for circle/containment.

        Any case where the envelope cannot be trusted degrades to a full scan --
        polar latitudes (``abs(lat) > 85``), an envelope wider than 90 degrees of
        longitude, a lon window crossing the antimeridian, or a negative radius.
        """
        if radius_km < 0:
            return self.alerts()
        if abs(lat) > 85:
            return self.alerts()
        cos_lat = math.cos(math.radians(lat))
        if cos_lat <= 0:
            return self.alerts()
        dlat = radius_km / _KM_PER_DEG * _NEAR_PAD
        dlon = radius_km / (_KM_PER_DEG * cos_lat) * _NEAR_PAD
        if dlon >= 90.0 or lon - dlon < -180.0 or lon + dlon > 180.0:
            return self.alerts()
        return self._load_entries(
            "minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ?",
            (lon + dlon, lon - dlon, lat + dlat, lat - dlat),
        )

    def response_bytes(self) -> bytes:
        """The packaged ``AlertsResponse`` JSON bytes (decompressed), or b""."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = 'response'"
                ).fetchone()
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            log.warning("Failed to read alerts artifact %s: %s", self._path, exc)
            return b""
        if not row or row[0] is None:
            return b""
        try:
            return zlib.decompress(bytes(row[0]))
        except zlib.error as exc:
            log.warning("Failed to decompress alerts response: %s", exc)
            return b""
