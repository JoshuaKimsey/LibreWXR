# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""EUMETSAT MTG Lightning Imager (LI) L2 flash source.

Meteosat-12 MTG LI L2 flashes from the EUMETSAT Data Store.  This is the
project's first credential path: the Data Store search endpoint is
anonymous, but downloading a product requires an OAuth2 client-credentials
bearer token (a free Data Store registration + API key).  It is a
deliberate one-off exception to the anonymous-access rule.

The adapter follows the
:class:`~librewxr.data.lightning_fetcher.LightningSource` protocol:
``fetch_cycle`` never raises, and the coordinator
(:class:`~librewxr.data.lightning_fetcher.LightningFetcher`) merges every
family's points into the shared on-disk artifact.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
import zipfile
from datetime import datetime, timezone
from urllib.parse import unquote

import h5py
import httpx
import numpy as np

from librewxr.data.lightning_fetcher import (
    Watermarks,
    _epoch_from_units,
    _scalar_attr,
)
from librewxr.data.lightning_store import _POINT_DTYPE
from librewxr.data.retry import retry_sync

logger = logging.getLogger(__name__)

# EUMETSAT Data Store endpoints.
_EUMETSAT_API_BASE = "https://api.eumetsat.int"
_EUMETSAT_TOKEN_URL = f"{_EUMETSAT_API_BASE}/token"
_EUMETSAT_SEARCH_URL = f"{_EUMETSAT_API_BASE}/data/search-products/1.0.0/os"

# MTG LI L2 Lightning Flashes (NRT, netCDF-4).  One stream today:
# Meteosat-12 at 0E.
_LI_COLLECTION = "EO:EUM:DAT:0691"
_LI_SAT_ID = 12                    # Meteosat-12 bird number (internal identity)
_LI_EPOCH_FALLBACK = 946684800.0   # 2000-01-01 00:00:00 UTC
_LI_UNITS = "mW.m-2.sr-1"

# Search pagination.  ``c`` caps at 500 on the Data Store; the page cap is a
# safety valve, not an expected path.
_MAX_SEARCH_PAGES = 10
_SEARCH_PAGE_SIZE = 500

# Refresh a cached bearer token this long before its advertised expiry.
_TOKEN_MARGIN_S = 300.0

# A stream whose newest published window end is older than this many fetch
# intervals is reported stale (a degraded feed, never an error).
_STALE_INTERVALS = 3


class _AuthError(Exception):
    """Token exchange or an authenticated download was rejected."""


class _TokenManager:
    """Cached OAuth2 client-credentials bearer token.

    The token is fetched lazily and refreshed when within
    ``_TOKEN_MARGIN_S`` of expiry.  No retries: a failed exchange surfaces
    as a failed fetch cycle and the next cycle tries again.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get(self) -> str:
        """Return a valid bearer token, refreshing it when near expiry."""
        with self._lock:
            token = self._token
            if (
                token is not None
                and time.time() < self._expires_at - _TOKEN_MARGIN_S
            ):
                return token
            token, expires_at = self._fetch()
            self._token = token
            self._expires_at = expires_at
            return token

    def invalidate(self) -> None:
        """Drop the cached token so the next ``get`` re-exchanges."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def _fetch(self) -> tuple[str, float]:
        """Exchange the API key/secret for ``(token, expires_at_epoch)``."""
        raw = f"{self._api_key}:{self._api_secret}".encode("utf-8")
        basic = base64.b64encode(raw).decode("ascii")
        try:
            resp = httpx.post(
                _EUMETSAT_TOKEN_URL,
                headers={"Authorization": f"Basic {basic}"},
                data={"grant_type": "client_credentials"},
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        except httpx.HTTPError as exc:
            raise _AuthError(f"token exchange failed: {exc}") from exc
        if resp.status_code != 200:
            raise _AuthError(f"token exchange HTTP {resp.status_code}")
        try:
            payload = resp.json()
            token = payload["access_token"]
            expires_in = float(payload.get("expires_in", 0))
        except (ValueError, KeyError, TypeError) as exc:
            raise _AuthError(f"malformed token response: {exc}") from exc
        if not isinstance(token, str) or not token:
            raise _AuthError("malformed token response: no access_token")
        return token, time.time() + expires_in


# ---------------------------------------------------------------------------
# Feature parsing helpers
# ---------------------------------------------------------------------------

def _feature_href(feature: dict) -> str | None:
    """Extract the download href from a GeoJSON feature's ``links.data``.

    The Data Store nests ``links`` under ``properties`` (verified against
    the live API 2026-11-13); a top-level ``links`` is also accepted
    defensively.
    """
    for container in (feature.get("properties"), feature):
        if not isinstance(container, dict):
            continue
        links = container.get("links")
        if not isinstance(links, dict):
            continue
        data = links.get("data")
        if isinstance(data, list):
            data = data[0] if data else None
        if isinstance(data, dict):
            href = data.get("href")
            if isinstance(href, str) and href:
                return href
    return None


def _parse_feature(feature) -> dict | None:
    """Parse one search feature to ``{product_id, href, window_*}``.

    Returns ``None`` for an unparseable feature so the caller can skip it.
    """
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        return None
    href = _feature_href(feature)
    if href is None:
        return None
    date = props.get("date")
    if not isinstance(date, str) or "/" not in date:
        return None
    start_iso, end_iso = date.split("/", 1)
    try:
        window_start = datetime.fromisoformat(
            start_iso.replace("Z", "+00:00")
        ).timestamp()
        window_end = datetime.fromisoformat(
            end_iso.replace("Z", "+00:00")
        ).timestamp()
    except ValueError:
        return None

    product_id = props.get("identifier") or feature.get("id")
    if not isinstance(product_id, str) or not product_id:
        product_id = unquote(href.rstrip("/").rsplit("/", 1)[-1])
    return {
        "product_id": product_id,
        "href": href,
        "window_start": window_start,
        "window_end": window_end,
    }


def _parse_features(features: list) -> list[dict]:
    """Parse a page's raw features, skipping (and DEBUG-logging) bad ones."""
    parsed: list[dict] = []
    for feature in features:
        item = _parse_feature(feature)
        if item is None:
            logger.debug("MTG LI: skipping unparseable search feature")
            continue
        parsed.append(item)
    return parsed


def _iso(ts: float) -> str:
    """Format a unix epoch as the Data Store's ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _decode_scaled_var(var) -> np.ndarray:
    """Decode a scaled netCDF variable preserving storage signedness."""
    raw = np.asarray(var[()])
    scale = _scalar_attr(var, "scale_factor", 1.0)
    offset = _scalar_attr(var, "add_offset", 0.0)
    return raw.astype(np.float64) * scale + offset


# ---------------------------------------------------------------------------
# MTGLISource
# ---------------------------------------------------------------------------

class MTGLISource:
    """Meteosat-12 MTG LI L2 flashes from the EUMETSAT Data Store (API-key auth)."""

    def __init__(
        self,
        *,
        interval_s: int,
        max_age_s: int,
        api_key: str,
        api_secret: str,
        collection: str = _LI_COLLECTION,
        sat_id: int = _LI_SAT_ID,
        name: str = "meteosat12",
        download_retries: int | None = None,
    ) -> None:
        self.name = name
        self._collection = collection
        self._sat_id = sat_id
        self._interval_s = int(interval_s)
        self._max_age_s = int(max_age_s)
        self._download_retries = download_retries

        self._tokens = _TokenManager(api_key, api_secret)

        # Stale-warn latch: warn once per stale episode, INFO once on
        # recovery.
        self._stale_warned = False

        self._client: httpx.Client | None = None

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the lazily-created sync HTTP client (idempotent)."""
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
                headers={"User-Agent": "(LibreWXR, librewxr@localhost)"},
            )
        return self._client

    # -- fetch cycle ---------------------------------------------------------

    def fetch_cycle(
        self, now_s: int, watermarks: Watermarks,
    ) -> tuple[np.ndarray, dict]:
        """Fetch new MTG LI products; NEVER raises, returns (points, status)."""
        name = self.name
        status: dict = {
            "source": self._collection,
            "units": _LI_UNITS,
            "last_key": None,
            "last_e": None,
            "stale": False,
            "last_fetch_ok": False,
        }
        try:
            now = now_s
            cutoff = now - self._max_age_s - 2 * self._interval_s

            features = self._search_products(_iso(cutoff), _iso(now))
            # The API sorts sensing-start descending; process ascending by
            # window end.  Dedupe by product id so a pagination off-by-one
            # cannot process the same product twice.
            features = sorted(features, key=lambda f: f["window_end"])
            seen: set[str] = set()
            unique: list[dict] = []
            for feature in features:
                if feature["product_id"] in seen:
                    continue
                seen.add(feature["product_id"])
                unique.append(feature)
            features = unique

            wm = watermarks.get(name) or {}
            last_key = wm.get("last_key")
            last_e = wm.get("last_e")

            # -- staleness (newest published window end) ---------------------
            newest_e: float | None = (
                features[-1]["window_end"] if features else None
            )
            stale = False
            if features:
                if (
                    newest_e is not None
                    and newest_e < now - _STALE_INTERVALS * self._interval_s
                ):
                    stale = True
            elif last_key:
                # Watermark present but no data published this cycle.
                stale = True
            status["stale"] = stale
            if stale:
                if not self._stale_warned:
                    logger.warning(
                        "MTG LI %s: feed appears stale (newest window end %s, "
                        "now %s); holding last-known points",
                        name, newest_e, now,
                    )
                    self._stale_warned = True
            elif self._stale_warned:
                logger.info("MTG LI %s: feed recovered", name)
                self._stale_warned = False

            # -- tail the listing in ascending window-end order --------------
            points: list[np.ndarray] = []
            auth_failed = False
            for feature in features:
                window_end = feature["window_end"]
                if window_end < cutoff:
                    continue
                if last_e is not None and window_end <= last_e:
                    continue

                try:
                    data = self._download(feature["href"])
                except _AuthError:
                    # A cached token may have been revoked/expired early;
                    # one forced re-exchange, then give up for this cycle.
                    self._tokens.invalidate()
                    try:
                        data = self._download(feature["href"])
                    except _AuthError:
                        logger.warning(
                            "MTG LI %s: authentication failed downloading %s; "
                            "check LIBREWXR_LIGHTNING_EUMETSAT_API_KEY / "
                            "_API_SECRET and confirm the EUMETSAT Data Store "
                            "terms have been accepted for this account",
                            name, feature["product_id"],
                        )
                        auth_failed = True
                        break

                if data is None:
                    # Transient miss / not yet published: stop the tail so the
                    # next cycle re-tails from the last processed product and
                    # never skips past a gap.
                    break

                try:
                    pts = self._decode_file(data)
                except Exception as exc:  # noqa: BLE001 - corrupt file path
                    logger.warning(
                        "MTG LI %s: decode failed for %s (%s); skipping",
                        name, feature["product_id"], exc,
                    )
                    # Advance past a permanently corrupt product so it cannot
                    # wedge the loop.
                    last_key = feature["product_id"]
                    last_e = window_end
                    continue

                points.append(pts)
                last_key = feature["product_id"]
                last_e = window_end

            status["last_key"] = last_key
            status["last_e"] = last_e
            status["last_fetch_ok"] = not auth_failed

            if last_key is not None:
                watermarks.update(
                    name,
                    source=self._collection,
                    last_key=last_key,
                    last_e=last_e,
                )

            merged = (
                np.concatenate(points)
                if points
                else np.empty(0, dtype=_POINT_DTYPE)
            )
            return merged, status
        except Exception:
            logger.warning("MTG LI %s: fetch cycle failed", name, exc_info=True)
            status["last_fetch_ok"] = False
            return np.empty(0, dtype=_POINT_DTYPE), status

    # -- HTTP surface --------------------------------------------------------

    def _search_products(
        self, dtstart_iso: str, dtend_iso: str,
    ) -> list[dict]:
        """Search the Data Store and return parsed features (``[]`` on failure).

        A failed search is a degraded cycle, never an exception (the GLM
        listing semantics): the caller keeps its watermark and retries next
        cycle.
        """
        client = self._get_client()
        raw_features: list = []
        total = 0
        for page in range(_MAX_SEARCH_PAGES):
            params = {
                "pi": self._collection,
                "dtstart": dtstart_iso,
                "dtend": dtend_iso,
                "c": _SEARCH_PAGE_SIZE,
                "si": page * _SEARCH_PAGE_SIZE + 1,
                "sort": "start,time,0",
                "format": "json",
            }
            resp = retry_sync(
                client.get,
                _EUMETSAT_SEARCH_URL,
                params=params,
                retries=self._download_retries,
                log_name="eumetsat search",
            )
            if resp is None:
                logger.warning("MTG LI: product search request failed")
                return []
            if resp.status_code != 200:
                logger.warning(
                    "MTG LI: product search HTTP %d", resp.status_code,
                )
                return []
            try:
                payload = resp.json()
                page_features = payload.get("features") or []
                total = int(payload.get("totalResults", 0))
            except (ValueError, TypeError, AttributeError):
                logger.warning(
                    "MTG LI: malformed product search response", exc_info=True,
                )
                return []
            raw_features.extend(page_features)
            if not page_features or len(raw_features) >= total:
                break
        else:
            logger.warning(
                "MTG LI: search page cap (%d) hit", _MAX_SEARCH_PAGES,
            )
        return _parse_features(raw_features)

    def _download(self, href: str) -> bytes | None:
        """Download one product ZIP; ``None`` when not (yet) available.

        Raises :class:`_AuthError` on 401/403 so the caller can re-exchange
        the bearer token once.
        """
        client = self._get_client()
        token = self._tokens.get()
        resp = retry_sync(
            client.get,
            href,
            headers={"Authorization": f"Bearer {token}"},
            retries=self._download_retries,
            log_name="eumetsat mtg li download",
        )
        if resp is None:
            return None
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 404:
            # A not-yet-published product is a normal miss, not an error.
            logger.info("MTG LI: %s not available yet (HTTP 404)", href)
            return None
        if resp.status_code in (401, 403):
            raise _AuthError(f"download HTTP {resp.status_code}")
        logger.warning(
            "MTG LI: download HTTP %d (%s)", resp.status_code, href,
        )
        return None

    # -- decode --------------------------------------------------------------

    def _decode_file(self, data: bytes) -> np.ndarray:
        """Decode a product ZIP's netCDF members into flash points.

        The consolidated product carries a CHK-BODY and a CHK-TRAIL member
        over the same sensing window; only the BODY member holds the flash
        variables (the TRAIL member carries statistics), so when any member
        is named CHK-BODY only those are decoded and concatenated.  With no
        such member every ``.nc`` member is attempted, so a naming drift
        cannot silently blind ingestion.  Raises ``ValueError`` when the ZIP
        has no netCDF member or every attempted member fails to decode.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                nc_members = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_members:
                    raise ValueError("no netCDF member in MTG LI product")
                body_members = [n for n in nc_members if "CHK-BODY" in n]
                if body_members:
                    for name in nc_members:
                        if name not in body_members:
                            logger.debug(
                                "MTG LI: skipping non-body member %s", name,
                            )
                    members = body_members
                else:
                    members = nc_members
                buffers = [zf.read(n) for n in members]
        except zipfile.BadZipFile as exc:
            raise ValueError("MTG LI product is not a ZIP archive") from exc

        points: list[np.ndarray] = []
        for member, buf in zip(members, buffers):
            try:
                points.append(self._decode_nc(buf))
            except Exception as exc:  # noqa: BLE001 - corrupt member path
                logger.warning(
                    "MTG LI: decode failed for member %s (%s); continuing",
                    member, exc,
                )
        if not points:
            raise ValueError("no decodable netCDF member in MTG LI product")
        return np.concatenate(points) if len(points) > 1 else points[0]

    def _decode_nc(self, nc_bytes: bytes) -> np.ndarray:
        """Decode one LI L2 flashes netCDF buffer into flash points."""
        with h5py.File(io.BytesIO(nc_bytes), "r") as f:
            try:
                flash_time_var = f["flash_time"]
                lat_var = f["latitude"]
                lon_var = f["longitude"]
                radiance_var = f["radiance"]
            except KeyError as exc:
                raise ValueError("unexpected MTG LI schema") from exc

            flash_time = np.asarray(
                flash_time_var[()], dtype=np.float64,
            ).reshape(-1)
            epoch = _epoch_from_units(
                flash_time_var.attrs.get("units"), default=_LI_EPOCH_FALLBACK,
            )
            # LI latitude/longitude are SIGNED int16 (0.00275 deg) -- decode
            # them preserving signedness, unlike GLM's _Unsigned storage.
            lat = _decode_scaled_var(lat_var).reshape(-1)
            lon = _decode_scaled_var(lon_var).reshape(-1)
            radiance = _decode_scaled_var(radiance_var).reshape(-1)
            if "l1b_radiometric_warning" in f:
                qc = np.asarray(f["l1b_radiometric_warning"][()]).reshape(-1)
            else:
                qc = None
                logger.debug(
                    "MTG LI: product has no l1b_radiometric_warning variable; "
                    "skipping QC mask",
                )

        time_s = np.asarray(flash_time + epoch, dtype=np.int64)
        mask = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= -90.0)
            & (lat <= 90.0)
            & (lon >= -180.0)
            & (lon <= 180.0)
        )
        if qc is not None:
            mask &= qc == 0

        out = np.empty(int(mask.sum()), dtype=_POINT_DTYPE)
        out["time_s"] = time_s[mask]
        out["lat"] = lat[mask].astype(np.float32)
        out["lon"] = lon[mask].astype(np.float32)
        # LI radiance is an energy-like intensity in mW.m-2.sr-1, not GLM's
        # joules; it rides the shared ``energy`` slot unconverted.
        out["energy"] = radiance[mask].astype(np.float32)
        out["satellite"] = np.int8(self._sat_id)
        return out
