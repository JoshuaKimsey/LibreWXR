# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from lxml import etree
from shapely.geometry import Polygon, shape

from librewxr.config import settings
from librewxr.data.alerts_store import AlertEntry, AlertsStore
from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)

# WMO API constants (from WMO-alerts-research.md)
_WMO_BASE = "https://severeweather.wmo.int"
_SOURCES_URL = f"{_WMO_BASE}/json/sources.json"
_WMO_ALL_URL = f"{_WMO_BASE}/v2/json/wmo_all.json"

# NWS API (direct GeoJSON, avoids WMO lag for US alerts)
_NWS_API_URL = "https://api.weather.gov/alerts/active"
_NWS_USER_AGENT = "(LibreWXR, librewxr@localhost)"

# Excluded sources: known bad feeds, data quality issues, or sources handled
# directly via a separate pipeline (e.g., NWS API for US alerts).
_EXCLUDED_SOURCE_IDS = frozenset({
    "co-ungrd-es",
    "mv-ndmc-en",
    "us-noaa-nws-en-marine",
    "us-noaa-nws-en",  # handled directly via NWS API
    "cn-cma-xx",
    "mo-smg-xx",
})

# MeteoAlarm data URLs (Pirate Weather's GitHub repo, dev branch)
_METEOALARM_GEOJSON_URL = (
    "https://raw.githubusercontent.com/Pirate-Weather/"
    "pirate-weather-code/dev/API/data/meteoalarm_geocodes.json"
)
_METEOALARM_ALIASES_URL = (
    "https://raw.githubusercontent.com/Pirate-Weather/"
    "pirate-weather-code/dev/API/data/meteoalarm_aliases.csv"
)


# ---------------------------------------------------------------------------
# CAP XML parsing helpers (adapted from Pirate Weather WMO_Alerts_Local.py)
# ---------------------------------------------------------------------------

def _cap_text(elem: etree._Element, tag: str, ns: dict[str, str]) -> str:
    """Namespace-aware text extraction from a CAP XML element."""
    if ns:
        return (elem.findtext(f"cap:{tag}", default="", namespaces=ns) or "").strip()
    return (elem.findtext(tag, default="") or "").strip()


def _extract_polygons_from_cap(
    cap_xml: str,
    source_id: str,
    cap_url: str,
    meteoalarm: Optional[dict[str, Polygon]] = None,
) -> list[AlertEntry]:
    """Parse CAP XML and return a list of AlertEntry objects.

    Adapted from Pirate Weather's WMO_Alerts_Local.py.
    Handles CAP 1.1/1.2 namespaces, duplicate-language skipping,
    polygon extraction, and EMMA_ID → polygon conversion.
    """
    results: list[AlertEntry] = []
    try:
        root = etree.fromstring(cap_xml.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        logger.warning("Failed to parse CAP XML for %s: %s", source_id, exc)
        return results

    # Detect namespace (CAP 1.1 or 1.2)
    ns: dict[str, str] = {}
    if root.tag.startswith("{"):
        cap_ns = root.tag.split("}")[0].strip("{")
        ns = {"cap": cap_ns}

    seen_languages: set[str] = set()

    for info in root.findall(".//cap:info" if ns else ".//info", ns):
        lang_elem = info.find("cap:language" if ns else "language", ns)
        lang = (
            (lang_elem.text or "").strip().lower()
            if lang_elem is not None
            else "unknown"
        )

        # Use only the first language encountered for this alert
        if not seen_languages:
            seen_languages.add(lang)
        elif lang not in seen_languages:
            seen_languages.add(lang)
            continue  # Skip additional languages

        urgency = _cap_text(info, "urgency", ns)
        if urgency.lower() == "past":
            continue

        # Event / headline / description logic (from Pirate Weather)
        event = _cap_text(info, "event", ns) or None
        headline = _cap_text(info, "headline", ns) or None
        description = _cap_text(info, "description", ns) or None

        description_text = description or headline or ""
        if headline and description:
            event_text = headline
        else:
            event_text = event or ""

        severity = _cap_text(info, "severity", ns)
        effective = _cap_text(info, "effective", ns) or _cap_text(info, "onset", ns)
        expires = _cap_text(info, "expires", ns)

        for area in info.findall("cap:area" if ns else "area", ns):
            area_desc = (
                area.findtext("cap:areaDesc" if ns else "areaDesc", "", ns) or ""
            ).strip()

            # Extract geocode entries
            geocode_entries: list[tuple[Optional[str], Optional[str]]] = []
            seen_geocodes: set[tuple[str, str]] = set()
            for geocode_elem in area.findall("cap:geocode" if ns else "geocode", ns):
                value_name = (
                    geocode_elem.findtext("cap:valueName" if ns else "valueName", "", ns)
                    or ""
                ).strip()
                value = (
                    geocode_elem.findtext("cap:value" if ns else "value", "", ns) or ""
                ).strip()
                if not value:
                    continue
                normalized = (value_name.upper(), value.upper())
                if normalized in seen_geocodes:
                    continue
                seen_geocodes.add(normalized)
                geocode_entries.append((value_name or None, value))

            # Process polygons
            has_polygon = False
            for poly_elem in area.findall("cap:polygon" if ns else "polygon", ns):
                polygon_text = (poly_elem.text or "").strip()
                if not polygon_text:
                    continue
                coords: list[tuple[float, float]] = []
                for part in polygon_text.replace(";", " ").split():
                    if "," not in part:
                        continue
                    lat_str, lon_str = part.split(",", 1)
                    try:
                        lat, lon = float(lat_str), float(lon_str)
                    except ValueError:
                        continue
                    coords.append((lon, lat))  # GeoJSON order
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        has_polygon = True
                        results.append(
                            AlertEntry(
                                source_id=source_id,
                                event=event_text,
                                description=description_text,
                                severity=severity,
                                effective=effective,
                                expires=expires,
                                area_desc=area_desc,
                                url=cap_url,
                                polygon=poly,
                            )
                        )
                    except Exception as exc:
                        logger.warning("Polygon construction failed for %s: %s", source_id, exc)
                        continue

            # If no polygon but geocode exists, try to resolve via meteoalarm
            if not has_polygon:
                for geocode_name, geocode_value in geocode_entries:
                    if not geocode_name or not geocode_value:
                        continue
                    poly: Optional[Polygon] = None
                    if (
                        geocode_name.upper() in ("EMMA_ID", "NUTS3")
                        and meteoalarm is not None
                    ):
                        poly = meteoalarm.get(geocode_value)
                    if poly is not None:
                        results.append(
                            AlertEntry(
                                source_id=source_id,
                                event=event_text,
                                description=description_text,
                                severity=severity,
                                effective=effective,
                                expires=expires,
                                area_desc=area_desc,
                                url=cap_url,
                                polygon=poly,
                            )
                        )

    return results


# ---------------------------------------------------------------------------
# MeteoAlarm helpers
# ---------------------------------------------------------------------------

def _parse_meteoalarm_geojson(path: str) -> dict[str, Polygon]:
    """Parse MeteoAlarm geocodes GeoJSON into {EMMA_ID: Polygon} dict."""
    geocodes: dict[str, Polygon] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            code = str(props.get("code", "")).strip().upper()
            geom = feature.get("geometry")
            if code and geom:
                try:
                    poly = shape(geom)
                    if poly.is_valid:
                        geocodes[code] = poly
                except Exception:
                    continue
    except Exception as exc:
        logger.warning("Failed to parse MeteoAlarm geocodes: %s", exc)
    return geocodes


def _apply_meteoalarm_aliases(
    geocodes: dict[str, Polygon], alias_csv_path: str
) -> dict[str, Polygon]:
    """Duplicate polygon entries for aliased EMMA_ID codes."""
    try:
        with open(alias_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = str(row.get("CODE", "")).strip().upper()
                alias = str(row.get("ALIAS_CODE", "")).strip().upper()
                if code and alias and code in geocodes and alias not in geocodes:
                    geocodes[alias] = geocodes[code]
    except Exception as exc:
        logger.warning("Failed to apply MeteoAlarm aliases: %s", exc)
    return geocodes


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def _rss_item_links(feed_bytes: bytes) -> list[tuple[str, Optional[str]]]:
    """Extract (link, guid) tuples from RSS feed bytes."""
    try:
        root = etree.fromstring(feed_bytes)
    except etree.XMLSyntaxError:
        return []
    out: list[tuple[str, Optional[str]]] = []
    channel = root.find("channel")
    if channel is None:
        return out
    for item in channel.findall("item"):
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip() or None
        if link:
            out.append((link, guid))
    return out


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

def _parse_cap_time(value: str) -> int | None:
    """Parse CAP ISO 8601 time string to Unix epoch."""
    if not value:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(value)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class WMOAlertsFetcher:
    """Background fetcher for WMO CAP weather alerts.

    Mirrors the RadarFetcher pattern: created in lifespan(), started via
    start(), runs a background asyncio task that fetches at clock-aligned
    intervals.
    """

    def __init__(
        self,
        store: AlertsStore,
        cache_dir: Optional[str] = None,
        interval: int = 300,
        timeout: float = 30.0,
        concurrency: int = 5,
    ):
        self._store = store
        self._interval = interval
        self._timeout = timeout
        self._concurrency = concurrency
        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None
        self._meteoalarm: dict[str, Polygon] = {}
        self._meteoalarm_ready = asyncio.Event()
        self._cache_dir = Path(cache_dir) if cache_dir else None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(
                max_connections=self._concurrency,
                max_keepalive_connections=self._concurrency,
            )
            self._client = httpx.AsyncClient(limits=limits, timeout=self._timeout)
        return self._client

    async def _ensure_meteoalarm_data(self) -> None:
        """Download or load cached MeteoAlarm geocodes + aliases."""
        if self._cache_dir is not None:
            cache_subdir = self._cache_dir / "alerts"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            geojson_path = cache_subdir / "meteoalarm_geocodes.json"
            aliases_path = cache_subdir / "meteoalarm_aliases.csv"
        else:
            import tempfile
            tmp = Path(tempfile.gettempdir()) / "librewxr_alerts"
            tmp.mkdir(parents=True, exist_ok=True)
            geojson_path = tmp / "meteoalarm_geocodes.json"
            aliases_path = tmp / "meteoalarm_aliases.csv"

        # Download if not cached
        if not geojson_path.exists():
            logger.info("Downloading MeteoAlarm geocodes (~33 MB)...")
            await self._download_file(_METEOALARM_GEOJSON_URL, str(geojson_path))
        if not aliases_path.exists():
            logger.info("Downloading MeteoAlarm aliases...")
            await self._download_file(_METEOALARM_ALIASES_URL, str(aliases_path))

        # Parse
        logger.info("Loading MeteoAlarm geocodes...")
        geocodes = _parse_meteoalarm_geojson(str(geojson_path))
        geocodes = _apply_meteoalarm_aliases(geocodes, str(aliases_path))
        self._meteoalarm = geocodes
        self._meteoalarm_ready.set()
        logger.info("MeteoAlarm ready: %d region polygons", len(self._meteoalarm))

    async def _download_file(self, url: str, dest: str) -> None:
        client = await self._get_client()
        resp = await retry_get(client, url, log_name="meteoalarm")
        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"Failed to download {url}")
        with open(dest, "wb") as f:
            f.write(resp.content)

    async def _fetch_nws_alerts(self) -> list[AlertEntry]:
        """Fetch active US alerts directly from the NWS API.

        The NWS API returns native GeoJSON FeatureCollection with polygons
        already in [lon, lat] order, bypassing the WMO feed lag.
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                _NWS_API_URL,
                headers={"User-Agent": _NWS_USER_AGENT},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning("NWS API returned %d", resp.status_code)
                return []
            data = resp.json()
        except Exception as exc:
            logger.warning("NWS API request failed: %s", exc)
            return []

        entries: list[AlertEntry] = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry")

            # Skip cancelled or test alerts
            status = props.get("status", "").lower()
            msg_type = props.get("messageType", "").lower()
            if status == "cancel" or msg_type == "test":
                continue

            # Geometry already in GeoJSON [lon, lat] order
            polygon = None
            if geom is not None:
                try:
                    polygon = shape(geom)
                except Exception:
                    pass

            # Use headline if present, otherwise event, otherwise description
            headline = props.get("headline", "") or ""
            event = props.get("event", "") or ""
            description = props.get("description", "") or ""
            event_text = headline or event or ""
            description_text = description or headline or ""

            # Effective/expires
            effective = props.get("effective", "")
            expires = props.get("expires", "")

            entries.append(
                AlertEntry(
                    source_id="nws-api",
                    event=event_text,
                    description=description_text,
                    severity=props.get("severity", "Unknown"),
                    effective=effective,
                    expires=expires,
                    area_desc=props.get("areaDesc", ""),
                    url=props.get("id", "") or feature.get("id", ""),
                    polygon=polygon,
                )
            )

        logger.info("NWS API: %d active alerts", len(entries))
        return entries

    async def _fetch_once(self) -> None:
        """Full ingest pipeline."""
        client = await self._get_client()

        # Fetch NWS alerts in parallel with WMO (no CAP XML parsing needed)
        nws_task = asyncio.create_task(self._fetch_nws_alerts())

        # 1. Fetch sources.json
        resp = await retry_get(client, _SOURCES_URL, log_name="wmo_sources")
        if resp is None or resp.status_code != 200:
            logger.warning("Failed to fetch sources.json")
            self._store.mark_failed()
            return
        sources_data = resp.json()
        sources = sources_data.get("sources", [])

        # 2. Fetch wmo_all.json
        resp = await retry_get(client, _WMO_ALL_URL, log_name="wmo_all")
        if resp is None or resp.status_code != 200:
            logger.warning("Failed to fetch wmo_all.json")
            self._store.mark_failed()
            return
        wmo_all_data = resp.json()
        wmo_all_items = wmo_all_data.get("items", [])

        # Build set of current alert IDs
        current_ids = {item.get("id") for item in wmo_all_items if item.get("id")}
        current_agencies = set()
        for record in wmo_all_items:
            cap_url = record.get("capURL", "") or ""
            url = record.get("url", "") or ""
            if cap_url:
                current_agencies.add(cap_url.split("/")[0])
            elif url:
                current_agencies.add(url.split("/")[0])

        # Filter operating sources
        source_ids: list[str] = []
        for entry in sources:
            src = entry.get("source", {})
            sid = src.get("sourceId")
            status = src.get("capAlertFeedStatus")
            if not sid or status != "operating":
                continue
            if sid in _EXCLUDED_SOURCE_IDS:
                continue
            if sid in current_agencies:
                source_ids.append(sid)

        logger.info("Fetching alerts from %d WMO sources", len(source_ids))

        # 3. Fetch RSS feeds and CAP XMLs
        sem = asyncio.Semaphore(self._concurrency)
        all_alerts: list[AlertEntry] = []

        async def process_feed(sid: str) -> None:
            feed_url = f"{_WMO_BASE}/v2/cap-alerts/{sid}/rss.xml"
            async with sem:
                resp = await retry_get(client, feed_url, log_name=f"rss_{sid}")
            if resp is None or resp.status_code != 200:
                return
            feed_bytes = resp.content
            items = _rss_item_links(feed_bytes)

            # Filter to current alert IDs
            filtered: list[str] = []
            for link, guid in items:
                if not link:
                    continue
                if guid and guid in current_ids:
                    filtered.append(link)
                elif guid and any(guid in cid for cid in current_ids):
                    filtered.append(link)

            # Fetch CAP XMLs concurrently per feed
            async def fetch_and_extract(cap_link: str) -> list[AlertEntry]:
                async with sem:
                    resp = await retry_get(client, cap_link, log_name=f"cap_{sid}")
                if resp is None or resp.status_code != 200:
                    return []
                return _extract_polygons_from_cap(
                    resp.text, sid, cap_link, self._meteoalarm
                )

            tasks = [asyncio.create_task(fetch_and_extract(link)) for link in filtered]
            for coro in asyncio.as_completed(tasks):
                try:
                    entries = await coro
                    all_alerts.extend(entries)
                except Exception:
                    continue

        await asyncio.gather(*(process_feed(sid) for sid in source_ids))

        # 4. Merge NWS results
        nws_alerts = await nws_task
        all_alerts.extend(nws_alerts)

        # 5. BBOX geographic filter — when LIBREWXR_BBOX is configured,
        # discard alerts whose polygon does not intersect the bounding
        # box.  Alerts without polygon geometry are also dropped since
        # they cannot be placed on the map and their relevance to the
        # configured region cannot be determined.
        bbox = settings.get_bbox()
        pre_filter = len(all_alerts)
        if bbox is not None:
            from shapely.geometry import box
            south, west, north, east = bbox
            bbox_poly = box(west, south, east, north)
            all_alerts = [
                a for a in all_alerts
                if a.polygon is not None and a.polygon.intersects(bbox_poly)
            ]

        # 6. Replace store
        self._store.replace_all(all_alerts)
        if bbox is not None:
            logger.info(
                "Alerts updated: %d stored (%d before BBOX filter, "
                "%d WMO + %d NWS) from %d sources",
                len(all_alerts), pre_filter,
                pre_filter - len(nws_alerts), len(nws_alerts),
                len(source_ids),
            )
        else:
            logger.info(
                "Alerts updated: %d total (%d WMO + %d NWS) from %d sources",
                len(all_alerts),
                len(all_alerts) - len(nws_alerts),
                len(nws_alerts),
                len(source_ids),
            )

    async def _fetch_loop(self) -> None:
        """Background task: wait for meteoalarm, then fetch→sleep→repeat."""
        try:
            await self._ensure_meteoalarm_data()
        except Exception:
            logger.exception("Failed to load MeteoAlarm data, alerts disabled")
            return

        # Do an initial fetch immediately
        try:
            await self._fetch_once()
        except Exception:
            logger.exception("Initial WMO alert fetch failed")

        while True:
            # Sleep until next clock-aligned boundary
            now = time.time()
            interval = self._interval
            next_boundary = (int(now // interval) + 1) * interval
            sleep_secs = max(next_boundary - now, 1.0)
            logger.debug("Next WMO alert fetch in %.1fs", sleep_secs)
            await asyncio.sleep(sleep_secs)

            try:
                await self._fetch_once()
            except Exception:
                logger.exception("WMO alert fetch failed")

    async def start(self) -> None:
        """Kick off background fetch task."""
        self._task = asyncio.create_task(self._fetch_loop())

    async def close(self) -> None:
        """Cancel background task and close HTTP client."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
