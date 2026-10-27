# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from lxml import etree
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from librewxr.config import settings
from librewxr.data.alert_geocodes import (
    GeocodeProvider,
    build_default_providers,
    ensure_loaded,
    resolve,
)
from librewxr.data.alerts_store import AlertEntry, AlertsStore, ensure_valid_polygon
from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)

# WMO API constants (from WMO-alerts-research.md)
_WMO_BASE = "https://severeweather.wmo.int"
_SOURCES_URL = f"{_WMO_BASE}/json/sources.json"
_WMO_ALL_URL = f"{_WMO_BASE}/v2/json/wmo_all.json"

# NWS API (direct GeoJSON, avoids WMO lag for US alerts)
_NWS_API_URL = "https://api.weather.gov/alerts/active"
# Shared User-Agent for all outbound alert requests (WMO, NWS, MeteoAlarm,
# RSS/CAP).  NWS requires a contact UA; sending the default httpx UA to WMO
# was a likely trigger for rejections, so every request now identifies as
# LibreWXR.  Per-request headers (NWS) still override the client default.
_USER_AGENT = "(LibreWXR, librewxr@localhost)"

# Zone-based NWS alerts (Tornado Watches, Special Weather Statements, ...)
# ship with ``geometry: null`` — their polygons come from affected zones.
# Zone boundaries are near-static (county/forecast-zone geometry changes
# rarely), so resolved zone polygons are disk-cached for 30 days.  This
# replaces the old per-request ``?point=`` enrichment: no query path ever
# touches api.weather.gov at request time anymore.
_ZONE_CACHE_TTL = 30 * 24 * 60 * 60  # 30 days
_ZONE_FAILURE_TTL = 24 * 60 * 60  # 1 day - retry genuinely-broken zones daily, not every cycle

# UGC codes encode the zone type in the 3rd character: Z = forecast zone,
# C = county.  Only these two map to api.weather.gov zone URLs.
_UGC_ZONE_URL_PREFIXES = {
    "Z": "https://api.weather.gov/zones/forecast/",
    "C": "https://api.weather.gov/zones/county/",
}

# Excluded sources: known bad feeds, data quality issues, or sources handled
# directly via a separate pipeline (e.g., NWS API for US alerts).
_EXCLUDED_SOURCE_IDS = frozenset({
    "co-ungrd-es",
    "mv-ndmc-en",
    "us-noaa-nws-en-marine",
    "us-noaa-nws-en",  # handled directly via NWS API
    # China CMA: excluded for volume/shape, not data quality. Largest source on
    # the hub (~21% of active items), but its CAPs carry no <polygon>/<circle>
    # and only "CPEAS" 12-digit admin codes, plus dual-language info blocks --
    # every alert would be downloaded and discarded. A CPEAS resolver is a
    # trivial registry data drop now, but no openly licensed geometry source
    # exists: township polygons are not public and the county-level DataV/AMAP
    # table bars caching/redistribution. Parked until such a source appears.
    "cn-cma-xx",
    "mo-smg-xx",
})


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
    resolvers: list[GeocodeProvider],
) -> list[AlertEntry]:
    """Parse CAP XML and return a list of AlertEntry objects.

    Adapted from Pirate Weather's WMO_Alerts_Local.py.
    Handles CAP 1.1/1.2 namespaces, duplicate-language skipping, polygon
    extraction, and geocode -> polygon conversion via the registered
    ``resolvers`` tables.

    Each ``<info>`` block yields at most ONE ``AlertEntry``: the info's
    explicit ``<polygon>`` areas and its geocode-resolved polygons are
    collected together, resolved polygons already covered by an explicit one
    are pruned, and the remainder is merged -- kept as a single ``Polygon``
    when only one survives, otherwise assembled into a ``MultiPolygon`` from
    the exact parts (no ``unary_union``), then repaired to a valid polygon.
    That keeps a
    municipality-resolution feed like DWD at one entry per document instead
    of one per warn cell.
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

        # One entry per <info>: every area of the info contributes to a single
        # merged geometry.  Per-area entries multiplied the store ~8x for
        # municipality-resolution feeds (DWD ships one WARNCELLID area per
        # Gemeinde), so the areas are collected and merged instead.
        explicit_polys: list[Polygon] = []
        resolved_polys: list[Polygon] = []
        resolved_ids: set[int] = set()
        area_descs: list[str] = []

        for area in info.findall("cap:area" if ns else "area", ns):
            area_desc = (
                area.findtext("cap:areaDesc" if ns else "areaDesc", "", ns) or ""
            ).strip()
            if area_desc and area_desc not in area_descs:
                area_descs.append(area_desc)

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

            # Collect explicit <polygon> areas
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
                        poly_raw = Polygon(coords)
                    except Exception as exc:
                        logger.warning("Polygon construction failed for %s: %s", source_id, exc)
                        continue
                    # A self-intersecting CAP ring is invalid geometry that
                    # raises GEOSException at the request-time unary_union;
                    # repair it here, drop it if unrecoverable.
                    poly = ensure_valid_polygon(poly_raw)
                    if poly is None:
                        logger.warning("Dropping unrecoverable polygon for %s", source_id)
                        continue
                    explicit_polys.append(poly)

            # Resolve geocodes regardless of explicit polygons: a CAP may mix
            # <polygon> areas with geocode areas, and both describe coverage.
            for geocode_name, geocode_value in geocode_entries:
                if not geocode_name or not geocode_value:
                    continue
                poly: Optional[Polygon] = resolve(
                    resolvers, geocode_name, geocode_value
                )
                if poly is None:
                    continue
                # Two areas naming the same code resolve to the very same
                # table object; keep it once.
                if id(poly) in resolved_ids:
                    continue
                resolved_ids.add(id(poly))
                resolved_polys.append(poly)

        if not explicit_polys and not resolved_polys:
            # Fully-processed <info> that produced no entry (no usable
            # <polygon> and no resolvable geocode) is dropped — surface it at
            # DEBUG so dropped alerts are discoverable.
            logger.debug(
                "Dropped CAP alert for %s: no usable polygon or resolvable "
                "geocode (event=%r headline=%r)",
                source_id,
                event or "",
                headline or "",
            )
            continue

        # Containment pruning: DWD publishes one aggregate <polygon> plus a
        # WARNCELLID area per municipality; the warn cells sit entirely inside
        # the aggregate, so keeping both only duplicates coverage.
        if explicit_polys and resolved_polys:
            prepared = [prep(poly) for poly in explicit_polys]
            resolved_polys = [
                poly
                for poly in resolved_polys
                if not any(prep_explicit.contains(poly) for prep_explicit in prepared)
            ]

        polygons = explicit_polys + resolved_polys
        if len(polygons) == 1:
            merged: Optional[Polygon] = polygons[0]
        else:
            # Cheap exact assembly (no unary_union): the parts are already the
            # right geometries, and multi-part table polygons are flattened so
            # MultiPolygon() only ever sees Polygons.
            merged = MultiPolygon(
                [part for poly in polygons for part in getattr(poly, "geoms", [poly])]
            )

        # Overlapping parts make the assembled MultiPolygon invalid even when
        # every part is valid, and an invalid polygon raises
        # GEOSException (TopologyException: side location conflict) at the
        # request-time unary_union, 500ing the whole alerts endpoint.  Repair
        # the merged footprint; None keeps the alert geometry-less, an
        # already-supported case.
        merged = ensure_valid_polygon(merged)
        if merged is None:
            logger.warning(
                "Unrecoverable merged polygon for %s (event=%r): keeping the "
                "alert without geometry",
                source_id,
                event_text,
            )

        results.append(
            AlertEntry(
                source_id=source_id,
                event=event_text,
                description=description_text,
                severity=severity,
                effective=effective,
                expires=expires,
                area_desc="; ".join(area_descs),
                url=cap_url,
                polygon=merged,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Feed helpers
# ---------------------------------------------------------------------------

# CAP 1.2 XML namespace, used by the ``cap:identifier`` element some Atom
# feeds (MeteoAlarm) carry alongside the plain Atom ``<id>``.
_CAP_XML_NS = "urn:oasis:names:tc:emergency:cap:1.2"


def _normalize_alert_id(value: str) -> str:
    """Strip the ``urn:oid:`` prefix the hub applies inconsistently.

    ``wmo_all.json`` mixes ``urn:oid:2.49...`` and bare ``2.49...`` CAP
    identifiers, and feeds repeat whichever form they happen to publish,
    so every id is compared prefix-free.
    """
    return value.strip().removeprefix("urn:oid:")


def _local_name(tag: object) -> str:
    """Local part of a (possibly namespaced) lxml tag."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _child_text(elem: etree._Element, localname: str) -> str:
    """Text of the first direct child whose tag local name matches."""
    for child in elem:
        if _local_name(child.tag) == localname:
            return (child.text or "").strip()
    return ""


def _is_cap_identifier(tag: object) -> bool:
    """True for ``cap:identifier`` under any prefix (or no prefix)."""
    if not isinstance(tag, str):
        return False
    if not tag.startswith("{"):
        return _local_name(tag) == "identifier"
    ns = tag[1:].split("}", 1)[0]
    return ns == _CAP_XML_NS and _local_name(tag) == "identifier"


def _rss_item_links(feed_bytes: bytes) -> list[tuple[str, Optional[str]]]:
    """Extract (CAP-doc-href, identifier-candidate) tuples from feed bytes.

    Handles both RSS 2.0 (``<channel>`` / ``<item>``) and Atom
    (``<feed>`` / ``<entry>``) feeds -- WMO agencies publish both.  Tags are
    matched by local name so default namespaces, a ``cap:`` prefix, and
    unprefixed feeds all parse the same way.
    """
    try:
        root = etree.fromstring(feed_bytes)
    except etree.XMLSyntaxError:
        return []

    out: list[tuple[str, Optional[str]]] = []

    if _local_name(root.tag) == "feed":
        for entry in root:
            if _local_name(entry.tag) != "entry":
                continue
            links: list[tuple[str, str]] = []
            cap_identifier: Optional[str] = None
            entry_id: Optional[str] = None
            for child in entry:
                if _local_name(child.tag) == "link":
                    href = (child.get("href") or "").strip()
                    if href:
                        links.append((href, (child.get("type") or "").lower()))
                elif _is_cap_identifier(child.tag) and cap_identifier is None:
                    cap_identifier = (child.text or "").strip() or None
                elif _local_name(child.tag) == "id" and entry_id is None:
                    entry_id = (child.text or "").strip() or None

            # Prefer the declared CAP document link; then any .xml href; the
            # entry id is the last resort.  This ordering is what keeps the
            # decoy MeteoAlarm geocode/region links out.
            chosen: Optional[str] = None
            for href, link_type in links:
                if "cap+xml" in link_type:
                    chosen = href
                    break
            if chosen is None:
                for href, _link_type in links:
                    if href.split("?", 1)[0].lower().endswith(".xml"):
                        chosen = href
                        break
            if chosen is None:
                chosen = entry_id
            if not chosen:
                continue
            out.append((chosen, cap_identifier or entry_id or chosen))
        return out

    channel = None
    for child in root:
        if _local_name(child.tag) == "channel":
            channel = child
            break
    if channel is None:
        return out
    for item in channel:
        if _local_name(item.tag) != "item":
            continue
        link = _child_text(item, "link")
        guid = _child_text(item, "guid") or None
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
# NWS zone geometry helpers
# ---------------------------------------------------------------------------

def _polygonal(geom: dict | None) -> Optional[Polygon]:
    """Parse GeoJSON geometry to a Polygon/MultiPolygon, unwrapping
    GeometryCollections by unioning their polygonal members (NWS serves
    some zone geometries as GeometryCollection-wrapped MultiPolygons).
    Returns None for missing, unparseable, empty, or non-polygonal input."""
    if geom is None:
        return None
    try:
        parsed = shape(geom)
    except Exception:
        return None
    if parsed.geom_type == "GeometryCollection":
        members = [
            m for m in parsed.geoms
            if m.geom_type in ("Polygon", "MultiPolygon") and not m.is_empty
        ]
        if not members:
            return None
        try:
            parsed = unary_union(members)
        except GEOSException:
            return None
    if parsed.is_empty or parsed.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    # Zone geometries can be self-intersecting too; hand back valid geometry
    # or None so no invalid polygon ever reaches the store.
    return ensure_valid_polygon(parsed)


def _nws_zone_urls(props: dict) -> list[str]:
    """Resolve an NWS feature's zone references to api.weather.gov URLs.

    Prefers ``properties.affectedZones`` (full URLs).  Falls back to
    ``properties.geocode.UGC`` codes, mapping the 3rd character to the
    forecast (Z) or county (C) zone endpoint; anything else is skipped.
    """
    affected = props.get("affectedZones")
    if isinstance(affected, list):
        urls = [u for u in affected if isinstance(u, str) and u.strip()]
        if urls:
            return urls

    geocode = props.get("geocode")
    ugc = geocode.get("UGC") if isinstance(geocode, dict) else None
    if not isinstance(ugc, list):
        return []
    urls: list[str] = []
    for code in ugc:
        code = str(code).strip().upper()
        prefix = _UGC_ZONE_URL_PREFIXES.get(code[2:3] if len(code) >= 3 else "")
        if prefix:
            urls.append(f"{prefix}{code}")
    return urls


def _zone_id_from_url(url: str) -> str:
    """Zone id used for disk-cache naming: the last path segment (e.g. COZ041)."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _read_zone_cache(cache_path: Path) -> Optional[Polygon]:
    """Read a cached zone polygon; None when missing, stale, or corrupt.

    Corrupt/unparseable cache files are treated as a miss so a bad write
    can never wedge the ingest — the zone is simply refetched.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - float(data["fetched_at"]) > _ZONE_CACHE_TTL:
            return None
        return _polygonal(data.get("geometry"))
    except Exception:
        return None


def _write_zone_cache(cache_path: Path, polygon: Polygon) -> None:
    """Write a resolved zone polygon to the disk cache (atomic tmp + replace)."""
    tmp_path = cache_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"fetched_at": int(time.time()), "geometry": polygon.__geo_interface__},
            f,
        )
    os.replace(tmp_path, cache_path)


def _write_zone_failure(cache_path: Path) -> None:
    """Atomically write a failure marker so genuinely-broken zones are
    retried at most once per day instead of every fetch cycle."""
    tmp_path = cache_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"fetched_at": int(time.time()), "geometry": None, "failed": True},
            f,
        )
    os.replace(tmp_path, cache_path)


def _read_zone_failure(cache_path: Path) -> bool:
    """True iff a fresh failure marker exists for the zone.

    Any unparseable/missing marker or a marker older than
    ``_ZONE_FAILURE_TTL`` reads as False so the zone is refetched.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("failed"):
            return False
        return time.time() - float(data["fetched_at"]) <= _ZONE_FAILURE_TTL
    except Exception:
        return False


def _alert_entry_from_nws(
    feature: dict, props: dict, polygon: Optional[Polygon]
) -> AlertEntry:
    """Build an AlertEntry from a parsed NWS feature (shared by both paths)."""
    # Use headline if present, otherwise event, otherwise description
    headline = props.get("headline", "") or ""
    event = props.get("event", "") or ""
    description = props.get("description", "") or ""
    event_text = headline or event or ""
    description_text = description or headline or ""

    return AlertEntry(
        source_id="nws-api",
        event=event_text,
        description=description_text,
        severity=props.get("severity", "Unknown"),
        effective=props.get("effective", ""),
        expires=props.get("expires", ""),
        area_desc=props.get("areaDesc", ""),
        url=props.get("id", "") or feature.get("id", ""),
        polygon=polygon,
    )


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
        self._geocode_providers = build_default_providers()
        self._tables_ready = asyncio.Event()
        self._cache_dir = Path(cache_dir) if cache_dir else None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(
                max_connections=self._concurrency,
                max_keepalive_connections=self._concurrency,
            )
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=self._timeout,
                headers={"User-Agent": _USER_AGENT},
                # Some agency CAP feeds 301 to a www-canonical host, and this
                # module only issues GETs, so following redirects is safe.
                follow_redirects=True,
            )
        return self._client

    async def _ensure_geocode_tables(self) -> None:
        """Load every registered geocode table (cache-dir backed).

        Each provider is loaded independently: one broken table is logged and
        skipped so the remaining tables -- and the alerts cycle -- keep
        working.
        """
        client = await self._get_client()
        cache_base = self._alerts_cache_base()
        for provider in self._geocode_providers:
            try:
                await ensure_loaded(provider, client, cache_base)
            except Exception as exc:
                logger.warning(
                    "Geocode table %s failed to load: %s", provider.name, exc
                )
                continue
            logger.info(
                "Geocode table %s ready: %d polygons",
                provider.name,
                len(provider.polygons),
            )
        total = sum(len(p.polygons) for p in self._geocode_providers)
        self._tables_ready.set()
        logger.info("Geocode tables ready: %d polygons total", total)

    def _alerts_cache_base(self) -> Path:
        """Cache base shared with ``_ensure_geocode_tables`` (same fallback).

        Multi mode: the pipeline process owns the fetcher, so this disk
        cache has exactly one writer.
        """
        if self._cache_dir is not None:
            base = self._cache_dir / "alerts"
        else:
            import tempfile
            base = Path(tempfile.gettempdir()) / "librewxr_alerts"
        base.mkdir(parents=True, exist_ok=True)
        return base

    async def _fetch_zone_polygons(self, zone_urls: set[str]) -> dict[str, Polygon]:
        """Resolve NWS zone URLs to polygons, disk-cache-first.

        Zone polygons are near-static, so each zone's geometry is cached
        under ``{alerts_cache_base}/zones/{zone_id}.json`` for 30 days and
        refetched only when missing or stale.  Fail-soft: a zone that
        errors, non-200s, or carries null/unparseable geometry is skipped —
        alerts keep whatever other zones resolved, or stay polygon-less
        exactly like today.
        """
        zones_dir = self._alerts_cache_base() / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self._concurrency)
        polygons: dict[str, Polygon] = {}

        async def load_zone(url: str) -> None:
            zone_id = _zone_id_from_url(url)
            cache_path = zones_dir / f"{zone_id}.json"
            polygon = _read_zone_cache(cache_path)
            if polygon is not None:
                polygons[url] = polygon
                return
            if _read_zone_failure(cache_path):
                # Genuinely-broken zone: retried at most once per day, stays
                # silent (and HTTP-free) in between.
                return
            async with sem:
                client = await self._get_client()
                try:
                    resp = await client.get(
                        url,
                        headers={"User-Agent": _USER_AGENT},
                        timeout=self._timeout,
                    )
                except Exception as exc:
                    logger.debug("NWS zone fetch failed for %s: %s", zone_id, exc)
                    return
                if resp.status_code != 200:
                    logger.warning("NWS zone %s returned %d", zone_id, resp.status_code)
                    _write_zone_failure(cache_path)
                    return
                try:
                    polygon = _polygonal(resp.json().get("geometry"))
                except Exception as exc:
                    logger.warning(
                        "Failed to parse NWS zone %s geometry: %s", zone_id, exc
                    )
                    _write_zone_failure(cache_path)
                    return
                if polygon is None:
                    logger.warning("NWS zone %s has no usable geometry", zone_id)
                    _write_zone_failure(cache_path)
                    return
                _write_zone_cache(cache_path, polygon)
                polygons[url] = polygon

        await asyncio.gather(*(load_zone(url) for url in zone_urls))
        return polygons

    async def _fetch_nws_alerts(self) -> list[AlertEntry]:
        """Fetch active US alerts directly from the NWS API.

        The NWS API returns native GeoJSON FeatureCollection with polygons
        already in [lon, lat] order, bypassing the WMO feed lag.

        Zone-based alerts (Tornado Watches, Special Weather Statements, ...)
        ship with ``geometry: null``; their affected zones are resolved to a
        union polygon at ingest so they behave like polygon alerts in every
        query path.  Zone polygons are near-static, so they are disk-cached
        for 30 days — and because this runs in the periodic fetcher, no API
        request ever touches NWS (this replaces the old per-request
        ``?point=`` enrichment).
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                _NWS_API_URL,
                headers={"User-Agent": _USER_AGENT},
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
        # (feature, props, zone_urls) for geometry-less alerts awaiting zone
        # resolution — collected first so zone fetches can be deduped across
        # alerts (watches routinely share zones).
        pending_zones: list[tuple[dict, dict, tuple[str, ...]]] = []

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
                polygon = _polygonal(geom)

            zone_urls = _nws_zone_urls(props)
            if polygon is None and zone_urls:
                # Zone-based alert: resolve after the loop, once the unique
                # zone set is known (dedup across alerts sharing zones).
                pending_zones.append((feature, props, tuple(zone_urls)))
                continue

            entries.append(_alert_entry_from_nws(feature, props, polygon))

        if pending_zones:
            # Unique zone set first — each zone is fetched (or cache-loaded)
            # at most once per cycle even when many alerts share it.
            unique_urls = {u for _, _, urls in pending_zones for u in urls}
            polygons_by_url = await self._fetch_zone_polygons(unique_urls)
            for feature, props, urls in pending_zones:
                parts = [polygons_by_url[u] for u in urls if u in polygons_by_url]
                polygon = None
                if parts:
                    union = unary_union(parts)
                    if not union.is_valid:
                        # Adjacent zones can share boundary points; buffer(0)
                        # repairs the topology without moving the boundary.
                        union = union.buffer(0)
                    if not union.is_empty and union.geom_type in ("Polygon", "MultiPolygon"):
                        polygon = union
                entries.append(_alert_entry_from_nws(feature, props, polygon))

        logger.debug("NWS API: %d active alerts", len(entries))
        return entries

    async def _await_nws(self, nws_task: asyncio.Task) -> list[AlertEntry]:
        """Await the parallel NWS fetch task, never letting its failure kill
        the cycle — a WMO outage must not freeze the US slice either."""
        try:
            return await nws_task
        except Exception:
            logger.warning("NWS alert fetch failed", exc_info=True)
            return []

    async def _fetch_once(self) -> None:
        """Full ingest pipeline."""
        client = await self._get_client()

        # Fetch NWS alerts in parallel with WMO (no CAP XML parsing needed)
        nws_task = asyncio.create_task(self._fetch_nws_alerts())

        # 1. Fetch sources.json
        resp = await retry_get(client, _SOURCES_URL, log_name="wmo_sources")
        if resp is None or resp.status_code != 200:
            logger.warning("Failed to fetch sources.json")
            # WMO is down: still salvage the US slice from the NWS API and
            # keep last-known non-NWS alerts (expiry filtering on the serve
            # side hides stale ones).  fetch_success stays False so the
            # outage stays visible in /health.
            nws_alerts = await self._await_nws(nws_task)
            kept = [a for a in self._store.alerts if a.source_id != "nws-api"]
            self._store.replace_all(kept + nws_alerts, fetch_success=False)
            return
        sources_data = resp.json()
        sources = sources_data.get("sources", [])

        # 2. Fetch wmo_all.json
        resp = await retry_get(client, _WMO_ALL_URL, log_name="wmo_all")
        if resp is None or resp.status_code != 200:
            logger.warning("Failed to fetch wmo_all.json")
            nws_alerts = await self._await_nws(nws_task)
            kept = [a for a in self._store.alerts if a.source_id != "nws-api"]
            self._store.replace_all(kept + nws_alerts, fetch_success=False)
            return
        wmo_all_data = resp.json()
        wmo_all_items = wmo_all_data.get("items", [])

        # Build set of current alert IDs (normalized: the hub mixes
        # ``urn:oid:``-prefixed and bare CAP identifiers)
        current_ids = {
            _normalize_alert_id(item["id"])
            for item in wmo_all_items
            if item.get("id")
        }
        current_agencies = set()
        for record in wmo_all_items:
            cap_url = record.get("capURL", "") or ""
            url = record.get("url", "") or ""
            if cap_url:
                current_agencies.add(cap_url.split("/")[0])
            elif url:
                current_agencies.add(url.split("/")[0])

        # Agency families: sources.json sourceIds drift on the language
        # suffix (``de-dwd-en`` vs the hub's ``de-dwd-de``), so match on
        # (authorityCountry, authorityAbbrev) as a fallback.
        current_families: set[tuple[str, str]] = set()
        for segment in current_agencies:
            parts = segment.split("-")
            if len(parts) >= 2:
                current_families.add((parts[0], parts[1]))

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
            country = src.get("authorityCountry")
            abbrev = src.get("authorityAbbrev")
            family = (country, abbrev) if country and abbrev else None
            if sid in current_agencies or (
                family is not None and family in current_families
            ):
                source_ids.append(sid)

        logger.debug("Fetching alerts from %d WMO sources", len(source_ids))

        # 3. Fetch feeds and CAP XMLs
        # Per-agency feed URLs come from sources.json; a comma-separated
        # value carries several feeds.  The legacy synthesized URL is only a
        # fallback for agencies that declare no feed.
        feed_urls_by_sid: dict[str, list[str]] = {}
        for entry in sources:
            src = entry.get("source", {})
            sid = src.get("sourceId")
            if not sid:
                continue
            declared = src.get("capAlertFeed") or ""
            urls = [u.strip() for u in declared.split(",") if u.strip()]
            feed_urls_by_sid[sid] = urls or [
                f"{_WMO_BASE}/v2/cap-alerts/{sid}/rss.xml"
            ]

        sem = asyncio.Semaphore(self._concurrency)
        all_alerts: list[AlertEntry] = []
        feeds_ok = 0
        feeds_failed = 0
        failed_sids: list[str] = []

        def _matches_current(identifier: Optional[str]) -> bool:
            if not identifier:
                return False
            nid = _normalize_alert_id(identifier)
            if not nid:
                return False
            if nid in current_ids:
                return True
            # Bidirectional substring match: MeteoAlarm appends
            # ``?index_info=...`` to the hub id and Jamaica embeds the hub id
            # inside the CAP href, so neither direction alone is enough.
            return any(nid in cid or cid in nid for cid in current_ids)

        async def process_feed(sid: str, feed_url: str) -> None:
            nonlocal feeds_ok, feeds_failed
            async with sem:
                resp = await retry_get(client, feed_url, log_name=f"rss_{sid}")
            if resp is None or resp.status_code != 200:
                feeds_failed += 1
                if len(failed_sids) < 10:
                    failed_sids.append(sid)
                return
            feeds_ok += 1
            feed_bytes = resp.content
            items = _rss_item_links(feed_bytes)

            # Filter to current alert IDs
            filtered: list[str] = [
                link for link, identifier in items if _matches_current(identifier)
            ]
            if not filtered and items:
                # Tier 2: some agencies (e.g. INMET) publish URL guids whose
                # scheme never matches hub ids.  Take the newest entries --
                # feeds list newest first -- and let the serve side
                # expiry-filter the stale ones that slip through.
                filtered = [link for link, _identifier in items[:100] if link]

            # Fetch CAP XMLs concurrently per feed
            async def fetch_and_extract(cap_link: str) -> list[AlertEntry]:
                async with sem:
                    resp = await retry_get(client, cap_link, log_name=f"cap_{sid}")
                if resp is None or resp.status_code != 200:
                    return []
                return _extract_polygons_from_cap(
                    resp.text, sid, cap_link, self._geocode_providers
                )

            tasks = [asyncio.create_task(fetch_and_extract(link)) for link in filtered]
            for coro in asyncio.as_completed(tasks):
                try:
                    entries = await coro
                    all_alerts.extend(entries)
                except Exception:
                    continue

        await asyncio.gather(
            *(
                process_feed(sid, url)
                for sid in source_ids
                for url in feed_urls_by_sid.get(sid, [])
            )
        )

        # 4. Merge NWS results
        nws_alerts = await self._await_nws(nws_task)
        all_alerts.extend(nws_alerts)

        # 5. Replace store
        if feeds_failed:
            logger.warning(
                "WMO CAP feeds failing: %d of %d feed URLs unreachable (last failures: %s)",
                feeds_failed, feeds_ok + feeds_failed, ", ".join(dict.fromkeys(failed_sids)),
            )
        # Degraded store flag only when feeds were attempted and every feed
        # URL failed -- an empty feed from a healthy agency is not a failure.
        self._store.replace_all(
            all_alerts, fetch_success=(feeds_ok > 0) or not source_ids
        )
        logger.info(
            "Alerts updated: %d total (%d WMO + %d NWS) from %d/%d WMO feed URLs",
            len(all_alerts), len(all_alerts) - len(nws_alerts), len(nws_alerts),
            feeds_ok, feeds_ok + feeds_failed,
        )

    async def _fetch_loop(self) -> None:
        """Background task: wait for geocode tables, then fetch→sleep→repeat."""
        try:
            await self._ensure_geocode_tables()
        except Exception:
            logger.exception("Failed to load geocode tables, alerts disabled")
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
