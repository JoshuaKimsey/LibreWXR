# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Pluggable CAP geocode -> polygon resolver tables.

A CAP ``<area>`` often carries no ``<polygon>`` at all: it references a
national or regional area code through ``<geocode>`` instead (EMMA_ID in
Europe, DWD WARNCELLID in Germany, AMOC in Australia, CPEAS in China, ...).
Each table mapping such a code to a polygon is a :class:`GeocodeProvider`: a
set of claimed CAP ``valueName`` strings plus an async loader filling a
``{code: Polygon}`` dict.

Registering a new national table is therefore a data drop in
:func:`build_default_providers` -- no parser surgery required.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
from shapely.geometry import Polygon, shape

from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MeteoAlarm table (Pirate Weather's GitHub repo, dev branch)
# ---------------------------------------------------------------------------

_METEOALARM_GEOJSON_URL = (
    "https://raw.githubusercontent.com/Pirate-Weather/"
    "pirate-weather-code/dev/API/data/meteoalarm_geocodes.json"
)
_METEOALARM_ALIASES_URL = (
    "https://raw.githubusercontent.com/Pirate-Weather/"
    "pirate-weather-code/dev/API/data/meteoalarm_aliases.csv"
)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

def _default_normalize(value: str) -> str:
    """Default code normalization: trim + upper-case."""
    return value.strip().upper()


@dataclass
class GeocodeProvider:
    """One area-code table that can satisfy a CAP ``<geocode>``.

    ``value_names`` are the CAP ``valueName`` strings this table claims
    (stored upper-case, matched case-insensitively); ``polygons`` is the
    loaded ``{code: Polygon}`` table and starts empty until ``loader`` runs.
    """

    name: str
    value_names: frozenset[str]
    loader: Callable[[httpx.AsyncClient, Path], Awaitable[dict[str, Polygon]]]
    normalize: Callable[[str], str] = _default_normalize
    polygons: dict[str, Polygon] = field(default_factory=dict)


async def ensure_loaded(
    provider: GeocodeProvider,
    client: httpx.AsyncClient,
    cache_base: Path,
) -> None:
    """Load a provider's table on first use (idempotent).

    No-op once ``provider.polygons`` is non-empty, so repeated fetch cycles
    reuse the in-memory table instead of re-parsing the on-disk files.
    """
    if provider.polygons:
        return
    provider.polygons = await provider.loader(client, cache_base)


def resolve(
    providers: list[GeocodeProvider],
    value_name: str,
    code: str,
) -> Optional[Polygon]:
    """Resolve a CAP geocode to a polygon via the registered providers.

    Providers are tried in order.  The first claiming provider that holds the
    code wins; a claiming provider that misses does not stop the search,
    because different tables can legitimately claim the same ``valueName``
    with disjoint code sets.
    """
    if not value_name or not code:
        return None
    name = value_name.strip().upper()
    for provider in providers:
        if name not in provider.value_names:
            continue
        polygon = provider.polygons.get(provider.normalize(code))
        if polygon is not None:
            return polygon
    return None


# ---------------------------------------------------------------------------
# MeteoAlarm loader (EMMA_ID + NUTS2/NUTS3 aliases)
# ---------------------------------------------------------------------------

def _parse_meteoalarm_geojson(path: str) -> dict[str, Polygon]:
    """Parse MeteoAlarm geocodes GeoJSON into {EMMA_ID: Polygon} dict.

    A handful of table polygons are self-intersecting or otherwise invalid;
    those are repaired with ``buffer(0)`` instead of being dropped, which is
    what makes regions like Croatia's HR806 resolvable at all.
    """
    geocodes: dict[str, Polygon] = {}
    repaired = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            code = str(props.get("code", "")).strip().upper()
            geom = feature.get("geometry")
            if not code or not geom:
                continue
            try:
                poly = shape(geom)
            except Exception:
                continue
            if not poly.is_valid:
                try:
                    poly = poly.buffer(0)
                except Exception:
                    poly = None
                if poly is None or poly.is_empty or not poly.is_valid:
                    logger.debug(
                        "Dropped invalid MeteoAlarm geocode polygon %s (repair failed)",
                        code,
                    )
                    continue
                repaired += 1
            geocodes[code] = poly
        if repaired:
            logger.debug(
                "Repaired %d invalid MeteoAlarm geocode polygons", repaired
            )
    except Exception as exc:
        logger.warning("Failed to parse MeteoAlarm geocodes: %s", exc)
    return geocodes


def _apply_meteoalarm_aliases(
    geocodes: dict[str, Polygon], alias_csv_path: str
) -> dict[str, Polygon]:
    """Duplicate polygon entries for aliased codes (EMMA_ID -> NUTS2/NUTS3)."""
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


async def _download_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``; raises so the caller can skip the table."""
    resp = await retry_get(client, url, log_name="meteoalarm")
    if resp is None or resp.status_code != 200:
        raise RuntimeError(f"Failed to download {url}")
    with open(dest, "wb") as f:
        f.write(resp.content)


async def _load_meteoalarm(
    client: httpx.AsyncClient, cache_base: Path
) -> dict[str, Polygon]:
    """Load the MeteoAlarm geocode + alias tables, downloading if uncached."""
    geojson_path = cache_base / "meteoalarm_geocodes.json"
    aliases_path = cache_base / "meteoalarm_aliases.csv"

    if not geojson_path.exists():
        logger.info("Downloading MeteoAlarm geocodes (~33 MB)...")
        await _download_file(client, _METEOALARM_GEOJSON_URL, geojson_path)
    if not aliases_path.exists():
        logger.info("Downloading MeteoAlarm aliases...")
        await _download_file(client, _METEOALARM_ALIASES_URL, aliases_path)

    logger.info("Loading MeteoAlarm geocodes...")
    geocodes = _parse_meteoalarm_geojson(str(geojson_path))
    return _apply_meteoalarm_aliases(geocodes, str(aliases_path))


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

def build_default_providers() -> list[GeocodeProvider]:
    """The registered tables, in resolution order.

    Only MeteoAlarm ships today.  Future national tables register here as
    data drops, each with its own loader + claim set, e.g.:

    - DWD ``WARNCELLID`` (Germany, CAP geocode valueName field)
    - BoM ``AMOC-AreaCode`` / ``AMOC-AreaType`` (Australia)
    - CMA ``CPEAS Geographic Code`` (China; see the exclusion note in
      ``alerts_fetcher._EXCLUDED_SOURCE_IDS``)
    """
    return [
        GeocodeProvider(
            name="meteoalarm",
            # NUTS3 alongside EMMA_ID; NUTS2 because the MeteoAlarm table
            # carries NUTS2 aliases (Belgium publishes NUTS2 codes, e.g. BE23).
            value_names=frozenset({"EMMA_ID", "NUTS2", "NUTS3"}),
            loader=_load_meteoalarm,
        ),
    ]
