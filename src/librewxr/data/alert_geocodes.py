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

Registered tables
-----------------

``meteoalarm``
    EMMA_ID (+ NUTS2/NUTS3 aliases) region polygons for Europe, from Pirate
    Weather's derived MeteoAlarm tables.

``dwd_warncell``
    German DWD ``Warngebiete`` polygons keyed by ``WARNCELLID`` (Gemeinden,
    Kreise, coastal waters, inland lakes).  Warngebiete data (c) Deutscher
    Wetterdienst, CC BY 4.0.  Underlying administrative geometry:
    Geobasisdaten (c) GeoBasis-DE / BKG 2021 (Daten modifiziert), used under
    GeoNutzV -- both attributions must travel with derived products.

``bom_amoc``
    Australian Bureau of Meteorology AMOC/AAC areas keyed by
    ``AMOC-AreaCode`` (public weather districts, marine waters, fire weather
    areas, metro areas, flood catchments).  (c) Commonwealth of Australia,
    Bureau of Meteorology, CC BY 4.0.

Still future: CMA ``CPEAS Geographic Code`` (see the exclusion note in
``alerts_fetcher._EXCLUDED_SOURCE_IDS``).
"""

import asyncio
import csv
import json
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

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
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _repair_polygon(poly: Optional[Polygon]) -> Optional[Polygon]:
    """Repair invalid geometry with ``buffer(0)``; None when unrecoverable."""
    if poly is None or poly.is_empty:
        return None
    if poly.is_valid:
        return poly
    try:
        fixed = poly.buffer(0)
    except Exception:
        return None
    if fixed.is_empty or not fixed.is_valid:
        return None
    return fixed


def _shape_with_repair(geom: dict) -> Optional[Polygon]:
    """Build a shapely polygon from GeoJSON, repairing invalid geometry."""
    try:
        poly = shape(geom)
    except Exception:
        return None
    return _repair_polygon(poly)


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a temp file + replace (no torn cache)."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# DWD Warngebiete table (Germany) -- WARNCELLID -> polygon
# ---------------------------------------------------------------------------

_DWD_WFS_URL = (
    "https://maps.dwd.de/geoserver/wfs?service=WFS&version=2.0.0"
    "&request=GetFeature&typeNames=dwd:{layer}&outputFormat=application/json"
)

# Layers carrying warn-cell geometry.  ``Warngebiete_Bundeslaender`` is
# deliberately absent: CAP alerts never reference state-level codes, so the
# layer would only add unused polygons.
_DWD_LAYERS = (
    "Warngebiete_Gemeinden",   # municipalities (~11.2k cells, the bulk)
    "Warngebiete_Kreise",      # districts
    "Warngebiete_Kueste",      # coastal waters
    "Warngebiete_Binnenseen",  # inland lakes
)


def _parse_dwd_geojson(path: Path) -> dict[str, Polygon]:
    """Parse one DWD Warngebiete layer into {WARNCELLID: Polygon}.

    ``WARNCELLID`` arrives as a JSON number; keys are zero-padded to the
    9-digit form CAP alerts publish.  Invalid rings are repaired exactly like
    the MeteoAlarm table (``buffer(0)``).
    """
    cells: dict[str, Polygon] = {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        raw = props.get("WARNCELLID")
        geom = feature.get("geometry")
        if raw is None or not geom:
            continue
        if isinstance(raw, float):
            raw = int(raw)
        code = str(raw).strip().zfill(9)
        if not code:
            continue
        poly = _shape_with_repair(geom)
        if poly is None:
            logger.debug("Dropped DWD warn cell %s: unusable geometry", code)
            continue
        cells[code] = poly
    return cells


def _wfs_count(value: object) -> Optional[int]:
    """Parse a WFS 2.0 count field; None when absent or non-numeric.

    GeoServer emits these as ints, but the spec allows strings and the
    literal ``"unknown"``, so anything unparseable is treated as absent.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _check_wfs_complete(layer: str, payload: dict) -> None:
    """Fail loudly on a truncated FeatureCollection.

    A Gemeinden layer that silently returns fewer features than matched would
    under-cover Germany, so a comparable ``numberReturned < numberMatched``
    pair is fatal.  Missing/non-numeric values are accepted (lenient).
    """
    matched = _wfs_count(payload.get("numberMatched"))
    returned = _wfs_count(payload.get("numberReturned"))
    if matched is None or returned is None:
        return
    if returned < matched:
        raise RuntimeError(
            f"DWD layer {layer} truncated: "
            f"{returned} of {matched} features returned"
        )


async def _download_dwd_layer(
    client: httpx.AsyncClient, layer: str, dest: Path
) -> None:
    """Download one Warngebiete layer; raises so the loader can fail loudly."""
    resp = await retry_get(
        client, _DWD_WFS_URL.format(layer=layer), log_name=f"dwd_{layer}"
    )
    if resp is None or resp.status_code != 200:
        raise RuntimeError(f"Failed to download DWD layer {layer}")
    try:
        payload = json.loads(resp.content)
    except ValueError as exc:
        raise RuntimeError(f"DWD layer {layer} returned non-JSON: {exc}") from exc
    _check_wfs_complete(layer, payload)
    _atomic_write(dest, resp.content)


async def _load_dwd_warncell(
    client: httpx.AsyncClient, cache_base: Path
) -> dict[str, Polygon]:
    """Load the DWD Warngebiete layers, downloading any missing ones."""
    paths = {
        layer: cache_base / f"dwd_warncell_{layer.split('_', 1)[-1].lower()}.json"
        for layer in _DWD_LAYERS
    }
    missing = [layer for layer, path in paths.items() if not path.exists()]
    if missing:
        logger.info(
            "Downloading DWD Warngebiete layers (%s, ~12 MB)...", ", ".join(missing)
        )
        results = await asyncio.gather(
            *(_download_dwd_layer(client, layer, paths[layer]) for layer in missing),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            # Layers that did arrive stay cached; the next cycle retries only
            # the missing ones.
            raise failures[0]

    merged: dict[str, Polygon] = {}
    for layer in _DWD_LAYERS:
        try:
            cells = _parse_dwd_geojson(paths[layer])
        except Exception:
            # A corrupt cache file would stay broken forever otherwise: drop
            # it so the next cycle re-downloads this layer.
            paths[layer].unlink(missing_ok=True)
            raise
        for code, poly in cells.items():
            if code in merged:
                logger.debug("DWD WARNCELLID %s duplicated by %s", code, layer)
                continue
            merged[code] = poly
        logger.debug("DWD layer %s: %d warn cells", layer, len(cells))
    logger.info("DWD Warngebiete loaded: %d warn cells", len(merged))
    return merged


# ---------------------------------------------------------------------------
# BoM AMOC/AAC tables (Australia) -- AMOC-AreaCode -> polygon
# ---------------------------------------------------------------------------

# Official AAC geometry lives in ESRI shapefile zips on BoM's anonymous FTP.
# HTTPS is not an option in practice: ``www.bom.gov.au/anon/...`` answers 403
# (the anon area is FTP-only) and ``ftp.bom.gov.au`` does not listen on 443,
# so the HTTPS candidates are tried first only in case BoM ever exposes the
# path and plain FTP is the scheme that works today.
_BOM_HTTPS_HOST = "www.bom.gov.au"
_BOM_FTP_HOST = "ftp.bom.gov.au"
_BOM_HTTP_PATH = "/anon/home/adfd/spatial/{product}.zip"

# (product id, label).  ``IDM00013`` (PT place points) is deliberately absent:
# points cannot serve as alert polygons.
_BOM_PRODUCTS = (
    ("IDM00001", "PW public weather districts"),
    ("IDM00003", "MW marine waters"),
    ("IDM00007", "FW fire weather areas"),
    ("IDM00014", "ME metro areas"),
    ("IDM00017", "RC flood catchments"),
)


def _bom_url_candidates(product: str) -> list[str]:
    """Candidate download URLs for one BoM product, preferred scheme first."""
    path = _BOM_HTTP_PATH.format(product=product)
    return [
        f"https://{_BOM_HTTPS_HOST}{path}",
        f"ftp://{_BOM_FTP_HOST}{path}",
    ]


def _signed_ring_area(ring: list[tuple[float, float]]) -> float:
    """Shoelace area of a ring; negative means clockwise (ESRI exterior)."""
    total = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _polygon_from_parts(
    parts: list[int], points: list[tuple[float, float]]
) -> Optional[Polygon]:
    """Build one (Multi)Polygon from a shapefile feature's ring structure.

    ESRI convention: exterior rings run clockwise (negative shoelace area),
    interior rings (holes) anticlockwise.
    """
    rings: list[list[tuple[float, float]]] = []
    for index, start in enumerate(parts):
        end = parts[index + 1] if index + 1 < len(parts) else len(points)
        ring = [(float(x), float(y)) for x, y in points[start:end]]
        if len(ring) >= 4:
            rings.append(ring)
    if not rings:
        return None

    shells = [ring for ring in rings if _signed_ring_area(ring) < 0]
    holes = [ring for ring in rings if _signed_ring_area(ring) >= 0]
    if not shells:
        # Degenerate winding: treat every ring as its own shell.
        shells, holes = rings, []

    polygons: list[Polygon] = []
    for shell in shells:
        shell_poly = Polygon(shell)
        shell_holes = [
            hole for hole in holes
            if shell_poly.contains(Polygon(hole).representative_point())
        ]
        polygons.append(Polygon(shell, shell_holes) if shell_holes else shell_poly)
    merged = polygons[0] if len(polygons) == 1 else unary_union(polygons)
    return _repair_polygon(merged)


def _parse_bom_zip(path: Path, name: str) -> dict[str, Polygon]:
    """Parse one BoM AAC shapefile zip into {AAC: Polygon}.

    The code field is the first record field whose name starts with ``AAC``
    (the exact name varies per product: AAC, AAC_MW, AAC_FW, ...).
    """
    try:
        import shapefile  # pyshp -- optional, see pyproject dependencies
    except ImportError as exc:
        raise RuntimeError(
            "pyshp is required for the BoM AMOC geocode table: pip install pyshp"
        ) from exc

    with zipfile.ZipFile(path) as archive:
        shp_member = next(
            (m for m in archive.namelist() if Path(m).suffix.lower() == ".shp"),
            None,
        )
        if shp_member is None:
            logger.debug("BoM %s: %s has no .shp member", name, path.name)
            return {}
        # Only shp/shx/dbf: some products ship a .CPG whose "ansi 1252"
        # payload pyshp cannot resolve as a codec name.
        with tempfile.TemporaryDirectory(prefix="librewxr_bom_") as tmpdir:
            for member in archive.namelist():
                if Path(member).suffix.lower() in (".shp", ".shx", ".dbf"):
                    archive.extract(member, tmpdir)
            reader = shapefile.Reader(
                str(Path(tmpdir) / shp_member),
                encoding="latin-1",
                encodingErrors="replace",
            )
            fields = [
                str(f.name if hasattr(f, "name") else f[0])
                for f in reader.fields
            ]
            # pyshp lists a synthetic DeletionFlag first; records skip it.
            offset = 1 if fields and fields[0].lower() == "deletionflag" else 0
            aac_index = next(
                (
                    index
                    for index, field in enumerate(fields)
                    if field.upper().startswith("AAC")
                ),
                None,
            )
            if aac_index is None:
                logger.debug(
                    "BoM %s: no AAC field (fields=%s)", name, fields[:8]
                )
                return {}
            record_index = aac_index - offset

            cells: dict[str, Polygon] = {}
            skipped = 0
            for shape_record in reader.iterShapeRecords():
                shp = shape_record.shape
                if shp.shapeType not in (5, 15, 25):
                    skipped += 1
                    continue
                code = str(shape_record.record[record_index]).strip().upper()
                if not code:
                    skipped += 1
                    continue
                poly = _polygon_from_parts(list(shp.parts), shp.points)
                if poly is None:
                    skipped += 1
                    continue
                cells[code] = poly
            if skipped:
                logger.debug("BoM %s: skipped %d unusable shapes", name, skipped)
    return cells


# BoM FTP transfers: a per-socket timeout plus a hard deadline on the whole
# transfer.  ``urlretrieve`` has no timeout knob, so one stalled FTP session
# could otherwise pin a worker thread indefinitely.
_BOM_FTP_SOCKET_TIMEOUT = 60.0
_BOM_FTP_DEADLINE = 600.0


def _ftp_download(url: str, dest: Path, timeout: float) -> None:
    """Download an FTP URL to ``dest`` with a bounded socket timeout."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)


async def _download_bom_product(
    client: httpx.AsyncClient, product: str, dest: Path
) -> None:
    """Download one BoM shapefile zip, trying the candidate URLs in order."""
    part = dest.with_suffix(dest.suffix + ".part")
    errors: list[str] = []
    for url in _bom_url_candidates(product):
        try:
            if url.startswith("ftp://"):
                # wait_for is a second guard: the thread may outlive it (the
                # orphan simply finishes or dies on its socket timeout).
                await asyncio.wait_for(
                    asyncio.to_thread(
                        _ftp_download, url, part, _BOM_FTP_SOCKET_TIMEOUT
                    ),
                    timeout=_BOM_FTP_DEADLINE,
                )
            else:
                resp = await retry_get(client, url, log_name=f"bom_{product}")
                if resp is None or resp.status_code != 200:
                    status = "error" if resp is None else resp.status_code
                    errors.append(f"{url}: HTTP {status}")
                    continue
                part.write_bytes(resp.content)
            if not zipfile.is_zipfile(part):
                errors.append(f"{url}: payload is not a zip archive")
                continue
            part.replace(dest)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
    part.unlink(missing_ok=True)
    raise RuntimeError(
        f"Failed to download BoM {product} (" + "; ".join(errors) + ")"
    )


async def _load_bom_amoc(
    client: httpx.AsyncClient, cache_base: Path
) -> dict[str, Polygon]:
    """Load the BoM AAC shapefiles, downloading any missing ones."""
    paths = {
        product: cache_base / f"bom_amoc_{product}.zip"
        for product, _label in _BOM_PRODUCTS
    }
    missing = [
        (product, label)
        for product, label in _BOM_PRODUCTS
        if not paths[product].exists()
    ]
    if missing:
        logger.info(
            "Downloading BoM AAC shapefiles (%s, ~22 MB)...",
            ", ".join(product for product, _ in missing),
        )
        results = await asyncio.gather(
            *(
                _download_bom_product(client, product, paths[product])
                for product, _label in missing
            ),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            # Products that did arrive stay cached; the next cycle retries
            # only the missing ones.
            raise failures[0]

    merged: dict[str, Polygon] = {}
    for product, label in _BOM_PRODUCTS:
        try:
            cells = _parse_bom_zip(paths[product], label)
        except Exception:
            # A corrupt zip would stay broken forever otherwise: drop it so
            # the next cycle re-downloads this product.
            paths[product].unlink(missing_ok=True)
            raise
        for code, poly in cells.items():
            if code in merged:
                logger.debug("BoM AAC %s duplicated by %s", code, product)
                continue
            merged[code] = poly
        logger.debug("BoM %s (%s): %d areas", product, label, len(cells))
    logger.info("BoM AMOC areas loaded: %d", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

def build_default_providers() -> list[GeocodeProvider]:
    """The registered tables, in resolution order.

    MeteoAlarm first (its EMMA_ID/NUTS2/NUTS3 claims are disjoint from the
    national tables), then the national data drops:

    - DWD ``WARNCELLID`` (Germany)
    - BoM ``AMOC-AreaCode`` (Australia)

    Still future: CMA ``CPEAS Geographic Code`` (China; see the exclusion
    note in ``alerts_fetcher._EXCLUDED_SOURCE_IDS``).
    """
    return [
        GeocodeProvider(
            name="meteoalarm",
            # NUTS3 alongside EMMA_ID; NUTS2 because the MeteoAlarm table
            # carries NUTS2 aliases (Belgium publishes NUTS2 codes, e.g. BE23).
            value_names=frozenset({"EMMA_ID", "NUTS2", "NUTS3"}),
            loader=_load_meteoalarm,
        ),
        GeocodeProvider(
            name="dwd_warncell",
            value_names=frozenset({"WARNCELLID"}),
            loader=_load_dwd_warncell,
        ),
        GeocodeProvider(
            name="bom_amoc",
            # CAP spells the field "AMOC-AreaCode"; resolve() upper-cases the
            # incoming valueName before matching, so store it upper-case.
            value_names=frozenset({"AMOC-AREACODE"}),
            loader=_load_bom_amoc,
        ),
    ]
