# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionDef:
    """Definition of a radar composite region.

    Frozen so it's hashable and can be used as an LRU cache key.
    """

    name: str
    west: float  # geographic bounds in degrees (used for tile overlap checks)
    east: float
    south: float
    north: float
    pixel_size: float  # degrees per pixel (lon axis) for latlon grids
    group: str  # group this region belongs to (e.g. "US")
    pixel_size_y: float = 0.0  # degrees per pixel (lat axis); 0 = same as pixel_size
    # IEM directory names for URL construction (only used by IEM regions)
    live_dir: str = ""
    archive_dir: str = ""
    # Projected grid support
    proj: str = "latlon"  # "latlon", "laea", or "tmerc"
    grid_x_min: float = 0.0   # x of top-left pixel in projection meters
    grid_y_max: float = 0.0   # y of top-left pixel in projection meters
    grid_scale: float = 1000.0  # meters per pixel
    grid_width: int = 0   # explicit grid dimensions; 0 = compute from pixel_size
    grid_height: int = 0
    # Lambert Azimuthal Equal Area parameters (only used when proj="laea")
    laea_lat0: float = 0.0   # latitude of projection origin
    laea_lon0: float = 0.0   # central meridian
    laea_x0: float = 0.0     # false easting (meters)
    laea_y0: float = 0.0     # false northing (meters)
    # Transverse Mercator parameters (only used when proj="tmerc").
    # Spherical TM per Snyder §8 — appropriate for met composites that
    # specify their grid on a sphere (DPC Italy uses R=6371229).
    tmerc_lat0: float = 0.0           # latitude of natural origin
    tmerc_lon0: float = 0.0           # central meridian
    tmerc_radius: float = 6371229.0   # sphere radius (WMO met sphere default)
    tmerc_k0: float = 1.0             # scale factor at the central meridian

    @property
    def _ps_y(self) -> float:
        """Effective latitude pixel size."""
        return self.pixel_size_y if self.pixel_size_y > 0 else self.pixel_size

    @property
    def width(self) -> int:
        if self.grid_width > 0:
            return self.grid_width
        return int(round((self.east - self.west) / self.pixel_size))

    @property
    def height(self) -> int:
        if self.grid_height > 0:
            return self.grid_height
        return int(round((self.north - self.south) / self._ps_y))


# All available radar composite regions
REGIONS: dict[str, RegionDef] = {
    # USCOMP/AKCOMP/HICOMP/PRCOMP/GUCOMP — migrated to
    # ``sources/regional/north_america/usa/radar/`` (shared between
    # IEM and MRMS) and contributed via the discovery walker below.
    #
    # CACOMP (MSC Canada) — migrated to
    # ``sources/regional/north_america/canada/radar/msc_canada/`` and
    # contributed via the discovery walker below.
    #
    # El Salvador MARN/SNET (SVCOMP) — migrated to
    # ``sources/regional/central_america/el_salvador/radar/marn/`` and
    # contributed via the discovery walker below.
    # OPERA pan-European composite — migrated to
    # ``sources/regional/europe/radar/opera/`` and contributed via the
    # discovery walker below.
    #
    # Taiwan CWA (TWCOMP) — migrated to
    # ``sources/regional/east_asia/taiwan/radar/cwa/`` and contributed
    # via the discovery walker below.
    #
    # MET Malaysia (MYPENINSULAR, MYEAST) — migrated to
    # ``sources/regional/southeast_asia/malaysia/radar/mmd/`` and
    # contributed via the discovery walker below.
}

# Group aliases: shorthand names that expand to multiple regions.
# Keep entries in alphabetical order so the list stays scannable as new
# groups are added.
REGION_GROUPS: dict[str, list[str]] = {
    # CANADA contributed by sources/regional/north_america/canada/...
    # CENTRAL_AMERICA contributed by sources/regional/central_america/...
    # Curated alias kept here because it's a subset of US (not a
    # provider contribution).  US itself is built by the discovery
    # walker from ``sources/regional/north_america/usa/radar/``.
    "CONUS": ["USCOMP"],
    # EUROPE contributed by sources/regional/europe/radar/opera/
    # SOUTHEAST_ASIA contributed by sources/regional/southeast_asia/...
    # TAIWAN contributed by sources/regional/east_asia/taiwan/...
    # US contributed by sources/regional/north_america/usa/radar/
}


def _merge_discovered_regions() -> None:
    """Merge regions contributed by source packages into the global maps.

    Each source package may expose a module-level ``REGIONS`` list (of
    ``RegionDef``) and a ``REGION_GROUP`` string.  We pick them up here
    so that adding a new source = creating one directory; nothing in
    this file needs to change.

    The import is deferred to avoid a circular import: ``librewxr.sources``
    pulls in source subpackages, which in turn import ``RegionDef`` from
    this module.  Doing the walk after ``RegionDef``/``REGIONS`` are
    already defined keeps the cycle broken.
    """
    from librewxr.sources import iter_source_packages

    for mod in iter_source_packages():
        contributed = getattr(mod, "REGIONS", None)
        group = getattr(mod, "REGION_GROUP", None)
        if not contributed:
            continue
        for region in contributed:
            REGIONS.setdefault(region.name, region)
        if group:
            bucket = REGION_GROUPS.setdefault(group, [])
            for region in contributed:
                if region.name not in bucket:
                    bucket.append(region.name)
            bucket.sort()


_merge_discovered_regions()


def _apply_bbox_crop() -> None:
    """Crop all latlon regions to LIBREWXR_BBOX if configured.

    Regions entirely outside the box are removed from REGIONS and
    REGION_GROUPS.  Regions that overlap are replaced with a new
    RegionDef whose bounds are the intersection.  Non-latlon (projected)
    regions are left untouched.

    Called once at import time after _merge_discovered_regions().
    """
    try:
        from librewxr.config import settings
    except ImportError:
        return

    bbox = settings.get_bbox()
    if bbox is None:
        return

    south, west, north, east = bbox
    logger.info(
        "BBOX crop enabled: south=%.2f west=%.2f north=%.2f east=%.2f",
        south, west, north, east,
    )

    to_remove: list[str] = []
    to_replace: dict[str, RegionDef] = {}

    for name, region in REGIONS.items():
        if region.proj != "latlon":
            continue

        # No overlap → drop this region entirely
        if (region.east <= west or region.west >= east
                or region.north <= south or region.south >= north):
            to_remove.append(name)
            continue

        # Compute intersection
        new_west = max(region.west, west)
        new_east = min(region.east, east)
        new_south = max(region.south, south)
        new_north = min(region.north, north)

        if (new_west == region.west and new_east == region.east
                and new_south == region.south and new_north == region.north):
            continue  # fully contained — no crop needed

        cropped = replace(
            region,
            west=new_west, east=new_east,
            south=new_south, north=new_north,
        )
        to_replace[name] = cropped

    for name in to_remove:
        del REGIONS[name]
        logger.info("BBOX crop: removed %s (no overlap)", name)

    for name, cropped in to_replace.items():
        old = REGIONS[name]
        REGIONS[name] = cropped
        logger.info(
            "BBOX crop: %s [%.1f,%.1f,%.1f,%.1f] → [%.1f,%.1f,%.1f,%.1f] (%d×%d px)",
            name,
            old.south, old.west, old.north, old.east,
            cropped.south, cropped.west, cropped.north, cropped.east,
            cropped.width, cropped.height,
        )

    # Clean up REGION_GROUPS: remove references to dropped regions
    for group_name in list(REGION_GROUPS):
        REGION_GROUPS[group_name] = [
            m for m in REGION_GROUPS[group_name] if m in REGIONS
        ]
        if not REGION_GROUPS[group_name]:
            del REGION_GROUPS[group_name]


_apply_bbox_crop()


def resolve_regions(spec: str) -> list[str]:
    """Resolve a region spec string into a list of individual region names.

    The spec is a comma-separated list of region names, group aliases, or ALL.
    Examples:
        "CONUS"                -> ["USCOMP"]
        "US"                   -> ["USCOMP", "AKCOMP", "HICOMP", "PRCOMP", "GUCOMP"]
        "ALL"                  -> all regions
        "CONUS,HICOMP"         -> ["USCOMP", "HICOMP"]
        "USCOMP,AKCOMP"        -> ["USCOMP", "AKCOMP"]
    """
    tokens = [t.strip().upper() for t in spec.split(",") if t.strip()]
    result: list[str] = []

    for token in tokens:
        if token == "ALL":
            return list(REGIONS.keys())
        elif token in REGION_GROUPS:
            for name in REGION_GROUPS[token]:
                if name not in result:
                    result.append(name)
        elif token in REGIONS:
            if token not in result:
                result.append(token)
        else:
            logger.warning("Unknown region or group '%s', skipping", token)

    if not result:
        logger.warning("No valid regions resolved, defaulting to CONUS")
        result = ["USCOMP"]

    return result
