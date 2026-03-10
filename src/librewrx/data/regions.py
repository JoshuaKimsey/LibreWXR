# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionDef:
    """Definition of a radar composite region.

    Frozen so it's hashable and can be used as an LRU cache key.
    """

    name: str
    west: float
    east: float
    south: float
    north: float
    pixel_size: float  # degrees per pixel
    group: str  # group this region belongs to (e.g. "US")
    # IEM directory names for URL construction
    live_dir: str   # e.g. "USCOMP" (uppercase, used in live image path)
    archive_dir: str  # e.g. "uscomp" (lowercase, used in archive path)

    @property
    def width(self) -> int:
        return int(round((self.east - self.west) / self.pixel_size))

    @property
    def height(self) -> int:
        return int(round((self.north - self.south) / self.pixel_size))


# All available radar composite regions
REGIONS: dict[str, RegionDef] = {
    "USCOMP": RegionDef(
        name="USCOMP",
        west=-126.0, east=-65.0, south=23.0, north=50.0,
        pixel_size=0.005, group="US",
        live_dir="USCOMP", archive_dir="uscomp",
    ),
    "AKCOMP": RegionDef(
        name="AKCOMP",
        west=-170.5, east=-130.5, south=53.2, north=68.7,
        pixel_size=0.01, group="US",
        live_dir="AKCOMP", archive_dir="akcomp",
    ),
    "HICOMP": RegionDef(
        name="HICOMP",
        west=-162.4, east=-152.4, south=15.4, north=24.4,
        pixel_size=0.005, group="US",
        live_dir="HICOMP", archive_dir="hicomp",
    ),
    "PRCOMP": RegionDef(
        name="PRCOMP",
        west=-71.1, east=-61.1, south=13.1, north=23.1,
        pixel_size=0.01, group="US",
        live_dir="PRCOMP", archive_dir="prcomp",
    ),
    "GUCOMP": RegionDef(
        name="GUCOMP",
        west=140.5, east=149.0, south=9.2, north=17.7,
        pixel_size=0.0085, group="US",
        live_dir="GUCOMP", archive_dir="gucomp",
    ),
}

# Group aliases: shorthand names that expand to multiple regions
REGION_GROUPS: dict[str, list[str]] = {
    "CONUS": ["USCOMP"],
    "US": ["USCOMP", "AKCOMP", "HICOMP", "PRCOMP", "GUCOMP"],
}


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
