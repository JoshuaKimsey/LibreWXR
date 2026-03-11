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
    # Projected grid support (e.g. Lambert Conformal Conic)
    proj: str = "latlon"  # "latlon" or "lcc"
    # LCC parameters (only used when proj="lcc")
    lcc_lat0: float = 0.0   # latitude of projection origin
    lcc_lon0: float = 0.0   # central meridian
    lcc_lat1: float = 0.0   # standard parallel (lat1 = lat2 for this LCC)
    lcc_R: float = 6371000.0  # earth radius in meters
    grid_x_min: float = 0.0   # x of top-left pixel in projection meters
    grid_y_max: float = 0.0   # y of top-left pixel in projection meters
    grid_scale: float = 1000.0  # meters per pixel
    grid_width: int = 0   # explicit grid dimensions; 0 = compute from pixel_size
    grid_height: int = 0
    # Polar stereographic parameters (only used when proj="stere")
    stere_lat_ts: float = 0.0   # true-scale latitude
    stere_lon0: float = 0.0     # central meridian
    stere_x0: float = 0.0       # false easting (meters)
    stere_y0: float = 0.0       # false northing (meters)

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
    # Nordic countries composite (MET Norway)
    # Native Lambert Conformal Conic grid at 1km resolution
    # LCC params: lat_0=63, lon_0=15, lat_1=lat_2=63, R=6371000
    "NORDIC": RegionDef(
        name="NORDIC",
        west=-8.1, east=40.7, south=53.1, north=71.8,
        pixel_size=0.028808, group="NORDIC",
        pixel_size_y=0.009585,
        proj="lcc",
        lcc_lat0=63.0, lcc_lon0=15.0, lcc_lat1=63.0,
        lcc_R=6371000.0,
        grid_x_min=-796500.0, grid_y_max=1125500.0, grid_scale=1000.0,
        grid_width=1694, grid_height=1951,
    ),
    # Germany composite (DWD)
    # Polar stereographic DE4800 grid at 250m resolution
    # +proj=stere +lat_ts=60 +lat_0=90 +lon_0=10 +x_0=543571.835 +y_0=3622213.862
    "GERMANY": RegionDef(
        name="GERMANY",
        west=1.4, east=18.8, south=45.6, north=55.9,
        pixel_size=0.0035, group="GERMANY",
        pixel_size_y=0.00215,
        proj="stere",
        stere_lat_ts=60.0, stere_lon0=10.0,
        stere_x0=543571.83521776402, stere_y0=3622213.8619310022,
        grid_x_min=0.0, grid_y_max=0.0, grid_scale=250.0,
        grid_width=4400, grid_height=4800,
    ),
}

# Group aliases: shorthand names that expand to multiple regions
REGION_GROUPS: dict[str, list[str]] = {
    "CONUS": ["USCOMP"],
    "US": ["USCOMP", "AKCOMP", "HICOMP", "PRCOMP", "GUCOMP"],
    "NORDIC": ["NORDIC"],
    "GERMANY": ["GERMANY"],
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
