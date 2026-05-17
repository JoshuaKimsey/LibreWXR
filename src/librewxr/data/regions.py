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
    # Projected grid support
    proj: str = "latlon"  # "latlon" or "laea"
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
    # Canada composite (Environment and Climate Change Canada via MSC GeoMet WMS)
    # Latlon grid; MSC serves pre-colored PNG only, decoded via palette
    # reverse-engineering.  Resolution chosen for a ~3560x1720 single-request
    # WMS tile (under typical server size caps).
    "CACOMP": RegionDef(
        name="CACOMP",
        west=-141.0, east=-52.0, south=41.0, north=84.0,
        pixel_size=0.025, group="CANADA",
    ),
    # El Salvador MARN/SNET — single S-band radar at San Andrés, 120 km
    # product.  PNG with RGB palette; anonymous Google Cloud Storage bucket
    # (radar-images-sv); 5-min cadence.  Pixel grid is slightly anisotropic
    # (~0.926 km lon × ~1.02 km lat — set both pixel sizes explicitly).
    "SVCOMP": RegionDef(
        name="SVCOMP",
        west=-90.833, east=-87.044, south=12.112, north=15.244,
        pixel_size=0.00926, pixel_size_y=0.00916,
        group="CENTRAL_AMERICA",
        grid_width=409, grid_height=342,
    ),
    # OPERA pan-European CIRRUS composite via MeteoGate S3
    # LAEA projection: +proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84
    # 3800x4400 at 1 km, 5-minute cadence, ODIM HDF5 with float64 dBZ
    # bbox trimmed to actual European radar network extent (Iceland–Turkey,
    # southern Mediterranean–northern Scandinavia), NOT the full LAEA grid.
    "OPERA": RegionDef(
        name="OPERA",
        west=-25.0, east=45.0, south=34.0, north=72.0,
        pixel_size=0.01, group="EUROPE",
        proj="laea",
        laea_lat0=55.0, laea_lon0=10.0,
        laea_x0=1950000.0, laea_y0=-2100000.0,
        grid_x_min=0.0, grid_y_max=0.0, grid_scale=1000.0,
        grid_width=3800, grid_height=4400,
    ),
    # Taiwan CWA QPESUMS composite (O-A0059-001) via anonymous AWS S3
    # (cwaopendata in ap-northeast-1).  XML format with raw dBZ as
    # scientific-notation floats; 921x881 at 0.0125° (~1.4 km), 10-min
    # cadence.  Row-major south-to-north → vertical flip on decode.
    # Datum is TWD67 (sub-pixel offset vs WGS84 at this resolution).
    "TWCOMP": RegionDef(
        name="TWCOMP",
        west=115.0, east=126.5125, south=18.0, north=29.0125,
        pixel_size=0.0125, group="TAIWAN",
        grid_width=921, grid_height=881,
    ),
    # MET Malaysia (Jabatan Meteorologi Malaysia) composite covering
    # Peninsular Malaysia + N. Sumatra (MYPENINSULAR) and East Malaysia
    # / Borneo + Brunei (MYEAST).  Both regions are sub-rectangles of a
    # single 1352x570 animated GIF served under CC-BY-4.0 from
    # ``api.met.gov.my/static/images/radar-latest.gif`` (6 frames at
    # 10-min cadence, ~60 min of backfill per fetch).
    #
    # The combined GIF renders both coverage zones in a shared
    # equirectangular grid over the union bounding box, with a vertical
    # band of pure sea (South China Sea) between the peninsular and east
    # coverage.  Pixels are non-square (45.32 px/° lon, 53.47 px/° lat) —
    # ``pixel_size_y`` captures the latitude axis separately.  Bounds
    # match RainViewer's published ``MYCOMP71`` / ``MYCOMP72`` metadata
    # (data.rainviewer.com/images/MYCOMP71/0_products.json,
    # MYCOMP72/0_products.json) which mirrors the same MET Malaysia
    # products.  The internal Rainbow 5 / LEONARDO processing software's
    # native Albers Equal Area projection is treated as equirect for
    # serving purposes — same simplification RainViewer makes for the
    # other regional sources (e.g. MARN El Salvador).
    #
    # The Singapore MSS coverage that briefly lived in this group (2026-
    # 05-15 → 2026-05-16) was removed because MET Malaysia's KLIA radar
    # already covers the same area through MYPENINSULAR, and overlapping
    # the higher-resolution MSS box on top produced a visible seam at
    # the SGCOMP rectangle edges with no clean composite path.
    "MYPENINSULAR": RegionDef(
        name="MYPENINSULAR",
        west=96.92, east=106.28, south=-1.33, north=8.97,
        pixel_size=1.0 / 45.323,           # 0.02207°/px (lon axis)
        pixel_size_y=1.0 / 53.471,         # 0.01870°/px (lat axis)
        group="SOUTHEAST_ASIA",
        grid_width=424, grid_height=551,
    ),
    "MYEAST": RegionDef(
        name="MYEAST",
        west=107.08, east=121.19, south=-1.48, north=9.18,
        pixel_size=1.0 / 45.323,           # 0.02207°/px (lon axis)
        pixel_size_y=1.0 / 53.471,         # 0.01870°/px (lat axis)
        group="SOUTHEAST_ASIA",
        grid_width=640, grid_height=570,
    ),
}

# Group aliases: shorthand names that expand to multiple regions.
# Keep entries in alphabetical order so the list stays scannable as new
# groups are added.
REGION_GROUPS: dict[str, list[str]] = {
    "CANADA": ["CACOMP"],
    "CENTRAL_AMERICA": ["SVCOMP"],
    "CONUS": ["USCOMP"],
    "EUROPE": ["OPERA"],
    "SOUTHEAST_ASIA": ["MYPENINSULAR", "MYEAST"],
    "TAIWAN": ["TWCOMP"],
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
