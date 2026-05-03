# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "LIBREWXR_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 8080
    public_url: str = "http://localhost:8080"
    fetch_interval: int = 600  # seconds between fetches (10 min = radar frame cadence)
    max_frames: int = 12
    max_zoom: int = 12
    tile_cache_mb: int = 200  # Max tile cache size in MB (byte-capped)
    coord_cache_size: int = 2048  # LRU entries per coordinate cache (lower = less RAM)
    memory_limit_mb: int = 0  # Container memory limit in MB (0 = auto-detect)
    memory_pressure_check_interval: int = 30  # Seconds between memory pressure checks
    smooth_radius: float = 2.0  # Gaussian blur radius when smoothing is enabled
    noise_floor_dbz: float = 10.0  # Minimum dBZ to display; lower values are zeroed out
    despeckle_min_neighbors: int = 3  # Min non-zero neighbors (of 8) to keep a pixel; 0 to disable
    webp_quality: int = 65  # WebP quality: 100 = lossless, 1-99 = lossy at that quality
    workers: int = 1  # Number of uvicorn worker processes
    warmer_threads: int = 0  # Render thread pool size (0 = CPU count - 1)
    warm_coord_zoom: int = 6  # Pre-warm coordinate caches up to this zoom (0 = disable)
    warm_overview_zoom: int = 4  # Pre-render ALL tiles up to this zoom on each fetch (-1 = disable)
    warm_overview_zoom_regional: int = 6  # Pre-render tiles overlapping enabled regions up to this zoom (-1 = disable)
    enabled_regions: str = "ALL"  # Region spec: CONUS, US, ALL, or comma-separated region names
    # North American radar data source.  Three modes:
    #   mrms_fallback  - (default) MRMS primary + IEM fallback for USCOMP + MSC
    #                    blending for CACOMP.  Best coverage, slightly more bandwidth.
    #   mrms           - MRMS only, no fallback/blending.  Pure MRMS where available,
    #                    gaps show as empty (IFS fills in).  Least bandwidth.
    #   iem            - Legacy IEM N0Q for USCOMP, MSC standalone for CACOMP.
    #                    NEXRAD-only, no Canadian radar ingest.
    na_source: Literal["mrms", "mrms_fallback", "iem"] = "mrms_fallback"
    iem_base_url: str = "https://mesonet.agron.iastate.edu"
    msc_canada_base_url: str = "https://geo.weather.gc.ca"
    mrms_base_url: str = "https://mrms.ncep.noaa.gov/2D"
    opera_base_url: str = "https://s3.waw3-1.cloudferro.com"
    ecmwf_s3_bucket: str = "openmeteo"
    ecmwf_s3_region: str = "us-west-2"
    ecmwf_s3_prefix: str = "data_spatial/ecmwf_ifs"
    ecmwf_snow_ratio_threshold: float = 0.5
    ecmwf_max_timesteps: int = 0  # 0 = auto (derived from max_frames)
    ecmwf_interpolation: bool = True  # Optical flow interpolation of IFS hourly data to 10-min frames
    # North American NWP source for the chain. "ifs" uses ECMWF IFS as the
    # only source (current behavior). "hrrr" prepends NOAA HRRR-subh as the
    # CONUS-priority source, falling back to IFS outside HRRR's domain.
    na_nwp_source: Literal["ifs", "hrrr"] = "ifs"
    hrrr_s3_bucket: str = "noaa-hrrr-bdp-pds"
    hrrr_s3_region: str = "us-east-1"
    hrrr_publish_delay_minutes: int = 55  # subh files typically publish ~55 min after run init
    # European NWP source for the chain. "ifs" leaves IFS as the only
    # source over Europe. "icon_eu" prepends DWD ICON-EU (~7 km, 3-hourly
    # cycles) ahead of IFS for the European OPERA radar region.
    eu_nwp_source: Literal["ifs", "icon_eu"] = "ifs"
    icon_eu_base_url: str = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
    icon_eu_publish_delay_minutes: int = 240  # main runs typically publish ~3-4h after init; 4h is conservative
    nowcast_enabled: bool = True  # Generate precipitation nowcast via radar extrapolation + IFS
    nowcast_frames: int = 6  # Number of 10-min forecast frames (6 = 60 min)
    nowcast_blend_mode: str = "radar"  # "radar", "blended", or "model"
    satellite_enabled: bool = True  # Fetch and serve IFS-derived cloud cover as satellite tiles
    satellite_max_frames: int = 12  # Number of hourly IFS cloud timesteps to keep
    cache_dir: str = ""  # Persistent cache directory for satellite grids; empty = in-memory only
    # Tile request tracking — observational only; surfaces hot tiles in /health.
    # See docs/adaptive-tile-warming-design.md for the planned use of this data.
    tile_tracking_enabled: bool = True
    tile_tracking_min_zoom: int = 7  # Track only z >= this (overview zooms are pre-warmed anyway)
    tile_tracking_max_entries: int = 10_000  # Cap per-tile counters; halves when full
    cors_origins: list[str] = ["*"]

    def get_ecmwf_max_timesteps(self) -> int:
        """Return effective ECMWF timestep count.

        If ecmwf_max_timesteps > 0, use it as-is (user override).
        Otherwise auto-derive from max_frames + nowcast_frames, with two
        extra hourly buckets so the window still covers both ends of the
        radar/nowcast span when ``now`` falls between hourly IFS marks
        (worst case can drift the window up to ~1h on each side).
        """
        if self.ecmwf_max_timesteps > 0:
            return self.ecmwf_max_timesteps
        past_hours = math.ceil(self.max_frames * self.fetch_interval / 3600)
        future_hours = (
            math.ceil(self.nowcast_frames * self.fetch_interval / 3600)
            if self.nowcast_enabled else 0
        )
        return past_hours + future_hours + 2

    def get_enabled_regions(self) -> list[str]:
        """Resolve the region spec into individual region names."""
        from librewxr.data.regions import resolve_regions
        return resolve_regions(self.enabled_regions)


settings = Settings()
