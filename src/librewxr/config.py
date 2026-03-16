# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "LIBREWXR_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 8080
    public_url: str = "http://localhost:8080"
    fetch_interval: int = 300  # seconds between fetches
    max_frames: int = 12
    max_zoom: int = 12
    tile_cache_mb: int = 200  # Max tile cache size in MB (byte-capped)
    coord_cache_size: int = 2048  # LRU entries per coordinate cache (lower = less RAM)
    memory_limit_mb: int = 0  # Container memory limit in MB (0 = auto-detect)
    memory_pressure_check_interval: int = 30  # Seconds between memory pressure checks
    smooth_radius: float = 2.0  # Gaussian blur radius when smoothing is enabled
    noise_floor_dbz: float = 10.0  # Minimum dBZ to display; lower values are zeroed out
    despeckle_min_neighbors: int = 3  # Min non-zero neighbors (of 8) to keep a pixel; 0 to disable
    webp_quality: int = 100  # WebP quality: 100 = lossless, 1-99 = lossy at that quality
    workers: int = 1  # Number of uvicorn worker processes
    warmer_threads: int = 4  # Thread pool size for background tile warming
    enabled_regions: str = "CONUS"  # Region spec: CONUS, US, ALL, or comma-separated region names
    iem_base_url: str = "https://mesonet.agron.iastate.edu"
    met_nordic_base_url: str = "https://thredds.met.no"
    dwd_base_url: str = "https://opendata.dwd.de"
    cors_origins: list[str] = ["*"]

    def get_enabled_regions(self) -> list[str]:
        """Resolve the region spec into individual region names."""
        from librewxr.data.regions import resolve_regions
        return resolve_regions(self.enabled_regions)


settings = Settings()
