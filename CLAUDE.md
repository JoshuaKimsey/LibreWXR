# CLAUDE.md - LibreWXR

## Project Overview

LibreWXR is a self-hostable Rain Viewer API replacement. It fetches radar composites from public sources, composites them into map tiles, and serves a Rain Viewer-compatible JSON/tile API. Written in Python with FastAPI. No GDAL dependency.

- **License:** AGPL-3.0-or-later
- **Repo:** JoshuaKimsey/LibreWXR (public)
- **Python:** >=3.11 (Docker image uses 3.12)
- **Package manager:** pip with hatchling build backend

## Quick Start

```bash
# Local development
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m librewxr.main

# Docker
docker compose up --build
```

Configuration is via environment variables prefixed `LIBREWXR_` or a `.env` file. See `src/librewxr/config.py` for all settings.

## Project Structure

```
src/librewxr/
  main.py           # FastAPI app, lifespan, uvicorn entry point
  config.py          # Pydantic Settings (all LIBREWXR_* env vars)
  memory.py          # Memory pressure monitor
  api/
    routes.py        # API endpoints (Rain Viewer-compatible)
    models.py        # Pydantic response models
  data/
    regions.py       # RegionDef definitions and projection params
    sources.py       # Radar source classes (IEM, MSC Canada, OPERA)
    fetcher.py       # Multi-source fetch orchestrator
    store.py         # FrameStore (RadarFrame ring buffer)
    coverage.py      # Radar station coverage masks
    ecmwf_grid.py    # ECMWF IFS global precipitation grid
    ecmwf_interpolation.py  # Optical flow interpolation (hourly -> 10-min)
    nowcast.py       # Nowcast generation (radar extrapolation + IFS blend)
    cloud_grid.py    # IFS-derived cloud cover (satellite layer)
    cloud_cache.py   # Persistent disk cache for cloud grids
  tiles/
    renderer.py      # On-demand tile rendering
    satellite_renderer.py  # Cloud cover → IR-like satellite tiles
    cache.py         # Byte-capped LRU tile cache
    coordinates.py   # Tile/region coordinate transforms
    warmer.py        # Background tile pre-rendering
  colors/
    schemes.py       # Color scheme definitions
```

## Running Tests

```bash
# All tests
pytest

# By marker
pytest -m api
pytest -m ecmwf
pytest -m nowcast
pytest -m sources
pytest -m tiles
pytest -m store
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. Markers are defined in `pyproject.toml`.

## Architecture Notes

- **Multi-region:** US (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP), Canada (CACOMP), Europe (OPERA)
- **Region groups:** CONUS, US, CANADA, EUROPE, ALL (configured via `LIBREWXR_ENABLED_REGIONS`)
- **Source dispatch:** `RadarFetcher` routes each region group to the correct source class
- **Frame cadence:** 10 minutes, clock-aligned to match Rain Viewer
- **RadarFrame.regions:** `dict[str, np.ndarray]` keyed by region name, uint8 dBZ encoding
- **Projections:** RegionDef supports latlon, LCC (`proj="lcc"`), polar stereographic (`proj="stere"`), and LAEA
- **Tile rendering:** On-demand with byte-capped LRU cache + background tile warmer
- **ECMWF IFS:** 9km global precipitation from Open-Meteo S3; optical flow interpolation for 10-min frames; reference_time skip avoids redundant downloads
- **Nowcasting:** Radar extrapolation + IFS blending with spatial feathering at radar boundaries
- **Satellite:** IFS cloud cover (high/mid/low) composited into IR-like tiles; persistent disk cache with atomic writes and model-run backfill
- **Memory management:** Radar frames, ECMWF grids, and nowcast data use numpy memmap (temp files); radar fetcher skips timestamps already in store

## Adding a New Region

1. Define `RegionDef` in `data/regions.py` (with projection params if non-latlon)
2. Create source class in `data/sources.py` (fetch + parse + convert to uint8 dBZ)
3. Add group dispatch in `fetcher.py` `__init__`
4. Add config setting if a new base URL is needed
5. Everything downstream (store, renderer, API, cache) is source-agnostic

## Development Conventions

- Always use the project venv (`.venv/`) for pip installs; never install to the system Python
- Commit messages: imperative mood, concise (e.g., "Add precipitation motion arrows")
- SPDX license headers on source files: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Copyright line: `# Copyright (C) 2026 Joshua Kimsey`
