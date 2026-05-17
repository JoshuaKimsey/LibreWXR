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
    sources.py       # Radar source classes (MRMS, IEM, MSC Canada, MARN, OPERA, CWA)
    mmd_source.py    # MET Malaysia radar source (split out of sources.py)
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

- **Multi-region:** US (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP), Canada (CACOMP), Central America (SVCOMP), Europe (OPERA), Taiwan (TWCOMP), SE Asia (MYPENINSULAR, MYEAST)
- **Region groups:** CONUS, US, CANADA, CENTRAL_AMERICA, EUROPE, SOUTHEAST_ASIA, TAIWAN, ALL (configured via `LIBREWXR_ENABLED_REGIONS`)
- **NA source:** 3-way `LIBREWXR_NA_SOURCE` setting — `mrms_fallback` (default: MRMS + IEM/MSC fallback), `mrms` (MRMS only), `iem` (legacy IEM + MSC)
- **Source dispatch:** `RadarFetcher` routes each region group to the correct source class
- **Frame cadence:** 10 minutes, clock-aligned to match Rain Viewer
- **RadarFrame.regions:** `dict[str, np.ndarray]` keyed by region name, uint8 dBZ encoding
- **Projections:** RegionDef supports latlon, LCC (`proj="lcc"`), polar stereographic (`proj="stere"`), and LAEA
- **Tile rendering:** On-demand with byte-capped LRU cache + background tile warmer
- **ECMWF IFS:** 9km global precipitation from Open-Meteo S3; optical flow interpolation for 10-min frames; reference_time skip avoids redundant downloads
- **Nowcasting:** Radar extrapolation + IFS blending with spatial feathering at radar boundaries
- **Satellite:** IFS cloud cover (high/mid/low) composited into IR-like tiles; persistent disk cache with atomic writes and model-run backfill
- **Memory management:** Radar frames, ECMWF grids, and nowcast data use numpy memmap (temp files); radar fetcher skips timestamps already in store
- **MRMS:** Region-aware — separate `MRMSSource` per product path (CONUS, ALASKA, HAWAII, CARIB, GUAM); directory listing with bisect for archive lookups; gzip retry + eccodes stderr suppression
- **MARN/SNET (El Salvador):** Single S-band radar at San Andrés, 120 km product (`esar82/Images/`) from anonymous GCS bucket `radar-images-sv`; 5-min cadence; filename embeds local time (UTC-6, no DST); decoder maps HSV-style continuous hue gradient (green→cyan→blue→magenta) to dBZ; bucket archive depth ~24 h; MARN license requires citation
- **CWA (Taiwan):** 7-radar QPESUMS composite (`O-A0059-001` / 雷達合成回波) from anonymous AWS S3 bucket `cwaopendata` in `ap-northeast-1`; 10-min cadence; archive key uses UTC+8 timestamp with no separator dot (`{YYYYMMDDHHMM}compref_mosaic.xml`); XML format with raw dBZ as comma-separated scientific-notation floats; data is row-major south-to-north → vertical flip on decode; sentinels `-99`/`-999`; OGDL v1.0 license, attribution required
- **MMD (MET Malaysia):** 12-radar national composite covering Peninsular Malaysia + Borneo + Brunei + Singapore + N. Sumatra via anonymous HTTPS at `api.met.gov.my`; 10-min native cadence; one animated GIF per fetch (1352×570, 6 frames, ~60 min of backfill); 18-stop palette → dBZ Marshall-Palmer table; combined GIF split into MYPENINSULAR + MYEAST sub-rects (peer regions, one fetch shared between them); per-frame timestamps anchored at the current wall-clock 10-min slot because MET publishes each slot ~11 min after its real data time (anchoring on real time leaves the "current" slot perpetually empty); post-decode morphological close fills hairline gaps left by burned-in state-boundary lines; CC-BY-4.0 attribution required; lives in `data/mmd_source.py` (first radar source split out of `sources.py`)

## Adding a New Region

1. Define `RegionDef` in `data/regions.py` (with projection params if non-latlon)
2. Create source class in `data/sources.py` (fetch + parse + convert to uint8 dBZ)
3. Add group dispatch in `fetcher.py` `__init__`
4. Add config setting if a new base URL is needed
5. Everything downstream (store, renderer, API, cache) is source-agnostic
6. Add the new domain to `scripts/generate_coverage_map.py` and regenerate `docs/coverage-map-radar.png` / `docs/coverage-map-models.png` (script header documents the throwaway-venv recipe)

## Development Conventions

- Always use the project venv (`.venv/`) for pip installs; never install to the system Python
- Commit messages: imperative mood, concise (e.g., "Add precipitation motion arrows")
- SPDX license headers on source files: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Copyright line: `# Copyright (C) 2026 Joshua Kimsey`
