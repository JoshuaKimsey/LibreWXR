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
  main.py            # FastAPI app, lifespan, uvicorn entry point
  data_pipeline.py   # Standalone fetcher process (multi-worker deployment)
  config.py          # Pydantic Settings (all LIBREWXR_* env vars)
  memory.py          # Memory pressure monitor
  api/
    routes.py        # API endpoints (Rain Viewer-compatible)
    models.py        # Pydantic response models
  sources/                                # Per-source packages (auto-discovered)
    _base.py                              # Protocols + contribution dataclasses
    _helpers.py                           # Shared dBZ encoder + GRIB stderr muzzle
    _shared/                              # Base classes for source families
      arome.py                            # AROMEOverseasGrid (all 5 AROME-OM variants)
    __init__.py                           # Discovery walker, registry, helpers
    world/ifs/                            # ECMWF IFS (global, NWP)
    satellite/gmgsi/                      # NOAA GMGSI (global, LW + VIS composite)
    regional/
      africa/nwp/arome_indien/            # AROME Indien (RE+YT+KM+MG+SW Indian Ocean)
      caribbean/nwp/arome_antilles/       # AROME Antilles (FR-GP+MQ)
      central_america/el_salvador/radar/marn/
      east_asia/taiwan/radar/cwa/
      europe/radar/opera/
      europe/nwp/{icon_eu,dmi_dini}/
      north_america/
        canada/radar/msc_canada/
        canada/nwp/hrdps/
        usa/radar/{iem,mrms}/
        usa/nwp/{hrrr,hrrr_alaska}/
      oceania/nwp/{arome_ncaled,arome_polyn}/  # AROME New Caledonia + French Polynesia
      south_america/nwp/{wrf_smn,arome_guyane}/  # WRF-SMN (cone) + AROME Guyane (FR-GF)
      southeast_asia/malaysia/radar/mmd/
  data/                                   # Cross-cutting infra only
    regions.py       # RegionDef base + REGIONS dict (built from discovery)
    fetcher.py       # Multi-source fetch orchestrator
    store.py         # FrameStore (RadarFrame ring buffer)
    coverage.py      # Radar station coverage masks (parameter-driven)
    nowcast.py       # Nowcast generation (radar extrapolation + IFS blend)
    nwp_source.py    # NWPSource Protocol + NWPChain dispatcher
    nwp_interpolation.py  # Shared optical-flow helper for regional NWP
    radar_cache.py   # Persistent disk cache for radar frames
    alerts_fetcher.py / alerts_store.py   # WMO weather alerts
    master_state.py  # Multi-worker state.json snapshot
    retry.py         # Backoff helper
  tiles/
    renderer.py      # On-demand tile rendering
    satellite_renderer.py  # GMGSI VIS-over-LW composite tiles
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
- **Source dispatch:** `RadarFetcher` walks the auto-discovered radar providers under `sources/` and routes each region to the contributing source. NWP grids work the same way via `collect_nwp_contributions()`.
- **Frame cadence:** 10 minutes, clock-aligned to match Rain Viewer
- **RadarFrame.regions:** `dict[str, np.ndarray]` keyed by region name, uint8 dBZ encoding
- **Projections:** RegionDef supports latlon, LCC (`proj="lcc"`), polar stereographic (`proj="stere"`), and LAEA
- **Tile rendering:** On-demand with byte-capped LRU cache + background tile warmer; Gaussian smoothing radius auto-scales from the local Jacobian (`_compute_blur_radius` in `tiles/renderer.py`) so coarse-grid sources (OPERA LAEA, MRMS, MMD) get more blur at high zoom without over-blurring fine sources at low zoom. Radar sampling under `smooth=1` is bilinear in both padded and unpadded paths
- **ECMWF IFS:** 9km global precipitation from Open-Meteo S3; optical flow interpolation for 10-min frames; reference_time skip avoids redundant downloads
- **Nowcasting:** Radar extrapolation + IFS blending with spatial feathering at radar boundaries
- **Satellite:** NOAA GMGSI hourly global mosaic (LW + VIS), composited at render time as VIS-over-LW with a natural day/night terminator. Latitude grid is Mercator-spaced (`y=atanh(sin(lat))`) — `sample()` inverts Mercator on the queried lat. Disk-edge feathering smooths the ±72.7° cutoff. `LIBREWXR_SATELLITE_ENABLED=false` returns 503 + empty `satellite.infrared` array
- **Memory management:** Radar frames, ECMWF grids, and nowcast data use numpy memmap (temp files); radar fetcher skips timestamps already in store
- **MRMS:** Region-aware — separate `MRMSSource` per product path (CONUS, ALASKA, HAWAII, CARIB, GUAM); directory listing with bisect for archive lookups; gzip retry + eccodes stderr suppression
- **MARN/SNET (El Salvador):** Single S-band radar at San Andrés, 120 km product (`esar82/Images/`) from anonymous GCS bucket `radar-images-sv`; 5-min cadence; filename embeds local time (UTC-6, no DST); decoder maps HSV-style continuous hue gradient (green→cyan→blue→magenta) to dBZ; bucket archive depth ~24 h; MARN license requires citation
- **CWA (Taiwan):** 7-radar QPESUMS composite (`O-A0059-001` / 雷達合成回波) from anonymous AWS S3 bucket `cwaopendata` in `ap-northeast-1`; 10-min cadence; archive key uses UTC+8 timestamp with no separator dot (`{YYYYMMDDHHMM}compref_mosaic.xml`); XML format with raw dBZ as comma-separated scientific-notation floats; data is row-major south-to-north → vertical flip on decode; sentinels `-99`/`-999`; OGDL v1.0 license, attribution required
- **MMD (MET Malaysia):** 12-radar national composite covering Peninsular Malaysia + Borneo + Brunei + Singapore + N. Sumatra via anonymous HTTPS at `api.met.gov.my`; 10-min native cadence; one animated GIF per fetch (1352×570, 6 frames, ~60 min of backfill); 18-stop palette → dBZ Marshall-Palmer table; combined GIF split into MYPENINSULAR + MYEAST sub-rects (peer regions, one fetch shared between them); per-frame timestamps anchored at the current wall-clock 10-min slot because MET publishes each slot ~11 min after its real data time (anchoring on real time leaves the "current" slot perpetually empty); post-decode morphological close fills hairline gaps left by burned-in state-boundary lines; CC-BY-4.0 attribution required

## Adding a New Source

See **`docs/adding-a-source.md`** for the full walkthrough (directory layout, provider function shape, country-dir convention, station + range overrides, NWP priority assignment, worked examples).

Short version:
1. Create a self-contained package under `sources/regional/<continent>/<country>/{radar,nwp}/<source_name>/` (or `sources/world/<source>/` for global sources). Multi-country sources skip the country directory.
2. Implement the fetcher/decoder in `source.py` (radar) or `grid.py` (NWP). Radar sources also need `regions.py` (`RegionDef` definitions + `REGIONS` list + `REGION_GROUP` string) and `stations.py` (`STATION_MAP` + optional `RANGE_OVERRIDES`).
3. In the package `__init__.py`, expose a `radar_provider(settings)` or `nwp_provider(settings, cache_dir)` that returns a `RadarSourceContribution` / `NWPContribution` (or `None` when disabled).
4. Add any new env vars to `config.py`. New sources default to enabled by `*_enabled = True` per project convention.
5. Add a coverage polygon to `scripts/generate_coverage_map.py` and regenerate `docs/coverage-map-radar.png` / `docs/coverage-map-models.png` (script header documents the throwaway-venv recipe).

The discovery walker in `sources/__init__.py` picks up the new package automatically. `data/fetcher.py`, `data/regions.py`, `data/coverage.py`, `api/routes.py`, `main.py`, and `data_pipeline.py` all iterate the provider list / contribution slugs — no per-source plumbing is required. If a new NWP source has a `fetch()` signature different from the standard `(history_seconds, horizon_seconds)` shape, the fetcher uses `inspect.signature` to pass only the kwargs that signature accepts. If a contribution's display name doesn't slug cleanly (non-ASCII, abbreviations), set `slug="…_grid"` on the `NWPContribution` to pin the snapshot / `/health` key.

## Development Conventions

- Always use the project venv (`.venv/`) for pip installs; never install to the system Python
- Commit messages: imperative mood, concise (e.g., "Add precipitation motion arrows")
- SPDX license headers on source files: `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Copyright line: `# Copyright (C) 2026 Joshua Kimsey`
