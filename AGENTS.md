# AGENTS.md - LibreWRX

## Project Overview

LibreWRX is a self-hostable Rain Viewer API replacement. It fetches radar composites from public sources, composites them into map tiles, and serves a Rain Viewer-compatible JSON/tile API. Python + FastAPI, no GDAL dependency.

- **License:** AGPL-3.0-or-later
- **Python:** >=3.11 (Docker uses 3.12)
- **Package manager:** pip with hatchling build backend

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Always use the project venv `.venv/`; never install to system Python.

## Running & Testing

```bash
python -m librewxr.main          # dev server (uvicorn, single mode)
python -m librewxr.data_pipeline # standalone fetcher (multi mode only)
pytest                            # all tests
pytest -m api                     # by marker
pytest tests/test_renderer.py     # single file
pytest -k "test_tile_render"      # by name pattern
```

Test markers (defined in `pyproject.toml`): `api`, `ecmwf`, `nowcast`, `sources`, `tiles`, `store`, `hrrr`, `hrrr_alaska`, `icon_eu`, `dmi_dini`, `hrdps`, `arome_antilles`, `arome_guyane`, `arome_indien`, `arome_ncaled`, `arome_polyn`, `wrf_smn`, `jma_msm`, `alerts`.

All tests are auto-async (`asyncio_mode = "auto"` in pyproject.toml). No explicit `@pytest.mark.asyncio` needed on individual async tests (though some older tests still have it).

No linter/formatter/typechecker is configured — there is no `ruff`, `mypy`, `black`, etc. in the project.

## Project Structure

```
src/librewxr/
  main.py            # FastAPI app, lifespan, uvicorn entry point
  data_pipeline.py   # Standalone fetcher process (multi-worker deployment)
  config.py          # Pydantic Settings (all LIBREWXR_* env vars)
  memory.py          # Memory pressure monitor (cgroup-aware)
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
      east_asia/japan/nwp/jma_msm/          # JMA MSM (Japan + Korean Peninsula + Taiwan + Yellow Sea)
      east_asia/japan/radar/jma/            # JMA HRPN composite (analysis leg → JPCOMP)
      east_asia/taiwan/radar/cwa/
      europe/radar/opera/
      europe/italy/radar/dpc/               # DPC Italy national VMI (ITCOMP)
      europe/nwp/{icon_eu,dmi_dini}/
      north_america/
        canada/radar/msc_canada/
        canada/nwp/hrdps/
        usa/radar/{iem,mrms}/
        usa/nwp/{hrrr,hrrr_alaska}/
      oceania/nwp/{arome_ncaled,arome_polyn}/
      south_america/nwp/{wrf_smn,arome_guyane}/
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
    renderer.py      # On-demand tile rendering (compute / present split)
    satellite_renderer.py  # GMGSI VIS-over-LW composite tiles
    cache.py         # Byte-capped LRU tile cache (stores TileGeometry)
    coordinates.py   # Tile/region coordinate transforms
    warmer.py        # Background tile pre-rendering
    request_tracker.py  # Hot-tile counters for /health diagnostics
  colors/
    schemes.py       # Color scheme definitions
```

## Deployment Modes

Two deployment shapes, selected by `LIBREWXR_MODE` or `COMPOSE_PROFILES`:

- **single** (default): One container, fetcher + renderer in the same process. `python -m librewxr.main`.
- **multi**: Pipeline sidecar (`python -m librewxr.data_pipeline`) + N renderer workers (`python -m librewxr.main` with `LIBREWXR_RENDER_ONLY=1`). Workers memmap shared files and refresh via `state.json` mtime polling. Bypasses the Python GIL on the render path.

Docker Compose uses profiles: `COMPOSE_PROFILES=single` or `COMPOSE_PROFILES=multi`. A gitignored `docker-compose.override.yml` holds host-specific deltas (bind mounts, tunnels).

## Key Architecture Facts

- **Source layout:** `src/librewxr/` (hatchling build backend, editable install via `pip install -e ".[dev]"`)
- **Entry points:** `python -m librewxr.main` (renderer/server); `python -m librewxr.data_pipeline` (multi-mode fetcher)
- **Auto-discovery:** `sources/__init__.py` walks the `sources/` tree and registers radar/NWP/satellite providers automatically. Adding a source requires no changes to `fetcher.py`, `routes.py`, or `main.py`.
- **Shared state wiring:** Lifespan in `main.py` creates all singletons and assigns them to `routes` module-level vars — dependencies are NOT injected via FastAPI's DI. Key vars: `frame_store`, `tile_cache`, `nwp_grids` (dict by slug), `ecmwf_grid`, `nwp_chain`, `satellite_grids`, `nowcast_store`, `alerts_store`, `alerts_fetcher`, `tile_request_tracker`.
- **NWP chain:** Priority-ordered sources: HRRR (10) → HRRR-Alaska (11) → HRDPS (20) → JMA MSM (20) → AROME Antilles (25) → AROME Guyane (26) → AROME Indien (27) → AROME Ncaled (28) → AROME Polyn (29) → DMI DINI (30) → ICON-EU (35) → WRF-SMN (40) → IFS (1000, global catch-all). `NWPChain` dispatches narrowest-domain-first.
- **Radar regions:** US (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP), Canada (CACOMP), Central America (SVCOMP), Europe (OPERA + ITCOMP — Italy via DPC, finer `pixel_size` so it precedes OPERA in the multi-region compositor), Japan (JPCOMP — JMA HRPN analysis leg), Taiwan (TWCOMP), SE Asia (MYPENINSULAR, MYEAST). Region groups: CONUS, US, CANADA, CENTRAL_AMERICA, EUROPE, JAPAN, SOUTHEAST_ASIA, TAIWAN, ALL.
- **Data encoding:** Radar frames are `dict[str, np.ndarray]` keyed by region name, stored as uint8 dBZ values.
- **Tile rendering:** Compute / present split — `compute_tile_geometry` does the expensive work (region sampling, multi-region compositing, NWP fill/blend, noise-floor masking, optional snow mask) and returns a `TileGeometry` dataclass. `present_tile` does the cheap per-request tail (LUT colorize, Gaussian blur, optional motion-arrow overlay, encode). The `TileCache` stores `TileGeometry` records (not encoded bytes) so one cached entry serves every visual variant.
- **Satellite:** NOAA GMGSI hourly global mosaic (LW + VIS), composited at render time as VIS-over-LW with a natural day/night terminator. Latitude grid is Mercator-spaced.
- **Nowcasting:** Radar extrapolation + IFS blending with spatial feathering at radar boundaries.
- **Memory:** Heavily uses numpy memmap (temp files) for radar frames, ECMWF grids, and nowcast data. Memory monitor is cgroup-aware for multi-worker. See docker-compose.yml for RAM guidance.
- **Tile warming:** Two separate thread pools — one for on-demand requests, one for background tile warming — so requests never queue behind warming tasks. Warmer pre-computes geometry only.
- **Weather alerts:** WMO CAP alerts via `alerts_fetcher.py` (async HTTP) + `alerts_store.py`. In multi mode, pipeline owns fetching; render workers read via `state.json` snapshot.

## Configuration

All config via `LIBREWXR_*` env vars or `.env` file. Settings defined in `src/librewxr/config.py`. Full reference: `docs/configuration-reference.md`.

**Deployment:**
- `LIBREWXR_MODE`: `single` (default) or `multi` — drives per-mode defaults for workers, cache, threads
- `LIBREWXR_WORKERS`: uvicorn worker count (0 = mode default: 1 single, 16 multi)
- `LIBREWXR_TILE_CACHE_MB`: tile cache size (0 = mode default: 200 single, 128 multi)
- `LIBREWXR_WARMER_THREADS`: render thread pool size (0 = mode default: auto single, 4 multi)
- `LIBREWXR_RENDER_ONLY`: `true` — skip fetcher init, memmap pipeline snapshot (multi mode)
- `LIBREWXR_SSL_CERTFILE` / `LIBREWXR_SSL_KEYFILE`: optional direct TLS termination (paths to cert + key; leave both unset to serve plain HTTP behind a reverse proxy)

**Radar:**
- `LIBREWXR_ENABLED_REGIONS`: `ALL`, `CONUS`, `US`, `CANADA`, `EUROPE`, or comma-separated region names
- `LIBREWXR_RADAR_ENABLED`: global radar toggle (false = satellite/NWP only)
- `LIBREWXR_NA_SOURCE`: `mrms_fallback` (default), `mrms`, or `iem` — US-side radar source
- `LIBREWXR_CA_SOURCE`: `mrms_with_msc_blend` (default), `mrms`, or `msc` — Canada-side radar source
- `LIBREWXR_MAX_FRAMES`: default 12 (past radar frames to keep)
- `LIBREWXR_MAX_ZOOM`: default 12

**NWP:**
- `LIBREWXR_REGIONAL_NWP_ENABLED`: master switch for all regional NWP (false = IFS only)
- `LIBREWXR_NA_NWP_SOURCE`: `ifs` (default) or `hrrr` — North American NWP source
- `LIBREWXR_EU_NWP_PROFILE`: `ifs`, `icon_eu_only`, or `dini_with_icon_eu` — European NWP profile
- `LIBREWXR_ECMWF_ENABLED`: disable IFS global precipitation (debug use)

**Satellite:**
- `LIBREWXR_SATELLITE_ENABLED`: master toggle for GMGSI satellite layer
- `LIBREWXR_GMGSI_LW_ENABLED` / `LIBREWXR_GMGSI_VIS_ENABLED`: per-channel toggles

**Nowcast:**
- `LIBREWXR_NOWCAST_ENABLED`: default true
- `LIBREWXR_NOWCAST_FRAMES`: default 6 (10-min forecast frames)

**Alerts:**
- `LIBREWXR_ALERTS_ENABLED`: WMO CAP weather alerts toggle
- `LIBREWXR_ALERTS_FETCH_INTERVAL`: default 300s

**Other:**
- `LIBREWXR_CACHE_DIR`: persistent disk cache; empty = in-memory only
- `LIBREWXR_NWP_FETCH_CONCURRENCY`: max parallel NWP grid decodes (default 4)

## Adding a New Source

See `docs/adding-a-source.md` for the full walkthrough. Short version:

1. Create a self-contained package under `sources/regional/<continent>/<country>/{radar,nwp}/<source_name>/` (or `sources/world/<source>/` for global sources).
2. Implement the fetcher/decoder in `source.py` (radar) or `grid.py` (NWP). Radar sources also need `regions.py` and `stations.py`.
3. In the package `__init__.py`, expose a `radar_provider(settings)` or `nwp_provider(settings, cache_dir)` returning a contribution dataclass (or `None` when disabled).
4. Add any new env vars to `config.py`. New sources default to enabled by convention.
5. Add a coverage polygon to `scripts/generate_coverage_map.py` and regenerate coverage maps.

The discovery walker picks up the new package automatically — no per-source plumbing in `fetcher.py`, `routes.py`, or `main.py`.

## Conventions

- **File headers:** `# SPDX-License-Identifier: AGPL-3.0-or-later` + `# Copyright (C) 2026 Joshua Kimsey` on every source file
- **Commit style:** imperative mood, concise (e.g., "Add precipitation motion arrows")
- **Docker:** `docker compose up --build` with `COMPOSE_PROFILES=single` (default) or `COMPOSE_PROFILES=multi`. Exposes port 8080 (configurable via `LIBREWXR_PORT`). Use `docker compose run --rm clear-cache` to wipe caches.
- **Docs:** `docs/adding-a-source.md`, `docs/configuration-reference.md`, `docs/satellite-implementation-plan.md`, `docs/coverage.md`, `docs/rainviewer-migration-guide.md`, `docs/web-integration-guide.md`, `docs/source-survey.md`