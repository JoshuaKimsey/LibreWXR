# AGENTS.md - LibreWRX

## Project Overview

LibreWRX is a self-hostable Rain Viewer API replacement. It fetches radar composites from public sources, composites them into map tiles, and serves a Rain Viewer-compatible JSON/tile API. Python + FastAPI, no GDAL dependency.

- **License:** AGPL-3.0-or-later
- **Python:** >=3.11 (Docker uses 3.12)
- **Package manager:** pip with hatchling build backend
- **MCP server:** Model Context Protocol tools (point nowcast sampling, alerts, storm cells, lightning strikes) over HTTP at `/mcp` or stdio (`python -m librewxr.mcp`); optional `fastmcp` via the `mcp` extra (installed by default in Docker)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Always use the project venv `.venv/`; never install to system Python.

## Running & Testing

```bash
python -m librewxr.main          # dev server (auto-spawns the data pipeline as a child; render worker)
python -m librewxr.data_pipeline # standalone fetcher (the pipeline main.py auto-spawns)
python -m librewxr.mcp           # standalone MCP stdio server (also the librewxr-mcp console script)
pytest                            # all tests
pytest -m api                     # by marker
pytest tests/test_renderer.py     # single file
pytest -k "test_tile_render"      # by name pattern
```

Test markers (defined in `pyproject.toml`): `api`, `ecmwf`, `nowcast`, `sources`, `tiles`, `store`, `hrrr`, `hrrr_alaska`, `icon_eu`, `dmi_dini`, `hrdps`, `arome_antilles`, `arome_guyane`, `arome_indien`, `arome_ncaled`, `arome_polyn`, `wrf_smn`, `jma_msm`, `rrqpe`, `alerts`, `mcp`, `storm_cells`, `lightning`.

All tests are auto-async (`asyncio_mode = "auto"` in pyproject.toml). No explicit `@pytest.mark.asyncio` needed on individual async tests (though some older tests still have it).

No linter/formatter/typechecker is configured — there is no `ruff`, `mypy`, `black`, etc. in the project.

## Project Structure

```
src/librewxr/
  main.py            # FastAPI app, lifespan, uvicorn entry point
  data_pipeline.py   # Standalone fetcher process (multi-worker deployment)
  config.py          # Pydantic Settings (all LIBREWXR_* env vars)
  memory.py          # Memory pressure monitor (cgroup-aware)
  logging_setup.py   # Shared log-level normalization + rotating file setup
  api/
    routes.py        # API endpoints (Rain Viewer-compatible)
    models.py        # Pydantic response models
    conditional.py   # ETag / If-None-Match conditional-request helpers
  sources/                                # Per-source packages (auto-discovered)
    _base.py                              # Protocols + contribution dataclasses
    _helpers.py                           # Shared dBZ encoder + GRIB stderr muzzle
    _shared/                              # Base classes for source families
      arome.py                            # AROMEOverseasGrid (all 5 AROME-OM variants)
    __init__.py                           # Discovery walker, registry, helpers
    world/ifs/                            # ECMWF IFS (global, NWP)
    world/rrqpe/                          # NOAA Enterprise Rain Rate blend (global observed-precip radar region)
    satellite/gmgsi/                      # NOAA GMGSI (global, LW + VIS composite)
    regional/
      africa/nwp/arome_indien/            # AROME Indien (RE+YT+KM+MG+SW Indian Ocean)
      caribbean/nwp/arome_antilles/       # AROME Antilles (FR-GP+MQ)
      central_america/el_salvador/radar/marn/
      east_asia/japan/nwp/jma_msm/          # JMA MSM (Japan + Korean Peninsula + Taiwan + Yellow Sea)
      east_asia/japan/radar/jma/            # JMA HRPN composite (analysis leg → JPCOMP; forecast leg NOT ingested)
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
      southeast_asia/malaysia/radar/mmd/    # MET Malaysia (MYPENINSULAR + MYEAST)
      southeast_asia/philippines/radar/pagasa/  # PAGASA PANAHON mosaic (PHCOMP)
  data/                                   # Cross-cutting infra only
    regions.py       # RegionDef base + REGIONS dict (built from discovery)
    fetcher.py       # Multi-source fetch orchestrator
    store.py         # FrameStore (RadarFrame ring buffer)
    coverage.py      # Radar station coverage masks (parameter-driven)
    nowcast.py       # Nowcast generation (radar extrapolation + IFS blend)
    nwp_source.py    # NWPSource Protocol + NWPChain dispatcher
    nwp_interpolation.py  # Shared optical-flow helper for regional NWP
    radar_cache.py   # Persistent disk cache for radar frames
    coord_store.py   # Shared on-disk coordinate-array store
    pagecache.py     # posix_fadvise(WILLNEED) page-cache priming after fetch cycles
    precip_mask.py   # Per-timestamp global precip mask (multi-worker empty-tile fast path)
    storm_cells.py   # Storm-cell detection + StormCellStore (?cells= overlay)
    worker_pulse.py  # Per-render-worker health pulses (<cache_dir>/workers/)
    alerts_fetcher.py / alerts_store.py   # WMO weather alerts
    lightning_fetcher.py   # GOES GLM flash fetch loop (pipeline-owned, clock-aligned)
    lightning_store.py    # Flash-point store (on-disk artifact handoff, no state.json section)
    master_state.py  # Multi-worker state.json snapshot
    retry.py         # Backoff helper
  tiles/
    renderer.py      # On-demand tile rendering (compute / present split)
    window.py        # Lat/lon-centered window stitching (RainViewer point-tile API)
    satellite_renderer.py  # GMGSI VIS-over-LW composite tiles
    cache.py         # Byte-capped LRU tile cache (stores TileGeometry)
    shared_tile_store.py  # Shared on-disk encoded-tile store (multi-mode only)
    png_palette.py   # Adaptive lossless PNG8 palette encoding for tile output
    coordinates.py   # Tile/region coordinate transforms
    request_tracker.py  # Hot-tile counters for /health diagnostics
  mcp/
    server.py        # FastMCP server (HTTP app mounted at /mcp + stdio main)
    __main__.py       # stdio entry point (python -m librewxr.mcp)
    tools.py         # Pure store-passing MCP tool functions (nowcast, alerts, storm cells)
    sampling.py      # Point sampling helpers (dBZ decode, rain rate, region resolve)
    discovery.py     # SEP-2127 server card + /.well-known/ai-catalog.json
    context.py       # Stdio-mode lifespan (builds stores from state.json)
    alerts_query.py  # Alerts-within-radius queries (merged WMO + NWS store)
    storm_cells.py   # Inverse-projection helpers for get_storm_cells
  colors/
    schemes.py       # Color scheme definitions
    color_table.csv  # Shared color stop table for the scheme generators
```

The repo root also holds `tests/`, `examples/` (example frontends/viewers), `scripts/` (generators — see Conventions), and assorted design notes; `librewxr-site/` is a gitignored landing site.

## Deployment Modes

One architecture: a data pipeline (`python -m librewxr.data_pipeline`) fetches all radar / NWP / satellite / alerts data and writes a shared `state.json` snapshot; N render workers (`python -m librewxr.main`) memmap the shared files and refresh via `state.json` mtime polling. Bypasses the Python GIL on the render path.

Bare metal / dev, `python -m librewxr.main` with no flags auto-spawns the pipeline as a child process and runs this process as a render worker with 1 uvicorn worker unless `LIBREWXR_WORKERS` is explicitly set. `LIBREWXR_RENDER_ONLY=1` is only for dedicated render workers that read an already-running pipeline's snapshot (the Docker renderer service sets it).

Docker Compose uses profiles: `COMPOSE_PROFILES=multi` starts the `pipeline` + `renderer` services. A legacy `COMPOSE_PROFILES=single` still works through a compatibility alias (same pair, 1 render worker, legacy-single cache defaults, startup warning); `mode` always resolves to `multi`. See `docs/single-mode-migration.md`. A gitignored `docker-compose.override.yml` holds host-specific deltas (bind mounts, tunnels).

## Key Architecture Facts

- **Source layout:** `src/librewxr/` (hatchling build backend, editable install via `pip install -e ".[dev]"`)
- **Entry points:** `python -m librewxr.main` (renderer/server; auto-spawns the data pipeline unless `LIBREWXR_RENDER_ONLY` is set); `python -m librewxr.data_pipeline` (the fetcher)
- **Auto-discovery:** `sources/__init__.py` walks the `sources/` tree and registers radar/NWP/satellite providers automatically. Adding a source requires no changes to `fetcher.py`, `routes.py`, or `main.py`.
- **Shared state wiring:** The lifespan in `main.py` creates all singletons and assigns them to `routes` module-level vars — dependencies are NOT injected via FastAPI's DI. There is now only one lifespan: `lifespan` delegates to `_render_only_lifespan`, which builds the render-side singletons and leaves fetch-side singletons as `None` (the pipeline owns fetching). Key vars: `frame_store`, `tile_cache`, `nwp_grids` (dict by slug), `ecmwf_grid`, `nwp_chain`, `satellite_grids`, `nowcast_store`, `alerts_store`, `alerts_fetcher`, `tile_request_tracker`, `radar_cache`, `radar_fetcher`, `enabled_regions`, `alerts_enabled`, `storm_cell_store`, `precip_mask`, `shared_tile_store`, `memory_monitor`, `present_executor` / `io_executor`.
- **NWP chain:** Priority-ordered sources: HRRR (10) → HRRR-Alaska (11) → HRDPS (20) → JMA MSM (20) → AROME Antilles (25) → AROME Guyane (26) → AROME Indien (27) → AROME Ncaled (28) → AROME Polyn (29) → DMI DINI (30) → ICON-EU (35) → WRF-SMN (40) → IFS (1000, the terminal model of the chain). `NWPChain` dispatches narrowest-domain-first. The model layer fills past frames only (a) poleward of the RRQPE band, (b) in the 2-degree fringe excluded by RRQPE's coverage polygon (68-70N, -60 to -58S), and (c) when RRQPE declines (missed scans / stale store) — within the 60S-70N band, RRQPE (next bullet) is the always-on global observed radar region at the bottom compositing tier.
- **Radar regions:** US (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP), Canada (CACOMP), Central America (SVCOMP), Europe (OPERA + ITCOMP — Italy via DPC, finer `pixel_size` so it precedes OPERA in the multi-region compositor), Japan (JPCOMP — JMA HRPN analysis leg; the HRPN forecast leg is not ingested — JPCOMP nowcast comes from internal optical-flow extrapolation blended with JMA MSM), Taiwan (TWCOMP), SE Asia (MYPENINSULAR, MYEAST — MET Malaysia; PHCOMP — PAGASA Philippines), plus RRQPE — the always-on global observed-precip band (60S-70N, all longitudes) whose coarsest `pixel_size` sorts it last in the multi-region compositor (it fills only pixels no finer radar region claims) and joins nowcast extrapolation like any region. Region groups: CONUS, US, CANADA, CENTRAL_AMERICA, EUROPE, JAPAN, SOUTHEAST_ASIA, TAIWAN, plus the special `ALL` token.
- **Data encoding:** Radar frames are `dict[str, np.ndarray]` keyed by region name, stored as uint8 dBZ values.
- **Tile rendering:** Compute / present split — `compute_tile_geometry` does the expensive work (region sampling, multi-region compositing — RRQPE is the always-on global observed bottom tier within the 60S-70N band, filling only pixels no finer radar region claims — NWP fill/blend, noise-floor masking, optional snow mask) and returns a `TileGeometry` dataclass. `present_tile` does the cheap per-request tail (LUT colorize, Gaussian blur, optional motion-arrow overlay, encode). The `TileCache` stores `TileGeometry` records (not encoded bytes) so one cached entry serves every visual variant. NWP fill/blend therefore applies to the polar fringe outside the RRQPE band, to RRQPE-decline pixels, and to the nowcast blend tail.
- **Satellite:** NOAA GMGSI hourly global mosaic (LW + VIS), composited at render time as VIS-over-LW with a natural day/night terminator. Latitude grid is Mercator-spaced.
- **Nowcasting:** Radar extrapolation + IFS blending with spatial feathering at radar boundaries. Full-longitude regions (`RegionDef.is_global`, e.g. the global RRQPE band) get wrap-aware optical flow and remap so content advecting across the ±180° seam re-enters on the other side instead of zeroing at a hard edge.
- **Memory:** Heavily uses numpy memmap (temp files) for radar frames, ECMWF grids, and nowcast data. Memory monitor is cgroup-aware for multi-worker. See docker-compose.yml for RAM guidance.
- **Weather alerts:** WMO CAP alerts via `alerts_fetcher.py` (async HTTP) + `alerts_store.py`. The pipeline owns fetching; render workers read via the `state.json` snapshot.
- **Worker pulses:** Every render process writes a small JSON pulse to `<cache_dir>/workers/worker_<pid>.json` every ~15s (jittered, atomic tmp+os.replace; see `src/librewxr/data/worker_pulse.py`); `/health` aggregates fresh pulses (mtime-filtered, lock-free) into an additive top-level `cluster` section - workers_reporting, container cgroup anon/file/shmem split, summed per-worker RSS / tile-cache / coord-cache / request counters with hit ratios recomputed from sums. The pulse loop runs in the render lifespan (both the auto-spawned bare-metal process and Docker render workers); not in the pipeline.
- **Shared stores:** three best-effort on-disk stores under `cache_dir` shared by pipeline + renderers — `data/coord_store.py` (tile-coordinate arrays computed once per fetch cycle globally instead of per worker), `tiles/shared_tile_store.py` (encoded tile bytes, multi-mode only, content-versioned keys so stale entries are unreachable between fetch cycles), and `data/precip_mask.py` (per-timestamp global 0.5° precip mask gating the multi-worker empty-tile fast path).
- **Storm cells:** `data/storm_cells.py` thresholds the latest radar frame per region (default 40 dBZ), groups contiguous pixels with cv2 connected components, filters cells under 25 km², and derives each cell's motion vector from the same optical-flow field as `?arrows=`; cells live in `StormCellStore` and are drawn as a present-time `?cells=light|dark` tile overlay (does not affect the geometry cache). See `docs/storm-cells.md`.
- **Lightning:** NOAA GOES GLM flash points (GOES-East + GOES-West) fetched by `data/lightning_fetcher.py` on their own clock-aligned loop (default 300 s) into a shared on-disk artifact `<cache_dir>/lightning/current.npz` with per-satellite watermarks in `<cache_dir>/lightning/watermark.json` — deliberately NO `state.json` section (the artifact is the cross-process handoff, the shared-tile-store/precip-mask pattern). The store (`data/lightning_store.py`) holds structured flash points under a 500k RAM ceiling; render workers and the stdio MCP transport refresh via a cheap `mtime` stat on demand. The `?lightning=1|true|dots|bolts` tile overlay draws dots or a hand-authored vector-bolt glyph (constant brightness, no age fade) on the newest analysis frame only, with a presentational strongest-by-energy draw cap (`LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE`) — the store always keeps every flash in the window; unknown values fall back to off silently. MCP `get_recent_lightning` runs on both transports; `/health` gains an additive `lightning` section.
- **MCP server:** FastMCP HTTP transport mounted inside the FastAPI app at `LIBREWXR_MCP_PATH` (default `/mcp`) when `LIBREWXR_MCP_ENABLED`; stdio transport via `python -m librewxr.mcp` or the `librewxr-mcp` console script. Tools in `mcp/tools.py` are pure store-passing functions shared by both transports (HTTP reads the `routes` globals; stdio builds its own stores from `state.json`). Discovery metadata: MCP Server Card at `<mcp_path>/server-card` (SEP-2127 draft) and AI Catalog at `/.well-known/ai-catalog.json`. Requires the `fastmcp` `[mcp]` extra (the Dockerfile installs it). See `docs/mcp-server.md`.

## Configuration

All config via `LIBREWXR_*` env vars or `.env` file. Settings defined in `src/librewxr/config.py`. Full reference: `docs/configuration-reference.md`.

**Deployment:**
- `LIBREWXR_MODE`: always resolves to `multi`; a legacy `single` value runs the multi architecture with the legacy-single defaults profile (1 worker + legacy cache sizes) and logs a warning
- `LIBREWXR_WORKERS`: uvicorn render-worker count (0 = profile default: 16 multi, 1 legacy-single)
- `LIBREWXR_TILE_CACHE_MB`: tile cache size (0 = profile default: 128 multi, 200 legacy-single)
- `LIBREWXR_RENDER_THREADS`: per-render-worker geometry compute pool size (0 = profile default: 4 multi, 0 = auto legacy-single; legacy alias `LIBREWXR_WARMER_THREADS`)
- `LIBREWXR_RENDER_ONLY`: `true` — dedicated render workers only; skip fetcher init and memmap the pipeline snapshot (`main.py` auto-spawns the pipeline when unset)
- `LIBREWXR_SSL_CERTFILE` / `LIBREWXR_SSL_KEYFILE`: optional direct TLS termination (paths to cert + key; leave both unset to serve plain HTTP behind a reverse proxy)
- `LIBREWXR_HOST`: bind host (default unset → uvicorn dual-stack default; set `0.0.0.0` to restore the pre-`f1eea96` IPv4-only behaviour — useful for IPv4-only reverse proxies)
- `LIBREWXR_PORT`: bind port (default 8080)
- `LIBREWXR_PUBLIC_URL`: public base URL advertised by the JSON API (default `http://localhost:8080`)
- `LIBREWXR_CORS_ORIGINS`: comma-separated allowed origins (default `["*"]`)
- `LIBREWXR_FETCH_INTERVAL`: radar/satellite fetch cadence, seconds (default 600)
- `LIBREWXR_STATE_POLL_INTERVAL` / `LIBREWXR_STATE_WAIT_TIMEOUT`: render-worker `state.json` mtime poll interval / startup wait before fresh snapshot (defaults 1.0s / 300.0s; `0` wait = forever)
- `LIBREWXR_SHARED_TILE_STORE_MB`: shared encoded-tile store budget, multi-mode only (unset = auto: 2048 MB in render-only workers, disabled for legacy 1-worker; `0` or negative disables)
- `LIBREWXR_MEMORY_LIMIT_MB`: container memory limit for the pressure monitor (0 = auto-detect)
- `LIBREWXR_MEMORY_PRESSURE_CHECK_INTERVAL`: seconds between memory pressure checks (default 30)
- `LIBREWXR_WORKER_HEALTHCHECK_TIMEOUT`: uvicorn worker healthcheck timeout in seconds (default 30; 0 = uvicorn's built-in 5 s)
- `LIBREWXR_PAGECACHE_PRIME_ENABLED`: pipeline-only `posix_fadvise(WILLNEED)` page-cache priming after fetch cycles (default true)

**Radar:**
- `LIBREWXR_ENABLED_REGIONS`: `ALL`, a region group (`CONUS`, `US`, `CANADA`, `CENTRAL_AMERICA`, `EUROPE`, `JAPAN`, `SOUTHEAST_ASIA`, `TAIWAN`), or comma-separated region names
- `LIBREWXR_RADAR_ENABLED`: global radar toggle (false = satellite/NWP only)
- `LIBREWXR_NA_SOURCE`: `mrms_fallback` (default), `mrms`, or `iem` — US-side radar source
- `LIBREWXR_CA_SOURCE`: `mrms_with_msc_blend` (default), `mrms`, or `msc` — Canada-side radar source
- `LIBREWXR_MAX_FRAMES`: default 12 (past radar frames to keep)
- `LIBREWXR_MAX_ZOOM`: default 12
- `LIBREWXR_RADAR_FETCH_CONCURRENCY`: max parallel radar region fetches per fetch cycle (default 8)
- `LIBREWXR_DOWNLOAD_RETRIES`: retries on transient download errors (default 1)

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
- `LIBREWXR_NOWCAST_BLEND_MODE`: `radar`, `blended` (default), or `model` — radar extrapolation only, radar+IFS blend, or IFS-only nowcast composition
- `LIBREWXR_NOWCAST_COARSEN_ENABLED`: lead-time-ramped Gaussian coarsening of extrapolated radar (default true)
- `LIBREWXR_NOWCAST_COARSEN_MAX_KM`: effective resolution floor at the last blend step (default 3.0)
- `LIBREWXR_ARROW_FLOW_ENABLED`: render motion-arrow overlay on nowcast tiles (default true)

**MCP:**
- `LIBREWXR_MCP_ENABLED`: mount the MCP HTTP transport inside the FastAPI app (default true)
- `LIBREWXR_MCP_PATH`: mount path for the MCP HTTP transport (default `/mcp`)

**Storm cells:**
- `LIBREWXR_STORM_CELLS_ENABLED`: storm-cell detection feeding the `?cells=` tile overlay (default true)
- `LIBREWXR_STORM_CELLS_MIN_DBZ`: minimum dBZ for a pixel to be part of a cell (default 40)
- `LIBREWXR_STORM_CELLS_MIN_AREA_KM2`: minimum cell area in km²; smaller cells are filtered as noise (default 25.0)

**Lightning:**
- `LIBREWXR_LIGHTNING_ENABLED`: master toggle for the GOES GLM lightning overlay + MCP tool (default true)
- `LIBREWXR_LIGHTNING_NOAA_ENABLED`: enable the NOAA GOES GLM flash family (default true); satellite identity is internal — per-family booleans are the knob, never cherry-picked satellites
- `LIBREWXR_LIGHTNING_FETCH_INTERVAL`: clock-aligned GLM fetch cadence, seconds (default 300)
- `LIBREWXR_LIGHTNING_MAX_AGE`: seconds of held flash history (default 1800 = 30 min)
- `LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE`: presentational strongest-by-energy draw cap per tile (default 1000); the store always holds every flash in the window

**Alerts:**
- `LIBREWXR_ALERTS_ENABLED`: WMO CAP weather alerts toggle
- `LIBREWXR_ALERTS_FETCH_INTERVAL`: default 300s
- `LIBREWXR_ALERTS_CACHE_DIR`: local WMO CAP cache dir (default empty = system temp)
- `LIBREWXR_ALERTS_CONCURRENCY`: parallel alert fetches (default 5)

**Other:**
- `LIBREWXR_CACHE_DIR`: persistent disk cache shared by the pipeline + renderers; empty = per-host tempdir fallback with a one-time warning
- `LIBREWXR_VOLATILE_CACHE_DIR`: optional RAM-backed directory for per-cycle regenerated stores (nowcast/flows, precip masks, storm cells, RRQPE scan cache); empty = keep them under `LIBREWXR_CACHE_DIR`
- `LIBREWXR_NWP_FETCH_CONCURRENCY`: max parallel NWP grid decodes (default 4)
- `LIBREWXR_TILE_TRACKING_ENABLED`: hot-tile counters surfaced in `/health` diagnostics (default true; adaptive warming policy not currently shipping)
- `LIBREWXR_COORD_STORE_ENABLED`: shared on-disk coordinate store (default true; false reverts to per-worker in-process caches; requires `LIBREWXR_CACHE_DIR`)
- `LIBREWXR_COORD_STORE_MB`: shared coord-store size cap (0 = profile default: 4096 legacy-single / 8192 multi; soft cap, pruned once per fetch cycle; multi budget shared by all render workers)
- `LIBREWXR_COORD_CACHE_SIZE`: per-coordinate-cache LRU entries (0 = profile default: 512 multi, 2048 legacy-single)
- `LIBREWXR_WARM_COORD_ZOOM`: startup coordinate-cache warm zoom (0 = profile default: no eager warm in multi, 4 in legacy-single; negative disables)
- `LIBREWXR_LOG_LEVEL`: root log level (DEBUG/INFO/WARNING/ERROR/CRITICAL, default INFO; case-insensitive; per-cycle noise logs at DEBUG)
- `LIBREWXR_LOG_FILE`: rotating WARNING+ log file (default logs/librewxr.log; empty = disabled; compose bind-mounts ./logs so it lands in the clone directory)

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
- **Commit sign-off:** every commit must carry a `Signed-off-by:` trailer (`git commit -s`) whose name and email match the commit author identity; enforced on pull requests by `.github/workflows/dco.yml`.
- **Contribution licensing:** LibreWXR is dual-licensed (AGPL-3.0-or-later plus a separate commercial license offered by the maintainer). Contributions are governed by the license grant in CONTRIBUTING.md, restated as required checkboxes in `.github/PULL_REQUEST_TEMPLATE.md`; do not weaken or bypass those terms.
- **Docker:** `docker compose up --build` with `COMPOSE_PROFILES=multi` (default). A legacy `COMPOSE_PROFILES=single` works through a compatibility alias (same pipeline + renderer pair, 1 render worker). Exposes port 8080 (configurable via `LIBREWXR_PORT`). Use `docker compose run --rm clear-cache` to wipe caches.
- **Scripts:** `scripts/generate_coverage_map.py` regenerates the `docs/coverage-map-*.png` files (needs the `maps` extra); `generate_scheme_docs.py` + `generate_color_scheme_previews.py` regenerate the color-scheme docs and preview PNGs (a CI workflow checks the previews aren't stale); `refresh_dpc_coverage.py` / `refresh_jma_coverage.py` refresh per-source coverage data; `auto-update.sh` is the server self-update helper
- **Docs:** `docs/adding-a-source.md`, `docs/configuration-reference.md`, `docs/satellite-implementation-plan.md`, `docs/coverage.md`, `docs/rainviewer-migration-guide.md`, `docs/web-integration-guide.md`, `docs/source-survey.md`, `docs/self-host-sizing.md`, `docs/single-mode-migration.md`, `docs/mcp-server.md`, `docs/storm-cells.md`