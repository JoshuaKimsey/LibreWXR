<p align="center">
  <img src="LibreWXR-Logo.png" alt="LibreWXR" width="200">
</p>

# LibreWXR

A self-hostable, drop-in replacement for the [Rain Viewer](https://www.rainviewer.com/) API. LibreWXR serves weather radar tiles using freely available radar composite data from multiple sources, with full compatibility for any client built against the Rain Viewer v2 API.

## Why?

Rain Viewer recently (as of January 1st, 2026) restricted their free API tier: maximum zoom 7, single color scheme, no satellite, no forecast, PNG only. LibreWXR restores the full pre-restriction functionality as a self-hosted service.

Beyond this though, is the goal of creating a far more customizable API backend for self hosters. The ability to specify regions, radar styles, denoising levels, and more to come as well. With the goal being self-hosting, there are far greater possibilities for what can be both ingested and output via the API, and there is no need to offer any limitations on what is provided, aside from the technicality of the implementation of such features.

## Features

- **Rain Viewer v2 API compatible** — drop-in replacement, no client changes needed
- **All 9 color schemes** — Black & White, Rainviewer Original, Universal Blue, TITAN, TWC, Meteored, NEXRAD III, Rainbow, Dark Sky, plus raw grayscale
- **Tile sizes** — 256px and 512px
- **Image formats** — PNG and WebP (with configurable lossy/lossless quality)
- **Smoothing** — zoom-adaptive Gaussian blur with seamless tile boundaries
- **Multi-region coverage** — US (CONUS, Alaska, Hawaii, Puerto Rico, Guam) via NOAA MRMS quality-controlled mosaics with IEM fallback, Europe (OPERA pan-European composite, ~155 radars across 24 countries), Canada (MSC GeoMet with MRMS blending), Central America (MARN/SNET El Salvador, 120 km), Taiwan (CWA QPESUMS 7-radar composite, 1.4 km observed dBZ), and SE Asia (MET Malaysia 12-radar composite covering Peninsular Malaysia, Borneo, Brunei, and N. Sumatra)
- **Regional NWP chain** — high-resolution rapid-refresh NWP models layered specificity-first: NOAA HRRR (CONUS + Alaska), ECCC HRDPS (Canada + N. CONUS), DMI HARMONIE-AROME DINI (most of populated Europe), DWD ICON-EU (the European remainder), Météo-France AROME Antilles (eastern Caribbean), and SMN WRF-DET (Argentina + S. American Cone), all on top of ECMWF IFS for global coverage. Soft-feathering at each domain edge prevents visible seams
- **ECMWF IFS global coverage** — ECMWF IFS 9 km precipitation data provides global precipitation animation and powers the nowcast everywhere the regional NWP chain doesn't reach. Multi-timestep animation auto-scales to match radar history length
- **Optical flow interpolation** — hourly ECMWF IFS frames are interpolated to 10-minute steps using dense motion vectors, so global IFS coverage animates smoothly like real radar data instead of jumping hour-to-hour (configurable, enabled by default)
- **Precipitation nowcasting (experimental)** — 60-minute short-range forecast by extrapolating recent radar forward using optical flow, with configurable blend mode: smooth radar-to-model blending (default), pure radar extrapolation (closest to Rain Viewer), or pure NWP forecast. The model side is taken from the active NWP chain — HRRR over CONUS, ICON-EU/DINI over Europe, WRF-SMN over the S. American Cone, IFS elsewhere. Beyond 60 minutes, always uses pure model. Quality varies by weather pattern — works best for steady, organized precipitation; less reliable for fast-developing convection
- **Precipitation motion arrows** — optional Dark Sky-style arrows showing storm movement direction and speed, derived from optical flow. Available for both radar and ECMWF data globally. Supports light and dark styles for different map themes via `?arrows=light` or `?arrows=dark` query parameter
- **IFS-derived satellite imagery** — global cloud cover tiles composited from ECMWF IFS high/mid/low cloud layers, approximating infrared satellite imagery. Up to 12 hours of hourly animation with persistent disk caching for instant restarts. Populates the Rain Viewer-compatible `satellite.infrared` endpoint
- **Weather alerts (WMO CAP)** — global weather alerts polled every 5 minutes from severeweather.wmo.int, with MeteoAlarm geocodes for European polygon resolution. Surfaced through a Rain Viewer-extension alerts API (`/v2/alerts/...`). Configurable via `LIBREWXR_ALERTS_ENABLED`
- **Snow detection** — per-pixel snow/rain classification. Regional NWP sources classify natively from their own 2-metre temperature field (HRRR-CONUS, HRRR-Alaska, WRF-SMN, DMI DINI, ICON-EU); ECMWF IFS snowfall ratio fills everywhere else
- **Noise filtering** — configurable dBZ noise floor and speckle removal
- **Tile cache warming** — background pre-rendering for smooth animation playback
- **Multi-worker tile-server split** — optional production deployment splits the data pipeline from a pool of render workers that share state via memmap files. Lets every core actually do work instead of being GIL-bound at one. Run with `docker-compose.multiworker.yml`
- **Persistent disk cache** — radar / NWP / satellite / alerts data are cached to disk with atomic writes, surviving restarts and container recreation without re-downloading from upstream. Configurable via `LIBREWXR_CACHE_DIR` (required for multi-worker mode)
- **Memory-efficient storage** — radar frames, NWP grids, cloud cover, and nowcast data are all backed by memory-mapped files, letting the OS page cache manage physical RAM instead of pinning data on the heap. Pages are reclaimed under memory pressure and re-faulted on access
- **Smart fetch optimization** — radar sources skip re-downloading frames already in memory (only ~1 of 12 frames is new each cycle), NWP models skip redundant S3 fetches when the model run hasn't changed, and parallel NWP fetches are concurrency-capped via `LIBREWXR_NWP_FETCH_CONCURRENCY` so peak transient RAM stays bounded
- **Health endpoint** — `/health` for monitoring uptime, per-component memory breakdown, frame count, NWP chain status, alerts status, and cache state
- **Fully configurable** — all tunable parameters exposed via environment variables

## Quick Start

LibreWXR runs in either of two deployment modes — pick the one that
matches your hardware. See [Architecture → Deployment modes](#deployment-modes)
for the full comparison.

### Docker — single-container (personal / small-scale)

One process handles everything. Best for laptops, small VPSes, home
servers, or anywhere a few GB of RAM is plenty.

```bash
git clone https://github.com/JoshuaKimsey/LibreWXR.git
cd LibreWXR
cp .env.example .env
# Edit .env to taste
docker compose up -d
```

### Docker — multi-worker (production / multi-core)

The data pipeline and tile renderers run as separate containers, sharing
state via memmap files on a shared volume. The right choice for any
deployment with 8+ cores and meaningful traffic — single-process mode
caps at one core under load because of the Python GIL, multi-worker
scales across the whole rack.

```bash
git clone https://github.com/JoshuaKimsey/LibreWXR.git
cd LibreWXR
cp .env.example .env
# Edit .env to taste (LIBREWXR_WORKERS=N for the render pool size)
docker compose -f docker-compose.multiworker.yml up -d
```

Defaults target an 80-core / 32 GB rack (32 render workers, 12 GB
pipeline cap, 18 GB render cap). Tune `LIBREWXR_WORKERS` and the
`LIBREWXR_PIPELINE_MEMORY` / `LIBREWXR_RENDER_MEMORY` env vars down
for smaller hardware.

### Manual

Requires Python 3.11+.

```bash
git clone https://github.com/JoshuaKimsey/LibreWXR.git
cd LibreWXR
python3 -m venv .venv
source .venv/bin/activate
pip install .
cp .env.example .env
# Edit .env to taste
python -m librewxr.main
```

The server starts at `http://localhost:8080` by default. It will fetch
radar data on startup (takes a few seconds), then begin serving tiles.

For multi-worker mode without Docker, run the data pipeline as a
sidecar and start the render workers with `LIBREWXR_RENDER_ONLY=1`:

```bash
# Terminal 1 — data pipeline
python -m librewxr.data_pipeline

# Terminal 2 — tile-server worker pool
LIBREWXR_RENDER_ONLY=1 LIBREWXR_WORKERS=8 python -m librewxr.main
```

Both processes need the same `LIBREWXR_CACHE_DIR` pointed at a shared
directory.

### Auto-updating a Docker deployment

`scripts/auto-update.sh` is an optional helper for self-hosters running LibreWXR from a git checkout with `docker compose`. When run, it:

1. Fetches `origin` and checks whether the tracked branch has new commits.
2. If so, fast-forwards the working tree and runs `docker compose up -d --build` to rebuild and redeploy.
3. Otherwise exits quietly — it's safe to schedule via cron or a systemd timer.

The script is a **no-op by default** on any host. To opt in on a production host:

```bash
touch /path/to/LibreWXR/.auto-update-enabled
```

(Or export `LIBREWXR_AUTO_UPDATE=1` in the environment.) This sentinel is in `.gitignore`, so cloning the repo on a development machine will *not* accidentally enable auto-updates.

A typical cron entry for hourly updates:

```cron
0 * * * * /path/to/LibreWXR/scripts/auto-update.sh >> /var/log/librewxr-update.log 2>&1
```

Use `scripts/auto-update.sh --dry-run` to see what the script would do without making any changes; dry-run is always allowed regardless of the sentinel.

## Usage

### As a Rain Viewer replacement

Point any Rain Viewer-compatible client at your LibreWXR instance. The only change needed is replacing the Rain Viewer host URL with your LibreWXR URL.

For example, in JavaScript:

```javascript
// Before (Rain Viewer)
const apiUrl = "https://tilecache.rainviewer.com";

// After (LibreWXR)
const apiUrl = "http://localhost:8080";
```

### API Endpoints

#### Metadata

```
GET /public/weather-maps.json
```

Returns available radar timestamps and the host URL, matching Rain Viewer's response format:

```json
{
  "version": "2.0",
  "generated": 1773037528,
  "host": "http://localhost:8080",
  "radar": {
    "past": [
      {"time": 1773030600, "path": "/v2/radar/1773030600"},
      ...
    ],
    "nowcast": [
      {"time": 1773038400, "path": "/v2/radar/1773038400"},
      ...
    ],
    "colorSchemes": [
      {"id": 0, "name": "Black and White"},
      {"id": 7, "name": "Rainbow @ Selex SI"},
      ...
    ]
  },
  "satellite": {
    "infrared": [
      {"time": 1773030600, "path": "/v2/satellite/1773030600"},
      ...
    ]
  }
}
```

#### Radar Tiles

```
GET /v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.{ext}
```

| Parameter | Values | Description |
|---|---|---|
| `timestamp` | Unix timestamp | From the metadata endpoint |
| `size` | `256`, `512` | Tile size in pixels |
| `z`, `x`, `y` | integers | Standard slippy map tile coordinates |
| `color` | `0`-`8`, `255` | Color scheme (see below) |
| `smooth` | `0`, `1` | Enable smoothing |
| `snow` | `0`, `1` | Enable snow precipitation colors |
| `ext` | `png`, `webp` | Image format |

**Optional query parameters:**

| Parameter | Values | Description |
|---|---|---|
| `arrows` | `light`, `dark` | Draw precipitation motion arrows (light for dark maps, dark for light maps) |

**Color schemes:**

| ID | Name |
|---|---|
| 0 | Black and White |
| 1 | Rainviewer Original |
| 2 | Universal Blue |
| 3 | TITAN |
| 4 | The Weather Channel |
| 5 | Meteored |
| 6 | NEXRAD Level III |
| 7 | Rainbow |
| 8 | Dark Sky |
| 255 | Raw (grayscale) |

#### Satellite Tiles

```
GET /v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}
```

| Parameter | Values | Description |
|---|---|---|
| `timestamp` | Unix timestamp | From `satellite.infrared` in the metadata endpoint |
| `size` | `256`, `512` | Tile size in pixels |
| `z`, `x`, `y` | integers | Standard slippy map tile coordinates |
| `ext` | `png`, `webp` | Image format |

Returns IFS-derived cloud cover tiles that approximate infrared satellite imagery. High clouds render bright white, mid clouds light gray, low clouds darker gray. Global coverage with hourly updates.

#### Coverage Tiles

```
GET /v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png
```

Returns tiles showing where radar data exists (white semi-transparent overlay).

#### Weather Alerts (LibreWXR extension)

```
GET /v2/alerts
GET /v2/alerts?lat={lat}&lon={lon}
GET /v2/alerts?bbox=west,south,east,north
```

Returns active weather alerts as a GeoJSON `FeatureCollection`, with
each feature carrying the alert polygon plus CAP metadata (severity,
urgency, certainty, event, headline, sender, expiry). Fed by WMO CAP
(global) and the NWS point endpoint (US locations, used to surface
non-polygon alerts like Tornado Watches).

| Query parameter | Description |
|---|---|
| *(none)* | All active alerts worldwide |
| `lat` + `lon` | Alerts whose polygon contains the point |
| `bbox=W,S,E,N` | Alerts whose polygon intersects the bounding box |
| `simplify` | Polygon simplification tolerance in meters (default 1000, `0` = full resolution) |

Returns `503` if `LIBREWXR_ALERTS_ENABLED=false`.

#### Health

```
GET /health
```

Returns server status, frame count, cache usage, NWP chain state, satellite cache state, alerts status, and per-component memory breakdown.

## Configuration

All settings are configured via environment variables (or a `.env` file). Copy `.env.example` to `.env` and adjust as needed. Every setting has a sensible default.

The table below covers the **commonly-tuned settings**. The full surface
(including per-source NWP publish delays, dBZ calibration offsets, and
source base URLs) lives in
[`docs/configuration-reference.md`](docs/configuration-reference.md) and
the inline comments in [`src/librewxr/config.py`](src/librewxr/config.py).

| Variable | Default | Description |
|---|---|---|
| **Server** | | |
| `LIBREWXR_PUBLIC_URL` | `http://localhost:8080` | Public URL for metadata responses |
| `LIBREWXR_PORT` | `8080` | Server listen port |
| `LIBREWXR_MAX_ZOOM` | `12` | Maximum tile zoom level |
| **Radar** | | |
| `LIBREWXR_ENABLED_REGIONS` | `ALL` | Radar region spec (see below) |
| `LIBREWXR_NA_SOURCE` | `mrms_fallback` | North-American radar source: `mrms_fallback`, `mrms`, or `iem` |
| `LIBREWXR_FETCH_INTERVAL` | `600` | Seconds between radar data fetches (10 min, clock-aligned) |
| `LIBREWXR_MAX_FRAMES` | `12` | Radar frames in memory (2h at default 10-min cadence) |
| **Regional NWP chain** | | |
| `LIBREWXR_NA_NWP_SOURCE` | `ifs` | `hrrr` to enable NOAA HRRR (CONUS + Alaska) |
| `LIBREWXR_EU_NWP_PROFILE` | `ifs` | `icon_eu_only` or `dini_with_icon_eu` to enable European NWP |
| `LIBREWXR_HRDPS_ENABLED` | `true` | ECCC HRDPS-Continental (Canada) |
| `LIBREWXR_AROME_ANTILLES_ENABLED` | `true` | Météo-France AROME Antilles (eastern Caribbean) |
| `LIBREWXR_WRF_SMN_ENABLED` | `true` | SMN Argentina WRF-DET (South American Cone) |
| `LIBREWXR_ECMWF_ENABLED` | `true` | ECMWF IFS global precipitation + cloud (disable for regional-only debugging) |
| `LIBREWXR_NWP_FETCH_CONCURRENCY` | `4` | Max parallel NWP grid fetches per cycle |
| **Nowcast** | | |
| `LIBREWXR_NOWCAST_ENABLED` | `true` | Enable experimental precipitation nowcast |
| `LIBREWXR_NOWCAST_FRAMES` | `6` | Number of nowcast frames (6 × 10 min = 60 min forecast) |
| `LIBREWXR_NOWCAST_BLEND_MODE` | `blended` | `radar`, `blended`, or `model`. Beyond 60 min always uses pure model |
| **Satellite + alerts** | | |
| `LIBREWXR_SATELLITE_ENABLED` | `true` | Enable IFS-derived cloud cover tiles |
| `LIBREWXR_SATELLITE_MAX_FRAMES` | `12` | Hourly IFS cloud timesteps to keep (12 = 12 hours) |
| `LIBREWXR_ALERTS_ENABLED` | `true` | Enable WMO CAP weather alerts |
| `LIBREWXR_ALERTS_FETCH_INTERVAL` | `300` | Alerts refresh interval in seconds |
| **Tile rendering** | | |
| `LIBREWXR_TILE_CACHE_MB` | `200` | Max tile cache size in MB per worker (byte-capped) |
| `LIBREWXR_COORD_CACHE_SIZE` | `2048` | Coordinate cache entries per cache (lower = less RAM) |
| `LIBREWXR_SMOOTH_RADIUS` | `2.0` | Gaussian blur radius (0 = disabled) |
| `LIBREWXR_NOISE_FLOOR_DBZ` | `10.0` | Min dBZ to display (-32 = disabled) |
| `LIBREWXR_DESPECKLE_MIN_NEIGHBORS` | `3` | Speckle filter strength (0 = disabled) |
| `LIBREWXR_WEBP_QUALITY` | `65` | WebP quality (100 = lossless, <100 = lossy) |
| `LIBREWXR_WARMER_THREADS` | `0` | Background tile warming threads per worker (0 = CPU count - 1) |
| `LIBREWXR_WARM_COORD_ZOOM` | `6` | Pre-warm coordinate caches up to this zoom at startup (0 = disable) |
| `LIBREWXR_WARM_OVERVIEW_ZOOM` | `4` | Pre-render overview tiles up to this zoom after each fetch (-1 = disable) |
| **Workers + memory** | | |
| `LIBREWXR_WORKERS` | `1` | Uvicorn worker processes (32 recommended in multi-worker mode) |
| `LIBREWXR_MEMORY_LIMIT_MB` | `0` | Memory limit in MB (0 = auto-detect from Docker/cgroup) |
| `LIBREWXR_CACHE_DIR` | *(empty)* | Persistent cache directory. **Required** in multi-worker mode. Empty = in-memory only |
| **Multi-worker tile-server split** | | |
| `LIBREWXR_RENDER_ONLY` | `false` | When `1`, skip fetcher / NWP / cloud init and only render tiles from the snapshot |
| `LIBREWXR_STATE_POLL_INTERVAL` | `1.0` | Seconds between state.json mtime polls in render-only mode |
| `LIBREWXR_STATE_WAIT_TIMEOUT` | `300` | Seconds to wait for the first state.json on cold start (0 = forever) |

**Radar regions:**

| Code | Region | Source | Resolution | RAM per frame |
|---|---|---|---|---|
| `USCOMP` | Continental US | NCEP MRMS (IEM fallback) | 0.005° (~500m) | ~63 MB |
| `AKCOMP` | Alaska | NCEP MRMS (IEM fallback) | 0.01° (~1km) | ~6 MB |
| `HICOMP` | Hawaii | NCEP MRMS (IEM fallback) | 0.005° (~500m) | ~3.4 MB |
| `PRCOMP` | Puerto Rico | NCEP MRMS (IEM fallback) | 0.01° (~1km) | ~1 MB |
| `GUCOMP` | Guam | NCEP MRMS (IEM fallback) | 0.0085° (~850m) | ~1 MB |
| `CACOMP` | Canada | MSC GeoMet (MRMS blending) | 0.025° (~2.5km) | ~6 MB |
| `SVCOMP` | El Salvador + neighbours | MARN/SNET (San Andrés, 120 km) | 0.00926° (~1km) | <1 MB |
| `OPERA` | Europe (24 countries) | EUMETNET OPERA | 1km | ~16 MB |
| `TWCOMP` | Taiwan + W. Pacific buffer | CWA QPESUMS (7-radar composite) | 0.0125° (~1.4km) | ~3 MB |
| `MYPENINSULAR` | Peninsular Malaysia + N. Sumatra | MET Malaysia (12-radar composite) | 0.022° lon / 0.019° lat (~2.5km) | <1 MB |
| `MYEAST` | East Malaysia (Borneo) + Brunei | MET Malaysia (12-radar composite) | 0.022° lon / 0.019° lat (~2.5km) | <1 MB |

Group aliases: `CONUS` (continental US only), `US` (all US regions), `CANADA` (Canada), `CENTRAL_AMERICA` (El Salvador + W. Honduras + S. Guatemala + offshore Pacific), `EUROPE` (OPERA pan-European composite), `SOUTHEAST_ASIA` (MET Malaysia peninsular + east composites — Peninsular Malaysia, Borneo, Brunei, N. Sumatra), `TAIWAN` (CWA QPESUMS composite covering Taiwan + offshore Pacific), `ALL` (everything).
You can also mix groups and individual regions: `CONUS,EUROPE,CANADA`.

Examples:
```bash
LIBREWXR_ENABLED_REGIONS=CONUS          # just continental US
LIBREWXR_ENABLED_REGIONS=US             # all US regions
LIBREWXR_ENABLED_REGIONS=EUROPE         # Europe only (OPERA composite)
LIBREWXR_ENABLED_REGIONS=CANADA         # Canada only
LIBREWXR_ENABLED_REGIONS=CONUS,EUROPE   # continental US + Europe
LIBREWXR_ENABLED_REGIONS=ALL            # everything available (default)
```

**RAM requirements (single-container mode):**

In single-container mode each worker process holds its own copy of
radar frames, NWP grids, coordinate caches, and tile caches. RAM usage
grows under real traffic as caches fill up.

| Configuration | Estimated RAM |
|---|---|
| CONUS + IFS only, 1 worker, 12 frames | ~3-4 GB |
| CONUS + HRRR + IFS, 1 worker, 12 frames | ~4-5 GB |
| ALL regions + IFS only, 1 worker, 12 frames | ~7-8 GB |
| ALL regions + full NWP chain, 1 worker, 12 frames | ~9-10 GB |
| ALL regions + full NWP chain, 2 workers, 12 frames | ~16-18 GB |

**RAM requirements (multi-worker mode):**

Multi-worker mode shares the radar / NWP / cloud / alert state across
all render workers via memmap, so adding workers doesn't multiply the
data RAM — only the per-worker tile cache (default 64 MB) and Python
interpreter overhead (~80 MB).

| Configuration | Pipeline RAM | Render RAM | Total |
|---|---|---|---|
| ALL regions + full NWP chain, 8 workers | ~8-10 GB | ~3-4 GB | ~12-14 GB |
| ALL regions + full NWP chain, 16 workers | ~8-10 GB | ~5-6 GB | ~14-16 GB |
| ALL regions + full NWP chain, 32 workers | ~8-10 GB | ~7-8 GB | ~16-18 GB |

Production observation on an 80-core / 32 GB rack (32 workers): ~16 GB
total RSS across both containers.

See `.env.example` for detailed descriptions and tuning guidance for
each setting.

### Scaling

| Users | Mode | Workers | RAM (ALL regions + full NWP) |
|---|---|---|---|
| 1-5 (personal) | single-container | 1 | ~9-10 GB |
| 5-50 (small community) | single-container | 1-2 (with CDN) | ~9-18 GB |
| 50-500 (medium) | multi-worker | 8-16 | ~12-16 GB |
| 500+ (large) | multi-worker | 24-32+ (with CDN) | ~16-20 GB |

Tiles are served with `Cache-Control: public, max-age=300`, so any caching reverse proxy (nginx, Cloudflare, etc.) will work out of the box for high-traffic deployments. A CDN like Cloudflare (free tier works) absorbs most tile requests at the edge, meaning a single worker can serve far more users than the table above suggests. Using a Cloudflare Tunnel also provides free HTTPS with no certificate management. For most self-hosting scenarios, 1 worker behind Cloudflare is sufficient.

## Architecture

### Data flow

```
[MRMS / IEM]       ──┐
[MSC Canada]       ──┼──> [Radar Frames (memmap)] ───┐
[OPERA S3]         ──┘     (N frames, multi-region)  │
                                                      │
[HRRR / HRRR-AK]   ──┐                                ├──> [Nowcast Store (memmap)]
[HRDPS]            ──┤                                │     (radar extrap + NWP blend,
[DMI DINI]         ──┤   ┌─> [NWP Chain (memmap)] ───┤      6 frames / 60 min)
[ICON-EU]          ──┼───┤    (per-source feather +   │
[AROME Antilles]   ──┤   │     specificity-first      │
[WRF-SMN]          ──┤   │     blending)              ├──> [FastAPI + Tile Renderer]
[ECMWF IFS]        ──┘   │                            │      (LRU cache + tile warmer)
                          │                            │
                          └─> [Optical Flow Interp] ───┤      [Satellite Tile Renderer]
                              (hourly → 10-min)        │       (IR-like cloud tiles)
                                                       │
[Open-Meteo S3] ─> [IFS Cloud Cover] ─> [Disk Cache] ──┤
   (high/mid/low)                       (atomic writes)│
                                                       │
[WMO CAP] ───────> [Alert Store] ──────────────────────┘
   (severeweather.wmo.int + MeteoAlarm geocodes)
```

All stores are memmap-backed — the OS page cache manages physical RAM
and pages are reclaimed under pressure, so total resident set scales
with what's actually being touched, not with what's loaded.

### Coverage

LibreWXR fuses native radar from three regional networks and a chain of
high-resolution NWP models. ECMWF IFS provides global coverage
everywhere else and isn't shown.

**Radar composites:**

![Radar coverage map](docs/coverage-map-radar.png)

**Regional NWP models:**

![Regional NWP coverage map](docs/coverage-map-models.png)

Polygon shapes follow each grid's actual projected domain (LCC, polar
stereographic, LAEA, rotated lat/lon, or regular lat/lon) — not a
misleading lat/lon bounding box — so the curved edges visible on HRRR,
HRDPS, DMI DINI, OPERA, and WRF-SMN are the real coverage boundaries.
See [`docs/coverage.md`](docs/coverage.md) for a labelled breakdown of
each polygon (resolution, projection, cycle cadence). Regenerate both
PNGs with
[`scripts/generate_coverage_map.py`](scripts/generate_coverage_map.py)
after adding or changing a radar source or NWP grid (the script header
documents the throwaway venv recipe).

### Deployment modes

LibreWXR supports two deployment modes.

#### Single-container (personal / small-scale)

One process handles everything: fetching, nowcast generation, tile
rendering, and the HTTP server.

```
                       ┌────────────────────────────────────────┐
                       │   librewxr (one asyncio process)       │
[radar / NWP / cloud   │                                        │
 / alerts upstreams] ──┼──> [Fetchers + memmap stores]          ├──> tiles + API
                       │            │                           │
                       │            └──> [Tile Renderer +       │
                       │                  uvicorn HTTP server]  │
                       └────────────────────────────────────────┘
```

Best for laptops, small VPSes, home servers — anywhere a few GB of RAM
is plenty and you don't need to scale across cores. Run with:

```bash
docker compose up -d
```

#### Multi-worker (production / multi-core)

The data pipeline and tile renderers run as separate containers,
sharing state via memmap files + a `state.json` snapshot on a shared
volume. The render side scales to N worker processes that each map the
same files, so 32 workers don't cost 32× the radar/NWP RAM — just the
per-worker tile cache and Python interpreter overhead.

```
┌────────────────────────────────────────┐    ┌────────────────────────────────────────┐
│   data-pipeline                        │    │   tile-server                          │
│   (one asyncio process)                │    │   (N uvicorn workers,                  │
│                                        │    │    LIBREWXR_RENDER_ONLY=1)             │
│   [radar / NWP / cloud / alerts] ──┐   │    │                                        │
│                                    │   │    │       ┌──> Worker 1 ─┐                 │
│   [Fetchers] ──> [Memmap stores +  │   │    │       ├──> Worker 2 ─┤                 │
│                   state.json       │   │    │       ├──> Worker 3 ─┼──> tiles + API  │
│                   snapshot]        │   │    │       ├──> ...      ─┤                 │
│                                    │   │    │       └──> Worker N ─┘                 │
└────────────────────────────────────────┘    │       (each polls state.json mtime and │
                  │                           │        re-loads stores on change)      │
                  │                           └────────────────────────────────────────┘
                  │                                            │
                  └──────────── shared volume (LIBREWXR_CACHE_DIR) ────────────┘
                                  (memmap files + state.json)
```

This is the right choice for any deployment with 8+ cores and meaningful
traffic — the single-process renderer gets GIL-bound at one core no
matter how many threads its pool has, but multi-worker mode hands one
core to each render process. Production observation on an 80-core /
32 GB rack: ~16 GB total RSS, all cores active under load. Run with:

```bash
docker compose -f docker-compose.multiworker.yml up -d
```

### Data sources

- **US:** NCEP MRMS MultiSensor/3DReflectivity quality-controlled mosaics (default) — 2-min cadence, includes Canadian radar ingest. Falls back to IEM NEXRAD N0Q if MRMS is unavailable (`LIBREWXR_NA_SOURCE`).
- **Canada:** ECCC MSC GeoMet WMS — pre-colored PNG precipitation composite, decoded via palette reverse-engineering back to dBZ. MRMS blending fills gaps in northern Canada and the Atlantic coast.
- **Central America (El Salvador):** MARN/SNET San Andrés radar via anonymous Google Cloud Storage bucket — 120 km product covering all of El Salvador + western Honduras + southern Guatemala + offshore Pacific, 5-min cadence. Continuous HSV hue gradient decoded back to dBZ. Use of MARN/SNET data requires citation per their open-data terms.
- **Europe:** EUMETNET OPERA CIRRUS composite via MeteoGate S3 — ODIM HDF5, 3800×4400 at 1 km (LAEA), ~155 radars across 24 countries.
- **Taiwan:** CWA QPESUMS 7-radar composite (`O-A0059-001` / 雷達合成回波) via anonymous AWS S3 bucket `cwaopendata` in `ap-northeast-1` — UTF-8 XML with raw dBZ at 1.4 km / 10-min cadence, covering Taiwan + a substantial western Pacific buffer for typhoon tracking. Filename timestamps are Taipei local time (UTC+8); data uses TWD67 datum (sub-pixel offset vs WGS84 at this resolution). OGDL v1.0 attribution required.
- **SE Asia (Malaysia, Borneo, Brunei, Singapore):** MET Malaysia (Jabatan Meteorologi Malaysia) 12-radar national composite via anonymous HTTPS at `api.met.gov.my` — 1352×570 animated GIF carrying 6 frames at 10-min cadence (~60 min of backfill per fetch), decoded via 18-stop palette → dBZ table. Split into two regions (`MYPENINSULAR` and `MYEAST`) covering Peninsular Malaysia + N. Sumatra and East Malaysia (Borneo) + Brunei respectively. Singapore sits within `MYPENINSULAR`'s KLIA-radar coverage. CC-BY-4.0 — attribution required.
- **Regional NWP (layered ahead of IFS):**
  - **NOAA HRRR-CONUS** (3 km LCC, 15-min subh, hourly cycles) — anonymous AWS Open Data
  - **NOAA HRRR-Alaska** (3 km polar stereographic, hourly wrfsfcf, 3-hourly cycles)
  - **ECCC HRDPS-Continental** (2.5 km rotated lat/lon, 6-hourly cycles) — anonymous HTTPS via dd.weather.gc.ca
  - **DMI HARMONIE-AROME DINI** (2 km native LCC, 3-hourly cycles) — anonymous AWS Open Data, covers most of populated Europe
  - **DWD ICON-EU** (~7 km, 3-hourly cycles) — DWD opendata, fills the European gaps DINI doesn't reach
  - **Météo-France AROME Antilles** (1.3 km native, 4 cycles/day) — anonymous via data.gouv.fr, covers the eastern Caribbean
  - **SMN Argentina WRF-DET** (4 km LCC, 4 cycles/day) — anonymous AWS Open Data, covers the South American Cone
- **ECMWF IFS (global coverage):** native 9 km resolution via [Open-Meteo](https://open-meteo.com/) S3 — Marshall-Palmer Z-R conversion with snow/rain classification from snowfall ratio. Hourly frames optical-flow-interpolated to 10-min steps. Covers everywhere the regional NWP chain doesn't reach, and feeds both the precipitation animation and the satellite cloud layer.
- **Satellite (IFS-derived):** cloud_cover_high/mid/low composited into IR-like cloud tiles. Persistent disk cache survives restarts; backfills from previous model runs.
- **Weather alerts:** WMO CAP feed at severeweather.wmo.int (global) with MeteoAlarm geocodes for European polygon resolution. Updates every 5 minutes, surfaced through a Rain Viewer-extension alerts API.
- **No external dependencies beyond pip** — no GDAL, rasterio, or system geo libraries needed.

### Regional NWP chain

LibreWXR layers regional NWP models on top of the global IFS layer using
a **specificity-first chain**: at each pixel, the chain dispatches to
the narrowest model whose domain covers it, falling through to wider
models elsewhere. Soft feathering at every domain edge prevents visible
seams where domains meet.

| Region | Source | Resolution | Cycles | Toggle |
|---|---|---|---|---|
| Continental US | NOAA HRRR-CONUS | 3 km LCC | hourly | `LIBREWXR_NA_NWP_SOURCE=hrrr` |
| Alaska | NOAA HRRR-Alaska | 3 km polar stereo | 3-hourly | *(bundled with HRRR)* |
| Canada + N. fringe of CONUS | ECCC HRDPS-Continental | 2.5 km rotated lat/lon | 6-hourly | `LIBREWXR_HRDPS_ENABLED=true` |
| Most of populated Europe | DMI HARMONIE-AROME DINI | 2 km LCC | 3-hourly | `LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu` |
| European remainder | DWD ICON-EU | ~7 km | 3-hourly | `LIBREWXR_EU_NWP_PROFILE=icon_eu_only` *(or `dini_with_icon_eu`)* |
| Eastern Caribbean | Météo-France AROME Antilles | 1.3 km lat/lon | 6-hourly | `LIBREWXR_AROME_ANTILLES_ENABLED=true` |
| South American Cone | SMN Argentina WRF-DET | 4 km LCC | 6-hourly | `LIBREWXR_WRF_SMN_ENABLED=true` |
| Everywhere else | ECMWF IFS | 9 km global | 6-hourly | `LIBREWXR_ECMWF_ENABLED=true` |

Each model gets its own `LIBREWXR_*_PUBLISH_DELAY_MINUTES` and
`LIBREWXR_*_DBZ_OFFSET` for fine tuning — see `src/librewxr/config.py`
or [`docs/configuration-reference.md`](docs/configuration-reference.md)
for the full surface.

**Default recipe for broad global coverage:**

```bash
LIBREWXR_NA_NWP_SOURCE=hrrr               # CONUS + Alaska
LIBREWXR_HRDPS_ENABLED=true               # Canada
LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu # Europe (DINI + ICON-EU)
LIBREWXR_AROME_ANTILLES_ENABLED=true      # eastern Caribbean
LIBREWXR_WRF_SMN_ENABLED=true             # South American Cone
LIBREWXR_ECMWF_ENABLED=true               # global IFS layer (everywhere else)
```

The active chain is logged at startup as `NWP chain: [...]` and
surfaced under `/health` for verification.

## Examples

The `examples/` directory contains two self-contained HTML files showcasing the full LibreWXR feature set:

- **`leaflet.html`** — Leaflet-based weather map
- **`maplibre.html`** — MapLibre GL JS-based weather map

Both examples include:
- **Source selector** — switch between local (`localhost:8080`) and the public instance (`api.librewxr.net`) with auto-detection
- **Layer modes** — Radar, Satellite, or Radar + Satellite (satellite as background under radar)
- **Light/dark theme** — toggles both the base map and UI styling
- **Color scheme selector** — all 10 color schemes
- **Motion arrows** — off, light, or dark
- **Scrubber bar** — draggable timeline with past/nowcast visual distinction and tick labels
- **Background preloading** — pre-renders all frames with a progress indicator for smooth animation
- **Keyboard shortcuts** — Space to play/pause, arrow keys to step through frames
- **Locate Me** — geolocate and zoom to your position
- **Auto-refresh** — metadata refreshes every 5 minutes to stay current

Open either file in a browser — it auto-detects whether to use your local server or the public instance based on how the file is loaded.

## Data Sources

LibreWXR uses the following freely available data:

- **[NCEP MRMS](https://www.ncep.noaa.gov/products/mrms/)** — MultiSensor Reanalysis/3DReflectivity quality-controlled radar composites (US + Canadian radar ingest, default source for North American regions)
- **[Iowa Environmental Mesonet (IEM)](https://mesonet.agron.iastate.edu/)** — NEXRAD N0Q composite radar imagery (US regions, legacy fallback for MRMS)
- **[ECCC MSC GeoMet](https://eccc-msc.github.io/open-data/msc-geomet/readme_en/)** — Canadian weather radar composite (RADAR_1KM_RRAI via WMS)
- **[MARN / SNET](https://www.snet.gob.sv/)** — Servicio Nacional de Estudios Territoriales (Ministerio de Medio Ambiente y Recursos Naturales, El Salvador), San Andrés 120 km radar product. Reproduced with attribution per MARN's open-data permission.
- **[EUMETNET OPERA](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/)** — Pan-European CIRRUS radar composite via [MeteoGate](https://meteogate.eu/) S3 (~155 radars, 24 countries, ODIM HDF5)
- **[CWA QPESUMS](https://www.cwa.gov.tw/)** — Central Weather Administration of Taiwan, 7-radar composite reflectivity product `O-A0059-001` via the `cwaopendata` AWS bucket. Licensed under the [Open Government Data License v1.0](https://data.gov.tw/license) (資料來源：中央氣象署 / Source: Central Weather Administration, Taiwan).
- **[MET Malaysia](https://www.met.gov.my/)** — Jabatan Meteorologi Malaysia, 12-radar national composite (CAPPI 1 km, Rainbow 5 / LEONARDO Germany GmbH processing) via anonymous HTTPS at `api.met.gov.my`. Licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (Radar data © Jabatan Meteorologi Malaysia / METMalaysia).
- **[ECMWF IFS](https://www.ecmwf.int/) via [Open-Meteo](https://open-meteo.com/)** — ECMWF IFS 9km global precipitation, snowfall, and cloud cover. The global base layer for precipitation animation, nowcast extrapolation outside the regional NWP chain, and the satellite cloud tiles (CC-BY-4.0, data provided by Open-Meteo.com)

All sources are provided by government-funded institutions and are freely available for any use. ECMWF IFS data is provided by Open-Meteo under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Current Limitations

- **Limited radar coverage outside US / Canada / Europe / Central America / Taiwan / SE Asia** — real radar composites cover the US (CONUS, Alaska, Hawaii, Puerto Rico, Guam), Canada, El Salvador and its neighbours, Europe (via OPERA), Taiwan (CWA QPESUMS), and Malaysia + Borneo + Brunei + Singapore + N. Sumatra (MET Malaysia). Everywhere else uses the regional NWP chain on top of ECMWF IFS for the precipitation layer — that's a complete picture of global precipitation, but it's modelled output, not direct radar observation
- **Experimental nowcasting** — precipitation nowcast uses optical flow extrapolation blended with ECMWF IFS, which works well for steady, organized precipitation but is less reliable for fast-developing convection, cell initiation/dissipation, or complex terrain effects
- **Satellite is IFS-derived, not real imagery** — the satellite layer composites ECMWF IFS cloud cover fields rather than using actual satellite observations, so it reflects model output rather than real-time conditions. Update cadence is hourly (matching IFS), not the ~15-minute cadence of real geostationary satellites

## License

LibreWXR is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
