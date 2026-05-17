# LibreWXR Configuration Reference

All settings are configured via environment variables prefixed with `LIBREWXR_` or through a `.env` file. Copy `.env.example` to `.env` and adjust as needed. Every setting has a sensible default — you only need to set what you want to change.

LibreWXR uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration. Environment variables take precedence over `.env` file values.

This document is the **full** reference for every setting LibreWXR understands. The trimmed `.env.example` only covers the commonly-tuned subset; the advanced knobs (per-source publish delays, dBZ calibration offsets, source base URLs) are documented here.

## Table of Contents

- [Server](#server)
- [Radar Data](#radar-data)
- [Regions](#regions)
- [Tile Rendering](#tile-rendering)
- [Workers and Memory](#workers-and-memory)
- [Multi-worker Tile-Server Split](#multi-worker-tile-server-split)
- [ECMWF IFS Global Coverage](#ecmwf-ifs-global-coverage)
- [Regional NWP Sources](#regional-nwp-sources)
  - [North American: HRRR / HRRR-Alaska](#north-american-hrrr--hrrr-alaska)
  - [North American: HRDPS](#north-american-hrdps)
  - [European: DMI DINI + ICON-EU](#european-dmi-dini--icon-eu)
  - [Caribbean: AROME Antilles](#caribbean-arome-antilles)
  - [South American: WRF-SMN](#south-american-wrf-smn)
- [Nowcasting](#nowcasting)
- [Satellite (IFS-Derived Clouds)](#satellite-ifs-derived-clouds)
- [Weather Alerts (WMO CAP)](#weather-alerts-wmo-cap)
- [Persistent Cache](#persistent-cache)
- [Performance and Reliability](#performance-and-reliability)
- [Tile Request Tracking](#tile-request-tracking)
- [RAM Sizing Guide](#ram-sizing-guide)
- [Example Configurations](#example-configurations)

---

## Server

### `LIBREWXR_HOST`

The address the server binds to.

| | |
|---|---|
| **Default** | `0.0.0.0` |
| **Type** | string |

### `LIBREWXR_PORT`

The port the server listens on.

| | |
|---|---|
| **Default** | `8080` |
| **Type** | integer |

### `LIBREWXR_PUBLIC_URL`

The public-facing URL of your LibreWXR instance. This value is returned in the `host` field of `/public/weather-maps.json` responses. Clients use it to construct full tile URLs.

Set this to whatever URL users will use to reach your instance (e.g., your domain name, Cloudflare Tunnel URL, or reverse proxy address).

| | |
|---|---|
| **Default** | `http://localhost:8080` |
| **Type** | string |

**Example:**
```bash
LIBREWXR_PUBLIC_URL=https://radar.example.com
```

### `LIBREWXR_CORS_ORIGINS`

Allowed CORS origins for cross-origin requests from web browsers.

| | |
|---|---|
| **Default** | `["*"]` (all origins) |
| **Type** | list of strings |

If you restrict this, make sure your web app's origin is included or tile requests from browsers will fail silently.

---

## Radar Data

### `LIBREWXR_FETCH_INTERVAL`

Seconds between radar data fetches. Frame timestamps are always aligned to clock boundaries (e.g., :00, :10, :20) regardless of when the server starts.

The default of 600 seconds (10 minutes) matches Rain Viewer's cadence. IEM publishes US composites every 5 minutes; MRMS publishes every 2 minutes. Setting the interval below 300 seconds is not recommended as most sources don't update faster than that.

| | |
|---|---|
| **Default** | `600` |
| **Type** | integer |
| **Unit** | seconds |

### `LIBREWXR_MAX_FRAMES`

Number of past radar frames to keep in memory. Each frame stores radar data for all enabled regions.

At the default 10-minute cadence:
- 12 frames = 2 hours of history
- 18 frames = 3 hours
- 24 frames = 4 hours

More frames = longer animation history = more RAM usage.

| | |
|---|---|
| **Default** | `12` |
| **Type** | integer |

### `LIBREWXR_NA_SOURCE`

Data source for North American radar composites (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP, and CACOMP). Three modes:

- **`mrms_fallback`** (default) — NCEP MRMS quality-controlled mosaics as the primary source, with IEM NEXRAD fallback for US regions and MSC blending for Canadian gaps. Best coverage: MRMS includes Canadian radar ingest and quality control. IEM is only fetched when MRMS fails for a specific frame.
- **`mrms`** — NCEP MRMS only, no fallback or blending. Pure MRMS where available; gaps show as empty (the global ECMWF IFS layer still fills in outside radar coverage). Least bandwidth.
- **`iem`** — Legacy mode. IEM NEXRAD N0Q for US regions, MSC GeoMet standalone for Canada. NEXRAD-only without Canadian radar ingest. Simplest and most battle-tested, but fewer radars and no QC.

| | |
|---|---|
| **Default** | `mrms_fallback` |
| **Type** | string |
| **Values** | `mrms_fallback`, `mrms`, `iem` |

**Note:** This setting does not affect the OPERA (Europe) source, which always uses EUMETNET OPERA via MeteoGate S3.

### `LIBREWXR_MRMS_BASE_URL`

Base URL for NCEP MRMS data products. Each region (CONUS, Alaska, Hawaii, Caribbean, Guam) has its own subdirectory under this path.

| | |
|---|---|
| **Default** | `https://mrms.ncep.noaa.gov/2D` |
| **Type** | string |

Only change this if you're mirroring MRMS data to a custom endpoint.

### `LIBREWXR_IEM_BASE_URL`

Base URL for the Iowa Environmental Mesonet NEXRAD composites (US regions). Only used when `LIBREWXR_NA_SOURCE` is `iem` or `mrms_fallback`.

| | |
|---|---|
| **Default** | `https://mesonet.agron.iastate.edu` |
| **Type** | string |

### `LIBREWXR_MSC_CANADA_BASE_URL`

Base URL for the Environment and Climate Change Canada MSC GeoMet WMS service (Canadian radar). Only used when `LIBREWXR_NA_SOURCE` is `iem` or `mrms_fallback`.

| | |
|---|---|
| **Default** | `https://geo.weather.gc.ca` |
| **Type** | string |

### `LIBREWXR_OPERA_BASE_URL`

Base URL for the OPERA CIRRUS composite S3 bucket (European radar via MeteoGate).

| | |
|---|---|
| **Default** | `https://s3.waw3-1.cloudferro.com` |
| **Type** | string |

### `LIBREWXR_MARN_BASE_URL`

Base URL for the MARN/SNET (El Salvador) radar bucket on Google Cloud Storage. The source reads the `radar-images-sv` bucket anonymously under this host. Only used when `SVCOMP` or the `CENTRAL_AMERICA` group is in `LIBREWXR_ENABLED_REGIONS`.

| | |
|---|---|
| **Default** | `https://storage.googleapis.com` |
| **Type** | string |

### `LIBREWXR_CWA_BASE_URL`

Base URL for the Taiwan CWA QPESUMS composite bucket on AWS S3 (`cwaopendata` in `ap-northeast-1`). The source reads archive XML keys at `/history/Observation/{YYYYMMDDHHMM}compref_mosaic.xml` (no separator dot between the timestamp and the product name). Only used when `TWCOMP` or the `TAIWAN` group is in `LIBREWXR_ENABLED_REGIONS`.

| | |
|---|---|
| **Default** | `https://cwaopendata.s3.ap-northeast-1.amazonaws.com` |
| **Type** | string |

### `LIBREWXR_MMD_BASE_URL`

Base URL for the MET Malaysia radar composite endpoint. The animated GIF at `{base}/static/images/radar-latest.gif` carries 6 frames at 10-min cadence (~60 min of backfill per fetch). CC-BY-4.0 — attribution required. Only used when `MYPENINSULAR`, `MYEAST`, or the `SOUTHEAST_ASIA` group is in `LIBREWXR_ENABLED_REGIONS`.

| | |
|---|---|
| **Default** | `https://api.met.gov.my` |
| **Type** | string |

### `LIBREWXR_MMD_ENABLED`

Master toggle for the MET Malaysia source. When `false`, drops `MYPENINSULAR` and `MYEAST` from the active region set even if a group alias (`SOUTHEAST_ASIA`, `ALL`) would otherwise pull them in.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_MMD_PUBLISH_LAG_SEC`

Estimated publication lag (seconds) between a MET Malaysia frame's data time and when the carrying GIF lands at `api.met.gov.my`. The GIF carries no structured per-frame timestamps, so the newest frame's UTC time is derived from `floor(Last-Modified - mmd_publish_lag_sec, 10min)`. The empirically observed lag is ~11 min; 600 s gives a safe rounding margin. Bump if you observe the latest store slot stuck behind by one frame.

| | |
|---|---|
| **Default** | `600` |
| **Type** | integer (seconds) |

---

## Regions

### `LIBREWXR_ENABLED_REGIONS`

Which radar regions to fetch and serve. Accepts group aliases, individual region codes, or comma-separated combinations.

| | |
|---|---|
| **Default** | `ALL` |
| **Type** | string |

**Group aliases:**

| Group | Expands to | Description |
|-------|-----------|-------------|
| `CONUS` | `USCOMP` | Continental US only (lightest option) |
| `US` | `USCOMP`, `AKCOMP`, `HICOMP`, `PRCOMP`, `GUCOMP` | All US regions |
| `CANADA` | `CACOMP` | Canada |
| `CENTRAL_AMERICA` | `SVCOMP` | El Salvador + W. Honduras + S. Guatemala + offshore Pacific |
| `EUROPE` | `OPERA` | Pan-European composite (~155 radars, 24 countries) |
| `SOUTHEAST_ASIA` | `MYPENINSULAR`, `MYEAST` | Peninsular Malaysia + N. Sumatra + all of Borneo + Brunei + Singapore (MET Malaysia 12-radar composite) |
| `TAIWAN` | `TWCOMP` | Taiwan + W. Pacific buffer (CWA QPESUMS 7-radar composite) |
| `ALL` | All of the above | Every available region |

**Individual regions:**

| Region | Area | Source | Grid Size | Resolution | RAM / Frame |
|--------|------|--------|-----------|------------|-------------|
| `USCOMP` | Continental US | NCEP MRMS (IEM fallback) | 12200 x 5400 | 0.005° (~500m) | ~63 MB |
| `AKCOMP` | Alaska | NCEP MRMS (IEM fallback) | 4000 x 1550 | 0.01° (~1km) | ~6 MB |
| `HICOMP` | Hawaii | NCEP MRMS (IEM fallback) | 2000 x 1800 | 0.005° (~500m) | ~3.4 MB |
| `PRCOMP` | Puerto Rico | NCEP MRMS (IEM fallback) | 1000 x 1000 | 0.01° (~1km) | ~1 MB |
| `GUCOMP` | Guam | NCEP MRMS (IEM fallback) | 1000 x 1000 | 0.0085° (~850m) | ~1 MB |
| `CACOMP` | Canada | MSC GeoMet (MRMS blending) | 3560 x 1720 | 0.025° (~2.5km) | ~6 MB |
| `SVCOMP` | El Salvador + neighbours | MARN/SNET (San Andrés, 120 km) | 409 x 342 | 0.00926° (~1km) | <1 MB |
| `OPERA` | Europe | EUMETNET OPERA (MeteoGate S3) | 3800 x 4400 | 1km (LAEA) | ~16 MB |
| `TWCOMP` | Taiwan + W. Pacific | CWA QPESUMS (cwaopendata S3) | 921 x 881 | 0.0125° (~1.4km) | ~3 MB |
| `MYPENINSULAR` | Peninsular Malaysia + N. Sumatra | MET Malaysia (12-radar composite) | 424 x 551 | 0.022° lon / 0.019° lat (~2.5km) | <1 MB |
| `MYEAST` | East Malaysia (Borneo) + Brunei | MET Malaysia (12-radar composite) | 640 x 570 | 0.022° lon / 0.019° lat (~2.5km) | <1 MB |

**Examples:**
```bash
LIBREWXR_ENABLED_REGIONS=CONUS            # Continental US only
LIBREWXR_ENABLED_REGIONS=US               # All US regions
LIBREWXR_ENABLED_REGIONS=EUROPE           # Europe only
LIBREWXR_ENABLED_REGIONS=CANADA           # Canada only
LIBREWXR_ENABLED_REGIONS=CONUS,EUROPE     # Continental US + Europe
LIBREWXR_ENABLED_REGIONS=US,CANADA        # US + Canada
LIBREWXR_ENABLED_REGIONS=ALL              # Everything
```

---

## Tile Rendering

### `LIBREWXR_MAX_ZOOM`

Maximum tile zoom level. Higher values allow more detail when zoomed in but use more memory for cached tiles. 12 is the maximum supported by the source data resolution.

| | |
|---|---|
| **Default** | `12` |
| **Type** | integer |
| **Range** | 0 - 12 |

### `LIBREWXR_SMOOTH_RADIUS`

Gaussian blur radius applied when smoothing is enabled in the tile URL (`smooth=1`). Higher values produce a softer appearance. Set to 0 to disable smoothing entirely, even when clients request it.

| | |
|---|---|
| **Default** | `2.0` |
| **Type** | float |

**Recommended range:** 2.0 - 4.0. Rain Viewer used approximately 3.0.

### `LIBREWXR_NOISE_FLOOR_DBZ`

Minimum dBZ value to display. Pixels below this threshold are made transparent. Filters out ground clutter, anomalous propagation, and weak noise.

For reference on the dBZ scale:
- 5 dBZ = barely detectable
- 10 dBZ = very light precipitation
- 20 dBZ = light rain

| | |
|---|---|
| **Default** | `10.0` |
| **Type** | float |

Set to `-32` to disable and show everything.

### `LIBREWXR_DESPECKLE_MIN_NEIGHBORS`

Speckle filter strength. A pixel is removed if it has fewer than this many non-zero neighbors (out of 8 surrounding pixels). Removes isolated radar artifacts and ground clutter.

| | |
|---|---|
| **Default** | `3` |
| **Type** | integer |
| **Range** | 0 - 8 |

- `0` = disabled
- `2` = light filtering
- `3` = moderate (recommended)
- `4+` = aggressive (may remove edges of real precipitation)

### `LIBREWXR_WEBP_QUALITY`

WebP encoding quality for tiles requested in `.webp` format. Does not affect PNG tiles.

| | |
|---|---|
| **Default** | `65` |
| **Type** | integer |
| **Range** | 1 - 100 |

- `100` = lossless (best quality, larger files)
- `65` = lossy (visually identical for radar imagery, ~4-6x smaller than PNG)
- `1-64` = increasingly lossy

### `LIBREWXR_TILE_CACHE_MB`

Maximum tile cache size in megabytes, **per worker**. The cache stores rendered tile images (PNG/WebP bytes) and evicts the oldest entries when this byte limit is reached.

Higher values mean faster tile serving for repeat requests; lower values save RAM. In multi-worker mode the default is reduced to 64 MB per worker (set via `docker-compose.multiworker.yml`) since many workers share the rack.

| | |
|---|---|
| **Default** | `200` |
| **Type** | integer |
| **Unit** | megabytes |

### `LIBREWXR_COORD_CACHE_SIZE`

Maximum entries per coordinate LRU cache, **per worker**. Controls how many tile-coordinate mappings are kept in memory. There are 6 internal coordinate caches, and each entry is 0.5-2 MB depending on tile size.

These caches are the largest RAM consumer after frame data. Reducing this saves significant RAM at the cost of occasional recomputation (~5-20 ms per cache miss). In multi-worker mode the default is reduced to 256 per worker.

| | |
|---|---|
| **Default** | `2048` |
| **Type** | integer |

### `LIBREWXR_WARMER_THREADS`

Thread pool size for background tile cache warming, **per worker**. When a tile is requested, the warmer pre-renders that same tile position for all other timestamps in the background, so animation playback is smooth without waiting for each frame to render on demand.

| | |
|---|---|
| **Default** | `0` (auto: CPU count - 1) |
| **Type** | integer |

In multi-worker mode the default is reduced to 4 per worker — many workers each with a full thread pool would oversubscribe the rack.

### `LIBREWXR_WARM_COORD_ZOOM`

Pre-warm coordinate caches up to this zoom level at startup. Coordinate caches store tile-to-region pixel index mappings; warming them eliminates cold-start latency from trigonometric projections.

| | |
|---|---|
| **Default** | `6` |
| **Type** | integer |

Each zoom level adds ~4x the tiles of the previous (zoom 6 = ~5,500 tiles). Set to `0` to disable.

### `LIBREWXR_WARM_OVERVIEW_ZOOM`

Pre-render overview tiles up to this zoom level after each fetch cycle. Ensures zoomed-out views are served instantly from cache.

| | |
|---|---|
| **Default** | `4` |
| **Type** | integer |

At zoom 4, ~341 tiles per timestamp. Set to `-1` to disable.

### `LIBREWXR_WARM_OVERVIEW_ZOOM_REGIONAL`

Pre-render higher-zoom tiles ONLY where they overlap an enabled region's bounding box. Skips ocean / desert / unpopulated tiles that no one would zoom into.

Applies between `LIBREWXR_WARM_OVERVIEW_ZOOM` (exclusive) and this value (inclusive).

| | |
|---|---|
| **Default** | `6` |
| **Type** | integer |

Set to `-1` (or any value `<= warm_overview_zoom`) to disable. At zoom 6 with all regions enabled, the filter typically drops 80-85% of tiles.

---

## Workers and Memory

### `LIBREWXR_WORKERS`

Number of uvicorn worker processes.

| | |
|---|---|
| **Default** | `1` |
| **Type** | integer |

- **Single-process mode** (`docker-compose.yml`): each worker is a fully independent copy of LibreWXR with its own frame store, caches, and fetcher. More workers = more concurrency at ~1.3 GB+ RAM each. Recommended: 1 worker per 2 CPU cores; put a caching proxy in front for high traffic.
- **Multi-worker mode** (`docker-compose.multiworker.yml`): render workers share radar/NWP/cloud state via memmap snapshots written by a sidecar `data-pipeline` process. Scale workers across many cores without the per-worker data RAM cost — total RSS ≈ workers × (interpreter ~80 MB + tile cache + coord cache) + a single shared page-cache backing the memmap. Recommended: 8-32 workers depending on rack size.

### `LIBREWXR_MEMORY_LIMIT_MB`

Memory limit in MB for the in-process memory pressure monitor. When RSS usage exceeds 85% of this limit, the tile cache and coordinate caches are automatically trimmed.

| | |
|---|---|
| **Default** | `0` (auto-detect) |
| **Type** | integer |
| **Unit** | megabytes |

When set to `0`, the limit is auto-detected from Docker/cgroup limits or falls back to system RAM.

### `LIBREWXR_MEMORY_PRESSURE_CHECK_INTERVAL`

Seconds between memory pressure checks.

| | |
|---|---|
| **Default** | `30` |
| **Type** | integer |
| **Unit** | seconds |

### Docker memory limits

The compose files cap the containers using these env vars (not LIBREWXR_* settings — they're consumed by `deploy.resources.limits` in the YAML directly):

| Var | Default | Compose file |
|---|---|---|
| `LIBREWXR_MEMORY` | `7G` | `docker-compose.yml` (single-container) |
| `LIBREWXR_PIPELINE_MEMORY` | `12G` | `docker-compose.multiworker.yml` (pipeline container) |
| `LIBREWXR_RENDER_MEMORY` | `18G` | `docker-compose.multiworker.yml` (render container) |

Production observation on an 80-core / 32 GB rack with multi-worker mode (32 render workers): ~16 GB total RSS settled across both containers under continuous traffic.

---

## Multi-worker Tile-Server Split

Runs the data pipeline as one process and N tile-server worker processes alongside it, all sharing `LIBREWXR_CACHE_DIR` via memmap files + a single `state.json` snapshot. Bypasses Python's GIL on the tile-render path so the rack's full core count can actually do work.

To enable:
1. Set `LIBREWXR_CACHE_DIR` to a shared directory (required).
2. Use `docker-compose.multiworker.yml`, or run the two processes manually:
   ```bash
   python -m librewxr.data_pipeline                       # sidecar
   LIBREWXR_RENDER_ONLY=1 LIBREWXR_WORKERS=N \
     python -m librewxr.main                              # tile server
   ```

### `LIBREWXR_RENDER_ONLY`

When `true` (or `1`), the worker skips fetcher / NWP grid / cloud / nowcast initialization entirely. It only memory-maps the snapshot the pipeline writes and renders tiles from it.

| | |
|---|---|
| **Default** | `false` |
| **Type** | boolean |

### `LIBREWXR_STATE_POLL_INTERVAL`

Seconds between `state.json` mtime polls in render-only mode. The pipeline rewrites `state.json` once per `LIBREWXR_FETCH_INTERVAL` (default 600 s), so 1 s polls are responsive without burning CPU.

| | |
|---|---|
| **Default** | `1.0` |
| **Type** | float |
| **Unit** | seconds |

### `LIBREWXR_STATE_WAIT_TIMEOUT`

Seconds for render workers to wait for the first `state.json` on cold start before failing loudly. `0` = wait forever.

| | |
|---|---|
| **Default** | `300` |
| **Type** | float |
| **Unit** | seconds |

---

## ECMWF IFS Global Coverage

LibreWXR uses ECMWF IFS 9 km global data from [Open-Meteo](https://open-meteo.com/) S3 as the global base layer of its NWP chain. IFS provides:

- Precipitation animation everywhere the regional NWP chain doesn't reach
- Per-pixel snow/rain classification
- Cloud cover for the IR-like satellite tile layer
- Nowcast extrapolation outside regional model coverage

### `LIBREWXR_ECMWF_ENABLED`

Disable ECMWF IFS entirely. Useful only for isolating regional NWP layers during debugging — anywhere outside the regional models will then simply show zero precipitation and no satellite cloud.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_ECMWF_S3_BUCKET`

S3 bucket name for Open-Meteo ECMWF data.

| | |
|---|---|
| **Default** | `openmeteo` |
| **Type** | string |

### `LIBREWXR_ECMWF_S3_REGION`

AWS region of the Open-Meteo S3 bucket.

| | |
|---|---|
| **Default** | `us-west-2` |
| **Type** | string |

### `LIBREWXR_ECMWF_S3_PREFIX`

S3 key prefix for ECMWF IFS data.

| | |
|---|---|
| **Default** | `data_spatial/ecmwf_ifs` |
| **Type** | string |

### `LIBREWXR_ECMWF_SNOW_RATIO_THRESHOLD`

Snowfall fraction threshold for per-pixel snow/rain classification. When the snow-to-total precipitation ratio exceeds this value, the pixel is classified as snow and rendered with the snow color palette (when `snow=1` in the tile URL).

| | |
|---|---|
| **Default** | `0.5` |
| **Type** | float |
| **Range** | 0.0 - 1.0 |

### `LIBREWXR_ECMWF_MAX_TIMESTEPS`

Number of ECMWF IFS hourly timesteps to fetch for global precipitation animation.

| | |
|---|---|
| **Default** | `0` (auto) |
| **Type** | integer |

When set to `0` (recommended), the count is derived automatically from `LIBREWXR_MAX_FRAMES` + nowcast frames so the IFS animation covers the same time window as radar.

### `LIBREWXR_ECMWF_INTERPOLATION`

Enable optical flow interpolation of ECMWF IFS hourly data to 10-minute frames. Uses dense motion vectors (OpenCV Farneback) to animate precipitation movement between IFS hours, so the global IFS layer animates smoothly like real radar data instead of jumping hour-to-hour.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

Adds ~130 MB RAM for synthetic frames and ~5-10 seconds of compute per IFS fetch cycle.

---

## Regional NWP Sources

LibreWXR layers a chain of regional rapid-refresh NWP models on top of the global ECMWF IFS layer. At each pixel, the chain dispatches to the **narrowest** model whose domain covers it, soft-feathering at every domain edge so seams don't show. See [`coverage.md`](coverage.md) for visual maps of every radar + NWP domain.

Most regional sources also classify each pixel as snow vs rain from their own 2-metre temperature field. The threshold is shared across all of them via [`LIBREWXR_REGIONAL_SNOW_TEMP_THRESHOLD`](#librewxr_regional_snow_temp_threshold). HRRR-CONUS, HRRR-Alaska, WRF-SMN, DMI DINI, and ICON-EU all classify natively; HRDPS and AROME Antilles fall through to IFS for snow detection (HRDPS is expected to be replaced by RRFSv1 mid-2026; AROME Antilles is tropical so the question rarely matters).

Each regional source supports the same set of advanced tuning knobs:

- `<SOURCE>_PUBLISH_DELAY_MINUTES` — how long after a model run's init time its files become available upstream. The fetcher won't try to read a run published more recently than this.
- `<SOURCE>_DBZ_OFFSET` — a dBZ calibration shift applied after Marshall-Palmer Z-R conversion (only for sources that derive reflectivity from precipitation rate, not those with native composite reflectivity). Marshall-Palmer is for stratiform rain at the surface; radar reads 5-10 dBZ higher at the brightest part of the storm column, so a positive offset brings model output closer to OPERA / NEXRAD radar in colour.
- `<SOURCE>_BASE_URL` (HTTPS sources) or `<SOURCE>_S3_BUCKET` + `<SOURCE>_S3_REGION` (AWS Open Data sources) — should rarely need changing; the defaults point at the upstream-provider buckets.

### North American: HRRR / HRRR-Alaska

NOAA HRRR runs at 3 km native resolution on disjoint CONUS (LCC) and Alaska (polar stereographic) domains, both via the same anonymous AWS Open Data bucket. CONUS uses the `wrfsubhf` 15-min sub-hourly product; Alaska uses hourly `wrfsfcf`. The two domains share one toggle: enabling `hrrr` turns on both.

#### `LIBREWXR_NA_NWP_SOURCE`

| | |
|---|---|
| **Default** | `ifs` |
| **Type** | string |
| **Values** | `ifs`, `hrrr` |

- **`ifs`** — IFS only; no regional NWP over CONUS or Alaska.
- **`hrrr`** — Adds HRRR-CONUS (3 km LCC, 15-min subh, hourly cycles) and HRRR-Alaska (3 km polar stereo, hourly wrfsfcf, 3-hourly cycles) to the chain.

#### `LIBREWXR_HRRR_S3_BUCKET`

| | |
|---|---|
| **Default** | `noaa-hrrr-bdp-pds` |
| **Type** | string |

#### `LIBREWXR_HRRR_S3_REGION`

| | |
|---|---|
| **Default** | `us-east-1` |
| **Type** | string |

#### `LIBREWXR_HRRR_PUBLISH_DELAY_MINUTES`

Minutes after HRRR-CONUS run init before the `wrfsubhf` files are typically published.

| | |
|---|---|
| **Default** | `55` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_HRRR_ALASKA_PUBLISH_DELAY_MINUTES`

HRRR-Alaska run takes longer than CONUS subh — the full 0–48 h horizon is typically published ~80 min after run init. Bump higher if you see 404s on the freshest cycle.

| | |
|---|---|
| **Default** | `80` |
| **Type** | integer |
| **Unit** | minutes |

HRRR's native composite reflectivity field is used directly — no `DBZ_OFFSET` needed (no Marshall-Palmer conversion).

### North American: HRDPS

ECCC HRDPS Continental at 2.5 km rotated lat/lon. 4 cycles/day (00/06/12/18 UTC), 48 h horizon, 1-hour APCP accumulation. Anonymous HTTPS via dd.weather.gc.ca. Covers Canada + the northern fringe of CONUS — disjoint enough from HRRR's CONUS focus that they layer cleanly (HRRR first inside CONUS where it's denser, HRDPS second to fill Canada).

#### `LIBREWXR_HRDPS_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

#### `LIBREWXR_HRDPS_BASE_URL`

| | |
|---|---|
| **Default** | `https://dd.weather.gc.ca` |
| **Type** | string |

The URL builder appends the date-prefixed archive path so backfill spans midnight UTC cleanly without the `/today/` tree rolling out from under an in-flight fetch.

#### `LIBREWXR_HRDPS_PUBLISH_DELAY_MINUTES`

| | |
|---|---|
| **Default** | `240` (~4 hours) |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_HRDPS_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

### European: DMI DINI + ICON-EU

LibreWXR's European NWP chain uses both **DMI HARMONIE-AROME DINI** (2 km native LCC) and **DWD ICON-EU** (~7 km regridded lat/lon). DINI covers most of populated Europe (UK, France, Benelux, Germany, Alps, Czechia, Poland, southern Scandinavia, Iceland); ICON-EU fills the European remainder DINI doesn't reach (Iberia, southern Italy, Greece, the Balkans, and eastern Europe past Poland).

#### `LIBREWXR_EU_NWP_PROFILE`

| | |
|---|---|
| **Default** | `ifs` |
| **Type** | string |
| **Values** | `ifs`, `icon_eu_only`, `dini_with_icon_eu` |

- **`ifs`** — IFS only; no regional NWP over Europe.
- **`icon_eu_only`** — DWD ICON-EU ahead of IFS. Free DWD opendata HTTPS — no auth. Covers all of Europe broadly.
- **`dini_with_icon_eu`** — DMI HARMONIE-AROME DINI ahead of ICON-EU ahead of IFS. Anonymous AWS Open Data S3. Best European coverage; adds ~250 MB RAM total.

(Renamed from `LIBREWXR_EU_NWP_SOURCE` on 2026-05-03 — the old `dmi_dini` value implicitly loaded ICON-EU too, which was surprising. The new profile names make the loaded set obvious.)

#### `LIBREWXR_ICON_EU_BASE_URL`

| | |
|---|---|
| **Default** | `https://opendata.dwd.de/weather/nwp/icon-eu/grib` |
| **Type** | string |

#### `LIBREWXR_ICON_EU_PUBLISH_DELAY_MINUTES`

DWD main runs typically publish ~3-4 h after init.

| | |
|---|---|
| **Default** | `240` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_ICON_EU_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

#### `LIBREWXR_DMI_DINI_S3_BUCKET`

| | |
|---|---|
| **Default** | `dmi-opendata` |
| **Type** | string |

#### `LIBREWXR_DMI_DINI_S3_REGION`

| | |
|---|---|
| **Default** | `eu-north-1` |
| **Type** | string |

#### `LIBREWXR_DMI_DINI_PUBLISH_DELAY_MINUTES`

DMI files publish ~3 h after run init.

| | |
|---|---|
| **Default** | `180` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_DMI_DINI_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

### Caribbean: AROME Antilles

Météo-France AROME Antilles at 1.3 km native resolution, public-dist as 0.025° regular lat/lon. 4 cycles/day (00/06/12/18 UTC), 48 h horizon. Anonymous via the data.gouv.fr open-data portal. Covers Guadeloupe, Martinique, Saint Martin, Saint-Barthélemy, and the surrounding waters of the eastern Caribbean (~22.9°N → 9.7°N, -75.3°E → -51.7°E). Tiny in-memory cost since the domain is small.

#### `LIBREWXR_AROME_ANTILLES_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

#### `LIBREWXR_AROME_ANTILLES_BASE_URL`

| | |
|---|---|
| **Default** | `https://object.data.gouv.fr/meteofrance-pnt` |
| **Type** | string |

#### `LIBREWXR_AROME_ANTILLES_PUBLISH_DELAY_MINUTES`

Full 0..48h files publish ~6-7 h after init; 7 h is conservative.

| | |
|---|---|
| **Default** | `420` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_AROME_ANTILLES_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

### South American: WRF-SMN

Servicio Meteorológico Nacional Argentina WRF-DET at 4 km LCC. First regional NWP for the South American Cone — covers Argentina, Chile, Uruguay, Paraguay, Bolivia, southern Brazil + adjacent oceans. Anonymous AWS Open Data (smn-ar-wrf in us-west-2). 4 cycles/day, 72 h horizon. Files are NetCDF4 (~34 MB each — the only non-GRIB source in the chain).

#### `LIBREWXR_WRF_SMN_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

#### `LIBREWXR_WRF_SMN_S3_BUCKET`

| | |
|---|---|
| **Default** | `smn-ar-wrf` |
| **Type** | string |

#### `LIBREWXR_WRF_SMN_S3_REGION`

| | |
|---|---|
| **Default** | `us-west-2` |
| **Type** | string |

Note the bucket lives in **us-west-2**, not us-east-1 as the AWS Open Data Registry page suggests.

#### `LIBREWXR_WRF_SMN_PUBLISH_DELAY_MINUTES`

Full 0..72h files publish ~3-4 h after init; 4 h is conservative.

| | |
|---|---|
| **Default** | `240` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_WRF_SMN_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

### `LIBREWXR_REGIONAL_SNOW_TEMP_THRESHOLD`

Temperature threshold for native snow/rain classification across every regional NWP source that derives a snow mask from its own 2-metre temperature field (HRRR-CONUS, HRRR-Alaska, WRF-SMN, DMI DINI, ICON-EU). Pixels colder than this threshold are tagged as snow and rendered with the snow palette when `snow=1` is set on the tile URL.

| | |
|---|---|
| **Default** | `1.5` |
| **Type** | float |
| **Unit** | degrees Celsius |

1.5 °C is a typical near-surface snow-vs-rain transition line. Drop towards 0 °C for a stricter "only true freezing" definition; raise to 2-3 °C to catch wet snow / sleet conditions that visually behave like snow on the ground.

### `LIBREWXR_REGIONAL_INTERPOLATION`

Enable optical-flow temporal interpolation of hourly regional NWP frames to 10-minute steps. Uses the same OpenCV Farneback dense flow we apply to ECMWF IFS, applied at the end of each fetch cycle to every regional source whose native cadence is hourly (currently WRF-SMN, DMI DINI, ICON-EU).

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

Without this, a moving precip cell appears to cross-fade between hourly bracket frames at intermediate query times, producing a visible "two faint copies" ghost. With it, the cell translates smoothly along motion vectors. Adds ~2-10 s of CPU per source per fetch cycle (smallest grid ~2 s, largest ~10 s). Snow masks ride alongside precip through the same interpolation step.

### `LIBREWXR_NWP_FETCH_CONCURRENCY`

Maximum number of NWP grid fetches running in parallel inside one fetch cycle. Each grid loads tens-to-hundreds of MB during decode, so this caps peak transient RAM at ~N × per-grid working set.

| | |
|---|---|
| **Default** | `4` |
| **Type** | integer |

4 fits comfortably in 8 GB; bump to 6-8 on bigger rigs (the multi-worker compose file overrides this to 8) to bring cycle wall time closer to the slowest single source.

---

## Nowcasting

Precipitation nowcasting is an experimental feature that extrapolates recent radar data forward using optical flow to generate short-range forecast frames. The frames can optionally be blended with the active NWP model's forecast.

### `LIBREWXR_NOWCAST_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

When enabled, nowcast frames appear in the `radar.nowcast` array of the `/public/weather-maps.json` response.

### `LIBREWXR_NOWCAST_FRAMES`

Number of nowcast frames to generate. Each frame covers one `LIBREWXR_FETCH_INTERVAL`.

| | |
|---|---|
| **Default** | `6` |
| **Type** | integer |

At the default 10-minute cadence, 6 frames = 60 minutes of forecast. More frames extend the forecast range but accuracy decreases at the far end.

### `LIBREWXR_NOWCAST_BLEND_MODE`

Controls how radar extrapolation and the NWP model forecast are combined during the first 60 minutes of the nowcast window. Beyond 60 minutes, the pure NWP model is always used regardless of this setting.

The model side is taken from the active NWP chain — **HRRR over CONUS, HRDPS over Canada, DINI/ICON-EU over Europe, AROME Antilles over the Caribbean, WRF-SMN over the S. American Cone, and ECMWF IFS everywhere else.**

| | |
|---|---|
| **Default** | `blended` |
| **Type** | string |
| **Values** | `radar`, `blended`, `model` |

- **`radar`** — Pure radar extrapolation for the first 60 minutes. Closest to Rain Viewer behavior. Visibly diverges from reality past ~30 minutes for fast-moving convection, since the extrapolation has no skill at cell initiation or dissipation.
- **`blended`** (default) — Smooth transition from radar-heavy to model-heavy. The blend curve is `0.20 + 0.80 * (1 - t)^1.4` where `t` is normalized time from 0 to 1 across the 60-min window — about 100% radar at T+0, ~82% radar at T+10 min, ~50% at T+30 min, ~20% radar at T+60 min. Spatial feathering at radar coverage boundaries prevents hard seams. Leverages the regional NWP chain quality for the far end of the window.
- **`model`** — Pure NWP forecast for all nowcast frames. Most spatially consistent but misses fine detail from recent radar observations.

(Value renamed from `ifs` to `model` after the regional NWP chain shipped — the model side is no longer IFS-only.)

---

## Satellite (IFS-Derived Clouds)

Composites ECMWF IFS cloud_cover_high / mid / low into IR-like cloud tiles. Populates the Rain Viewer-compatible `satellite.infrared` endpoint.

### `LIBREWXR_SATELLITE_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_SATELLITE_MAX_FRAMES`

Hourly IFS cloud cover timesteps to keep. Each timestep adds ~20 MB RAM (three 3600x1801 uint8 grids).

| | |
|---|---|
| **Default** | `12` |
| **Type** | integer |

12 timesteps = 12 hours of cloud cover animation.

---

## Weather Alerts (WMO CAP)

Fetches global weather alerts from severeweather.wmo.int. MeteoAlarm geocodes are downloaded on first startup and cached locally. Updates are clock-aligned (:00, :05, :10, …).

For US locations, point lookups also query the NWS point endpoint at api.weather.gov to surface non-polygon alerts (e.g. Tornado Watches) that lack geometry in the global feed.

### `LIBREWXR_ALERTS_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_ALERTS_FETCH_INTERVAL`

How often (in seconds) to refresh alerts. Matches the upstream update cadence at 300 s; setting to 600 s halves the request volume.

| | |
|---|---|
| **Default** | `300` |
| **Type** | integer |
| **Unit** | seconds |

### `LIBREWXR_ALERTS_CONCURRENCY`

Max concurrent HTTP connections when polling the WMO endpoints.

| | |
|---|---|
| **Default** | `5` |
| **Type** | integer |

### `LIBREWXR_ALERTS_CACHE_DIR`

Cache directory for the downloaded MeteoAlarm geocode data. Empty = system temp.

| | |
|---|---|
| **Default** | *(empty)* |
| **Type** | string |

---

## Persistent Cache

### `LIBREWXR_CACHE_DIR`

Cache directory for processed grids (satellite cloud, NWP, alerts geocodes, master state snapshot). When set, data is saved as memory-mapped files that survive restarts, crashes, and container recreation — no need to re-download from upstream on startup.

| | |
|---|---|
| **Default** | *(empty — in-memory only)* |
| **Type** | string |

**Required** in multi-worker mode. Both the pipeline and render containers must share this directory via a named volume.

- Docker (single-container): set automatically via a named volume in `docker-compose.yml`.
- Docker (multi-worker): set automatically via the shared named volume in `docker-compose.multiworker.yml`.
- Local dev: set to a local path like `./cache`.

---

## Performance and Reliability

### `LIBREWXR_DOWNLOAD_RETRIES`

Number of retries on transient download errors (connection refused, timeout, DNS failure, truncated response). Each retry waits 1 second before trying again. Applies to all data sources: radar, NWP, satellite, alerts.

| | |
|---|---|
| **Default** | `1` |
| **Type** | integer |

`0` = fail immediately, `1` = one retry (2 total attempts).

---

## Tile Request Tracking

When enabled, per-tile request counts at high zooms are recorded in memory and surfaced in `/health` under `tile_requests`. Observational only — used to identify hotspots for a future adaptive pre-warming pass. Counters reset on restart.

### `LIBREWXR_TILE_TRACKING_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_TILE_TRACKING_MIN_ZOOM`

Track only zoom levels at or above this value. Lower zooms are pre-warmed already so they don't need observation.

| | |
|---|---|
| **Default** | `7` |
| **Type** | integer |

### `LIBREWXR_TILE_TRACKING_MAX_ENTRIES`

Cap on per-tile counter entries. When full, the table halves (drops the lower half of counters) and continues.

| | |
|---|---|
| **Default** | `10000` |
| **Type** | integer |

---

## RAM Sizing Guide

### Single-container mode

Each worker process holds its own copy of radar frames, NWP grids, coordinate caches, and tile caches. RAM grows under real traffic as caches fill up.

| Configuration | Estimated RAM |
|---|---|
| CONUS + IFS only, 1 worker, 12 frames | ~3-4 GB |
| CONUS + HRRR + IFS, 1 worker, 12 frames | ~4-5 GB |
| ALL regions + IFS only, 1 worker, 12 frames | ~7-8 GB |
| ALL regions + full NWP chain, 1 worker, 12 frames | ~9-10 GB |
| ALL regions + full NWP chain, 2 workers, 12 frames | ~16-18 GB |

### Multi-worker mode

Render workers share radar/NWP/cloud state via memmap, so adding workers doesn't multiply the data RAM — only the per-worker tile cache and Python interpreter overhead (~80 MB).

| Configuration | Pipeline RAM | Render RAM (total) | Total |
|---|---|---|---|
| ALL regions + full NWP chain, 8 workers | ~8-10 GB | ~3-4 GB | ~12-14 GB |
| ALL regions + full NWP chain, 16 workers | ~8-10 GB | ~5-6 GB | ~14-16 GB |
| ALL regions + full NWP chain, 32 workers | ~8-10 GB | ~7-8 GB | ~16-18 GB |

Production observation on an 80-core / 32 GB rack with 32 workers: ~16 GB total RSS settled across both containers.

---

## Example Configurations

### Minimal (personal use, US only, single-container)

```bash
LIBREWXR_PUBLIC_URL=http://localhost:8080
LIBREWXR_ENABLED_REGIONS=CONUS
LIBREWXR_NA_NWP_SOURCE=hrrr           # adds HRRR-CONUS for high-res forecasts
LIBREWXR_HRDPS_ENABLED=false          # not needed without Canada
LIBREWXR_AROME_ANTILLES_ENABLED=false
LIBREWXR_WRF_SMN_ENABLED=false
LIBREWXR_EU_NWP_PROFILE=ifs           # IFS only over Europe (we don't show it)
```

Docker memory limit: ~5 GB

### Full coverage, personal / small server (single-container)

```bash
LIBREWXR_PUBLIC_URL=https://radar.example.com
LIBREWXR_ENABLED_REGIONS=ALL
LIBREWXR_NA_NWP_SOURCE=hrrr
LIBREWXR_HRDPS_ENABLED=true
LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu
LIBREWXR_AROME_ANTILLES_ENABLED=true
LIBREWXR_WRF_SMN_ENABLED=true
```

Docker memory limit: ~10 GB

### Production / multi-worker (full coverage, busy public instance)

In `.env`:
```bash
LIBREWXR_PUBLIC_URL=https://radar.example.com
LIBREWXR_ENABLED_REGIONS=ALL
LIBREWXR_NA_NWP_SOURCE=hrrr
LIBREWXR_HRDPS_ENABLED=true
LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu
LIBREWXR_AROME_ANTILLES_ENABLED=true
LIBREWXR_WRF_SMN_ENABLED=true
LIBREWXR_WORKERS=32                   # render worker pool size
```

Then run:
```bash
docker compose -f docker-compose.multiworker.yml up -d
```

The compose file sets `LIBREWXR_NWP_FETCH_CONCURRENCY=8` and the per-worker tile cache / coord cache / warmer thread defaults automatically.

Defaults: pipeline cap 12 GB, render cap 18 GB, total ~16 GB RSS in practice.

### Lightweight / low-RAM

```bash
LIBREWXR_ENABLED_REGIONS=CONUS
LIBREWXR_MAX_FRAMES=6
LIBREWXR_COORD_CACHE_SIZE=512
LIBREWXR_TILE_CACHE_MB=50
LIBREWXR_ECMWF_INTERPOLATION=false
LIBREWXR_NOWCAST_ENABLED=false
LIBREWXR_NA_NWP_SOURCE=ifs            # skip HRRR — IFS only
LIBREWXR_HRDPS_ENABLED=false
LIBREWXR_AROME_ANTILLES_ENABLED=false
LIBREWXR_WRF_SMN_ENABLED=false
LIBREWXR_SATELLITE_ENABLED=false
LIBREWXR_ALERTS_ENABLED=false
```

Minimizes RAM at the cost of shorter history (1 hour), slower cache hits, no interpolation/nowcast, no regional NWP, no satellite, no alerts. Docker memory limit: ~1.5 GB.
