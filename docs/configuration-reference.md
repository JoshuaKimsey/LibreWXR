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
- [Multi-mode Tile-Server Split](#multi-mode-tile-server-split)
- [ECMWF IFS Global Coverage](#ecmwf-ifs-global-coverage)
  - [Global: NOAA RRQPE](#global-noaa-rrqpe)
- [Regional NWP Sources](#regional-nwp-sources)
  - [North American: HRRR / HRRR-Alaska](#north-american-hrrr--hrrr-alaska)
  - [North American: HRDPS](#north-american-hrdps)
  - [European: DMI DINI + ICON-EU](#european-dmi-dini--icon-eu)
  - [Caribbean: AROME Antilles](#caribbean-arome-antilles)
  - [South American: WRF-SMN](#south-american-wrf-smn)
  - [East Asia: JMA MSM](#east-asia-jma-msm)
- [Nowcasting](#nowcasting)
- [Storm-Cell Detection](#storm-cell-detection)
- [Satellite (GMGSI)](#satellite-gmgsi)
- [Weather Alerts (WMO CAP)](#weather-alerts-wmo-cap)
- [Persistent Cache](#persistent-cache)
- [Performance and Reliability](#performance-and-reliability)
- [Tile Request Tracking](#tile-request-tracking)
- [MCP Server](#mcp-server)
- [RAM Sizing Guide](#ram-sizing-guide)
- [Example Configurations](#example-configurations)

---

## Server

### `LIBREWXR_HOST`

The network interface the server binds to. When unset (`None`), the value is passed straight to uvicorn, which selects a dual-stack listen — both IPv4 and IPv6 on capable systems. This replaced the previous IPv4-only `0.0.0.0` default in commit `f1eea96` ("Default host to None (dual-stack) instead of 0.0.0.0 (IPv4-only)").

Most self-hosters want the new behaviour, but if your reverse proxy or network only speaks IPv4, set this explicitly:

| Value | Behaviour |
|---|---|
| `0.0.0.0` | IPv4 wildcard (all interfaces). Backwards-compatible with pre-`f1eea96` behaviour — what most IPv4-only deployments behind nginx / cloudflared will want. |
| `127.0.0.1` | IPv4 loopback only — useful when a sidecar proxy on the same host is the only client that should reach LibreWXR. |
| `::` | IPv6 wildcard (implies dual-stack on most kernels). |
| `<other IP>` | Bind to one specific interface. |

| | |
|---|---|
| **Default** | `None` (uvicorn dual-stack default) |
| **Type** | string \| None |

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

### `LIBREWXR_LOG_LEVEL`

Root log level for the Rich-tagged console output (the `[tag] message` format shared with uvicorn's own loggers).  One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` — case-insensitive, normalized to uppercase.  At the default `INFO`, boot status, fetch-cycle timing, and per-source fetch durations are visible while per-cycle noise (memmap-directory lines, per-source ingest summaries, fetch-cycle-start banners, retry attempts) stays at `DEBUG`.

| | |
|---|---|
| **Default** | `INFO` |
| **Type** | string |

**Example:**
```bash
LIBREWXR_LOG_LEVEL=DEBUG
```

---

### `LIBREWXR_LOG_FILE`

Path to a rotating log file capturing WARNING and above (warnings, errors, and exception tracebacks) in addition to the Rich-tagged console output. Enabled by default at `logs/librewxr.log`; each file is capped at 5 MB with 3 rotated backups (`librewxr.log`, `.1`, `.2`, `.3`). Set it to an empty value to disable the file entirely - console behaviour is unchanged.

| | |
|---|---|
| **Default** | `logs/librewxr.log` (enabled; empty disables) |
| **Type** | string (file path) |

Relative paths resolve against the process working directory - the project root for local runs, `/app` in the container. The stock docker-compose.yml sets `LIBREWXR_LOG_FILE=/logs/librewxr.log` and bind-mounts `./logs:/logs`, so every Docker deployment maps the container log to `./logs/` in the clone directory on the host with zero setup. In multi mode every process (pipeline and all render workers) appends to the same file.

**Example:**
```bash
LIBREWXR_LOG_FILE=logs/librewxr.log
```

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

US-side radar data source — applies to USCOMP, AKCOMP, HICOMP, PRCOMP, and GUCOMP only. **Canada (CACOMP) is controlled independently** by `LIBREWXR_CA_SOURCE`. Three modes:

- **`mrms_fallback`** (default) — NCEP MRMS quality-controlled mosaics as the primary source, with IEM NEXRAD fallback when MRMS fails for a specific frame. Best coverage.
- **`mrms`** — NCEP MRMS only, no fallback. Pure MRMS where available; gaps inside the RRQPE band fall through to the global observed RRQPE layer first, then ECMWF IFS (poleward / fringe / RRQPE-decline). Least bandwidth.
- **`iem`** — Legacy mode. IEM NEXRAD N0Q only. NEXRAD-only without quality control. Simplest and most battle-tested, but fewer radars and no QC.

| | |
|---|---|
| **Default** | `mrms_fallback` |
| **Type** | string |
| **Values** | `mrms_fallback`, `mrms`, `iem` |

**Note:** This setting does not affect the OPERA (Europe) source, which always uses EUMETNET OPERA via MeteoGate S3.

### `LIBREWXR_CA_SOURCE`

Canada-side radar data source — applies to CACOMP only. Fully independent of `LIBREWXR_NA_SOURCE`: any US choice can be combined with any Canada choice. Three modes:

- **`mrms_with_msc_blend`** (default) — NCEP MRMS as the primary source covering southern Canada via its CONUS product, with MSC Canada blended in to fill gaps north of MRMS's bbox (latitudes north of ~55°N) and as a fallback if MRMS fails. Best coverage.
- **`mrms`** — NCEP MRMS only via the CONUS product. Southern Canada is covered; northern Canada (outside the MRMS bbox) falls through to the global observed RRQPE layer first, then ECMWF IFS. No MSC fetched.
- **`msc`** — MSC Canada standalone — Environment and Climate Change Canada's native composite covering all of Canada (RADAR_1KM_RRAI via WMS, MRMS makes no contribution to CACOMP).

| | |
|---|---|
| **Default** | `mrms_with_msc_blend` |
| **Type** | string |
| **Values** | `mrms_with_msc_blend`, `mrms`, `msc` |

**Combinations:** With independent US/Canada knobs you can, for example, run `NA_SOURCE=mrms_fallback` + `CA_SOURCE=msc` to use MRMS for the US but stay on ECCC's native composite for Canada, or `NA_SOURCE=iem` + `CA_SOURCE=mrms` to use legacy IEM for the US while still getting MRMS-quality data for southern Canada.

### `LIBREWXR_MRMS_BASE_URL`

Base URL for NCEP MRMS data products. Each region (CONUS, Alaska, Hawaii, Caribbean, Guam) has its own subdirectory under this path.

| | |
|---|---|
| **Default** | `https://mrms.ncep.noaa.gov/2D` |
| **Type** | string |

Only change this if you're mirroring MRMS data to a custom endpoint.

### `LIBREWXR_IEM_BASE_URL`

Base URL for the Iowa Environmental Mesonet NEXRAD composites (US regions). Only used when `LIBREWXR_NA_SOURCE` is `iem` (primary) or `mrms_fallback` (US-side fallback).

| | |
|---|---|
| **Default** | `https://mesonet.agron.iastate.edu` |
| **Type** | string |

### `LIBREWXR_MSC_CANADA_BASE_URL`

Base URL for the Environment and Climate Change Canada MSC GeoMet WMS service (Canadian radar). Only used when `LIBREWXR_CA_SOURCE` is `msc` (primary) or `mrms_with_msc_blend` (blend partner + fallback).

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

### `LIBREWXR_DPC_BASE_URL`

Base URL for the DPC Italian national radar composite REST API. The source hits `{base}/findLastProductByType?type=VMI` for the latest timestamp, then `POST {base}/downloadProduct` for a 300–900 s pre-signed S3 URL. Anonymous, no API key. **CC-BY-SA 4.0** — attribution "Radar-DPC" required and derivative tiles inherit the share-alike clause. Only used when `ITCOMP` or the `EUROPE` group is in `LIBREWXR_ENABLED_REGIONS`.

| | |
|---|---|
| **Default** | `https://radar-api.protezionecivile.it` |
| **Type** | string |

### `LIBREWXR_DPC_ENABLED`

Master toggle for the DPC Italy source. When `false`, drops `ITCOMP` from the active region set even if the `EUROPE` group alias or `ALL` would otherwise pull it in. OPERA continues to cover the rest of Europe — note that with DPC disabled, the layer over Italian airspace will be edge-of-range data from neighbouring countries' radars rather than native Italian data.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

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
| `EUROPE` | `ITCOMP`, `OPERA` | DPC Italian national composite (24 radars) + OPERA pan-European composite (~155 radars, 24 countries). ITCOMP wins precedence over OPERA where it covers — Italy is not in the EUMETNET OPERA station list. |
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
| `ITCOMP` | Italy | DPC (Radar-DPC v2 REST API) | 1200 x 1400 | 1km (tmerc) | ~7 MB |
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

Baseline Gaussian blur radius applied when smoothing is enabled in the tile URL (`smooth=1`). The renderer auto-scales this up at high zoom on coarse sources (OPERA's 2 km LAEA grid, MRMS, MMD, etc.) by measuring how many tile pixels each region pixel covers — so this value is the floor for fine sources at low zoom, not the cap. Set to 0 to disable smoothing entirely, even when clients request it.

| | |
|---|---|
| **Default** | `1.0` |
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
| **Default** | `100` |
| **Type** | integer |
| **Range** | 1 - 100 |

- `100` = lossless (default; best quality, larger files)
- `1-99` = lossy at that quality (e.g. `65` is roughly 4x smaller than lossless but visibly softens saturated radar colors)

The lossless path uses libwebp's fast `method=1` preset: sizes stay within ~1% of max-effort lossless while encoding is 1.5-4x faster.

PNG tiles are encoded adaptively and losslessly: when a tile's final pixels contain at most 256 unique RGBA colors (typical for unsmoothed radar tiles), the encoder builds an exact 8-bit palette from those colors and writes a palette (P-mode) PNG with a `tRNS` chunk carrying full 8-bit alpha; otherwise it writes a plain 32-bit RGBA PNG. No configuration knob is needed — the encoder selects the smaller representation automatically, and both paths reproduce the input pixels bit-for-bit.

### `LIBREWXR_TILE_CACHE_MB`

Maximum tile cache size in megabytes, **per worker**. The cache stores pre-presentation `TileGeometry` records — uint8 pixel values plus an optional snow mask — keyed on `(timestamp, z, x, y, tile_size, smooth, snow)`. Color scheme, output format, and arrow style are applied per request in the cheap `present_tile` step, so one cached entry serves every variant of a given viewport. Oldest entries are evicted when this byte limit is reached.

Higher values mean faster tile serving for repeat requests; lower values save RAM. The default tracks `LIBREWXR_MODE`: 200 MB total in single mode, 128 MB per worker in multi mode (where many workers share the rack). At a 512² tile size each geometry entry is ~256 KB, so 200 MB holds ~800 viewport geometries.

The tile cache holds two kinds of entries: computed `TileGeometry` records (the expensive per-tile compositing result) and cached encoded tile bytes (rendered tiles kept for HTTP ETag reuse so repeat requests skip re-encoding — covering present, overlay, and lat/lon-window renders, with `/health` reporting each kind's count and bytes separately via `geometry_entries`, `present_entries`, `overlay_entries`, `window_entries`, and `satellite_entries`). Both share this single byte budget, and the half that overflows the byte cap is evicted via LRU when the cache is full. There is no separate config knob for the encoded-byte cache.

| | |
|---|---|
| **Default** | `200` (single) / `128` (multi) — set 0 or unset to use the mode default |
| **Type** | integer |
| **Unit** | megabytes |

### `LIBREWXR_COORD_CACHE_SIZE`

Maximum entries per coordinate LRU cache, **per worker**. Controls how many tile-coordinate mappings are kept in memory. There are 6 internal coordinate caches, and each entry is 0.5-2 MB depending on tile size.

These caches are the largest RAM consumer after frame data. Reducing this saves significant RAM at the cost of occasional recomputation (~5-20 ms per cache miss). The default tracks `LIBREWXR_MODE`: 2048 in single mode, 512 per worker in multi mode.

| | |
|---|---|
| **Default** | `2048` (single) / `512` (multi) — set 0 or unset to use the mode default |
| **Type** | integer |

### `LIBREWXR_COORD_STORE_ENABLED`

Master switch for the shared on-disk coordinate store (`data/coord_store.py`). When enabled, the six cached tile-coordinate functions in `tiles/coordinates.py` publish their computed arrays to a shared store under `LIBREWXR_CACHE_DIR` and read them back as read-only memmaps, so multi-worker deployments compute each array once globally instead of once per render worker. When `false`, the per-worker in-process coordinate LRU caches are used exactly as before the store existed.

Best-effort: any store failure (unwritable cache dir, corrupt files, version mismatch) is logged once and falls back to the in-process compute path — the store is never a single point of failure. Requires `LIBREWXR_CACHE_DIR`; the store disables itself when the cache dir is unset.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_COORD_STORE_MB`

Size cap of the shared on-disk coordinate store, in megabytes. The cap is **soft**: the store is pruned once per fetch cycle by whichever process owns store maintenance (the pipeline in multi mode, the main process in single mode — via the ~30 s-debounced cycle hook), so it can briefly overshoot between prunes.

The default tracks `LIBREWXR_MODE`: 1024 in single mode, 8192 in multi mode. In multi mode the budget is **shared by ALL render workers** — every worker reads the same on-disk store, so the 8192 MB default covers the combined warm set rather than 8192 MB per worker. Settable via `.env` like any knob; a restart applies the change. Requires `LIBREWXR_CACHE_DIR`; the store disables itself when the cache dir is unset.

| | |
|---|---|
| **Default** | `1024` (single) / `8192` (multi) — set 0 or unset to use the mode default |
| **Type** | integer |
| **Unit** | megabytes |

### `LIBREWXR_WARMER_THREADS`

Thread pool size for background tile cache warming, **single mode only** — in multi mode no `TileWarmer` is instantiated in render workers, and the 4-thread multi default sizes the request-executor pool used to compute tile geometry, not a warming pool. When a tile is requested, the warmer pre-computes the geometry for that same tile position at all other timestamps in the background, so animation playback is smooth without waiting for each frame to render on demand. Warming covers all color schemes and output formats automatically because the cache stores pre-presentation geometry, not encoded bytes.

| | |
|---|---|
| **Default** | `0` (single: auto = CPU count - 1) / `4` (multi) — set 0 or unset to use the mode default |
| **Type** | integer |

The empty-tile fast path (see `tile_requests.fast_path` in `/health`) and per-worker LRU caches cover the cold-render case in multi mode.

### `LIBREWXR_WARM_COORD_ZOOM`

Pre-warm coordinate caches up to this zoom level at startup, as a **background task**: the server starts accepting requests immediately and the warm proceeds alongside serving, so a slow warm on cold storage (ZFS/HDD) never blocks boot. Coordinate caches store tile-to-region pixel index mappings; warming them eliminates cold-start latency from trigonometric projections. Coordinate wrappers handle unwarmed entries gracefully — they compute on demand and publish to the shared on-disk coord store — so lazy loading is always safe.

| | |
|---|---|
| **Default** | `0` (mode default: `6` in single / no eager warm in multi) |
| **Type** | integer |

Resolution:

- `0` (or unset) — use the per-mode default: **single** warms up to zoom 6 in the background; **multi** render workers do no eager warm at all, building their coordinate caches lazily on first request.
- Negative (e.g. `-1`) — disable the warm entirely in either mode.
- Positive — force that zoom in either mode (e.g. `4` in multi re-enables a background warm; `-1` in single turns the warm off).

Each zoom level adds ~4x the tiles of the previous (zoom 6 = ~5,500 tiles).

> **Note:** this changes the meaning of `0` relative to earlier releases — `0` previously meant "disabled"; it now means "use the mode default". Use a negative value to disable.

### `LIBREWXR_WARM_OVERVIEW_ZOOM`

**Single mode only.** Pre-render overview tiles up to this zoom level after each fetch cycle. Ensures zoomed-out views are served instantly from cache. In multi mode the fetch cycle lives in a separate pipeline process with `warmer=None`, so these settings do nothing; overview tiles in multi mode are served cold on first request and then cached per-worker. The empty-tile fast path (`tile_requests.fast_path` in `/health`) makes cold renders cheap for precip-empty tiles.

| | |
|---|---|
| **Default** | `4` |
| **Type** | integer |

At zoom 4, ~341 tiles per timestamp. Set to `-1` to disable.

### `LIBREWXR_WARM_OVERVIEW_ZOOM_REGIONAL`

**Single mode only.** Pre-render higher-zoom tiles ONLY where they overlap an enabled region's bounding box. Skips ocean / desert / unpopulated tiles that no one would zoom into.

Applies between `LIBREWXR_WARM_OVERVIEW_ZOOM` (exclusive) and this value (inclusive). In multi mode these settings do nothing, as described under `LIBREWXR_WARM_OVERVIEW_ZOOM` above.

| | |
|---|---|
| **Default** | `6` |
| **Type** | integer |

Set to `-1` (or any value `<= warm_overview_zoom`) to disable. At zoom 6 with all regions enabled, the filter typically drops 80-85% of tiles.

---

## Deployment Mode, Workers, and Memory

### `COMPOSE_PROFILES` / `LIBREWXR_MODE`

Picks the deployment shape. Both names resolve to the same `mode` setting; `LIBREWXR_MODE` takes precedence when both are set. Docker Compose reads `COMPOSE_PROFILES` natively to pick which services start, and the app reads it as a fallback so docker users only set one env var.

| | |
|---|---|
| **Default** | `single` |
| **Type** | `single` or `multi` |

- **`single`**: fetcher + renderer in one process. Personal / small-scale self-hosting.
- **`multi`**: pipeline sidecar + N renderer workers sharing memmap state. Production deployment that bypasses the Python GIL on the render path.

`LIBREWXR_WORKERS`, `LIBREWXR_TILE_CACHE_MB`, `LIBREWXR_COORD_CACHE_SIZE`, and `LIBREWXR_WARMER_THREADS` all pick mode-appropriate defaults from this setting when left at `0` (or unset).

### `LIBREWXR_WORKERS`

Number of uvicorn worker processes. The default tracks `LIBREWXR_MODE`.

| | |
|---|---|
| **Default** | `1` (single) / `16` (multi) — set 0 or unset to use the mode default |
| **Type** | integer |

- **single**: each worker is a fully independent copy of LibreWXR with its own frame store, caches, and fetcher. More workers = more concurrency at ~1.3 GB+ RAM each. Recommended: 1 worker per 2 CPU cores; put a caching proxy in front for high traffic.
- **multi**: renderer workers share radar/NWP/satellite state via memmap snapshots written by a sidecar `pipeline` process. Scale workers across many cores without the per-worker data RAM cost — total RSS ≈ workers × (interpreter ~80 MB + tile cache + coord cache) + a single shared page-cache backing the memmap. Recommended: 8-32 workers depending on rack size.

### `LIBREWXR_MEMORY_LIMIT_MB`

Memory limit in MB for the memory pressure monitor. The monitor checks the container's cgroup usage against this limit using the kernel-irreclaimable share as the decision metric: `anon + shmem` on cgroup v2 (from `memory.stat`), `rss + shmem` on cgroup v1, falling back to the raw usage file (`memory.current` / `memory.usage_in_bytes`) when the stat file can't be parsed, and finally to the worker's own RSS outside containers. Clean file-backed page cache is excluded from the decision metric because in multi mode all render workers memmap the same snapshot files — those pages are shared, clean, and kernel-reclaimable, so they are not actionable pressure (tmpfs-backed cache dirs stay counted via `shmem`). Thresholds: at ~80% it logs a warning; at ~85% each worker evicts half its tile cache and runs `malloc_trim(0)` to return freed pages to the OS; at ~90% the tile and coordinate caches are cleared entirely. Each worker applies a small fixed random offset to its thresholds (warn ±1 percentage point, evict ±2) so workers in a shared cgroup don't trip in lock-step, and an eviction level only acts after two consecutive checks above it (the warn-level log fires on the first crossing). In multi mode every worker reads the same cgroup figure, so the thresholds fire across all workers in the same check window — the cache evictions add up to a container-wide drop.

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

### `LIBREWXR_SHARED_TILE_STORE_MB`

Budget, in megabytes, of the shared on-disk store of **encoded** tile bytes under `LIBREWXR_CACHE_DIR` (`tiles_shared/`). Multi-mode render workers publish their freshly-encoded plain past-frame tiles here and read back bytes published by any other worker — one encode serves the whole fleet — instead of each worker redundantly colorizing and encoding the same viewport. The store is disabled in single mode (one process — the in-memory cache is enough).

Semantics: unset (`None`) = auto, which resolves to **2048 MB for render-only workers** and **disabled in single mode**; `0` or any negative value disables the store entirely; a positive value sets the MB budget explicitly. Content-versioned keys (the frame's content version is folded into each key) make stale entries unreachable between fetch cycles, and the render workers' state poller invalidates + prunes the store with the same cadence as the in-memory tile cache. Requires `LIBREWXR_CACHE_DIR` (a shared volume) — render-only mode already requires it.

| | |
|---|---|
| **Default** | unset (auto: `2048` in multi / disabled in single) |
| **Type** | integer (or unset) |
| **Unit** | megabytes |

### Docker memory limits

The compose file caps each container using these env vars (not LIBREWXR_* settings — they're consumed by `deploy.resources.limits` in the YAML directly). Which one applies depends on which profile is active.

| Var | Default | Profile |
|---|---|---|
| `LIBREWXR_MEMORY` | `7G` | `single` (the librewxr container) |
| `LIBREWXR_PIPELINE_MEMORY` | `12G` | `multi` (the pipeline container) |
| `LIBREWXR_RENDER_MEMORY` | `18G` | `multi` (the renderer container) |

Production observation on an 80-core / 32 GB rack in multi mode (32 render workers): ~16 GB total RSS settled across both containers under continuous traffic.

---

## Multi-mode Tile-Server Split

Runs the data pipeline as one process and N tile-server worker processes alongside it, all sharing `LIBREWXR_CACHE_DIR` via memmap files + a single `state.json` snapshot. Bypasses Python's GIL on the tile-render path so the rack's full core count can actually do work.

To enable:
1. Set `LIBREWXR_CACHE_DIR` to a shared directory (required).
2. Set `COMPOSE_PROFILES=multi` in `.env` and run `docker compose up -d`, or run the two processes manually:
   ```bash
   export LIBREWXR_MODE=multi
   python -m librewxr.data_pipeline                       # sidecar
   LIBREWXR_RENDER_ONLY=1 python -m librewxr.main         # tile server
   ```

### `LIBREWXR_RENDER_ONLY`

When `true` (or `1`), the worker skips fetcher / NWP grid / satellite / nowcast initialization entirely. It only memory-maps the snapshot the pipeline writes and renders tiles from it.

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

### `LIBREWXR_WORKER_HEALTHCHECK_TIMEOUT`

Seconds uvicorn's master process waits for a worker healthcheck ping before killing and respawning the worker (applies whenever `LIBREWXR_WORKERS` > 1, i.e. multi mode). Render workers can stall well past the default when they page-fault freshly written memmap frame files off a slow backing disk while holding the GIL; raising this to 30 s lets a stalled worker recover instead of being SIGKILLed. `0` = uvicorn's built-in default (5 s).

| | |
|---|---|
| **Default** | `30` |
| **Type** | integer |
| **Unit** | seconds |

### `LIBREWXR_PAGECACHE_PRIME_ENABLED`

When `true` (default), the data pipeline primes freshly written memmap frame files (radar, NWP, satellite, nowcast, precip-mask) into the host page cache after each fetch cycle via `posix_fadvise(WILLNEED)`. The host page cache is shared between the pipeline and renderer containers, so render workers serve those frames without cold page faults on slow backing disks. Consumed only by the multi-mode pipeline process; single mode never runs it.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

---

## ECMWF IFS Global Coverage

LibreWXR uses ECMWF IFS 9 km global data from [Open-Meteo](https://open-meteo.com/) S3 as the terminal model of its NWP chain. IFS provides:

- Model precipitation everywhere the regional NWP chain doesn't reach — for past frames that means poleward of the RRQPE band, the 2-degree fringe excluded by RRQPE's coverage polygon (68-70N, -60 to -58S), and wherever RRQPE declines (missed scans / stale store); within the band, past/current frames come from the always-on observed radar region NOAA RRQPE (below)
- Per-pixel snow/rain classification
- The model side of the nowcast blend tail (RRQPE joins nowcast extrapolation like any radar region)

### `LIBREWXR_ECMWF_ENABLED`

Disable ECMWF IFS entirely. Useful only for isolating regional NWP layers during debugging — the always-on observed radar region RRQPE (see below) still renders the 60S-70N band, so only pixels poleward of the band or in the fringe excluded by RRQPE's coverage polygon will then simply show zero precipitation.

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

### Global: NOAA RRQPE

NOAA's Enterprise Rain Rate (RRQPE) GLB-5 blend is ingested as a single coarse global **radar** region (lat 60°S-70°N, all longitudes): satellite-derived **observed** precipitation on a global 0.02° grid, block-averaged to 0.04° at the default downsample, consumed from the anonymous NOAA Open Data bucket `noaa-enterprise-rainrate-pds`. It sorts **last** in the multi-region compositor — the bottom tier that fills only pixels no finer radar region claims, so it never overwrites a Doppler composite's authoritative "no echo" zeros. Because it is observations rather than model output it only ever answers for past / observed frame times; it joins radar nowcast extrapolation and blend-weight fade like any other region, the fetcher's carry-forward covers late scans, and the region is **always-on** — it keeps fetching and rendering even when `LIBREWXR_ENABLED_REGIONS` is a narrow group.

It is an IR-based satellite **estimate**, not a measurement: it underestimates warm / stratiform rain, is unreliable over snow and ice surfaces, and only covers the 60°S-70°N geostationary ring. Scans publish on a 10-min cadence with ~17-min median latency.

Data is distributed under the NOAA Open Data Dissemination (NODD) program. Attribution is requested: "Precipitation data from NOAA Enterprise Rain Rate (RRQPE)". No endorsement by NOAA is implied, and don't present modified data as unaltered NOAA data. Blend inputs include JMA Himawari-9 and EUMETSAT Meteosat-9/10; courtesy attribution to the contributing agencies is appreciated but not required.

#### `LIBREWXR_RRQPE_ENABLED`

Master switch for the RRQPE layer.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

#### `LIBREWXR_RRQPE_BASE_URL`

S3 bucket for the NOAA Enterprise Rain Rate GLB-5 files.

| | |
|---|---|
| **Default** | `https://noaa-enterprise-rainrate-pds.s3.amazonaws.com` |
| **Type** | string |

#### `LIBREWXR_RRQPE_PUBLISH_DELAY_MINUTES`

How long after a 10-min scan start the file is considered safely published. The fetch window ends at `now - publish_delay`, so not-yet-published slots are never requested; a missed scan simply has no key in its hour directory and is skipped.

| | |
|---|---|
| **Default** | `15` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_RRQPE_DBZ_OFFSET`

dBZ calibration shift applied after Z-R conversion of RRQPE rain rates (Marshall-Palmer 200·R^1.6). Satellite QPE is a surface rain rate; radar reflectivity samples the storm column and reads higher, so nudge the derived dBZ up to match.

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

#### `LIBREWXR_RRQPE_DOWNSAMPLE`

Integer block-averaging factor for the 0.02° native grid (1/2/4 → 0.02°/0.04°/0.08°). 2 is the default: each decoded ~117 MB float32 frame becomes a ~29 MB uint8 store.

| | |
|---|---|
| **Default** | `2` |
| **Type** | integer |
| **Values** | `1` (0.02° native) · `2` (0.04°) · `4` (0.08°) |

#### `LIBREWXR_RRQPE_MATCH_TOLERANCE_SECONDS`

Match slack around the ideal constant-shift target slot. Every frame is served the scan exactly `RRQPE_LAG_SECONDS` (30 min) its senior — a **constant shift** that keeps the frame → scan mapping deterministic 1:1, so consecutive frames step one scan per frame (no freezing, no skipping). The target 30-min-old scan is essentially always published given the product's ~13-25 min publish latency, so the target slot exists every cycle; the fib is a constant ~30 min — honest staleness over fabricated motion (previously the shift wobbled between 2-3 slots as latency varied, freezing or skipping frames). This value bounds how far the nearest stored scan may sit from that ideal target: at the default it tolerates up to ~2 consecutive missed scan slots before the region declines for the affected frames (carry-forward / NWP fill take over until the next fetch cycle heals). It is not a publish-lag cap — the shift is constant by design.

| | |
|---|---|
| **Default** | `1800` |
| **Type** | integer |
| **Unit** | seconds |

---

## Regional NWP Sources

LibreWXR layers a chain of regional rapid-refresh NWP models on top of the global ECMWF IFS layer. At each pixel, the chain dispatches to the **narrowest** model whose domain covers it, soft-feathering at every domain edge so seams don't show. See [`coverage.md`](coverage.md) for visual maps of every radar + NWP domain.

Most regional sources also classify each pixel as snow vs rain from their own 2-metre temperature field. The threshold is shared across all of them via [`LIBREWXR_REGIONAL_SNOW_TEMP_THRESHOLD`](#librewxr_regional_snow_temp_threshold). HRRR-CONUS, HRRR-Alaska, WRF-SMN, DMI DINI, ICON-EU, and JMA MSM all classify natively; HRDPS and AROME Antilles fall through to IFS for snow detection (HRDPS is expected to be replaced by RRFSv1 mid-2026; AROME Antilles is tropical so the question rarely matters).

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
| **Default** | `12.0` |
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
| **Default** | `12.0` |
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
| **Default** | `https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net` |
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

### Météo-France AROME Outre-Mer (other variants)

The remaining four AROME-OM variants share the same upstream, file
format, cadence, and decoder as Antilles (via the
`AROMEOverseasGrid` family base in `sources/_shared/arome.py`).
Each is independently toggleable. All four are tropical /
sub-tropical and skip snow-mask classification + optical-flow
interpolation (their natively-hourly cadence is fine for animation
at small domain sizes).

| Variant | Token | Domain (~km E-W × N-S) | Coverage | Chain priority |
|---|---|---|---|---|
| AROME Guyane | `GUYANE` | 1156 × 877 | French Guiana + Suriname + Amapá borders | 26 |
| AROME Indien | `INDIEN` | 3742 × 2492 (largest AROME-OM) | Réunion + Mayotte + Comoros + most of Madagascar + Tanzania coast | 27 |
| AROME Nouvelle-Calédonie | `NCALED` | 1357 × 1360 | New Caledonia + Loyalty Islands + Vanuatu side | 28 |
| AROME Polynésie | `POLYN` | 1365 × 1404 | Society + Tuamotu archipelagoes | 29 |

Each variant exposes four settings analogous to Antilles:
`LIBREWXR_AROME_{VARIANT}_ENABLED`,
`LIBREWXR_AROME_{VARIANT}_BASE_URL`,
`LIBREWXR_AROME_{VARIANT}_PUBLISH_DELAY_MINUTES`, and
`LIBREWXR_AROME_{VARIANT}_DBZ_OFFSET`. Defaults match Antilles
(`true`, `https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net`, `420`,
`6.0`).

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

### East Asia: JMA MSM

Japan Meteorological Agency Mesoscale Model at native 5 km (0.0625° lon × 0.05° lat) over 22.4–47.6°N × 120–150°E — Japan + Korean Peninsula + Taiwan + Yellow Sea + adjacent waters of the western Pacific. 8 cycles/day (00/03/06/09/12/15/18/21 UTC), hourly forecast steps, 78 h horizon from the main 00Z/12Z runs (39 h from the others). Distributed via Open-Meteo's anonymous AWS Open Data mirror (`openmeteo` bucket in us-west-2 — same bucket as IFS). Pairs with the JMA HRPN radar composite (JPCOMP) to give Japan a proper mesoscale NWP overlay instead of falling through to global IFS. JMA's direct JMBSC feed is paid-contract-only; the Open-Meteo mirror is the workaround.

#### `LIBREWXR_JMA_MSM_ENABLED`

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

#### `LIBREWXR_JMA_MSM_S3_BUCKET`

| | |
|---|---|
| **Default** | `openmeteo` |
| **Type** | string |

#### `LIBREWXR_JMA_MSM_S3_REGION`

| | |
|---|---|
| **Default** | `us-west-2` |
| **Type** | string |

#### `LIBREWXR_JMA_MSM_S3_PREFIX`

| | |
|---|---|
| **Default** | `data_spatial/jma_msm` |
| **Type** | string |

#### `LIBREWXR_JMA_MSM_PUBLISH_DELAY_MINUTES`

Full run write-out completes ~5 h after init (the published `latest.json`'s `last_modified_time` typically lags `reference_time` by 5h).

| | |
|---|---|
| **Default** | `300` |
| **Type** | integer |
| **Unit** | minutes |

#### `LIBREWXR_JMA_MSM_DBZ_OFFSET`

| | |
|---|---|
| **Default** | `6.0` |
| **Type** | float |
| **Unit** | dBZ |

### `LIBREWXR_REGIONAL_SNOW_TEMP_THRESHOLD`

Temperature threshold for native snow/rain classification across every regional NWP source that derives a snow mask from its own 2-metre temperature field (HRRR-CONUS, HRRR-Alaska, WRF-SMN, DMI DINI, ICON-EU, JMA MSM). Pixels colder than this threshold are tagged as snow and rendered with the snow palette when `snow=1` is set on the tile URL.

| | |
|---|---|
| **Default** | `1.5` |
| **Type** | float |
| **Unit** | degrees Celsius |

1.5 °C is a typical near-surface snow-vs-rain transition line. Drop towards 0 °C for a stricter "only true freezing" definition; raise to 2-3 °C to catch wet snow / sleet conditions that visually behave like snow on the ground.

### `LIBREWXR_REGIONAL_INTERPOLATION`

Enable optical-flow temporal interpolation of hourly regional NWP frames to 10-minute steps. Uses the same OpenCV Farneback dense flow we apply to ECMWF IFS, applied at the end of each fetch cycle to every regional source whose native cadence is hourly (currently WRF-SMN, DMI DINI, ICON-EU, JMA MSM).

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

4 fits comfortably in 8 GB; bump to 6-8 on bigger rigs (multi mode has a separate pipeline container with its own memory budget, so it can usually go higher) to bring cycle wall time closer to the slowest single source.

### `LIBREWXR_RADAR_FETCH_CONCURRENCY`

Maximum number of radar region-frame fetches (live or archive) running in parallel inside one fetch cycle. Each in-flight fetch can hold 100-200 MB during decode (MRMS), so this caps peak transient RAM at ~N x per-frame working set.

| | |
|---|---|
| **Default** | `8` |
| **Type** | integer |

8 caps transient decode RAM around 1.6 GB; raise on fatter rigs to shorten backfill wall time.

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

### `LIBREWXR_NOWCAST_COARSEN_ENABLED`

Progressive spatial coarsening of the optical-flow-extrapolated radar fields in the nowcast pipeline.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

When enabled, each extrapolated forecast frame is Gaussian-smoothed with a sigma that ramps quadratically with lead time — negligible at T+10, the full `LIBREWXR_NOWCAST_COARSEN_MAX_KM` effective-resolution floor at the last blend step. Farneback optical flow produces melted/filamented high-spatial-frequency warping artifacts at long lead times; the lead-time-ramped low-pass attenuates exactly those artifacts and honestly encodes the growing positional uncertainty of the extrapolation. Early frames stay crisp; only the internal optical-flow path is smoothed — external nowcast contribution frames (e.g. JMA HRPN for JPCOMP) pass through untouched.

### `LIBREWXR_NOWCAST_COARSEN_MAX_KM`

Effective resolution floor reached at the last blend step, in kilometres.

| | |
|---|---|
| **Default** | `3.0` |
| **Type** | float |

The Gaussian sigma at forecast step `t` (normalized to the blend window) is `max_km * t²` in kilometres — so at the default 3.0 km and 10-minute cadence the T+60 field is smoothed to roughly the resolution of a 3 km NWP grid while the T+10 field is left effectively untouched. Setting this to `0` (or disabling `LIBREWXR_NOWCAST_COARSEN_ENABLED`) disables the smoothing entirely.

### `LIBREWXR_ARROW_FLOW_ENABLED`

The `/v2/radar/...` tile endpoint accepts an `?arrows=` query param that overlays semi-transparent precipitation-direction arrows on areas with active precipitation. Arrows key off per-region optical flow computed between the two most recent radar frames; outside radar coverage, a single **composite NWP flow raster** (built from `NWPChain.sample()` at T and T−1) drives the arrows — reflecting whichever regional NWP source is active at each point (HRRR over CONUS, ICON-EU over Europe, JMA MSM over Japan, IFS elsewhere) rather than IFS alone.

Before this toggle existed, optical flow was computed only as a byproduct of nowcast generation — so disabling `LIBREWXR_NOWCAST_ENABLED` silently broke arrow direction (every tile fell through to the coarse ECMWF IFS wind field, producing the "incorrect, randomly placed" arrows reported in [issue #7](https://github.com/JoshuaKimsey/LibreWXR/issues/7)).

With this toggle on (the default), arrows get real storm motion regardless of the nowcast state — both from radar (per-region, fine-grained) and from the composite NWP raster (one global field, coarse but covers every NWP source at once):

| `nowcast_enabled` | `arrow_flow_enabled` | Flow CPU | Arrows | Nowcast frames |
|---|---|---|---|---|
| `true`  | (any)   | full | correct (radar + composite NWP flow) | generated |
| `false` | `true`  | ~25-35% of full nowcast | **correct** (radar + composite NWP flow at reduced resolution) | none |
| `false` | `false` | zero | suppressed entirely | none |

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

Note: Farneback optical flow is the expensive part of the nowcast pipeline (~90% of the per-cycle ~44s in the default 13-region config); extrapolation/IFS-blend is comparatively cheap. Disabling nowcast but keeping arrow flow on therefore saves only **~10-20% CPU** vs full nowcast — not ~90%. **The real escape hatch for CPU-conscious operators is disabling both `nowcast_enabled` and `arrow_flow_enabled`** — that fully suppresses the overlay (the `?arrows=` query becomes a no-op) and pays zero flow CPU.

Inside radar coverage, the per-region radar flow wins (arrows reflect observed storm motion, even when stationary). Outside radar coverage, the composite NWP flow fills in — arrow presence is gated on the NWP chain's own precip at the point (above the noise floor), so arrows now appear wherever the chain shows precip, not just where IFS does. This fixes the long-standing "HRRR precip present but IFS dry → no arrows" gap.

### `LIBREWXR_ARROW_FLOW_TARGET_DIM`

Longest dimension (in pixels) passed to the Farneback optical-flow algorithm when computing flow for the arrow-only path (`nowcast_enabled=false, arrow_flow_enabled=true`). Larger grids are downscaled to this before the flow computation, then the flow vectors are upscaled back to the original resolution.

| | |
|---|---|
| **Default** | `500` |
| **Type** | integer |

Arrows draw on a 32-pixel (256-tile) or 48-pixel (512-tile) grid and downsample flow ~10-30x while drawing, so a high-resolution flow field is wasted work for arrow purposes — 500 reaches the arrow-visible quality ceiling at roughly 4× the speed of the 1000-pixel field used for nowcast extrapolation (which feeds `cv2.remap` and needs the pixel sharpness).

**No effect when nowcast is on** — that path uses the module constant `_TARGET_FLOW_DIM = 1000` in `nowcast.py`, so the two paths no longer share tuning. If you turn nowcast back on, you don't need to revisit this setting.

### `LIBREWXR_ARROW_NWP_FLOW_RESOLUTION_DEG`

Resolution of the global composite NWP flow raster used by the arrow overlay outside radar coverage, in degrees. One Farneback pass per fetch cycle over two `NWPChain.sample()` snapshots at T and T−1.

| | |
|---|---|
| **Default** | `0.25` |
| **Type** | float |

At 0.25° the raster is 721×1440 float32 (~8 MB). The 32/48px arrow draw grid can't resolve finer detail at most zooms, so coarser is cheaper for no visible loss. Finer values help only at high zoom inside small convective cells — and inside radar coverage those cells already get the fine per-region radar flow (which wins by construction), so the composite only fills NWP-only regions where sub-0.25° detail doesn't matter. This is an advanced tuning knob not surfaced in `.env.example`.

---

## Storm-Cell Detection

Storm-cell detection uses OpenCV `connectedComponentsWithStats` on the latest radar frame to identify convective cells, compute their centroids and motion vectors (from the nowcast optical flow), and store them in `StormCellStore`. The renderer overlays cell outlines and arrows on radar tiles via the `?cells=light|dark` query parameter.

Detection runs once per fetch cycle, after nowcast generation, so it can reuse the just-computed optical flow. The data is included in the `state.json` snapshot so multi-mode render workers can serve it without running detection themselves.

### `LIBREWXR_STORM_CELLS_ENABLED`

Master switch for storm-cell detection. When `false`, no detection runs and the `?cells=` query parameter is a no-op on the tile endpoint.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_STORM_CELLS_MIN_DBZ`

Minimum dBZ threshold for a pixel to be considered part of a storm cell. Pixels below this value are ignored by the connected-components algorithm.

| | |
|---|---|
| **Default** | `40` |
| **Type** | integer |
| **Unit** | dBZ |

40 dBZ corresponds to moderate convection — the typical boundary between light stratiform rain and organized convective cores. Lower values (e.g., 35) will detect more diffuse cells but may increase false positives from bright-band contamination and ground clutter.

### `LIBREWXR_STORM_CELLS_MIN_AREA_KM2`

Minimum area for a connected component to be reported as a storm cell, in square kilometres. Filters out noise, speckle, and very small convective cores.

| | |
|---|---|
| **Default** | `25.0` |
| **Type** | float |
| **Unit** | km² |

At 25 km², a cell needs to be roughly 5×5 km to register — well below the size of a single thunderstorm cell (~10-50 km² at the lower end), while still filtering out speckle and isolated clutter pixels. Increase to e.g. 100 km² to report only the largest organized mesoscale features. Decrease to e.g. 5 km² for very fine-grained detection, at the cost of more noise.

---

## Satellite (GMGSI)

Real satellite imagery backed by NOAA's GMGSI hourly global mosaic — GOES-East + GOES-West + Meteosat-9 + Meteosat-10 + Himawari-9, composited and re-projected by NESDIS into a single equirectangular file per hour per channel. LibreWXR ingests two channels (longwave IR + visible) and renders the `/v2/satellite/...` tile endpoint as a VIS-over-LW composite with a natural day/night terminator crossfade. Coverage extends to ±72.7° latitude.

When the satellite layer is disabled, the endpoint returns 503 and the catalog's `satellite.infrared` array is empty (mirrors the `LIBREWXR_RADAR_ENABLED=false` behaviour).

### `LIBREWXR_SATELLITE_ENABLED`

Master switch for the satellite layer.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_GMGSI_LW_ENABLED`

Per-channel toggle for GMGSI longwave IR (~12 µm). LW is the 24/7 base of the composite and works on the night side too. Disabling it alongside VIS effectively disables the layer.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_GMGSI_VIS_ENABLED`

Per-channel toggle for GMGSI visible (~0.6 µm). VIS adds the daytime reflected-sunlight overlay; on the night side it contributes nothing. Disabling VIS while LW stays on degrades the composite to LW-only without breaking the endpoint.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_SATELLITE_MAX_FRAMES`

Number of hourly satellite frames retained per channel. GMGSI publishes one frame per hour, so 12 ≈ 12 hours of animation. Each frame is ~15 MB, so 12 frames × 2 channels ≈ 360 MB resident.

| | |
|---|---|
| **Default** | `12` |
| **Type** | integer |

---

## Weather Alerts (WMO CAP)

Fetches global weather alerts from severeweather.wmo.int. MeteoAlarm geocodes are downloaded on first startup and cached locally. Updates are clock-aligned (:00, :05, :10, …). US alerts come directly from the NWS API; zone-based alerts (e.g. Tornado Watches) are resolved to zone polygons at ingest, with zone geometries disk-cached for 30 days — no per-request NWS queries are needed.

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

Cache directory for processed grids (GMGSI satellite, NWP, alerts geocodes, master state snapshot). When set, data is saved as memory-mapped files that survive restarts, crashes, and container recreation — no need to re-download from upstream on startup.

| | |
|---|---|
| **Default** | *(empty — in-memory only)* |
| **Type** | string |

**Required** in multi mode. Both the pipeline and renderer containers must share this directory via a named volume.

- Docker: set automatically via a named volume in `docker-compose.yml` (both modes — only the service layout differs between profiles).
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

Track only zoom levels at or above this value. Lower zooms are already cheap to render so they don't need observation.

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

## MCP Server

LibreWXR ships a built-in [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that exposes data tools to LLM agents. Two transports are available: an **HTTP transport** mounted inside the main FastAPI app (reachable by any HTTP-based MCP client — n8n, Claude Code in proxy mode, custom scripts), and a **stdio transport** that runs as a standalone process for local desktop agents like Claude Desktop.

Both transports require the optional `[mcp]` extra:

```bash
pip install -e ".[mcp]"
```

Without the extra, the HTTP transport is silently disabled at startup with a logged warning, and the stdio entry point (`python -m librewxr.mcp` / `librewxr-mcp`) won't import.

### `LIBREWXR_MCP_ENABLED`

Master switch for the MCP HTTP transport. When `false`, no MCP route is mounted and the MCP lifespan is not combined — the app boots lean without any MCP overhead.

The standalone stdio entry point (`python -m librewxr.mcp` / `librewxr-mcp`) is **unaffected** by this flag — it runs independently, reads `state.json` from `LIBREWXR_CACHE_DIR`, and polls its mtime to stay in sync with the running server.

| | |
|---|---|
| **Default** | `true` |
| **Type** | boolean |

### `LIBREWXR_MCP_PATH`

URL path where the MCP HTTP transport is mounted inside the FastAPI app. Change this if you need a different endpoint path (e.g. `/api/mcp`) — for example, if your reverse proxy already uses `/mcp` for something else, or you want the endpoint under an API prefix.

| | |
|---|---|
| **Default** | `/mcp` |
| **Type** | string |

---

### Single-container mode

Each worker process holds its own copy of radar frames, NWP grids, coordinate caches, and tile caches. RAM grows under real traffic as caches fill up.

| Configuration | Estimated RAM |
|---|---|
| CONUS + IFS only, 1 worker, 12 frames | ~3-4 GB |
| CONUS + HRRR + IFS, 1 worker, 12 frames | ~4-5 GB |
| ALL regions + IFS only, 1 worker, 12 frames | ~7-8 GB |
| ALL regions + full NWP chain, 1 worker, 12 frames | ~9-10 GB |
| ALL regions + full NWP chain, 2 workers, 12 frames | ~16-18 GB |

> **Note:** The "ALL regions" rows include the always-on RRQPE global observed region — ~350 MB of frame store (12 × ~29 MB past frames) plus ~175 MB of nowcast-extrapolated frames, per [`self-host-sizing.md`](self-host-sizing.md).

### Multi-worker mode

Render workers share radar/NWP/satellite state via memmap, so adding workers doesn't multiply the data RAM — only the per-worker tile cache and Python interpreter overhead (~80 MB).

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

### Full coverage, personal / small server (single mode)

```bash
COMPOSE_PROFILES=single               # one container, fetcher + renderer
LIBREWXR_PUBLIC_URL=https://radar.example.com
LIBREWXR_ENABLED_REGIONS=ALL
LIBREWXR_NA_NWP_SOURCE=hrrr
LIBREWXR_HRDPS_ENABLED=true
LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu
LIBREWXR_AROME_ANTILLES_ENABLED=true
LIBREWXR_WRF_SMN_ENABLED=true
```

Docker memory limit: ~10 GB

### Production / multi mode (full coverage, busy public instance)

In `.env`:
```bash
COMPOSE_PROFILES=multi                # pipeline + N renderer workers
LIBREWXR_PUBLIC_URL=https://radar.example.com
LIBREWXR_ENABLED_REGIONS=ALL
LIBREWXR_NA_NWP_SOURCE=hrrr
LIBREWXR_HRDPS_ENABLED=true
LIBREWXR_EU_NWP_PROFILE=dini_with_icon_eu
LIBREWXR_AROME_ANTILLES_ENABLED=true
LIBREWXR_WRF_SMN_ENABLED=true
# Optional — bigger box than the 16-worker default:
#LIBREWXR_WORKERS=32
```

Then run:
```bash
docker compose up -d
```

The mode automatically picks per-worker tile cache, coord cache, and render thread defaults (128 MB / 512 entries / 4 threads per worker). Bump `LIBREWXR_NWP_FETCH_CONCURRENCY` above the default 4 if your pipeline container has the RAM headroom.

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
