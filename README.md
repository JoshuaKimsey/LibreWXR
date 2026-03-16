# LibreWXR

A self-hostable, drop-in replacement for the [Rain Viewer](https://www.rainviewer.com/) API. LibreWXR serves weather radar tiles using freely available radar composite data from multiple sources, with full compatibility for any client built against the Rain Viewer v2 API.

## Why?

Rain Viewer recently (as of January 1st, 2026) restricted their free API tier: maximum zoom 7, single color scheme, no satellite, no forecast, PNG only. LibreWXR restores the full pre-restriction functionality as a self-hosted service.

Beyond this though, is the goal of creating a far more customizable API backend for self hosters. The ability to specify regions, radar styles, denoising levels, and more to come as well. With the goal being self-hosting, there are far greater possibilities for what can be both ingested and output via the API, and there is no need to offer any limitations on what is provided, aside from the technicality of the implementation of such features.

## Features

- **Rain Viewer v2 API compatible** — drop-in replacement, no client changes needed
- **All 9 color schemes** — Black & White, Original, Universal Blue, TITAN, TWC, Meteored, NEXRAD III, Rainbow, Dark Sky, plus raw grayscale
- **Tile sizes** — 256px and 512px
- **Image formats** — PNG and WebP (with configurable lossy/lossless quality)
- **Smoothing** — zoom-adaptive Gaussian blur with seamless tile boundaries
- **Multi-region coverage** — US (CONUS, Alaska, Hawaii, Puerto Rico, Guam), Nordic countries (Norway, Sweden, Finland, Denmark), and Germany
- **GFS global fallback** — GFS simulated reflectivity fills in worldwide coverage where no radar composite exists
- **Snow detection** — per-pixel snow/rain classification using GFS global surface temperature data
- **Noise filtering** — configurable dBZ noise floor and speckle removal
- **Tile cache warming** — background pre-rendering for smooth animation playback
- **Health endpoint** — `/health` for monitoring uptime, RAM usage, frame count, and cache status
- **Fully configurable** — all tunable parameters exposed via environment variables

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/JoshuaKimsey/LibreWXR.git
cd LibreWXR
cp .env.example .env
# Edit .env to taste
docker compose up -d
```

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

The server starts at `http://localhost:8080` by default. It will fetch radar data on startup (takes a few seconds), then begin serving tiles.

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
    "nowcast": []
  },
  "satellite": {"infrared": []}
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

**Color schemes:**

| ID | Name |
|---|---|
| 0 | Black and White |
| 1 | Original |
| 2 | Universal Blue |
| 3 | TITAN |
| 4 | The Weather Channel |
| 5 | Meteored |
| 6 | NEXRAD Level III |
| 7 | Rainbow |
| 8 | Dark Sky |
| 255 | Raw (grayscale) |

#### Coverage Tiles

```
GET /v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png
```

Returns tiles showing where radar data exists (white semi-transparent overlay).

#### Health

```
GET /health
```

Returns server status, frame count, cache usage, and temperature grid status.

## Configuration

All settings are configured via environment variables (or a `.env` file). Copy `.env.example` to `.env` and adjust as needed. Every setting has a sensible default.

| Variable | Default | Description |
|---|---|---|
| `LIBREWXR_PUBLIC_URL` | `http://localhost:8080` | Public URL for metadata responses |
| `LIBREWXR_PORT` | `8080` | Server listen port |
| `LIBREWXR_MAX_ZOOM` | `12` | Maximum tile zoom level |
| `LIBREWXR_FETCH_INTERVAL` | `300` | Seconds between radar data fetches |
| `LIBREWXR_MAX_FRAMES` | `12` | Radar frames in memory (~97 MB each with ALL regions) |
| `LIBREWXR_COORD_CACHE_SIZE` | `2048` | Coordinate cache entries per cache (lower = less RAM) |
| `LIBREWXR_TILE_CACHE_SIZE` | `50000` | Max cached rendered tiles |
| `LIBREWXR_SMOOTH_RADIUS` | `3.0` | Gaussian blur radius (0 = disabled) |
| `LIBREWXR_NOISE_FLOOR_DBZ` | `5.0` | Min dBZ to display (-32 = disabled) |
| `LIBREWXR_DESPECKLE_MIN_NEIGHBORS` | `3` | Speckle filter strength (0 = disabled) |
| `LIBREWXR_WEBP_QUALITY` | `100` | WebP quality (100 = lossless, <100 = lossy) |
| `LIBREWXR_WORKERS` | `1` | Uvicorn worker processes |
| `LIBREWXR_ENABLED_REGIONS` | `CONUS` | Radar region spec (see below) |
| `LIBREWXR_WARMER_THREADS` | `4` | Background tile warming threads |

**Radar regions:**

| Code | Region | Source | Resolution | RAM per frame |
|---|---|---|---|---|
| `USCOMP` | Continental US | IEM | 0.005° (~500m) | ~63 MB |
| `AKCOMP` | Alaska | IEM | 0.01° (~1km) | ~6 MB |
| `HICOMP` | Hawaii | IEM | 0.005° (~500m) | ~3.4 MB |
| `PRCOMP` | Puerto Rico | IEM | 0.01° (~1km) | ~1 MB |
| `GUCOMP` | Guam | IEM | 0.0085° (~850m) | ~1 MB |
| `NORDIC` | Norway, Sweden, Finland, Denmark | MET Norway | ~1km | ~3.2 MB |
| `GERMANY` | Germany | DWD | 250m | ~20 MB |

Group aliases: `CONUS` (continental US only), `US` (all US regions), `NORDIC` (Nordic countries), `GERMANY` (Germany), `ALL` (everything).
You can also mix groups and individual regions: `CONUS,NORDIC,GERMANY`.

Examples:
```bash
LIBREWXR_ENABLED_REGIONS=CONUS        # just continental US (default)
LIBREWXR_ENABLED_REGIONS=US           # all US regions
LIBREWXR_ENABLED_REGIONS=NORDIC       # Nordic countries only
LIBREWXR_ENABLED_REGIONS=CONUS,NORDIC # continental US + Nordic
LIBREWXR_ENABLED_REGIONS=GERMANY      # Germany only
LIBREWXR_ENABLED_REGIONS=ALL          # everything available
```

**RAM requirements:**

Each worker process holds its own copy of all radar frames, coordinate caches, and tile caches. RAM usage grows significantly under real traffic as caches fill up.

| Configuration | Estimated RAM Needed |
|---|---|
| CONUS, 1 worker, 12 frames | ~3 GB |
| CONUS, 1 worker, 20 frames | ~4 GB |
| ALL regions, 1 worker, 12 frames | ~7 GB |
| ALL regions, 1 worker, 20 frames | ~8 GB |
| ALL regions, 2 workers, 12 frames | ~12 GB |
| ALL regions, 2 workers, 20 frames | ~14 GB |

See `.env.example` for detailed descriptions and tuning guidance for each setting.

### Scaling

| Users | Workers | RAM (ALL regions) | RAM (CONUS only) |
|---|---|---|---|
| 1-5 (personal) | 1 | ~7 GB | ~3 GB |
| 5-50 (small community) | 2 | ~12 GB | ~5 GB |
| 50-200 (medium) | 4 with CDN | ~22 GB | ~9 GB |
| 200+ (large) | 8+ with CDN | 40+ GB | 16+ GB |

Tiles are served with `Cache-Control: public, max-age=300`, so any caching reverse proxy (nginx, Cloudflare, etc.) will work out of the box for high-traffic deployments. A CDN like Cloudflare (free tier works) absorbs most tile requests at the edge, meaning a single worker can serve far more users than the table above suggests. Using a Cloudflare Tunnel also provides free HTTPS with no certificate management. For most self-hosting scenarios, 1 worker behind Cloudflare is sufficient.

## Architecture

```
[IEM NEXRAD Fetcher]     --> [In-Memory Frame Store] --> [FastAPI + Tile Renderer]
[MET Norway WCS Fetcher] -->   (N frames, multi-region)    (LRU cache + tile warmer)
[DWD HX Fetcher]         -->     (every 5 min)

[UCAR THREDDS] --> [Temperature Grid]     --> [Per-pixel snow/rain classification]
  (every 5 min)     (GFS 2m temp)
               --> [Reflectivity Grid]    --> [Global fallback where no radar exists]
                    (GFS sim. reflectivity)
```

- **US data source:** IEM NEXRAD N0Q composites — 8-bit reflectivity, multiple regions (CONUS, Alaska, Hawaii, Puerto Rico, Guam)
- **Nordic data source:** MET Norway THREDDS WCS — float32 dBZ reflectivity composite (Norway, Sweden, Finland, Denmark), converted to uint8 on ingest
- **Germany data source:** DWD Open Data — HX reflectivity composite in OPERA HDF5 format (4800×4400 at 250m), converted to uint8 on ingest
- **Temperature source:** UCAR THREDDS GFS Global 0.25° 2m analysis — used for snow/rain precipitation classification (worldwide coverage)
- **GFS reflectivity fallback:** UCAR THREDDS GFS simulated reflectivity at 1000m — provides low-resolution (~25km) global coverage where no real radar composite exists
- **Tile rendering:** On-demand with LRU caching. Web Mercator reprojection via pure numpy (no GDAL required)
- **Tile warming:** Background thread pool pre-renders tiles for all timestamps when a new tile position is requested, ensuring smooth animation playback
- **No external dependencies beyond pip** — no GDAL, rasterio, or system geo libraries needed

## Examples

The `examples/` directory contains ready-to-use HTML files for local development:

- **`leaflet.html`** — Leaflet-based radar map (connects to `localhost:8080`)
- **`maplibre.html`** — MapLibre GL-based radar map (connects to `localhost:8080`)

Open either file in a browser while LibreWXR is running to see the radar overlay on an interactive map.

The `examples/live-demo/` directory contains the same examples pre-configured to use the public LibreWXR instance at `api.librewxr.net` — no local server required. Just open them in a browser to try it out.

## Data Sources

LibreWXR uses the following freely available data:

- **[Iowa Environmental Mesonet (IEM)](https://mesonet.agron.iastate.edu/)** — NEXRAD N0Q composite radar imagery (US regions)
- **[MET Norway THREDDS](https://thredds.met.no/)** — Nordic radar reflectivity composite (Norway, Sweden, Finland, Denmark)
- **[DWD Open Data](https://opendata.dwd.de/)** — HX radar reflectivity composite (Germany). License: [GeoNutzV](https://www.gesetze-im-internet.de/geonutzv/) (free, attribution required: "Deutscher Wetterdienst")
- **[UCAR THREDDS](https://thredds.ucar.edu/)** — GFS Global 0.25° 2m temperature analysis for snow classification, and GFS simulated reflectivity at 1000m for global fallback coverage

All sources are provided by government-funded institutions and are freely available for any use.

## Current Limitations

- **Limited high-resolution coverage** — real radar composites cover US territories (CONUS, Alaska, Hawaii, Puerto Rico, Guam), Nordic countries (Norway, Sweden, Finland, Denmark), and Germany; the rest of the world uses low-resolution GFS model data (~25km) as a fallback
- **No nowcast/forecast** — only past radar frames are available; precipitation prediction is not yet implemented
- **No satellite imagery** — the satellite infrared endpoint returns empty data

## License

LibreWXR is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
