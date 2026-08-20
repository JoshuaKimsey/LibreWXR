# Migrating from Rain Viewer to LibreWXR

LibreWXR is a drop-in replacement for the Rain Viewer v2 API. If you have an existing website or app that uses Rain Viewer for weather radar, switching to LibreWXR requires minimal changes — in most cases, just updating the server URL.

## Table of Contents

- [Quick Migration (TL;DR)](#quick-migration-tldr)
- [What Changed on Rain Viewer](#what-changed-on-rain-viewer)
- [What LibreWXR Restores](#what-librewxr-restores)
- [Step-by-Step Migration](#step-by-step-migration)
  - [1. Update the API URL](#1-update-the-api-url)
  - [2. Update the Tile Host](#2-update-the-tile-host)
  - [3. Test It](#3-test-it)
- [API Compatibility Reference](#api-compatibility-reference)
  - [Metadata Endpoint](#metadata-endpoint)
  - [Tile URL Format](#tile-url-format)
  - [Coverage Tiles](#coverage-tiles)
  - [Lat/Lon Window URLs (Point-Tile API)](#latlon-window-urls-point-tile-api)
- [Feature Comparison](#feature-comparison)
- [What's Different in LibreWXR](#whats-different-in-librewxr)
- [What's Not Supported](#whats-not-supported)
- [Common Migration Scenarios](#common-migration-scenarios)
  - [Leaflet](#leaflet)
  - [MapLibre GL JS](#maplibre-gl-js)
  - [Google Maps](#google-maps)
  - [Generic JavaScript](#generic-javascript)
- [Trying It Before Committing](#trying-it-before-committing)
- [Troubleshooting](#troubleshooting)

---

## Quick Migration (TL;DR)

Find where your code references the Rain Viewer API and change two things:

1. **Metadata URL:** Change `https://api.rainviewer.com/public/weather-maps.json` to `http://your-librewxr-server:8080/public/weather-maps.json`
2. **Tile host:** Use the `host` field from the metadata response instead of hardcoding `https://tilecache.rainviewer.com`

That's it. The endpoint paths, URL format, query parameters, color scheme IDs, and response structure are all identical.

---

## What Changed on Rain Viewer

As of January 1, 2026, Rain Viewer's free API tier was restricted:

- Maximum zoom level 7 (was 12)
- Only one color scheme (was 9)
- No satellite imagery
- No nowcast/forecast frames
- PNG only (no WebP)

Higher tiers still offer the full functionality but require a paid subscription.

## What LibreWXR Restores

LibreWXR provides everything the pre-restriction Rain Viewer API offered, self-hosted with no usage limits:

- All zoom levels up to 12
- All 13 color schemes + raw grayscale
- 256px and 512px tiles
- PNG and WebP formats
- Nowcast/forecast frames (up to 60 minutes)
- Smoothing and snow color options
- WMO CAP weather alerts as a GeoJSON feed (`/v2/alerts` with point, bounding-box, and polygon-simplification query params)

Plus additional features Rain Viewer didn't offer:
- Precipitation motion arrows (`?arrows=light` or `?arrows=dark`)
- Configurable noise filtering and speckle removal
- ECMWF IFS 9km global model layer + NOAA RRQPE global observed precipitation (60S-70N) + regional NWP layers (HRRR, HRRR-Alaska, HRDPS, JMA MSM, AROME Antilles, AROME Guyane, AROME Indien, AROME Ncaled, AROME Polyn, DMI DINI, ICON-EU, WRF-SMN)
- Optical flow interpolation for smooth global animation
- Fully configurable via environment variables

---

## Step-by-Step Migration

### 1. Update the API URL

Find where your code fetches the Rain Viewer metadata:

```javascript
// Before (Rain Viewer)
var apiUrl = "https://api.rainviewer.com/public/weather-maps.json";

// After (LibreWXR — self-hosted)
var apiUrl = "http://localhost:8080/public/weather-maps.json";

// After (LibreWXR — public instance, for testing)
var apiUrl = "https://api.librewxr.net/public/weather-maps.json";
```

### 2. Update the Tile Host

Rain Viewer returned `https://tilecache.rainviewer.com` as the `host` in its metadata response. LibreWXR returns your server's `LIBREWXR_PUBLIC_URL` instead.

**If your code already uses the `host` field from the API response** (the recommended approach), no tile URL changes are needed — it will automatically point to your LibreWXR instance.

**If your code hardcodes the tile host**, update it:

```javascript
// Before
var tileUrl = "https://tilecache.rainviewer.com" + frame.path + "/256/{z}/{x}/{y}/2/1_0.png";

// After
var tileUrl = apiData.host + frame.path + "/256/{z}/{x}/{y}/2/1_0.png";
```

### 3. Test It

1. Start your LibreWXR server (or use `https://api.librewxr.net` to test first)
2. Open your web page
3. Verify radar tiles appear and animation works

If tiles don't appear, check the browser developer console for CORS errors or failed requests. See [Troubleshooting](#troubleshooting) below.

---

## API Compatibility Reference

### Metadata Endpoint

| | Rain Viewer | LibreWXR |
|---|---|---|
| **URL** | `https://api.rainviewer.com/public/weather-maps.json` | `http://your-server:8080/public/weather-maps.json` |
| **Response format** | Identical | Identical |
| **`host` field** | `https://tilecache.rainviewer.com` | Your `LIBREWXR_PUBLIC_URL` value |
| **`radar.past`** | Array of `{time, path}` | Identical |
| **`radar.nowcast`** | Array of `{time, path}` (paid tier) | Identical (enabled by default) |
| **`satellite.infrared`** | Array of `{time, path}` (discontinued Jan 2026) | Array of `{time, path}` (up to 12 hourly GMGSI frames; empty when the satellite layer is disabled) |

### Tile URL Format

The tile URL format is identical:

```
{host}/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.{ext}
```

Every parameter works the same way:

| Parameter | Rain Viewer | LibreWXR |
|---|---|---|
| `timestamp` | Unix timestamp from metadata | Identical |
| `size` | `256` or `512` | Identical |
| `z`, `x`, `y` | Slippy map tile coordinates | Identical |
| `color` | `0`-`8` | `0`-`12` + `255` (raw grayscale) |
| `smooth` | `0` or `1` | Identical |
| `snow` | `0` or `1` | Identical |
| `ext` | `png` (free) / `webp` (paid) | `png` or `webp` (both always available) |

**LibreWXR addition:** The `?arrows=light` and `?arrows=dark` query parameters are new and optional. Rain Viewer clients that don't use them will work without changes.

### Coverage Tiles

| | Rain Viewer | LibreWXR |
|---|---|---|
| **URL** | `/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png` | Identical |
| **Response** | PNG tile showing radar coverage | Identical |

### Lat/Lon Window URLs (Point-Tile API)

Rain Viewer's lat/lon single-location image endpoint is supported:

```
{host}/v2/radar/{timestamp}/{size}/{z}/{lat}/{lon}/{color}/{smooth}_{snow}.{ext}
```

| | Rain Viewer | LibreWXR |
|---|---|---|
| **URL** | `/v2/radar/{timestamp}/{size}/{z}/{lat}/{lon}/{color}/{smooth}_{snow}.{ext}` | Identical |
| **Coverage** | `/v2/coverage/0/{size}/{z}/{lat}/{lon}/0/0_0.png` | Identical |
| **Response** | `size` x `size` image centered on the EPSG:4326 coordinate | Identical |

Returns a `size` x `size` PNG or WebP centered on the coordinate, for past radar frames and nowcast timestamps alike. `size` is `256` or `512` (values in between quantize: `< 512` becomes `256`, matching the tile route), and `{smooth}_{snow}` behaves exactly as on the tile route. Unknown timestamps return 404; areas with no data return a transparent 200 response in the requested format (PNG or WebP). The coverage layer has the same variant: `/v2/coverage/0/{size}/{z}/{lat}/{lon}/0/0_0.png`.

**LibreWXR addition:** timestamp `0` in the `{timestamp}` slot is an alias for the latest frame - radar resolves it to the newest past radar frame and the satellite endpoint to the latest GMGSI timestamp (RainViewer itself reserves `0` this way only on the coverage endpoint). Resolution happens before any caching, so alias URLs key and cache exactly like the canonical ones, and the resolved timestamp is returned in the `X-Frame-Timestamp` response header on both 200 and 304 responses. Any other unknown timestamp still returns 404.

Path segments containing a dot are treated as lat/lon; plain integer segments are x/y tile indices (a coordinate without a dot is an integer index, even if it names a latitude). The center is snapped to the nearest pixel at that zoom. Longitude wraps across the antimeridian (a window centered near +/-180 deg shows content from both sides of the seam, center preserved). Latitude is clamped to the Web Mercator limit (+/-85.0511 deg); lat beyond +/-90 deg is a 400, and windows at the poles clamp to the world edge. The `?arrows=` and `?cells=` query parameters are tile-mode only and are silently ignored on lat/lon window URLs.

Repeated requests for the same location hit the tile cache (the snapped origin is the cache key), so widgets polling a fixed location are cheap after the first render.

**Contract change:** the `x`/`y` path parameters are now string-typed in OpenAPI (previously integer). Malformed non-numeric values now return 400 instead of 422; negative integers still return 400, and out-of-range tile indices still return 400.

---

## Feature Comparison

| Feature | Rain Viewer (Free, Post-2026) | Rain Viewer (Paid) | LibreWXR |
|---|---|---|---|
| Max zoom | 7 | 12 | 12 |
| Color schemes | 1 | 9 | 13 + raw grayscale |
| Tile sizes | 256px | 256px, 512px | 256px, 512px |
| Image formats | PNG | PNG, WebP | PNG, WebP |
| Smoothing | No | Yes | Yes |
| Snow colors | No | Yes | Yes |
| Nowcast/Forecast | No | ~60 min | Up to 60 min |
| Satellite | No (discontinued Jan 2026) | Yes (IR, 10-min) | Yes (GMGSI LW+VIS composite, hourly) |
| Motion arrows | No | No | Yes |
| Coverage | Global | Global | US, Canada, Europe, El Salvador, Japan (JMA HRPN), Taiwan, SE Asia radar + global RRQPE observed + global ECMWF IFS + regional NWP |
| Rate limits | Yes | Higher limits | None (self-hosted) |
| Cost | Free | Subscription | Free (self-hosted) |

---

## What's Different in LibreWXR

These are things to be aware of but generally don't require code changes:

- **Coverage area**: Rain Viewer sourced radar data globally from many countries. LibreWXR has high-resolution radar composites for the US, Canada, Europe, El Salvador (MARN/SNET), Japan (JMA HRPN), Taiwan (CWA QPESUMS), Peninsular Malaysia + Borneo + Brunei + Singapore + N. Sumatra (MET Malaysia), and the Philippines (PAGASA PANAHON), plus NOAA RRQPE — a global observed (satellite-derived) precipitation radar region covering the 60S-70N band — and a chain of regional NWP models (HRRR, HRRR-Alaska, HRDPS, JMA MSM, AROME Antilles, AROME Guyane, AROME Indien, AROME Ncaled, AROME Polyn, DMI DINI, ICON-EU, WRF-SMN) layered on top of ECMWF IFS. Outside the radar domains but inside the band, the precipitation layer is observed (RRQPE, satellite-derived); it is modelled only poleward of the band, in the fringe excluded by RRQPE's coverage polygon, and when RRQPE declines — at a few-km resolution where regional NWP applies, and at IFS 9 km elsewhere. If your users are primarily in any of these radar regions, the experience is equivalent or better.

- **Data update cadence**: Both use 10-minute intervals. LibreWXR aligns to clock boundaries (:00, :10, :20, etc.) just like Rain Viewer.

- **Satellite imagery**: LibreWXR serves NOAA GMGSI where Rain Viewer's paid layer was 10-minute infrared-only (discontinued for everyone January 2026): a global (±72.7 deg) hourly LW+VIS composite — infrared at night, visible reflectance by day, with a natural day/night terminator — with up to 12 hourly frames of history. The `satellite.infrared` metadata array works the same way and tile URLs use fixed `0`/`0_0` color/options segments: `/v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}` (png or webp). `LIBREWXR_SATELLITE_ENABLED=false` empties the catalog array and makes tile requests return 503.

- **Color scheme rendering**: LibreWXR reproduces all 9 original Rain Viewer color schemes from the same color lookup tables, plus four contributed schemes (13 named total) and a raw grayscale mode (255). The visual output should be identical for a given scheme ID.

- **Tile caching headers**: LibreWXR serves tiles with `Cache-Control: public` — `max-age=300` (5 minutes) for the latest and nowcast frames, and `max-age=7200` (2 hours) for historical frames, which are immutable once backfill is complete. This is compatible with any CDN or caching proxy.

---

## What's Not Supported

- **Rain Viewer API key authentication** — LibreWXR has no authentication. If your code sends a Rain Viewer API key, it will be ignored harmlessly.

- **Rain Viewer webhooks or push notifications** — LibreWXR is a pull-based API only.

---

## Common Migration Scenarios

### Leaflet

```javascript
// Before (Rain Viewer)
var API_URL = "https://api.rainviewer.com/public/weather-maps.json";

// After (LibreWXR)
var API_URL = "http://localhost:8080/public/weather-maps.json";

// If you hardcode the tile host:
// Before
var tileUrl = "https://tilecache.rainviewer.com" + frame.path + "/256/{z}/{x}/{y}/2/1_0.png";
// After (use the host from the API response)
var tileUrl = apiData.host + frame.path + "/256/{z}/{x}/{y}/2/1_0.png";
```

No other changes needed. The `L.tileLayer` options, opacity settings, and animation logic all work identically.

### MapLibre GL JS

```javascript
// Before
var API_URL = "https://api.rainviewer.com/public/weather-maps.json";

// After
var API_URL = "http://localhost:8080/public/weather-maps.json";

// Tile source URLs
// Before
tiles: ["https://tilecache.rainviewer.com" + frame.path + "/256/{z}/{x}/{y}/2/1_0.png"]
// After
tiles: [apiData.host + frame.path + "/256/{z}/{x}/{y}/2/1_0.png"]
```

### Google Maps

```javascript
// Before
var API_URL = "https://api.rainviewer.com/public/weather-maps.json";

// After
var API_URL = "http://localhost:8080/public/weather-maps.json";

// In your ImageMapType getTileUrl function:
// Before
return "https://tilecache.rainviewer.com" + currentFrame.path + "/256/" + zoom + "/" + coord.x + "/" + coord.y + "/2/1_0.png";
// After
return apiData.host + currentFrame.path + "/256/" + zoom + "/" + coord.x + "/" + coord.y + "/2/1_0.png";
```

### Generic JavaScript

If you're using the Rain Viewer API directly with `fetch` or `XMLHttpRequest`, the only change is the URL:

```javascript
// Before
fetch("https://api.rainviewer.com/public/weather-maps.json")
    .then(r => r.json())
    .then(data => {
        // data.host was "https://tilecache.rainviewer.com"
        // Everything else works the same
    });

// After
fetch("http://localhost:8080/public/weather-maps.json")
    .then(r => r.json())
    .then(data => {
        // data.host is now your LibreWXR URL
        // Everything else works the same
    });
```

---

## Trying It Before Committing

You don't need to set up your own server to test the migration. Use the public LibreWXR instance:

```javascript
var API_URL = "https://api.librewxr.net/public/weather-maps.json";
```

This lets you verify your code works with LibreWXR before investing time in self-hosting. When ready, swap the URL to your own server.

The `examples/` directory in the repository contains ready-to-open Leaflet and MapLibre examples that auto-detect whether to use a local or public API endpoint.

---

## Troubleshooting

### Tiles don't appear

1. **Check the browser console** for errors. Look for CORS errors or 404s.
2. **Verify the metadata endpoint** works by opening `/public/weather-maps.json` directly in your browser. You should see a JSON response with `radar.past` entries.
3. **Verify a tile loads** by constructing a URL manually from the metadata. Take the `host` + a `path` from `radar.past` + `/256/3/2/3/7/1_0.png` and open it in your browser. You should see a small radar tile image.
4. **Check the `host` field** in the metadata response. It should match the URL that your browser can reach. If you're running behind a reverse proxy, make sure `LIBREWXR_PUBLIC_URL` is set correctly.

### CORS errors

LibreWXR allows all origins by default (`LIBREWXR_CORS_ORIGINS=["*"]`). If you've restricted it, ensure your web app's origin is in the list.

### Tiles are blank/transparent

This is normal for areas with no precipitation. Radar tiles are transparent where there is no rain or snow. Try zooming to an area with active weather, or check the `/health` endpoint to confirm the server has radar data loaded.

### "Frame not found" (404) errors

The requested timestamp doesn't exist in the server's frame store. This can happen if:
- Your cached metadata is stale — re-fetch `/public/weather-maps.json`
- The server recently restarted and hasn't accumulated frames yet — check `/health` for frame count

### Nowcast frames missing

Nowcasting is enabled by default. If `radar.nowcast` is empty in the metadata response, the server may still be generating its first nowcast frames (requires at least 2 past radar frames). Wait for 1-2 fetch cycles (~10-20 minutes) after server startup.
