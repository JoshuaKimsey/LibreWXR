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
- All 9 color schemes + raw grayscale
- 256px and 512px tiles
- PNG and WebP formats
- Nowcast/forecast frames (up to 60 minutes)
- Smoothing and snow color options

Plus additional features Rain Viewer didn't offer:
- Precipitation motion arrows (`?arrows=light` or `?arrows=dark`)
- Configurable noise filtering and speckle removal
- ECMWF IFS 9km global precipitation coverage + regional NWP layers (HRRR, HRDPS, DMI DINI, ICON-EU, AROME Antilles, WRF-SMN)
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
| **`satellite.infrared`** | Array of `{time, path}` | Empty array `[]` |

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
| `color` | `0`-`8` | `0`-`8` + `255` (raw grayscale) |
| `smooth` | `0` or `1` | Identical |
| `snow` | `0` or `1` | Identical |
| `ext` | `png` (free) / `webp` (paid) | `png` or `webp` (both always available) |

**LibreWXR addition:** The `?arrows=light` and `?arrows=dark` query parameters are new and optional. Rain Viewer clients that don't use them will work without changes.

### Coverage Tiles

| | Rain Viewer | LibreWXR |
|---|---|---|
| **URL** | `/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png` | Identical |
| **Response** | PNG tile showing radar coverage | Identical |

---

## Feature Comparison

| Feature | Rain Viewer (Free, Post-2026) | Rain Viewer (Paid) | LibreWXR |
|---|---|---|---|
| Max zoom | 7 | 12 | 12 |
| Color schemes | 1 | 9 | 9 + raw grayscale |
| Tile sizes | 256px | 256px, 512px | 256px, 512px |
| Image formats | PNG | PNG, WebP | PNG, WebP |
| Smoothing | No | Yes | Yes |
| Snow colors | No | Yes | Yes |
| Nowcast/Forecast | No | ~60 min | Up to 60 min |
| Satellite IR | No | Yes | Not yet |
| Motion arrows | No | No | Yes |
| Coverage | Global | Global | US, Canada, Europe, El Salvador, Taiwan, SE Asia radar + global ECMWF IFS + regional NWP |
| Rate limits | Yes | Higher limits | None (self-hosted) |
| Cost | Free | Subscription | Free (self-hosted) |

---

## What's Different in LibreWXR

These are things to be aware of but generally don't require code changes:

- **Coverage area**: Rain Viewer sourced radar data globally from many countries. LibreWXR has high-resolution radar composites for the US, Canada, Europe, El Salvador (MARN/SNET), Taiwan (CWA QPESUMS), and Peninsular Malaysia + Borneo + Brunei + Singapore + N. Sumatra (MET Malaysia), plus a chain of regional NWP models (HRRR, HRDPS, DMI DINI, ICON-EU, AROME Antilles, WRF-SMN) layered on top of ECMWF IFS for global precipitation coverage. Outside the radar domains, the precipitation layer is modelled rather than observed — at a few-km resolution where regional NWP applies, and at IFS 9 km globally. If your users are primarily in any of these radar regions, the experience is equivalent or better.

- **Data update cadence**: Both use 10-minute intervals. LibreWXR aligns to clock boundaries (:00, :10, :20, etc.) just like Rain Viewer.

- **Color scheme rendering**: LibreWXR reproduces all 9 Rain Viewer color schemes from the same color lookup tables. The visual output should be identical for a given scheme ID.

- **Tile caching headers**: LibreWXR serves tiles with `Cache-Control: public, max-age=300` (5 minutes). This is compatible with any CDN or caching proxy.

---

## What's Not Supported

- **Satellite infrared imagery** — The `satellite.infrared` array is always empty. If your code depends on satellite tiles, those requests will return empty/404 responses. Radar tiles are unaffected.

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
