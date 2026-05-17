# LibreWXR Web Integration Guide

A tutorial for adding live weather radar to a website using LibreWXR. No prior experience with Rain Viewer or weather tile APIs is assumed.

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [API Reference](#api-reference)
  - [Weather Maps Endpoint](#weather-maps-endpoint)
  - [Tile URL Format](#tile-url-format)
  - [Satellite Tile URL Format](#satellite-tile-url-format)
  - [Coverage Tile Endpoint](#coverage-tile-endpoint)
  - [Alerts Endpoint](#alerts-endpoint)
  - [Health Endpoint](#health-endpoint)
- [Step-by-Step: Leaflet Integration](#step-by-step-leaflet-integration)
  - [1. Basic Map Setup](#1-basic-map-setup)
  - [2. Fetching Radar Metadata](#2-fetching-radar-metadata)
  - [3. Displaying a Radar Frame](#3-displaying-a-radar-frame)
  - [4. Adding Animation Controls](#4-adding-animation-controls)
  - [5. Supporting HiDPI / Retina Displays](#5-supporting-hidpi--retina-displays)
  - [6. Adding Nowcast (Forecast) Frames](#6-adding-nowcast-forecast-frames)
  - [7. Precipitation Motion Arrows](#7-precipitation-motion-arrows)
  - [8. Adding a Satellite Layer](#8-adding-a-satellite-layer)
  - [9. Adding a Weather Alerts Overlay](#9-adding-a-weather-alerts-overlay)
- [Step-by-Step: MapLibre GL JS Integration](#step-by-step-maplibre-gl-js-integration)
  - [1. Basic Map Setup](#1-basic-map-setup-1)
  - [2. Adding a Radar Layer](#2-adding-a-radar-layer)
  - [3. Animating Frames](#3-animating-frames)
  - [4. Adding a Satellite Layer](#4-adding-a-satellite-layer)
- [Tile URL Parameters In Depth](#tile-url-parameters-in-depth)
  - [Color Schemes](#color-schemes)
  - [Smoothing and Snow](#smoothing-and-snow)
  - [Image Format](#image-format)
  - [Arrows Query Parameter](#arrows-query-parameter)
- [Tips and Best Practices](#tips-and-best-practices)
- [Refreshing Data](#refreshing-data)
- [Complete Working Examples](#complete-working-examples)

---

## How It Works

LibreWXR serves weather radar as **map tiles** — small square images (256x256 or 512x512 pixels) that slot into a standard web map (like Google Maps, Leaflet, or MapLibre). Each tile covers a specific geographic area at a specific zoom level using the standard "slippy map" coordinate system.

The basic flow for displaying radar on a web page is:

1. **Fetch metadata** — call the API to get a list of available radar timestamps
2. **Build tile URLs** — for a given timestamp, construct URLs that your map library will use to load tile images
3. **Add as a map layer** — the map library requests individual tiles as the user pans and zooms
4. **Animate** — cycle through timestamps to show radar movement over time

Each radar frame represents a 10-minute snapshot of precipitation. The API typically serves 12 past frames (2 hours of history) plus optional nowcast (forecast) frames up to 60 minutes into the future.

LibreWXR also serves **satellite tiles** — IFS-derived cloud cover that approximates infrared satellite imagery. These work the same way (fetch timestamps, build URLs, add as a layer) but use a different URL pattern and update hourly instead of every 10 minutes.

Beyond tiles, LibreWXR exposes a **weather alerts API** — a global feed of WMO CAP alerts (severe weather warnings, watches, advisories) returned as a GeoJSON FeatureCollection that you can drop into any map library that consumes GeoJSON.

---

## Prerequisites

- A LibreWXR server to connect to (see below)
- A web page where you can add HTML/CSS/JavaScript
- A map library — this guide covers [Leaflet](https://leafletjs.com/) and [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/)

No API key is required. LibreWXR has no rate limits beyond your server's capacity.

### Using the public instance

If you want to experiment before setting up your own server, you can use the public LibreWXR instance:

```
https://api.librewxr.net
```

Just use this URL wherever you see `http://localhost:8080` in the examples below. The `examples/` directory in the repository contains ready-to-open HTML files that auto-detect whether to use a local or public API endpoint — no setup needed.

When you're ready to self-host, swap the URL to your own server and everything works the same way.

---

## API Reference

### Weather Maps Endpoint

```
GET /public/weather-maps.json
```

This is the starting point for any integration. It returns metadata about all available radar frames. The response is compatible with the Rain Viewer v2 API format.

**Example response:**

```json
{
  "version": "2.0",
  "generated": 1700000000,
  "host": "http://localhost:8080",
  "radar": {
    "past": [
      { "time": 1699999200, "path": "/v2/radar/1699999200" },
      { "time": 1699999800, "path": "/v2/radar/1699999800" },
      { "time": 1700000400, "path": "/v2/radar/1700000400" }
    ],
    "nowcast": [
      { "time": 1700001000, "path": "/v2/radar/1700001000" },
      { "time": 1700001600, "path": "/v2/radar/1700001600" }
    ],
    "colorSchemes": [
      { "id": 0, "name": "Black and White" },
      { "id": 1, "name": "Rainviewer Original" },
      { "id": 7, "name": "Rainbow @ Selex SI" }
    ]
  },
  "satellite": {
    "infrared": [
      { "time": 1699963200, "path": "/v2/satellite/1699963200" },
      { "time": 1699966800, "path": "/v2/satellite/1699966800" }
    ]
  }
}
```

**Fields:**

| Field | Description |
|-------|-------------|
| `host` | The base URL of the server. Use this to construct full tile URLs. |
| `radar.past` | Array of past (observed) radar frames, oldest first. |
| `radar.nowcast` | Array of forecast frames, nearest future first. May be empty if nowcasting is disabled. |
| `radar.colorSchemes` | Array of available color schemes (`id` and `name`). Use for dynamically populating UI dropdowns. |
| `satellite.infrared` | Array of satellite (cloud cover) frames, oldest first. Hourly cadence, up to 12 hours. May be empty if satellite is disabled or still loading. |
| `time` | Unix timestamp (seconds) of the frame. |
| `path` | Path prefix for tile requests for this frame. |

### Tile URL Format

```
GET /v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.{ext}
```

This is where the actual tile images come from. Your map library will call this URL pattern for every visible tile.

**Path parameters:**

| Parameter | Description | Values |
|-----------|-------------|--------|
| `timestamp` | Unix timestamp from the metadata response | Integer |
| `size` | Tile pixel size | `256` or `512` |
| `z` | Zoom level | `0` to `12` (configurable max) |
| `x` | Tile column | `0` to `2^z - 1` |
| `y` | Tile row | `0` to `2^z - 1` |
| `color` | Color scheme ID | `0` to `8`, or `255` (see [Color Schemes](#color-schemes)) |
| `smooth_snow` | Smoothing and snow flags, joined with `_` | `{0 or 1}_{0 or 1}` |
| `ext` | Image format | `png` or `webp` |

**Query parameters:**

| Parameter | Description | Values |
|-----------|-------------|--------|
| `arrows` | Precipitation motion arrows | `""` (off), `light`, `dark`, `1`/`true` (alias for light) |

**Example tile URL:**

```
http://localhost:8080/v2/radar/1700000400/256/5/8/12/7/1_0.png
```

This requests a 256px PNG tile at zoom 5, column 8, row 12, using color scheme 7 (Rainbow @ Selex SI), with smoothing enabled and snow coloring disabled.

### Satellite Tile URL Format

```
GET /v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}
```

Satellite tiles use a simpler URL than radar — there are no color scheme, smoothing, or snow parameters. The `0/0_0` segment is fixed.

**Path parameters:**

| Parameter | Description | Values |
|-----------|-------------|--------|
| `timestamp` | Unix timestamp from `satellite.infrared` in the metadata response | Integer |
| `size` | Tile pixel size | `256` or `512` |
| `z` | Zoom level | `0` to `12` (configurable max) |
| `x` | Tile column | `0` to `2^z - 1` |
| `y` | Tile row | `0` to `2^z - 1` |
| `ext` | Image format | `png` or `webp` |

**Example tile URL:**

```
http://localhost:8080/v2/satellite/1700000400/256/5/8/12/0/0_0.png
```

The satellite layer renders IFS-derived cloud cover that approximates infrared satellite imagery. High clouds appear bright white, mid-altitude clouds light gray, and low clouds darker gray. Coverage is global with hourly updates.

### Coverage Tile Endpoint

```
GET /v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png
```

Returns a tile showing where radar data exists (useful for debugging or displaying coverage boundaries). The coverage tile is always PNG format.

### Alerts Endpoint

```
GET /v2/alerts
GET /v2/alerts?lat={lat}&lon={lon}
GET /v2/alerts?bbox=west,south,east,north
```

Returns active weather alerts as a GeoJSON `FeatureCollection`. Each feature carries the alert polygon plus CAP metadata in its `properties`. The data source is the WMO CAP feed at severeweather.wmo.int (global) plus the NWS point endpoint for US locations (which surfaces alerts like Tornado Watches that lack polygon geometry in the global feed).

**Query parameters:**

| Parameter | Description |
|-----------|-------------|
| *(none)* | All active alerts worldwide |
| `lat`, `lon` | Alerts whose polygon contains the point (and, for US points, also queries the NWS point endpoint) |
| `bbox` | `west,south,east,north` — alerts whose polygon intersects the box (polygon-only) |
| `simplify` | Polygon simplification tolerance in meters (default `1000`, set `0` for full resolution) |

Returns `503 Service Unavailable` if `LIBREWXR_ALERTS_ENABLED=false` on the server.

**Example response:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-95.0, 30.0], [-94.0, 30.0], [-94.0, 31.0], [-95.0, 31.0], [-95.0, 30.0]]]
      },
      "properties": {
        "identifier": "NWS-LCH-1234",
        "sender": "w-nws.webmaster@noaa.gov",
        "sent": "2026-05-13T14:00:00Z",
        "expires": "2026-05-13T22:00:00Z",
        "event": "Severe Thunderstorm Warning",
        "headline": "Severe Thunderstorm Warning issued May 13 at 2:00PM CDT",
        "description": "...",
        "severity": "Severe",
        "urgency": "Immediate",
        "certainty": "Likely",
        "areaDesc": "Jefferson County",
        "country": "US"
      }
    }
  ]
}
```

The `severity` / `urgency` / `certainty` fields follow the CAP 1.2 vocabulary, which is convenient for styling: colour by severity, only animate the `Immediate` ones, etc.

### Health Endpoint

```
GET /health
```

Returns server status, frame count, cache usage, RAM stats, NWP chain state, alerts status, and satellite cache state. Useful for monitoring, not typically needed for web integration.

---

## Step-by-Step: Leaflet Integration

### 1. Basic Map Setup

Start with a basic HTML page with Leaflet and a full-screen map:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Weather Radar</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <style>
        body { margin: 0; }
        #map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map', { maxZoom: 12 }).setView([39.83, -98.58], 5);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
    </script>
</body>
</html>
```

This gives you a base map centered on the US. You can change the `setView` coordinates and zoom to center on any region — for example, `[50.0, 10.0]` for Europe, `[56.0, -96.0]` for Canada, `[23.7, 121.0]` for Taiwan, `[13.7, -88.9]` for El Salvador, or `[4.0, 109.0]` for the MET Malaysia / Borneo composite.

### 2. Fetching Radar Metadata

Next, fetch the list of available radar frames from your LibreWXR server:

```javascript
var LIBREWXR_URL = "http://localhost:8080";

function loadRadarData(callback) {
    var request = new XMLHttpRequest();
    request.open("GET", LIBREWXR_URL + "/public/weather-maps.json", true);
    request.onload = function () {
        var data = JSON.parse(request.response);
        callback(data);
    };
    request.send();
}
```

Or with `fetch` (modern browsers):

```javascript
async function loadRadarData() {
    var response = await fetch(LIBREWXR_URL + "/public/weather-maps.json");
    return await response.json();
}
```

### 3. Displaying a Radar Frame

Once you have the API data, add a radar tile layer for a specific frame:

```javascript
var radarLayer = null;

function showRadarFrame(apiData, frameIndex) {
    // Remove previous layer if any
    if (radarLayer) {
        map.removeLayer(radarLayer);
    }

    var frame = apiData.radar.past[frameIndex];

    // Build the tile URL template
    // Format: {host}{path}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.{ext}
    var tileUrl = apiData.host + frame.path + "/256/{z}/{x}/{y}/7/1_0.png";

    radarLayer = L.tileLayer(tileUrl, {
        tileSize: 256,
        opacity: 0.8,
        maxZoom: 12
    }).addTo(map);
}

// Load and display the most recent frame
loadRadarData(function (data) {
    var lastIndex = data.radar.past.length - 1;
    showRadarFrame(data, lastIndex);
});
```

Key points:
- **Opacity** is set to `0.8` so the base map shows through. Adjust to taste.
- The `{z}/{x}/{y}` placeholders are filled by Leaflet automatically as the user pans and zooms.
- Color scheme `7` (Rainbow @ Selex SI) is a good default — it closely resembles a standard weather radar display. See [Color Schemes](#color-schemes) for all options.

### 4. Adding Animation Controls

To animate through frames, cycle the tile layer on a timer. The trick is to pre-load layers so transitions are smooth:

```javascript
var apiData = {};
var frames = [];
var animationPosition = 0;
var animationTimer = null;
var layerCache = {};
var currentLayer = null;

var ANIMATION_DELAY = 500;   // ms between frames
var PAUSE_AT_END = 1500;     // ms pause at the end before looping

function createLayer(frame) {
    return L.tileLayer(apiData.host + frame.path + "/256/{z}/{x}/{y}/7/1_0.png", {
        tileSize: 256,
        opacity: 0.001,  // start invisible; opacity 0 prevents tile loading
        maxZoom: 12
    });
}

function showFrame(position) {
    // Wrap around
    position = ((position % frames.length) + frames.length) % frames.length;
    animationPosition = position;

    var frame = frames[position];

    // Update timestamp display
    var time = new Date(frame.time * 1000);
    document.getElementById("timestamp").textContent =
        time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Hide current layer
    if (currentLayer) {
        currentLayer.setOpacity(0);
    }

    // Show cached layer or create new one
    if (layerCache[position]) {
        layerCache[position].setOpacity(0.8);
        currentLayer = layerCache[position];
        scheduleNext();
    } else {
        var layer = createLayer(frame);
        layer.on("load", function () {
            layer.setOpacity(0.8);
            if (currentLayer && currentLayer !== layer) {
                currentLayer.setOpacity(0);
            }
            currentLayer = layer;
            layerCache[position] = layer;
            scheduleNext();
        });
        layer.addTo(map);
    }
}

function scheduleNext() {
    if (!animationTimer) return;
    var delay = (animationPosition === frames.length - 1) ? PAUSE_AT_END : ANIMATION_DELAY;
    animationTimer = setTimeout(function () {
        showFrame(animationPosition + 1);
    }, delay);
}

function play() {
    animationTimer = true;
    showFrame(animationPosition + 1);
}

function stop() {
    if (animationTimer) {
        clearTimeout(animationTimer);
        animationTimer = null;
    }
}

// Clear cached layers when the user pans (tiles would be stale)
map.on("movestart", function () {
    stop();
    for (var pos in layerCache) {
        if (parseInt(pos) !== animationPosition) {
            map.removeLayer(layerCache[pos]);
            delete layerCache[pos];
        }
    }
});

// Initialize
loadRadarData(function (data) {
    apiData = data;
    frames = data.radar.past;
    showFrame(frames.length - 1);
});
```

Add controls to your HTML:

```html
<div id="controls" style="position:absolute; top:10px; left:60px; z-index:1000; background:white; padding:8px; border-radius:4px;">
    <button onclick="stop(); showFrame(animationPosition - 1);">&#x2039;</button>
    <button onclick="animationTimer ? stop() : play();">&#x25B6;</button>
    <button onclick="stop(); showFrame(animationPosition + 1);">&#x203A;</button>
    <span id="timestamp"></span>
</div>
```

### 5. Supporting HiDPI / Retina Displays

On high-DPI screens, 256px tiles look blurry. Request 512px tiles instead:

```javascript
var tileSize = window.devicePixelRatio >= 2 ? 512 : 256;

// Use tileSize in your URL but keep tileSize: 256 in Leaflet options
// so it displays at the right size:
var tileUrl = apiData.host + frame.path + "/" + tileSize + "/{z}/{x}/{y}/7/1_0.png";

var layer = L.tileLayer(tileUrl, {
    tileSize: 256,  // display size stays 256
    opacity: 0.8,
    maxZoom: 12
});
```

### 6. Adding Nowcast (Forecast) Frames

Nowcast frames work identically to past frames — they use the same tile URL format. The only difference is they represent predicted future precipitation rather than observed data.

```javascript
loadRadarData(function (data) {
    apiData = data;
    frames = data.radar.past.slice();  // copy past frames

    var nowcastStartIndex = -1;

    // Append nowcast frames if available
    if (data.radar.nowcast && data.radar.nowcast.length > 0) {
        nowcastStartIndex = frames.length;
        frames = frames.concat(data.radar.nowcast);
    }

    // Start on the most recent observed frame
    var startPos = nowcastStartIndex >= 0 ? nowcastStartIndex - 1 : frames.length - 1;
    showFrame(startPos);
});
```

To visually distinguish forecast frames, you can style the timestamp differently:

```javascript
function isNowcastFrame(position) {
    return nowcastStartIndex >= 0 && position >= nowcastStartIndex;
}

// In your showFrame function:
var el = document.getElementById("timestamp");
if (isNowcastFrame(position)) {
    el.style.color = "#0077cc";
    el.textContent += " (Forecast)";
} else {
    el.style.color = "#333";
}
```

During animation, pausing briefly at the boundary between past and nowcast frames (and at the end of the loop) gives users time to notice the transition:

```javascript
function getDelay(position) {
    if (position === nowcastStartIndex - 1) return 1500;  // past/nowcast boundary
    if (position === frames.length - 1) return 1500;       // end of loop
    return 500;
}
```

### 7. Precipitation Motion Arrows

LibreWXR can overlay arrows showing storm movement direction and speed on tiles. Add the `arrows` query parameter to the tile URL:

```javascript
// Append ?arrows=light or ?arrows=dark to the tile URL
var tileUrl = apiData.host + frame.path + "/256/{z}/{x}/{y}/7/1_0.png?arrows=light";
```

- `light` — white arrows, good for dark map themes
- `dark` — dark arrows, good for light map themes

The arrows are rendered server-side into the tile image. They show precipitation motion direction and relative speed based on optical flow analysis of consecutive radar frames.

**Tip:** During animation playback, the arrows can be visually distracting. Consider hiding them while animating and only showing them on the static current frame. To do this, construct the URL without `?arrows=` during playback and add it back when paused.

### 8. Adding a Satellite Layer

LibreWXR can serve IFS-derived cloud cover tiles that approximate infrared satellite imagery. These use a different tile URL pattern than radar and are typically shown as a separate, toggleable layer.

Satellite frames are listed under `satellite.infrared` in the metadata response. They use hourly timestamps (vs radar's 10-minute cadence) and can cover up to 12 hours of history.

```javascript
var satelliteLayer = null;
var satelliteFrames = [];
var satellitePosition = 0;

function loadSatelliteFrames(apiData) {
    satelliteFrames = apiData.satellite.infrared || [];
}

function showSatelliteFrame(apiData, position) {
    if (satelliteFrames.length === 0) return;

    position = ((position % satelliteFrames.length) + satelliteFrames.length) % satelliteFrames.length;
    satellitePosition = position;

    if (satelliteLayer) {
        map.removeLayer(satelliteLayer);
    }

    var frame = satelliteFrames[position];

    // Satellite URL: {host}{path}/{size}/{z}/{x}/{y}/0/0_0.{ext}
    // Note: no color scheme, smoothing, or snow parameters — the path ends with /0/0_0
    var tileUrl = apiData.host + frame.path + "/256/{z}/{x}/{y}/0/0_0.png";

    satelliteLayer = L.tileLayer(tileUrl, {
        tileSize: 256,
        opacity: 0.6,
        maxZoom: 12
    }).addTo(map);
}
```

**Key differences from radar tiles:**
- URL path ends with `/0/0_0.{ext}` (fixed) instead of `/{color}/{smooth}_{snow}.{ext}`
- No `?arrows=` query parameter
- Lower opacity (0.5–0.6) works well since cloud cover is visually dense
- Hourly cadence — animation is slower, so a longer delay between frames looks more natural

To add a toggle button:

```html
<button id="sat-toggle" onclick="toggleSatellite()">Satellite</button>
```

```javascript
var satelliteVisible = false;

function toggleSatellite() {
    satelliteVisible = !satelliteVisible;
    if (satelliteVisible) {
        showSatelliteFrame(apiData, satelliteFrames.length - 1);
    } else if (satelliteLayer) {
        map.removeLayer(satelliteLayer);
        satelliteLayer = null;
    }
    document.getElementById("sat-toggle").style.opacity = satelliteVisible ? 1.0 : 0.5;
}
```

**Animating satellite independently:** Since satellite updates hourly while radar updates every 10 minutes, you typically animate them on separate timers. The satellite animation loop is the same pattern as radar (section 4) but with a longer delay between frames (1–2 seconds) and using `satelliteFrames` instead of `frames`.

**Layering order:** Add the satellite layer *before* the radar layer so radar overlays on top of clouds. In Leaflet, you can control this with `layer.setZIndex()` or by adding layers in the right order.

### 9. Adding a Weather Alerts Overlay

LibreWXR's `/v2/alerts` endpoint returns active WMO CAP weather alerts as a GeoJSON `FeatureCollection`. Leaflet has built-in GeoJSON support, so dropping alerts onto your map is a few lines:

```javascript
var alertsLayer = null;

async function loadAlerts() {
    if (alertsLayer) {
        map.removeLayer(alertsLayer);
    }

    var response = await fetch(LIBREWXR_URL + "/v2/alerts");
    if (!response.ok) return;  // 503 = alerts disabled on the server

    var geojson = await response.json();

    alertsLayer = L.geoJSON(geojson, {
        style: function (feature) {
            return {
                color: severityColor(feature.properties.severity),
                weight: 2,
                fillOpacity: 0.15,
            };
        },
        onEachFeature: function (feature, layer) {
            var p = feature.properties;
            layer.bindPopup(
                "<strong>" + (p.event || "Alert") + "</strong><br>" +
                (p.headline || "") + "<br>" +
                "<em>" + (p.areaDesc || "") + "</em><br>" +
                "Severity: " + p.severity + " · Urgency: " + p.urgency
            );
        }
    }).addTo(map);
}

function severityColor(severity) {
    switch (severity) {
        case "Extreme":  return "#ff00ff";
        case "Severe":   return "#ff0000";
        case "Moderate": return "#ff8800";
        case "Minor":    return "#ffcc00";
        default:         return "#aaaaaa";
    }
}

// Initial load
loadAlerts();

// Refresh every 5 minutes to match the server's update cadence
setInterval(loadAlerts, 5 * 60 * 1000);
```

**Filtering by viewport:** If you only care about alerts visible on screen, use the `bbox` query parameter:

```javascript
var bounds = map.getBounds();
var bbox = [
    bounds.getWest(), bounds.getSouth(),
    bounds.getEast(), bounds.getNorth()
].join(",");

var response = await fetch(LIBREWXR_URL + "/v2/alerts?bbox=" + bbox);
```

This is much lighter than loading every active alert worldwide when the user is only looking at one country.

**Per-location lookup:** For "what alerts apply where I clicked?" use `lat`/`lon`:

```javascript
map.on("click", async function (e) {
    var response = await fetch(
        LIBREWXR_URL + "/v2/alerts?lat=" + e.latlng.lat + "&lon=" + e.latlng.lng
    );
    var geojson = await response.json();
    // ... show alerts in a popup
});
```

For US locations, point lookups also include alerts from the NWS point endpoint that lack polygon geometry in the global CAP feed (e.g., Tornado Watches). Outside the US the response is polygon-only from the WMO feed.

**Layer order:** Add alerts *after* radar/satellite so the polygons sit on top. Use a low fill opacity (~0.15) so radar remains visible beneath the alert area.

---

## Step-by-Step: MapLibre GL JS Integration

MapLibre uses a source/layer model instead of Leaflet's direct tile layers. The concepts are the same, but the API is different.

### 1. Basic Map Setup

```html
<!DOCTYPE html>
<html>
<head>
    <title>Weather Radar</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/maplibre-gl/dist/maplibre-gl.css"/>
    <script src="https://unpkg.com/maplibre-gl/dist/maplibre-gl.js"></script>
    <style>
        body { margin: 0; }
        #map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = new maplibregl.Map({
            container: 'map',
            style: {
                version: 8,
                sources: {
                    osm: {
                        type: 'raster',
                        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                        tileSize: 256,
                        attribution: '&copy; OpenStreetMap contributors'
                    }
                },
                layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
            },
            center: [-98.58, 39.83],
            zoom: 4,
            maxZoom: 12
        });

        map.addControl(new maplibregl.NavigationControl());
    </script>
</body>
</html>
```

### 2. Adding a Radar Layer

In MapLibre, you add a **source** (where tiles come from) and a **layer** (how to render them) separately:

```javascript
var LIBREWXR_URL = "http://localhost:8080";

map.on('load', async function () {
    var response = await fetch(LIBREWXR_URL + "/public/weather-maps.json");
    var data = await response.json();

    var latestFrame = data.radar.past[data.radar.past.length - 1];

    // Add the tile source
    map.addSource('radar', {
        type: 'raster',
        tiles: [data.host + latestFrame.path + '/256/{z}/{x}/{y}/7/1_0.png'],
        tileSize: 256,
        maxzoom: 12
    });

    // Add a layer that renders the source
    map.addLayer({
        id: 'radar-layer',
        type: 'raster',
        source: 'radar',
        paint: {
            'raster-opacity': 0.8,
            'raster-opacity-transition': { duration: 0, delay: 0 },
            'raster-fade-duration': 0
        }
    });
});
```

The transition and fade settings of `0` are important — without them, MapLibre will cross-fade between tiles, which looks wrong when switching radar frames.

### 3. Animating Frames

MapLibre requires you to add/remove sources for each frame. Here's the pattern:

```javascript
var loadedPositions = new Set();
var currentLayerId = null;
var frames = [];
var animationPosition = 0;
var animationTimer = null;
var apiData = {};

function showFrame(position) {
    position = ((position % frames.length) + frames.length) % frames.length;
    animationPosition = position;

    var frame = frames[position];
    var sourceId = 'radar-' + position;
    var layerId = 'radar-layer-' + position;

    // Hide previous layer
    if (currentLayerId && currentLayerId !== layerId && map.getLayer(currentLayerId)) {
        map.setPaintProperty(currentLayerId, 'raster-opacity', 0);
    }

    // If already loaded, just show it
    if (loadedPositions.has(position)) {
        map.setPaintProperty(layerId, 'raster-opacity', 0.8);
        currentLayerId = layerId;
        return;
    }

    // Create source + layer
    map.addSource(sourceId, {
        type: 'raster',
        tiles: [apiData.host + frame.path + '/256/{z}/{x}/{y}/7/1_0.png'],
        tileSize: 256,
        maxzoom: 12
    });

    map.addLayer({
        id: layerId,
        type: 'raster',
        source: sourceId,
        paint: {
            'raster-opacity': 0.001,
            'raster-opacity-transition': { duration: 0, delay: 0 },
            'raster-fade-duration': 0
        }
    });

    // Wait for tiles to load, then show
    function onSourceData(e) {
        if (e.sourceId === sourceId && map.isSourceLoaded(sourceId)) {
            map.off('sourcedata', onSourceData);
            map.once('idle', function () {
                map.setPaintProperty(layerId, 'raster-opacity', 0.8);
                if (currentLayerId && currentLayerId !== layerId && map.getLayer(currentLayerId)) {
                    map.setPaintProperty(currentLayerId, 'raster-opacity', 0);
                }
                loadedPositions.add(position);
                currentLayerId = layerId;
            });
        }
    }

    map.on('sourcedata', onSourceData);
}
```

Cleanup on pan (same idea as Leaflet — remove stale layers to free memory):

```javascript
map.on('movestart', function () {
    loadedPositions.forEach(function (pos) {
        if (pos !== animationPosition) {
            var layerId = 'radar-layer-' + pos;
            var sourceId = 'radar-' + pos;
            if (map.getLayer(layerId)) map.removeLayer(layerId);
            if (map.getSource(sourceId)) map.removeSource(sourceId);
        }
    });
    loadedPositions.clear();
    loadedPositions.add(animationPosition);
});
```

### 4. Adding a Satellite Layer

Adding satellite in MapLibre follows the same source/layer pattern. The tile URL is simpler than radar — no color or smoothing parameters.

```javascript
var satelliteSourceId = 'satellite';
var satelliteLayerId = 'satellite-layer';

function showSatelliteLayer(apiData) {
    var irFrames = apiData.satellite.infrared;
    if (!irFrames || irFrames.length === 0) return;

    // Show the most recent satellite frame
    var latestFrame = irFrames[irFrames.length - 1];

    // Remove existing satellite layer/source if present
    if (map.getLayer(satelliteLayerId)) map.removeLayer(satelliteLayerId);
    if (map.getSource(satelliteSourceId)) map.removeSource(satelliteSourceId);

    // Satellite URL: {host}{path}/{size}/{z}/{x}/{y}/0/0_0.{ext}
    map.addSource(satelliteSourceId, {
        type: 'raster',
        tiles: [apiData.host + latestFrame.path + '/256/{z}/{x}/{y}/0/0_0.png'],
        tileSize: 256,
        maxzoom: 12
    });

    // Insert satellite BEFORE radar so radar renders on top
    map.addLayer({
        id: satelliteLayerId,
        type: 'raster',
        source: satelliteSourceId,
        paint: {
            'raster-opacity': 0.6,
            'raster-opacity-transition': { duration: 0, delay: 0 },
            'raster-fade-duration': 0
        }
    }, 'radar-layer-0');  // 'beforeId' — places satellite below radar
}
```

To toggle the satellite layer on and off:

```javascript
function toggleSatellite() {
    if (!map.getLayer(satelliteLayerId)) {
        showSatelliteLayer(apiData);
    } else {
        var current = map.getPaintProperty(satelliteLayerId, 'raster-opacity');
        map.setPaintProperty(satelliteLayerId, 'raster-opacity', current > 0 ? 0 : 0.6);
    }
}
```

To animate through satellite frames, use the same pattern as radar animation (section 3) but with `satellite.infrared` as the frame source, a `'sat-'` prefix for source/layer IDs, and a longer frame delay (1–2 seconds) since satellite updates hourly.

---

## Tile URL Parameters In Depth

### Color Schemes

LibreWXR supports all 9 Rain Viewer color schemes plus a raw grayscale mode:

| ID | Name | Description |
|----|------|-------------|
| 0 | Black and White | Grayscale intensity |
| 1 | Rainviewer Original | Classic Rain Viewer colors |
| 2 | Universal Blue | Blue-to-red gradient |
| 3 | Titan | High-contrast scheme |
| 4 | The Weather Channel (TWC) | Matches TWC broadcast colors |
| 5 | Meteored | European-style colors |
| 6 | NEXRAD Level III | US NWS standard radar colors |
| 7 | Rainbow @ Selex SI | Full rainbow gradient (recommended default — closest to standard weather radar) |
| 8 | Dark Sky | Muted, minimal style |
| 255 | Raw | Grayscale proportional to dBZ — useful for custom client-side coloring |

Use the scheme ID as the `{color}` path parameter. If an invalid ID is provided, the server falls back to Rainbow @ Selex SI (7).

### Smoothing and Snow

The `{smooth}_{snow}` path segment controls two independent features:

- **Smooth** (`1` = on, `0` = off): Applies a Gaussian blur to soften the pixelated edges of radar data. Especially useful at higher zoom levels. The blur radius is configurable server-side via `LIBREWXR_SMOOTH_RADIUS`.

- **Snow** (`1` = on, `0` = off): When enabled, areas classified as snowfall use an alternate color palette (typically blues/purples instead of greens/yellows). Classification comes from the regional NWP source covering each pixel (HRRR-CONUS / HRRR-Alaska / WRF-SMN / DMI DINI / ICON-EU each classify natively from their own 2-metre temperature field), or from ECMWF IFS snowfall ratio everywhere else.

Common combinations:
- `0_0` — raw, no smoothing, rain colors only
- `1_0` — smoothed, rain colors only (most common)
- `1_1` — smoothed with snow coloring

### Image Format

- **PNG** (`.png`): Lossless, larger files, universal browser support. Best for exact color reproduction.
- **WebP** (`.webp`): Smaller files, supported by all modern browsers. Quality is configurable server-side via `LIBREWXR_WEBP_QUALITY` (default 80; set to 100 for lossless).

For most web applications, WebP is the better choice — tiles load faster and use less bandwidth with minimal visual difference.

### Arrows Query Parameter

Append `?arrows=light` or `?arrows=dark` to any tile URL to overlay precipitation motion arrows:

```
/v2/radar/1700000400/256/5/8/12/7/1_0.png?arrows=light
/v2/radar/1700000400/256/5/8/12/7/1_0.png?arrows=dark
```

The arrows indicate the direction and relative speed of precipitation movement, derived from optical flow analysis. They are rendered server-side directly into the tile image.

---

## Tips and Best Practices

### Pixelated rendering

Radar data is inherently blocky. For a crisp, weather-radar look (rather than blurry interpolation), add this CSS:

```css
/* Leaflet */
.leaflet-tile {
    image-rendering: pixelated;
}

/* MapLibre handles this internally */
```

Alternatively, enable smoothing (`1_0`) in the tile URL for a softer appearance.

### Tile size and DPI

- Use `256` for standard displays
- Use `512` for HiDPI / Retina displays (2x device pixel ratio)
- Always set `tileSize: 256` in your map library options — the larger tiles are displayed at the same logical size but with more detail

### Opacity

A radar opacity of `0.7` to `0.8` works well for most base maps. This lets map labels and roads show through the radar overlay. Adjust based on your base map style.

### Layer visibility trick

When pre-loading layers for animation, set opacity to `0.001` rather than `0`. Setting opacity to exactly `0` causes some map libraries to skip loading tiles entirely as an optimization, which defeats the purpose of pre-loading.

### Map pan cleanup

Always clear cached/pre-loaded radar layers when the user pans the map. The tiles are only valid for the viewport that was visible when they loaded. Both examples above demonstrate this pattern.

### CORS

LibreWXR allows all origins by default (`LIBREWXR_CORS_ORIGINS=["*"]`). If you restrict it, make sure your web app's origin is included or tile requests will fail silently.

---

## Refreshing Data

Radar data updates every 10 minutes. Satellite data updates hourly. To keep your map current, re-fetch the metadata periodically and rebuild the frame lists:

```javascript
// Refresh every 5 minutes (server fetches every 10, so 5 ensures prompt updates)
setInterval(function () {
    loadRadarData(function (data) {
        apiData = data;
        frames = data.radar.past.slice();
        if (data.radar.nowcast && data.radar.nowcast.length > 0) {
            frames = frames.concat(data.radar.nowcast);
        }
        // Update satellite frames too
        satelliteFrames = data.satellite.infrared || [];
        // Optionally restart animation
    });
}, 5 * 60 * 1000);
```

---

## Complete Working Examples

The `examples/` directory contains two self-contained HTML files that demonstrate every feature covered in this guide:

- **`examples/leaflet.html`** — Full Leaflet integration
- **`examples/maplibre.html`** — Full MapLibre GL JS integration

Both examples include:
- **Source selector** — switch between your local server and the public instance (`api.librewxr.net`) without editing code. Auto-detects the best default based on how the file is opened.
- **Layer modes** — Radar, Satellite, or Radar + Satellite (satellite as a cloud background under animated radar)
- **Light/dark theme** — toggles both the base map style and UI colors
- **Color scheme selector**, **motion arrows**, and **nowcast** with full animation support
- **Draggable scrubber** — timeline with past/nowcast visual distinction and tick labels
- **Background preloading** — pre-renders all frames with a progress indicator for smooth playback
- **Keyboard shortcuts** — Space to play/pause, arrow keys to step through frames
- **Auto-refresh** — metadata refreshes every 5 minutes to stay current

To use them:

1. Open either HTML file in your browser — no server setup needed if using the public instance
2. To use your local server, start LibreWXR and select "Local (localhost:8080)" from the source dropdown

These examples serve as reference implementations for production web integrations.
