<!-- SPDX-License-Identifier: MIT -->
<!DOCTYPE html>
<html>
<head>
    <title>LibreWXR - Radar Widget</title>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <style>
        /*__VIEWER_CSS__*/
    </style>
    <style>
        /* === WIDGET PAGE LAYOUT ===
           viewer.css above is written for the full-page map demos: it makes
           <body> a non-scrolling flex column filling the viewport. This page
           is a normal scrolling document, so reset those base rules and center
           a compact card column instead. Everything below uses viewer.css
           design tokens, so the widget matches the map demos visually even
           though it has no map. */

        body {
            overflow: auto;
            height: auto;
            display: block;
            min-height: 100vh;
            background: radial-gradient(1000px 500px at 50% -10%, var(--accent-glow), transparent 60%), var(--bg);
        }

        .widget-page {
            max-width: 720px;
            margin: 0 auto;
            padding: var(--space-5, 24px) var(--space-4) calc(var(--space-4) * 2);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--space-4);
        }

        .widget-header {
            text-align: center;
        }
        .widget-header h1 {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: var(--space-1);
        }
        .widget-header p {
            color: var(--text-secondary);
            font-size: var(--font-size-md);
            line-height: 1.5;
            max-width: 520px;
        }
        .widget-header code {
            color: var(--accent);
            font-size: 0.88em;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: var(--radius-sm);
            padding: 1px 5px;
        }

        /* Error banner (page variant of the map demos' .error-overlay) */
        .wb-error {
            width: 100%;
            max-width: 480px;
            display: none;
            background: var(--error-bg);
            border: 1px solid var(--error-border);
            border-radius: var(--radius-md);
            padding: var(--space-3) var(--space-4);
            color: var(--text);
            font-size: var(--font-size-md);
            box-shadow: var(--shadow-lg);
        }
        .wb-error.visible {
            display: block;
        }
        .wb-error .error-msg {
            line-height: 1.4;
        }
        .wb-error .retry-btn {
            height: 32px;
            padding: 0 var(--space-4);
            margin-top: var(--space-2);
            background: var(--accent);
            border: none;
            border-radius: var(--radius-sm);
            color: #fff;
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: background var(--transition-fast), transform var(--transition-fast), box-shadow var(--transition-fast);
        }
        .wb-error .retry-btn:hover {
            background: var(--accent-hover);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }
        .wb-error .retry-btn:active {
            transform: scale(0.96);
        }
        .wb-error .retry-btn:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }

        /* === RADAR CARD ===
           One square image, no map. The image is transparent wherever there
           is no precipitation, so a faint CSS grid + crosshair behind it makes
           the radar pixels read well against the plain dark panel. */
        .widget-card {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--space-3);
            padding: var(--space-4);
            background: var(--card-bg);
            border: 1px solid var(--toolbar-border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-md);
        }

        .radar-frame {
            position: relative;
            width: min(90vw, 256px);   /* display size; the requested server size is separate */
            aspect-ratio: 1;
            border: 1px solid var(--toolbar-border);
            border-radius: var(--radius-md);
            overflow: hidden;
            background: var(--replayer-bg);
        }
        .radar-frame[data-size="512"] {
            width: min(90vw, 512px);
        }

        /* Faint lat/lon grid + center crosshair (pure CSS, pointer-transparent) */
        .radar-grid {
            position: absolute;
            inset: 0;
            z-index: 1;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(78, 168, 222, 0.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(78, 168, 222, 0.10) 1px, transparent 1px);
            background-size: 25% 25%;
        }
        .radar-grid::before,
        .radar-grid::after {
            content: '';
            position: absolute;
            background: rgba(78, 168, 222, 0.35);
        }
        .radar-grid::before {
            left: 50%;
            top: 0;
            bottom: 0;
            width: 1px;
        }
        .radar-grid::after {
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
        }
        .radar-grid .grid-dot {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 5px;
            height: 5px;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        /* Two stacked layers; the .active one is opaque (crossfade via CSS) */
        .lw-img {
            position: absolute;
            inset: 0;
            z-index: 2;
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: opacity var(--transition-slow);
            mix-blend-mode: plus-lighter;   /* same additive blend the map demos use */
        }
        .lw-img.active {
            opacity: 1;
        }

        .lw-badge {
            position: absolute;
            left: var(--space-2);
            bottom: var(--space-2);
            z-index: 3;
            padding: 2px var(--space-2);
            background: var(--card-bg);
            border: 1px solid var(--btn-border);
            border-radius: var(--radius-full);
            font-size: var(--font-size-sm);
            color: var(--text);
            font-variant-numeric: tabular-nums;
            box-shadow: var(--shadow-sm);
            backdrop-filter: var(--backdrop-blur);
            -webkit-backdrop-filter: var(--backdrop-blur);
        }

        /* Optional OSM basemap: plain raster <img> tiles covering the same
           point-tile window the server renders (toggled by the Map pill). */
        .map-layer {
            position: absolute;
            inset: 0;
            z-index: 0;
            display: none;
            overflow: hidden;
            border-radius: inherit;
        }
        .radar-frame.has-map .map-layer {
            display: block;
        }
        .map-tile {
            position: absolute;
            display: block;
        }
        .map-attribution {
            position: absolute;
            right: 4px;
            bottom: 4px;
            z-index: 3;
            display: none;
            font-size: 10px;
            line-height: 1.4;
            padding: 1px 4px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.75);
            color: #333;
            text-decoration: none;
        }
        .radar-frame.has-map .map-attribution {
            display: block;
        }
        .radar-frame.has-map .lw-img {
            mix-blend-mode: normal;   /* plus-lighter washes precip out on a light map */
        }

        /* === CONTROLS (wraps into rows on narrow screens) === */
        .controls {
            width: 100%;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: var(--space-2) var(--space-3);
        }

        .wb-field {
            display: inline-flex;
            align-items: center;
            gap: var(--space-1);
            font-size: var(--font-size-sm);
            color: var(--text-secondary);
            cursor: pointer;
        }
        .wb-field select,
        .wb-field input[type="number"] {
            height: 32px;
            border: 1px solid var(--input-border);
            border-radius: var(--radius-sm);
            background: var(--input-bg);
            font-size: var(--font-size-md);
            color: var(--input-text);
            padding: 0 var(--space-2);
            outline: none;
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
        }
        .wb-field select {
            cursor: pointer;
            appearance: none;
            -webkit-appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 6px center;
            padding-right: 22px;
        }
        [data-theme="light"] .wb-field select {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        }
        .wb-field select:hover,
        .wb-field input[type="number"]:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .wb-field select:focus-visible,
        .wb-field input[type="number"]:focus-visible {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .wb-field input[type="number"] {
            width: 88px;
            font-variant-numeric: tabular-nums;
            cursor: text;
        }

        /* Checkbox pills (same pattern as .options-panel checkbox labels) */
        .wb-field.check {
            padding: var(--space-1) var(--space-2);
            border-radius: var(--radius-sm);
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            gap: var(--space-1);
            user-select: none;
            -webkit-user-select: none;
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
        }
        .wb-field.check:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }
        .wb-field.check input[type="checkbox"] {
            width: 14px;
            height: 14px;
            margin: 0;
            accent-color: var(--accent);
            cursor: pointer;
            flex-shrink: 0;
        }

        /* === URL PREVIEW (the drop-in centerpiece) === */
        .url-card {
            width: 100%;
            padding: var(--space-3) var(--space-4);
            background: var(--card-bg);
            border: 1px solid var(--toolbar-border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }
        .url-card-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: var(--space-2);
            flex-wrap: wrap;
            margin-bottom: var(--space-2);
        }
        .url-card-head h2 {
            font-size: var(--font-size-md);
            color: var(--text);
        }
        .url-card-head p {
            font-size: var(--font-size-xs);
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .url-box {
            width: 100%;
            height: 34px;
            padding: 0 var(--space-3);
            border: 1px solid var(--input-border);
            border-radius: var(--radius-sm);
            background: var(--input-bg);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
            font-size: 12px;
            color: var(--text);
            cursor: copy;
            outline: none;
            white-space: nowrap;
            overflow-x: auto;
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
        }
        .url-box:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .url-box:focus-visible {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .wb-status {
            min-height: 1em;
            margin-top: var(--space-1);
            font-size: var(--font-size-xs);
            color: var(--text-dim);
        }

        /* === FOOTER === */
        .widget-footer {
            max-width: 480px;
            font-size: var(--font-size-xs);
            color: var(--text-dim);
            text-align: center;
            line-height: 1.6;
        }
        .widget-footer a {
            color: var(--accent);
        }

        /* === RESPONSIVE === */
        @media (max-width: 600px) {
            .widget-page {
                padding: var(--space-3) var(--space-2) var(--space-4);
                gap: var(--space-3);
            }
            .widget-header h1 { font-size: 19px; }
            .widget-card { padding: var(--space-3); }
            .url-card { padding: var(--space-2) var(--space-3); }
            .controls { gap: var(--space-1) var(--space-2); }
            .lw-badge { font-size: var(--font-size-xs); }
        }

        @media (prefers-reduced-motion: reduce) {
            .lw-img { transition: none !important; }
        }
    </style>
</head>
<body data-theme="dark">

<main class="widget-page">
    <header class="widget-header">
        <h1>LibreWXR Radar Widget</h1>
        <p>An animated radar snapshot for any location - no map library, with an optional
           OpenStreetMap background. Powered by the lat/lon point-tile
           endpoint <code>.../{size}/{z}/{lat}/{lon}/{color}/{options}.png</code>.</p>
    </header>

    <!-- Inline error banner (hidden until something goes wrong) -->
    <div class="wb-error" id="lw-error" role="alert">
        <div class="error-msg" id="lw-error-msg"></div>
        <button type="button" class="retry-btn" id="lw-error-retry" style="display:none">Try again</button>
    </div>

    <!-- Radar card: one square image, crosshair behind it, age badge on top -->
    <section class="widget-card" aria-label="Radar display">
        <div class="radar-frame" id="lw-frame" data-size="256">
            <div class="map-layer" id="lw-map-layer" aria-hidden="true"></div>
            <div class="radar-grid" aria-hidden="true">
                <span class="grid-dot"></span>
            </div>
            <img id="lw-img-a" class="lw-img" alt="Radar">
            <img id="lw-img-b" class="lw-img" alt="Radar">
            <div class="lw-badge" id="lw-badge" role="status">loading...</div>
            <a class="map-attribution" id="lw-map-attribution"
               href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">&copy; OpenStreetMap contributors</a>
        </div>
    </section>

    <!-- Controls: one tidy, wrapping row -->
    <section class="controls" aria-label="Widget controls">
        <button type="button" class="play-btn" id="lw-play" aria-label="Play playback">
            <svg id="lw-play-icon" viewBox="0 0 16 16" style="display:none"><polygon points="4,2 14,8 4,14"/></svg>
            <svg id="lw-pause-icon" viewBox="0 0 16 16"><rect x="3" y="2" width="3.5" height="12"/><rect x="9.5" y="2" width="3.5" height="12"/></svg>
        </button>
        <!-- #lv-source block: removed in --site builds -->
        <label class="wb-field">API
            <select id="lw-source" aria-label="API source">
                <option value="public" selected>Public (api.librewxr.net)</option>
                <option value="local">Local (localhost:8080)</option>
            </select>
        </label>
        <!-- /#lv-source -->
        <label class="wb-field">Preset
            <select id="lw-preset" aria-label="Location preset"></select>
        </label>
        <label class="wb-field">Lat
            <input id="lw-lat" type="number" step="0.01" min="-90" max="90" inputmode="decimal" value="33.749">
        </label>
        <label class="wb-field">Lon
            <input id="lw-lon" type="number" step="0.01" inputmode="decimal" value="-84.388">
        </label>
        <button type="button" class="icon-btn" id="lw-locate" title="Use my location" aria-label="Use my location">
            <svg viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3"/>
                <line x1="12" y1="2" x2="12" y2="6"/>
                <line x1="12" y1="18" x2="12" y2="22"/>
                <line x1="2" y1="12" x2="6" y2="12"/>
                <line x1="18" y1="12" x2="22" y2="12"/>
            </svg>
        </button>
        <label class="wb-field">Zoom
            <select id="lw-zoom" aria-label="Zoom level"></select>
        </label>
        <label class="wb-field">Size
            <select id="lw-size" aria-label="Image size"></select>
        </label>
        <label class="wb-field">Format
            <select id="lw-format" aria-label="Image format"></select>
        </label>
        <label class="wb-field check"><input type="checkbox" id="lw-smooth" checked> Smooth</label>
        <label class="wb-field check"><input type="checkbox" id="lw-snow" checked> Snow</label>
        <label class="wb-field check"><input type="checkbox" id="lw-map" checked> Map</label>
    </section>

    <!-- URL preview: always shows the exact URL of the image on screen -->
    <section class="url-card" aria-label="Current frame URL">
        <div class="url-card-head">
            <h2>Current frame URL</h2>
            <p>click to copy</p>
        </div>
        <input id="lw-url" class="url-box" type="text" readonly aria-label="Current frame image URL"
               spellcheck="false" autocomplete="off" value="">
        <div class="wb-status" id="lw-status" role="status" aria-live="polite"></div>
    </section>

    <footer class="widget-footer">
        MIT-licensed drop-in code - see the
        <a href="https://github.com/JoshuaKimsey/LibreWXR/blob/main/docs/web-integration-guide.md">integration
        guide</a> for the full point-tile API reference.
    </footer>
</main>

<script>
/* ============================================================================
   LibreWXR Radar Widget

   A single animated radar image for a fixed location, powered by the
   lat/lon-centered "point tile" endpoint:

       {apiBase}{frame.path}/{size}/{z}/{lat}/{lon}/{color}/{smooth}_{snow}.{ext}

   No map library, no tile grid, no external dependencies: the radar image is a
   plain <img> whose src the server renders centered on your coordinates. The
   optional basemap is the same idea - plain OpenStreetMap raster <img> tiles
   aligned to that point-tile window, still no library or dependencies. This
   whole file is MIT-licensed drop-in code - change the CONFIG block, copy the
   markup plus this script into your page, and you have a RainViewer-style
   widget image.

   Section map:
     [1] CONFIG        - everything you might want to change
     [2] CATALOG       - fetch /public/weather-maps.json, retry, refresh
     [3] URL BUILDING  - the point-tile URL and the coordinate "dot rule"
     [4] ANIMATION     - playback loop, preload pool, crossfade
     [5] CONTROLS      - presets, manual coordinates, rendering options
     [6] BOOT          - wire up the DOM and start
   ============================================================================ */

// ----------------------------------------------------------------------------
// [1] CONFIG - edit this block to make the widget yours
// ----------------------------------------------------------------------------

var LVR_API_SOURCES = {
    public: 'https://api.librewxr.net',
    local: 'http://localhost:8080'
};
// The "API" selector in the controls picks between the two sources above and
// defaults to the public instance. Leave LVR_API_FIXED null for normal use;
// the --site build pins it to the public API and strips that selector.
var LVR_API_FIXED = null;

var DEFAULT_PRESET = 'Atlanta';   // first location shown on load
var LOCATION_PRESETS = [
    { name: 'Atlanta',      lat: 33.749, lon: -84.388 },
    { name: 'London',       lat: 51.507, lon: -0.128 },
    { name: 'Tokyo',        lat: 35.681, lon: 139.767 },
    { name: 'Taipei',       lat: 25.033, lon: 121.565 },
    { name: 'Toronto',      lat: 43.653, lon: -79.383 },
    { name: 'Kuala Lumpur', lat: 3.139,  lon: 101.687 }
];

var DEFAULT_ZOOM = 7;
var ZOOM_RANGE = [4, 5, 6, 7, 8, 9, 10];
var DEFAULT_SIZE = 256;             // requested server-side image resolution
var SIZE_OPTIONS = [256, 512];      // the on-screen size shrinks on phones
var DEFAULT_FORMAT = 'webp';        // 'webp' or 'png'
var FORMAT_OPTIONS = ['webp', 'png'];
var COLOR_SCHEME = 10;              // matches the viewer-core.js default scheme
var DEFAULT_SMOOTH = true;          // URL options segment: 1_1 when both on
var DEFAULT_SNOW = true;

// Optional OpenStreetMap raster basemap (the "Map" checkbox pill). OSM tiles
// are plain 256px <img> elements, so this stays zero-dependency.
var MAP_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
var MAP_DEFAULT_ON = true;
var WEBMERC_MAX_LAT = 85.05112878;  // Web Mercator latitude clamp (matches the server)

var FRAME_DWELL_MS = 500;           // dwell per frame (same as the map engine)
var PRELOAD_AHEAD = 3;              // frames preloaded ahead of playback
var CATALOG_POLL_MS = 300000;       // re-fetch the catalog every 5 minutes
var RETRY_DELAYS = [5000, 15000, 30000];  // catalog auto-retry backoff

// DOM references (the script sits at the end of <body>, so nodes exist).
var errorBox = document.getElementById('lw-error');
var errorMsg = document.getElementById('lw-error-msg');
var retryBtn = document.getElementById('lw-error-retry');
var frameEl = document.getElementById('lw-frame');
var badgeEl = document.getElementById('lw-badge');
var layerA = document.getElementById('lw-img-a');
var layerB = document.getElementById('lw-img-b');
var playBtn = document.getElementById('lw-play');
var playIcon = document.getElementById('lw-play-icon');
var pauseIcon = document.getElementById('lw-pause-icon');
var sourceSelect = document.getElementById('lw-source'); // null in --site builds
var presetSelect = document.getElementById('lw-preset');
var latInput = document.getElementById('lw-lat');
var lonInput = document.getElementById('lw-lon');
var locateBtn = document.getElementById('lw-locate');
var zoomSelect = document.getElementById('lw-zoom');
var sizeSelect = document.getElementById('lw-size');
var formatSelect = document.getElementById('lw-format');
var smoothInput = document.getElementById('lw-smooth');
var snowInput = document.getElementById('lw-snow');
var mapInput = document.getElementById('lw-map');
var mapLayer = document.getElementById('lw-map-layer');
var urlInput = document.getElementById('lw-url');
var statusEl = document.getElementById('lw-status');

// ----------------------------------------------------------------------------
// [2] CATALOG - the frame list comes from {apiBase}/public/weather-maps.json
// ----------------------------------------------------------------------------

var frames = [];            // radar.past ++ radar.nowcast, each {time, path}
var position = 0;           // index of the frame currently on screen
var currentFrame = null;    // the frame backing the visible image
var catalogInFlight = false;
var retryDelayIndex = 0;
var retryTimer = null;
var bootPlayed = false;

function apiBase() {
    // A pinned base (--site builds) always wins; those pages ship without the
    // selector. Leave LVR_API_FIXED null to let the selector control it.
    if (LVR_API_FIXED) return LVR_API_FIXED;
    // Otherwise the "API" selector decides, defaulting to the public instance
    // when the node is missing (e.g. the markup was trimmed to just the card).
    if (sourceSelect) return LVR_API_SOURCES[sourceSelect.value] || LVR_API_SOURCES.public;
    return LVR_API_SOURCES.public;
}

function prefersReducedMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

// Normalize the catalog to just the two fields the widget uses.
function buildFrameList(data) {
    var out = [];
    var radar = data && data.radar;
    if (radar && Array.isArray(radar.past)) out = out.concat(radar.past);
    if (radar && Array.isArray(radar.nowcast)) out = out.concat(radar.nowcast);
    return out
        .map(function (f) { return { time: f.time, path: f.path }; })
        .filter(function (f) { return Number.isFinite(f.time) && typeof f.path === 'string' && f.path.length > 0; });
}

function nearestFrameIndex(list, timeSec) {
    var best = 0;
    var bestDiff = Infinity;
    for (var i = 0; i < list.length; i++) {
        var d = Math.abs(list[i].time - timeSec);
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return best;
}

// Apply a freshly fetched catalog: reframe playback at the nearest equivalent
// time position so a refresh does not visibly jump the animation.
function mergeCatalog(data) {
    var newFrames = buildFrameList(data);
    if (newFrames.length === 0) {
        showError('The catalog has no radar frames yet. Is radar enabled on the server?', true);
        return;
    }
    var hadFrames = frames.length > 0;
    var anchorTime = currentFrame ? currentFrame.time : Date.now() / 1000;
    frames = newFrames;
    position = nearestFrameIndex(frames, anchorTime);
    invalidatePreloads();
    displayFrame(position);
    preloadAhead(position);
    // First successful load: start the loop (unless the user prefers reduced
    // motion or keeps the tab hidden). The button can pause it any time.
    if (!hadFrames && !bootPlayed && !document.hidden && !prefersReducedMotion()) {
        bootPlayed = true;
        play();
    }
}

function showError(msg, withRetry) {
    errorBox.classList.add('visible');
    errorMsg.textContent = msg;
    retryBtn.style.display = withRetry ? '' : 'none';
}

function hideError() {
    errorBox.classList.remove('visible');
}

// Fetch the catalog with the same auto-retry backoff the map demos use
// (5s, then 15s, then 30s; after that the Retry button is manual).
async function loadCatalog() {
    if (catalogInFlight) return;
    catalogInFlight = true;
    var url = apiBase() + '/public/weather-maps.json';
    try {
        var resp = await fetch(url);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var data = await resp.json();
        catalogInFlight = false;
        retryDelayIndex = 0;
        hideError();
        mergeCatalog(data);
    } catch (err) {
        catalogInFlight = false;
        if (retryDelayIndex < RETRY_DELAYS.length) {
            var delay = RETRY_DELAYS[retryDelayIndex++];
            showError('Could not load the radar catalog from ' + url + ' - retrying in '
                + Math.round(delay / 1000) + 's', true);
            clearTimeout(retryTimer);
            retryTimer = setTimeout(loadCatalog, delay);
        } else {
            showError('Could not load the radar catalog from ' + url + '. Is the server running?', true);
        }
    }
}

// ----------------------------------------------------------------------------
// [3] URL BUILDING - the point-tile endpoint and the coordinate "dot rule"
// ----------------------------------------------------------------------------

function coord(v) {
    // The server tells a lat/lon window apart from a plain x/y tile by the
    // dot in the segment: "52" is tile column 52, "52.00000" is latitude 52.
    // toFixed(5) both rounds to 5 decimal places (as recommended) and
    // guarantees the dot even for whole numbers like 90.
    return v.toFixed(5);
}

function currentLat() {
    var v = parseFloat(latInput.value);
    return Number.isFinite(v) ? v : NaN;
}

function currentLon() {
    var v = parseFloat(lonInput.value);
    return Number.isFinite(v) ? v : NaN;
}

function validLocation() {
    var lat = currentLat();
    var lon = currentLon();
    // The server 400s on |lat| > 90 (longitudes wrap server-side), so refuse
    // to build such URLs.
    return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90;
}

function frameUrl(frame) {
    return apiBase() + frame.path +
        '/' + sizeSelect.value +
        '/' + zoomSelect.value +
        '/' + coord(currentLat()) +
        '/' + coord(currentLon()) +
        '/' + COLOR_SCHEME +
        '/' + (smoothInput.checked ? 1 : 0) + '_' + (snowInput.checked ? 1 : 0) +
        '.' + formatSelect.value;
}

// Build the optional OSM basemap under the radar image. The server's point-tile
// window {size}x{size} px centered on (lat, lon) at zoom z is exactly the
// standard Web Mercator slippy-map window of the same pixel size at zoom z, so
// plain 256px OSM raster tiles at that zoom line up pixel-for-pixel with the
// radar image. Percentage positions let the tiles scale with the responsive
// frame (no resize handler), and the +1px on each tile plus row-major DOM order
// hide the sub-pixel seams between adjacent tiles.
function buildMapTiles() {
    var frame = frameEl;
    if (!mapInput.checked) {
        frame.classList.remove('has-map');
        mapLayer.innerHTML = '';
        return;
    }
    frame.classList.add('has-map');
    var z = parseInt(zoomSelect.value, 10);
    var size = parseInt(sizeSelect.value, 10);
    var lat = Math.max(-WEBMERC_MAX_LAT, Math.min(WEBMERC_MAX_LAT, currentLat()));
    var lon = currentLon();
    lon = ((lon + 180) % 360 + 360) % 360 - 180;   // wrap to [-180, 180)
    var n = Math.pow(2, z);
    var scale = 256 * n;
    var px = (lon + 180) / 360 * scale;
    var latRad = lat * Math.PI / 180;
    var py = (1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * scale;
    px = Math.round(px);  // server snaps the window center to the nearest pixel
    py = Math.round(py);
    var left = px - size / 2, top = py - size / 2;
    var x0 = Math.floor(left / 256), x1 = Math.floor((left + size - 1) / 256);
    var y0 = Math.floor(top / 256), y1 = Math.floor((top + size - 1) / 256);
    var html = '';
    for (var ty = y0; ty <= y1; ty++) {
        if (ty < 0 || ty >= n) continue;   // beyond the poles: no tile, matches server's empty fill
        for (var tx = x0; tx <= x1; tx++) {
            var wx = ((tx % n) + n) % n;   // antimeridian wrap
            var tileLeft = (tx * 256 - left) / size * 100;
            var tileTop = (ty * 256 - top) / size * 100;
            var url = MAP_TILE_URL.replace('{z}', z).replace('{x}', wx).replace('{y}', ty);
            html += '<img class="map-tile" alt="" src="' + url + '" style="left:' + tileLeft +
                '%;top:' + tileTop + '%;width:calc(' + (256 / size * 100) + '% + 1px);height:calc(' +
                (256 / size * 100) + '% + 1px);">';
        }
    }
    mapLayer.innerHTML = html;
}

// ----------------------------------------------------------------------------
// [4] ANIMATION - playback loop, a small preload pool, and a crossfade
//     between two stacked <img> layers (pure CSS transition).
// ----------------------------------------------------------------------------

var playing = false;
var userPaused = true;           // true only when the play button is paused
var playingBeforeHidden = false; // set while the tab is hidden
var frameTimer = null;
var preloadCache = {};           // url -> Image; keyed by URL so a config
                                 // change silently retires stale entries
var renderEpoch = 0;             // guards against stale onload callbacks
var activeLayer = null;          // the <img> currently on screen

function togglePlay() {
    if (playing) pause(); else play();
}

function play() {
    if (playing || frames.length === 0) return;
    userPaused = false;
    playing = true;
    syncPlayUI();
    clearTimeout(frameTimer);
    frameTimer = setTimeout(step, FRAME_DWELL_MS);
}

function pause() {
    if (!playing) return;
    userPaused = true;
    playing = false;
    syncPlayUI();
    clearTimeout(frameTimer);
    frameTimer = null;
}

function syncPlayUI() {
    playIcon.style.display = playing ? 'none' : '';
    pauseIcon.style.display = playing ? '' : 'none';
    playBtn.setAttribute('aria-label', playing ? 'Pause playback' : 'Play playback');
}

function step() {
    if (!playing || frames.length === 0) return;
    position = (position + 1) % frames.length;
    displayFrame(position);
    preloadAhead(position);
    frameTimer = setTimeout(step, FRAME_DWELL_MS);
}

// Put frame `index` on screen. The old layer stays visible until the new one
// has loaded; swapping the .active class triggers the CSS crossfade.
function displayFrame(index) {
    if (!frames.length || !validLocation()) return;
    var frame = frames[index];
    var url = frameUrl(frame);
    updateUrlBox(url);              // preview the URL immediately
    var epoch = ++renderEpoch;      // invalidate earlier in-flight loads
    var next = activeLayer === layerA ? layerB : layerA;
    var prev = activeLayer;

    next.onload = function () {
        if (epoch !== renderEpoch) return;   // a newer frame superseded us
        if (prev) prev.classList.remove('active');
        next.classList.add('active');
        activeLayer = next;
        currentFrame = frame;
        updateBadge(frame);
    };
    next.onerror = function () {
        if (epoch !== renderEpoch) return;
        setStatus('Frame failed to load: ' + url);
    };

    if (next.getAttribute('src') === url) {
        // Re-showing the same URL: browsers do not fire load for an identical
        // src assignment, so apply the outcome directly.
        if (next.complete && next.naturalWidth > 0) next.onload();
        else if (next.complete) next.onerror();
        return;
    }
    next.src = url;                 // starting a fetch
}

// Warm the next few frames so playback has no visible stutter. Preloads run
// regardless of play state, which is what makes scrubbing feel instant too.
function invalidatePreloads() {
    preloadCache = {};
}

function preloadAhead(startIndex) {
    if (!frames.length || !validLocation()) return;
    for (var i = 1; i <= PRELOAD_AHEAD; i++) {
        var frame = frames[(startIndex + i) % frames.length];
        var url = frameUrl(frame);
        if (!preloadCache[url]) {
            var im = new Image();
            im.src = url;           // assigning src starts the fetch
            preloadCache[url] = im;
        }
    }
}

// Age badge: "45 min ago" / "NOW" / "+20 min", relative to wall-clock time.
function updateBadge(frame) {
    var offsetSec = frame.time - Date.now() / 1000;
    badgeEl.textContent = frameLabel(offsetSec);
    badgeEl.title = new Date(frame.time * 1000).toISOString();
}

function frameLabel(offsetSec) {
    if (Math.abs(offsetSec) < 150) return 'NOW';      // within +-2.5 min
    var mins = Math.round(Math.abs(offsetSec) / 60);
    return offsetSec < 0 ? mins + ' min ago' : '+' + mins + ' min';
}

// Tip: freeze playback while the tab is hidden, resume where it left off.
document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
        playingBeforeHidden = playing;
        playing = false;            // quietly stop the timer
        clearTimeout(frameTimer);
        frameTimer = null;
        syncPlayUI();
    } else if (playingBeforeHidden) {
        playingBeforeHidden = false;
        playing = true;             // resume without touching userPaused
        syncPlayUI();
        clearTimeout(frameTimer);
        frameTimer = setTimeout(step, FRAME_DWELL_MS);
    }
});

// ----------------------------------------------------------------------------
// [5] CONTROLS - presets, manual coordinates, rendering options
// ----------------------------------------------------------------------------

function populateSelect(select, values) {
    values.forEach(function (v) {
        var opt = document.createElement('option');
        opt.value = String(v);
        opt.textContent = String(v);
        select.appendChild(opt);
    });
}

function applyFrameSize() {
    // The requested tile size drives the server render; the CSS scratches it
    // down to min(90vw, size) so a 512px image still fits a phone.
    frameEl.setAttribute('data-size', sizeSelect.value);
}

// Rebuild the URL and re-render the current frame with the current settings.
function applyControls() {
    applyFrameSize();
    syncPresetFromFields();         // typed coordinates flip the preset label
    if (!validLocation()) {
        setStatus('Enter a valid latitude (-90..90) and longitude.');
        return;
    }
    buildMapTiles();                // basemap follows zoom/size/location too
    invalidatePreloads();           // URLs changed - rebuild the pool
    if (frames.length > 0) {
        displayFrame(position);     // re-render the same frame at the new URL
        preloadAhead(position);
    }
}

var applyTimer = null;
function scheduleApply() {          // debounce rapid coordinate typing
    clearTimeout(applyTimer);
    applyTimer = setTimeout(applyControls, 300);
}

function fillPreset(name) {
    var preset = LOCATION_PRESETS.find(function (p) { return p.name === name; });
    if (!preset) return;            // 'custom' keeps whatever is in the inputs
    latInput.value = String(preset.lat);
    lonInput.value = String(preset.lon);
    applyControls();
}

// When the lat/lon inputs are edited by hand, the preset select moves to
// "Custom" unless the values happen to match a preset.
function syncPresetFromFields() {
    var lat = currentLat();
    var lon = currentLon();
    var match = LOCATION_PRESETS.find(function (p) {
        return Number.isFinite(lat) && Number.isFinite(lon)
            && Math.abs(p.lat - lat) < 0.005 && Math.abs(p.lon - lon) < 0.005;
    });
    presetSelect.value = match ? match.name : 'custom';
}

function locateMe() {
    if (!('geolocation' in navigator)) {
        setStatus('Geolocation is not available in this browser.');
        return;
    }
    locateBtn.classList.add('active');
    navigator.geolocation.getCurrentPosition(
        function (pos) {
            locateBtn.classList.remove('active');
            latInput.value = pos.coords.latitude.toFixed(5);
            lonInput.value = pos.coords.longitude.toFixed(5);
            applyControls();
        },
        function (err) {
            locateBtn.classList.remove('active');
            setStatus('Location unavailable: ' + geoErrorMessage(err));
        },
        { timeout: 10000, maximumAge: 60000 }
    );
}

function geoErrorMessage(err) {
    switch (err.code) {
        case err.PERMISSION_DENIED: return 'permission denied';
        case err.POSITION_UNAVAILABLE: return 'position unavailable';
        case err.TIMEOUT: return 'timed out';
        default: return 'unknown error';
    }
}

// URL box: click anywhere on it to copy the exact frame URL. This is the
// drop-in centerpiece - paste the URL into your own <img>, card, or email.
var statusTimer = null;
function setStatus(msg) {
    statusEl.textContent = msg;
    clearTimeout(statusTimer);
    statusTimer = setTimeout(function () { statusEl.textContent = ''; }, 2500);
}

function flashCopied() {
    setStatus('URL copied!');
}

// Clipboard API needs a secure context; fall back to the legacy textarea +
// execCommand path for file:// and plain http:// pages.
function legacyCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'absolute';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
        document.execCommand('copy');
        flashCopied();
    } catch (e) { /* clipboard blocked; the selection stays for manual copy */ }
    document.body.removeChild(ta);
}

function copyUrl() {
    var url = urlInput.value;
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(url).then(flashCopied, function () { legacyCopy(url); });
    } else {
        legacyCopy(url);
    }
}

function updateUrlBox(url) {
    urlInput.value = url;
    urlInput.title = 'Click to copy the exact frame URL';
}

// ----------------------------------------------------------------------------
// [6] BOOT - populate the controls, hook up events, start the catalog loop
// ----------------------------------------------------------------------------

function boot() {
    // Presets (name value) plus the "Custom" catch-all.
    LOCATION_PRESETS.forEach(function (p) {
        var opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name;
        presetSelect.appendChild(opt);
    });
    var customOpt = document.createElement('option');
    customOpt.value = 'custom';
    customOpt.textContent = 'Custom';
    presetSelect.appendChild(customOpt);
    populateSelect(zoomSelect, ZOOM_RANGE);
    populateSelect(sizeSelect, SIZE_OPTIONS);
    populateSelect(formatSelect, FORMAT_OPTIONS);

    zoomSelect.value = String(DEFAULT_ZOOM);
    sizeSelect.value = String(DEFAULT_SIZE);
    formatSelect.value = DEFAULT_FORMAT;
    smoothInput.checked = DEFAULT_SMOOTH;
    snowInput.checked = DEFAULT_SNOW;
    mapInput.checked = MAP_DEFAULT_ON;
    var def = LOCATION_PRESETS.find(function (p) { return p.name === DEFAULT_PRESET; }) || LOCATION_PRESETS[0];
    presetSelect.value = def.name;
    latInput.value = String(def.lat);
    lonInput.value = String(def.lon);

    playBtn.addEventListener('click', togglePlay);
    // API source switch: drop the catalog state of the old instance, pause
    // playback cleanly, and reload immediately - no page reload required.
    if (sourceSelect) {
        sourceSelect.addEventListener('change', function () {
            frames = [];
            position = 0;
            currentFrame = null;
            retryDelayIndex = 0;
            clearTimeout(retryTimer);
            invalidatePreloads();
            hideError();
            pause();
            loadCatalog();
        });
    }
    presetSelect.addEventListener('change', function () { fillPreset(presetSelect.value); });
    latInput.addEventListener('input', scheduleApply);
    lonInput.addEventListener('input', scheduleApply);
    locateBtn.addEventListener('click', locateMe);
    zoomSelect.addEventListener('change', applyControls);
    sizeSelect.addEventListener('change', applyControls);
    formatSelect.addEventListener('change', applyControls);
    smoothInput.addEventListener('change', applyControls);
    snowInput.addEventListener('change', applyControls);
    mapInput.addEventListener('change', buildMapTiles);  // map toggle alone must not touch the radar pool
    urlInput.addEventListener('click', function () { urlInput.select(); copyUrl(); });
    urlInput.addEventListener('focus', function () { urlInput.select(); });
    retryBtn.addEventListener('click', function () {
        hideError();
        retryDelayIndex = 0;
        clearTimeout(retryTimer);
        loadCatalog();
    });

    syncPlayUI();
    applyFrameSize();
    buildMapTiles();                            // draw the basemap once up front
    loadCatalog();                          // first fetch
    setInterval(loadCatalog, CATALOG_POLL_MS);  // then poll for new frames

    // Keep the age badge honest while a frame sits on screen.
    setInterval(function () {
        if (currentFrame) updateBadge(currentFrame);
    }, 30000);
}

boot();
</script>

</body>
</html>