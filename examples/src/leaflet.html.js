<!-- SPDX-License-Identifier: MIT -->
<!DOCTYPE html>
<html>
<head>
    <title>LibreWXR - Leaflet Example</title>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <link href="https://unpkg.com/leaflet/dist/leaflet.css" rel="stylesheet"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <style>
        /*__VIEWER_CSS__*/
    </style>
</head>
<body data-theme="dark">

<!-- Toolbar -->
<div class="toolbar">
    <span class="toolbar-title">LibreWXR</span>
    <!-- #lv-source block: removed in --site builds -->
    <select id="lv-source" aria-label="API source">
        <option value="local">Local (localhost:8080)</option>
        <option value="public">Public (api.librewxr.net)</option>
    </select>
    <!-- /#lv-source -->
    <select id="lv-layermode" aria-label="Layer mode">
        <option value="radar">Radar</option>
        <option value="satellite">Satellite</option>
        <option value="both">Radar + Satellite</option>
    </select>
    <select id="lv-scheme" aria-label="Color scheme">
        <option value="10">Loading...</option>
    </select>
    <select id="lv-arrows" aria-label="Motion arrows">
        <option value="">Arrows: Off</option>
        <option value="light">Arrows: Light</option>
        <option value="dark">Arrows: Dark</option>
    </select>
    <select id="lv-cells" aria-label="Cell detection">
        <option value="">Cells: Off</option>
        <option value="light">Cells: Light</option>
        <option value="dark">Cells: Dark</option>
    </select>
    <button type="button" class="icon-btn" id="lv-alerts" aria-pressed="false" aria-label="Toggle weather alerts" title="Weather alerts">
        <span class="btn-icon"><svg viewBox="0 0 24 24"><path d="M12 3 L20 19 H4 Z"/><line x1="12" y1="10" x2="12" y2="15"/><circle cx="12" cy="17.5" r="1"/></svg></span>
        Alerts
    </button>
    <button type="button" class="icon-btn" id="lv-theme" aria-label="Switch to light theme" title="Switch to light theme">
        <span class="btn-icon" id="lv-theme-sun"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/></svg></span>
        <span class="btn-icon" id="lv-theme-moon" style="display:none"><svg viewBox="0 0 24 24"><path d="M20 14.5 A8 8 0 0 1 9.5 4 A8 8 0 1 0 20 14.5 Z"/></svg></span>
    </button>
    <button type="button" class="icon-btn" id="lv-options-btn" aria-expanded="false" aria-label="Toggle options panel" title="Options">
        <span class="btn-icon"><svg viewBox="0 0 24 24"><line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2"/></svg></span>
        Options
    </button>
</div>

<!-- Options panel (collapsible) -->
<div class="options-panel" id="lv-options">
    <label><input type="checkbox" id="lv-smooth" checked/> Smoothing</label>
    <label><input type="checkbox" id="lv-snow"/> Snow mask</label>
    <label for="lv-format">Format</label>
    <select id="lv-format" aria-label="Tile format">
        <option value="webp">WebP</option>
        <option value="png">PNG</option>
    </select>
    <label for="lv-tilesize">Tile size</label>
    <select id="lv-tilesize" aria-label="Tile size">
        <option value="auto">Auto (device)</option>
        <option value="256">256</option>
        <option value="512">512</option>
    </select>
</div>

<!-- Map -->
<div id="lv-map">
    <div class="preload-indicator" id="lv-preload">
        <span id="lv-preload-text">Loading frames 0/0</span>
        <div class="preload-bar"><div class="preload-fill" id="lv-preload-fill"></div></div>
    </div>
    <div class="error-overlay" id="lv-error" role="alert">
        <div class="error-msg" id="lv-error-msg"></div>
        <button type="button" class="retry-btn" id="lv-error-retry" style="display:none">Retry</button>
    </div>
    <div class="refresh-status" id="lv-refresh-status" role="status"></div>
    <button type="button" class="locate-btn" id="lv-locate" aria-label="Locate me" title="Locate me">
        <svg viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="3"/>
            <line x1="12" y1="2" x2="12" y2="6"/>
            <line x1="12" y1="18" x2="12" y2="22"/>
            <line x1="2" y1="12" x2="6" y2="12"/>
            <line x1="18" y1="12" x2="22" y2="12"/>
        </svg>
    </button>
</div>

<!-- Replayer / Scrubber -->
<div class="replayer">
    <div class="replayer-top">
        <button type="button" class="play-btn" id="lv-play" aria-label="Play playback">
            <svg id="lv-play-icon" viewBox="0 0 16 16"><polygon points="4,2 14,8 4,14"/></svg>
            <svg id="lv-pause-icon" viewBox="0 0 16 16" style="display:none"><rect x="3" y="2" width="3.5" height="12"/><rect x="9.5" y="2" width="3.5" height="12"/></svg>
        </button>
        <div class="scrubber-wrap">
            <div class="scrubber-track" id="lv-scrubber-track"></div>
            <div class="scrubber-ticks" id="lv-scrubber-ticks"></div>
        </div>
        <div class="timestamp-display" id="lv-timestamp">Loading...</div>
    </div>
</div>

<script>
//__VIEWER_CORE__
</script>

<script>
// === API SOURCE CONFIG (build.py rewrites for --site) ===
var LVR_API_SOURCES = {
    local: 'http://localhost:8080',
    public: 'https://api.librewxr.net'
};
var LVR_API_FIXED = null;

// === LEAFLET ADAPTER ===
// Implements the viewer-core.js adapter contract with Leaflet 1.x primitives.
var LeafletAdapter = function () {
    var map = null;
    var maxZoom = 12;
    var baseMaps = {
        dark: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>'
        }),
        light: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors'
        })
    };
    var currentBaseMap = null;
    var alertLayers = [];

    // Pane z-values come from the CSS design tokens so theming stays in one place.
    function paneZ(name, fallback) {
        var val = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
        return isNaN(val) ? fallback : val;
    }

    return {
        // Leaflet maps are usable synchronously - no deferred boot needed.
        onMapReady: function (cb) { cb(); },

        createMap: function (containerId, view) {
            maxZoom = view.maxZoom;
            map = L.map(containerId, { maxZoom: view.maxZoom, zoomControl: true })
                .setView([view.lat, view.lon], view.zoom);

            // Custom panes give deterministic z-ordering: satellite under alerts
            // under radar. Values mirror the --leaflet-z-* CSS tokens.
            map.createPane('lv-satellite-pane');
            map.getPane('lv-satellite-pane').style.zIndex = paneZ('--leaflet-z-satellite', 350);
            map.createPane('lv-alerts-pane');
            map.getPane('lv-alerts-pane').style.zIndex = paneZ('--leaflet-z-alerts', 400);
            map.createPane('lv-radar-pane');
            map.getPane('lv-radar-pane').style.zIndex = paneZ('--leaflet-z-radar', 450);
            return map;
        },

        setBasemap: function (theme) {
            if (currentBaseMap) map.removeLayer(currentBaseMap);
            currentBaseMap = baseMaps[theme] || baseMaps.dark;
            currentBaseMap.addTo(map);
            currentBaseMap.bringToBack();
        },

        createFrameLayer: function (url, kind) {
            var pane = kind === 'satellite' ? 'lv-satellite-pane' : 'lv-radar-pane';
            var layer = new L.TileLayer(url, {
                // 512-url HiDPI trick: the tile URL embeds 256 or 512 (the
                // devicePixelRatio-aware size) but Leaflet is told the grid is
                // 256px, and createTile forces tiles to render at 256 CSS px -
                // crisp on retina without doubling the request count.
                tileSize: 256,
                opacity: 0, // created hidden; the engine fades in on ready
                maxZoom: maxZoom,
                pane: pane
            });
            layer.createTile = function (coords, done) {
                var tile = document.createElement('img');
                // .radar-tile / .satellite-tile get image-rendering: pixelated
                // and the 256px !important sizing from viewer.css.
                tile.className = kind === 'satellite' ? 'satellite-tile' : 'radar-tile';
                tile.alt = '';
                var key = coords.x + ':' + coords.y + ':' + coords.z;
                var aborted = false;
                var onLoad = function () {
                    tile.removeEventListener('load', onLoad);
                    tile.removeEventListener('error', onError);
                    if (aborted) return;
                    tile.style.width = '256px';
                    tile.style.height = '256px';
                    // Only hand the tile back to Leaflet if it is still tracked; a pruned
                    // tile would make _tileReady dereference a null _map.
                    if (!layer._tiles || !layer._tiles[key]) return;
                    if (done) done(null, tile);
                    layer.off('remove', abortTile);
                };
                var onError = function () {
                    tile.removeEventListener('load', onLoad);
                    tile.removeEventListener('error', onError);
                    if (aborted) return;
                    if (!layer._tiles || !layer._tiles[key]) return;
                    if (done) done(new Error('Tile load failed'), tile);
                    layer.off('remove', abortTile);
                };
                var abortTile = function () {
                    aborted = true;
                    layer.off('remove', abortTile);
                    tile.src = '';
                    tile.removeEventListener('load', onLoad);
                    tile.removeEventListener('error', onError);
                };
                layer.on('remove', abortTile);
                tile.addEventListener('load', onLoad);
                tile.addEventListener('error', onError);
                tile.src = this.getTileUrl(coords);
                return tile;
            };
            // MUST be attached before returning: until a layer is on the map
            // Leaflet requests no tiles, so no `load`/`tileerror` would ever
            // fire and the engine's onLayerReady callback would hang forever.
            layer.addTo(map);
            return layer;
        },

        onLayerReady: function (handle, cb) {
            // The handler must be dropped on its FIRST fire. Leaflet re-fires
            // `load` every time all visible tiles for a layer finish
            // requesting, and a cached layer still attached to the map with
            // opacity 0 keeps requesting tiles when the viewport changes.
            // Without first-fire-only semantics a paused animation would
            // fast-forward on pan and a live one would skip frames.
            // `load` and `tileerror` can also both fire for the same layer
            // when some tiles error and others succeed; the settled flag
            // stops the second one from double-counting.
            var settled = false;
            var waiter = setTimeout(finish, 25000); // safety net: never stall the
                                                    // engine on a dead tile host
            function finish() {
                if (settled) return;
                settled = true;
                clearTimeout(waiter);
                handle.off('load', finish);
                handle.off('tileerror', finish);
                setTimeout(cb, 0);   // run engine teardown outside Leaflet's tile-event dispatch
            }
            handle.on('load', finish);
            handle.on('tileerror', finish);
        },

        setFrameOpacity: function (handle, v) {
            handle.setOpacity(v);
        },

        destroyFrameLayer: function (handle) {
            if (handle && map && map.hasLayer(handle)) map.removeLayer(handle);
        },

        setAlertsOverlay: function (geojsonOrNull, styleFn) {
            // Clear existing layers
            if (alertLayers && alertLayers.length) {
                for (var i = 0; i < alertLayers.length; i++) map.removeLayer(alertLayers[i]);
                alertLayers = [];
            }
            if (!geojsonOrNull || !geojsonOrNull.features) return;

            // Sort features by severity (most severe last = rendered on top by z-index).
            // Emergency > Extreme > Severe > Moderate > Minor > Unknown.
            var features = geojsonOrNull.features.slice();
            var sevOrder = { emergency: 5, extreme: 4, severe: 3, moderate: 2, minor: 1 };
            features.sort(function (a, b) {
                var sa = sevOrder[(a.properties && a.properties.severity || '').toLowerCase()] || 0;
                var sb = sevOrder[(b.properties && b.properties.severity || '').toLowerCase()] || 0;
                return sa - sb;
            });

            // Per-feature layer with severity z-index so Emergency renders on top.
            for (var i = 0; i < features.length; i++) {
                var feature = features[i];
                // Skip features without geometry (the alerts-catalog contract is "polygon or null").
                if (!feature.geometry) continue;
                var sev = (feature.properties && feature.properties.severity || 'unknown').toLowerCase();
                var zIdx = sevOrder[sev] ? sevOrder[sev] * 200 + 200 : 300;
                var fc = { type: 'FeatureCollection', features: [feature] };
                var layer = L.geoJSON(fc, {
                    pane: 'lv-alerts-pane',
                    style: function (f) { return styleFn(f); },
                    onEachFeature: function (f, lyr) {
                        // The engine pre-bakes popup HTML into properties.__popup.
                        if (f.properties && f.properties.__popup) {
                            lyr.bindPopup(f.properties.__popup);
                        }
                    }
                });
                layer.setZIndex(zIdx);
                layer.addTo(map);
                alertLayers.push(layer);
            }
        },

        flyTo: function (lat, lon, zoom) {
            map.flyTo([lat, lon], zoom);
        },

        onViewportChange: function (cb) {
            map.on('moveend', cb);
        },

        getBounds: function () {
            var b = map.getBounds();
            return { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
        }
    };
};

// === VIEWER ===
LibreWXR.createViewer({
    apiSources: LVR_API_SOURCES,
    apiFixed: LVR_API_FIXED,
    view: { lat: 39.8283, lon: -98.5795, zoom: 5, maxZoom: 12 }
}, new LeafletAdapter());
</script>

</body>
</html>
