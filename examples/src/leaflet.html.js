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
    <select id="lv-basemap" aria-label="Base map">
        <option value="auto">Map: Auto</option>
        <option value="osm-standard">Map: OSM Standard</option>
        <option value="osm-humanitarian">Map: OSM Humanitarian</option>
        <option value="cyclosm">Map: CyclOSM</option>
        <option value="opentopomap">Map: OpenTopoMap</option>
        <option value="carto-positron">Map: CARTO Positron</option>
        <option value="carto-darkmatter">Map: CARTO Dark Matter</option>
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
    <label><input type="checkbox" id="lv-snow" checked/> Snow mask</label>
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
    var lvTileErrors = 0; // module-level tile-load failure counter (browser diagnostics)
    var maxZoom = 12;
    // === BASE MAPS ===
    // Explicit choices for the base-map selector; "auto" follows the theme
    // (dark maps get Dark Matter, light maps get OSM Standard).
    var BASEMAPS = {
        'osm-standard': {
            url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
            maxZoom: 19
        },
        'osm-humanitarian': {
            url: 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors <a href="https://www.hotosm.org/">Humanitarian OSM Team</a>',
            maxZoom: 20
        },
        'cyclosm': {
            url: 'https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png',
            attribution: '<a href="https://github.com/cyclosm/cyclosm-cartocss-style/releases">CyclOSM</a> | <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
            maxZoom: 20
        },
        'opentopomap': {
            url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attribution: '<a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
            maxZoom: 17
        },
        'carto-positron': {
            url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
            maxZoom: 20
        },
        'carto-darkmatter': {
            url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
            maxZoom: 20
        }
    };
    var AUTO_BASEMAPS = {
        dark: BASEMAPS['carto-darkmatter'],
        light: BASEMAPS['osm-standard']
    };
    var basemapChoice = 'auto';
    var currentBaseMap = null;
    var alertLayers = [];
    var alertClickCb = null;

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
            var entry = (basemapChoice !== 'auto' && BASEMAPS[basemapChoice])
                || AUTO_BASEMAPS[theme]
                || AUTO_BASEMAPS.dark;
            currentBaseMap = L.tileLayer(entry.url, {
                attribution: entry.attribution,
                maxNativeZoom: entry.maxZoom || 19,
                maxZoom: 19
            });
            currentBaseMap.addTo(map);
            currentBaseMap.bringToBack();
        },
        // Shell-side extension: choose an explicit base map ('auto' follows the theme).
        setBasemapChoice: function (id) {
            if (id === 'auto' || BASEMAPS[id]) basemapChoice = id;
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
                    // Make tile-load failures visible from the browser console:
                    // Leaflet swaps failed tiles for a 1x1 placeholder and the
                    // engine treats tileerror as settled, so without this a
                    // dead tile host is silent.
                    console.warn('[librewxr] tile load failed:', url);
                    lvTileErrors++;
                    if (!layer._tiles || !layer._tiles[key]) return;
                    if (done) done(new Error('Tile load failed'), tile);
                    layer.off('remove', abortTile);
                };
                var abortTile = function () {
                    aborted = true;
                    layer.off('remove', abortTile);
                    // Do NOT use tile.src = '': assigning an empty src makes the
                    // browser fetch the page's own URL as an image, which on
                    // file:// pages logs a spurious "Unsafe attempt to load
                    // URL file://..." console error.
                    tile.removeAttribute('src');
                    tile.removeEventListener('load', onLoad);
                    tile.removeEventListener('error', onError);
                };
                layer.on('remove', abortTile);
                tile.addEventListener('load', onLoad);
                tile.addEventListener('error', onError);
                var url = this.getTileUrl(coords);
                tile.src = url;
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
            // Persistent logging-only listener: records tile failures so a
            // broken tile host is visible in the browser console. It does NOT
            // settle the handoff - the first of load/tileerror still finishes
            // the layer handoff exactly once via `finish` above, and this
            // listener only logs (and stays attached for the layer's lifetime
            // so later pan-in failures are caught too).
            handle.on('tileerror', function (e) {
                var failedUrl = (e && e.tile && e.tile.src) ? e.tile.src : handle._url;
                console.warn('[librewxr] tile load failed:', failedUrl);
            });
        },

        setFrameOpacity: function (handle, v) {
            handle.setOpacity(v);
        },

        destroyFrameLayer: function (handle) {
            if (handle && map && map.hasLayer(handle)) map.removeLayer(handle);
        },

        onAlertClick: function (cb) { alertClickCb = cb; },

        setAlertsOverlay: function (geojsonOrNull, styleFn, reopenId) {
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
            // Paths are tracked by uri so a clicked popup can be reopened after
            // the overlay is rebuilt (e.g. the 5-minute refresh cadence).
            var pathsByUri = {};
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
                            lyr.bindPopup(f.properties.__popup, {
                                maxWidth: 320,
                                autoPan: true,
                                autoPanPadding: [20, 20],
                                closeOnClick: true
                            });
                        }
                        // Let the engine know which alert was clicked so the
                        // autoPan viewport change doesn't close the popup.
                        lyr.on('click', function (e) {
                            L.DomEvent.stopPropagation(e);
                            if (alertClickCb) {
                                alertClickCb(f.properties && (f.properties.uri || f.properties.title));
                            }
                        });
                        if (f.properties && f.properties.uri) {
                            pathsByUri[f.properties.uri] = lyr;
                        }
                    }
                });
                layer.setZIndex(zIdx);
                layer.addTo(map);
                alertLayers.push(layer);
            }

            // Reopen the popup the user had open before the rebuild, if the
            // alert is still in view.
            if (reopenId && pathsByUri[reopenId]) {
                var reopenPath = pathsByUri[reopenId];
                setTimeout(function () {
                    try {
                        if (reopenPath._map) reopenPath.openPopup();
                    } catch (e) { /* map or layer gone - fine */ }
                }, 0);
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
var adapter = new LeafletAdapter();
LibreWXR.createViewer({
    apiSources: LVR_API_SOURCES,
    apiFixed: LVR_API_FIXED,
    view: { lat: 39.8283, lon: -98.5795, zoom: 5, maxZoom: 12 },
    nowMarker: true,        // red line on the scrubber marking the current wall-clock time
    nowMarkerLabel: true    // small current-time label above the marker
}, adapter);

// === BASE MAP SELECTOR ===
// Engine theme switches call adapter.setBasemap(theme), which re-resolves the
// current choice, so only user picks need handling here.
(function () {
    var sel = document.getElementById('lv-basemap');
    if (!sel) return;
    sel.addEventListener('change', function () {
        adapter.setBasemapChoice(this.value);
        adapter.setBasemap(document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark');
    });
})();
</script>

</body>
</html>
