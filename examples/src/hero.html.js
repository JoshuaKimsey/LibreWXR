<!-- SPDX-License-Identifier: MIT -->
<!DOCTYPE html>
<html>
<head>
    <title>LibreWXR - Live Radar</title>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <link href="https://unpkg.com/leaflet/dist/leaflet.css" rel="stylesheet"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <style>
        /*__VIEWER_CSS__*/
    </style>
</head>
<body data-theme="dark">

<!-- Map (hero: no toolbar, no options panel - locked config below) -->
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
// === API SOURCE CONFIG (auto-detect local vs public; no selector in hero) ===
var LVR_API_SOURCES = {
    local: 'http://localhost:8080',
    public: 'https://api.librewxr.net'
};
var LVR_API_FIXED = 'https://api.librewxr.net';

// === LEAFLET ADAPTER (hero is Leaflet-based) ===
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
    var alertsLayer = null;

    function paneZ(name, fallback) {
        var val = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
        return isNaN(val) ? fallback : val;
    }

    return {
        onMapReady: function (cb) { cb(); },

        createMap: function (containerId, view) {
            maxZoom = view.maxZoom;
            map = L.map(containerId, { maxZoom: view.maxZoom, zoomControl: true })
                .setView([view.lat, view.lon], view.zoom);
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
                tileSize: 256, // 512-url HiDPI trick (see leaflet shell)
                opacity: 0,    // created hidden; the engine fades in on ready
                maxZoom: maxZoom,
                pane: pane
            });
            layer.createTile = function (coords, done) {
                var tile = document.createElement('img');
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
            // First-fire-only: Leaflet re-fires `load` on every viewport change
            // for layers still attached to the map; a lingering handler would
            // fast-forward paused animations / skip frames in live ones.
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
            if (alertsLayer) {
                map.removeLayer(alertsLayer);
                alertsLayer = null;
            }
            if (!geojsonOrNull) return;
            alertsLayer = L.geoJSON(geojsonOrNull, {
                pane: 'lv-alerts-pane',
                style: function (feature) { return styleFn(feature); },
                onEachFeature: function (feature, layer) {
                    if (feature.properties && feature.properties.__popup) {
                        layer.bindPopup(feature.properties.__popup);
                    }
                }
            }).addTo(map);
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

// === VIEWER (locked hero configuration) ===
LibreWXR.createViewer({
    apiSources: LVR_API_SOURCES,
    apiFixed: LVR_API_FIXED,
    view: { lat: 33.749, lon: -84.388, zoom: 7, maxZoom: 12 }, // Atlanta
    layerMode: 'radar',   // radar only - satellite/nowcast split still animates
    colorScheme: 10,      // default scheme
    smooth: true,         // 1_1 webp tiles
    snow: false,
    format: 'webp',
    tileSize: 512,        // fixed hi-res tiles
    theme: 'dark',        // CARTO dark basemap
    alerts: true,         // alerts overlay on by default (subtle styling below)
    alertsFillAlpha: 0.10, // hero keeps alerts visually quieter
    autoplay: true        // start playback once the catalog loads
}, new LeafletAdapter());
</script>

</body>
</html>
