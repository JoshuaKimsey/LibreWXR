<!-- SPDX-License-Identifier: MIT -->
<!DOCTYPE html>
<html>
<head>
    <title>LibreWXR - MapLibre Example</title>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <link href="https://unpkg.com/maplibre-gl@^6.0.0/dist/maplibre-gl.css" rel="stylesheet"/>
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

<script type="module">
// MapLibre v6 ESM: inline-event-handler targets used to have to be hoisted to
// window (module scope does not leak to the global object like classic scripts
// do - `window.changeSource = changeSource` etc. were required for onchange=""
// attributes). This build wires all controls with addEventListener from the
// engine instead, so no hoisting is needed - but if you ever reintroduce
// inline handlers in this module scope, remember the gotcha.
import * as maplibregl from 'https://unpkg.com/maplibre-gl@^6.0.0/dist/maplibre-gl.mjs';

// GeoJSON (the alerts overlay) is tiled in a Web Worker. When this page is
// opened from file:// or served cross-origin, the default worker resolution
// can be blocked by the browser's top-level worker same-origin rule, which
// silently kills GeoJSON rendering while raster tiles (basemap/radar) keep
// working. Pinning the worker URL lets MapLibre launder it through a
// same-origin blob: worker so alerts render everywhere.
maplibregl.setWorkerUrl('https://unpkg.com/maplibre-gl@^6.0.0/dist/maplibre-gl-worker.mjs');

// === API SOURCE CONFIG (build.py rewrites for --site) ===
var LVR_API_SOURCES = {
    local: 'http://localhost:8080',
    public: 'https://api.librewxr.net'
};
var LVR_API_FIXED = null;

// === MAPLIBRE ADAPTER ===
// Implements the viewer-core.js adapter contract with MapLibre GL v6.
var MaplibreAdapter = function () {
    var map = null;
    // One-shot latch: set true once the initial style has finished loading
    // (the map's single 'load' event). Do NOT poll isStyleLoaded() for this -
    // in MapLibre GL v6 it goes false whenever ANY tile source is mid-fetch,
    // so on a live radar map it is nearly always false and gating on it would
    // defer mutations onto a 'load' event that never re-fires.
    var styleReady = false;
    var maxZoom = 12;
    var frameSeq = 0;
    var popupHandler = null;

    var baseMapStyleUrls = {
        // Keyless OpenFreeMap hosted styles (vector, MapLibre-native). The
        // style JSON carries its own source attribution, rendered by
        // MapLibre's default attribution control.
        dark: 'https://tiles.openfreemap.org/styles/dark',
        light: 'https://tiles.openfreemap.org/styles/positron'
    };

    // Runtime-layer registry: the adapter's single source of truth for every
    // non-basemap source/layer. map.setStyle(url) wipes all runtime-added
    // sources and layers, so after a basemap swap restoreRuntimeStyle()
    // re-materializes them from here with their ids intact - engine handles
    // stay valid, and the stacking order is reproduced.
    var frameRegistry = [];   // runtime frame entries, creation order:
    // { sourceId, layerId, kind: 'radar'|'satellite', url, maxZoom, opacity }
    var alertsEntry = null;   // { geojson: <FeatureCollection> } when the alerts overlay is active
    var appliedStyleUrl = baseMapStyleUrls.dark;
    var desiredStyleUrl = appliedStyleUrl;
    var swapInProgress = false;

    // Layer-ID prefixes are kind-aware ('lv-layer-radar-*' vs
    // 'lv-layer-sat-*') so stacking helpers can target the right family.
    // Target order is: basemap < satellite < alerts < radar. Basemap theme
    // changes swap the whole style via setStyle(url); restoreRuntimeStyle()
    // re-materializes the runtime layers from the registry, preserving their
    // ids and this order.
    function firstRadarFrameLayerId() {
        var layers = map.getStyle().layers;
        for (var i = 0; i < layers.length; i++) {
            if (layers[i].id.indexOf('lv-layer-radar-') === 0) return layers[i].id;
        }
        return undefined;
    }

    function firstNonBasemapLayerId() {
        var layers = map.getStyle().layers;
        for (var i = 0; i < layers.length; i++) {
            if (layers[i].id.indexOf('lv-') === 0) return layers[i].id;
        }
        return undefined;
    }

    // Alerts must sit directly below the first radar frame layer and above
    // satellite/basemap no matter when layers were added. Radar frames appear
    // over time, so this is re-asserted after each radar frame creation.
    // MapLibre's moveLayer(id, beforeId) moves id to just below beforeId;
    // beforeId undefined moves it to the top (the fallback when no radar
    // frame layer exists yet). Moving fill then line both before the radar
    // layer yields [.., fill, line, radar].
    function assertAlertsPosition() {
        if (!map.getLayer('lv-alerts-fill') || !map.getLayer('lv-alerts-line')) return;
        var radar = firstRadarFrameLayerId();
        // v6 moveLayer(id, undefined) appends to the top without throwing, and
        // a missing beforeId fires a non-fatal map 'error' event rather than an
        // exception - but be explicit about the no-radar case (single-arg
        // moveLayer = move to top) and keep any unexpected throw from here
        // logged instead of fatal to the render pipeline.
        try {
            if (radar) map.moveLayer('lv-alerts-fill', radar);
            else map.moveLayer('lv-alerts-fill');
        } catch (e) {
            console.error('LibreWXR alerts: moveLayer(lv-alerts-fill) failed', e);
        }
        try {
            if (radar) map.moveLayer('lv-alerts-line', radar);
            else map.moveLayer('lv-alerts-line');
        } catch (e) {
            console.error('LibreWXR alerts: moveLayer(lv-alerts-line) failed', e);
        }
    }

    // Re-materialize one frame entry on the current style. Mirrors
    // createFrameLayer's addToMap mutations, except the initial raster-opacity
    // is the engine's last-set value from the registry (0 for new frames).
    // Idempotent: skips sources/layers that already exist, so it can run after
    // a style swap (or twice, belt-and-braces) without duplicate-source errors.
    function addFrameEntry(entry) {
        if (!map.getSource(entry.sourceId)) {
            map.addSource(entry.sourceId, {
                type: 'raster',
                tiles: [entry.url],
                tileSize: 256,
                maxzoom: entry.maxZoom
            });
        }
        if (map.getLayer(entry.layerId)) return;
        // beforeId keeps satellite frames below radar frames: satellite
        // inserts above the basemap/alerts, radar appends at the top.
        var beforeId = entry.kind === 'satellite' ? firstNonBasemapLayerId() : undefined;
        map.addLayer({
            id: entry.layerId,
            type: 'raster',
            source: entry.sourceId,
            paint: {
                'raster-opacity': entry.opacity, // restored at the engine's last-set opacity
                'raster-opacity-transition': { duration: 0, delay: 0 },
                'raster-fade-duration': 0
            }
        }, beforeId);
        // A radar frame added later lands on top of everything; pull
        // the alerts back up directly beneath it so they never end up
        // hidden under a raster layer. Satellite frames stay below
        // alerts, so only radar-kind frames re-assert.
        if (entry.kind !== 'satellite') assertAlertsPosition();
    }

    // Materialize the alerts overlay (source + fill + line) on the current
    // style. Idempotent, so it serves both the normal setAlertsOverlay path
    // and post-swap restoreRuntimeStyle() - the source is re-added from the
    // registry when a freshly swapped-in style has no 'lv-alerts' source yet.
    function addAlertsLayers() {
        if (map.getLayer('lv-alerts-fill')) return;
        if (!map.getSource('lv-alerts') && alertsEntry) {
            map.addSource('lv-alerts', {
                type: 'geojson',
                data: alertsEntry.geojson
            });
        }
        // Above satellite frames, below radar frames. Must anchor on the
        // first RADAR layer - anchoring on any frame layer would put the
        // alerts under the satellite stack in satellite-only mode.
        var beforeId = firstRadarFrameLayerId();
        map.addLayer({
            id: 'lv-alerts-fill',
            type: 'fill',
            source: 'lv-alerts',
            paint: {
                // Colors mirror the CSS --alert-* tokens in viewer.css.
                'fill-color': ['match', ['get', 'severity'], 'Extreme', '#d50000', 'Severe', '#ff6d00', 'Moderate', '#ffb300', 'Minor', '#8e24aa', '#546e7a'],
                'fill-opacity': 0.18,
                // Higher severity sorts on top within the single fill
                // layer (MapLibre v6 fill-sort-key; Emergency > Extreme,
                // etc.). Unknown severities fall through to 0.
                'fill-sort-key': ['match', ['get', 'severity'],
                    'Emergency', 5,
                    'Extreme', 4,
                    'Severe', 3,
                    'Moderate', 2,
                    'Minor', 1,
                    0
                ]
            }
        }, beforeId);
        map.addLayer({
            id: 'lv-alerts-line',
            type: 'line',
            source: 'lv-alerts',
            paint: {
                // Colors mirror the CSS --alert-* tokens in viewer.css.
                'line-color': ['match', ['get', 'severity'], 'Extreme', '#d50000', 'Severe', '#ff6d00', 'Moderate', '#ffb300', 'Minor', '#8e24aa', '#546e7a'],
                'line-width': 2
            }
        }, beforeId);
    }

    // Register (once) the alerts click-popup handler. Safe to call repeatedly:
    // no-ops when a handler exists or the overlay is absent. Survives style
    // swaps only if registered after the swap - restoreRuntimeStyle calls
    // this to cover overlay adds that happened mid-swap.
    function registerAlertsPopup() {
        if (popupHandler || !alertsEntry) return;
        popupHandler = function (e) {
            var hits = map.queryRenderedFeatures(e.point, { layers: ['lv-alerts-fill'] });
            var feature = hits && hits[0];
            if (!feature || !feature.properties || !feature.properties.__popup) return;
            new maplibregl.Popup({ closeButton: true, maxWidth: '300px' })
                .setLngLat(e.lngLat)
                .setHTML(feature.properties.__popup)
                .addTo(map);
        };
        map.on('click', popupHandler);
    }

    // Swap the whole style to `url` (a keyless OpenFreeMap style URL). A style
    // swap wipes every runtime-added source/layer, so the registry restore
    // happens once the swapped-in style is usable.
    function swapToStyle(url) {
        swapInProgress = true;
        var finished = false;
        var finish = function () {
            if (finished) return;
            try {
                restoreRuntimeStyle(url);
                finished = true;
            } catch (e) {
                // The restore can only throw if the style is still loading
                // (e.g. the fallback timer fired early). Retry; every add in
                // the restore is idempotent.
                setTimeout(finish, 2000);
            }
        };
        // MapLibre re-fires 'load' once the swapped-in style is usable.
        map.once('load', finish);
        // Belt-and-braces: if 'load' never arrives (stalled style fetch),
        // attempt the restore anyway - every add is idempotent.
        setTimeout(finish, 10000);
        map.setStyle(url);
    }

    // Re-materialize every runtime layer on the freshly swapped-in style.
    // Ids are preserved, so engine-side handles stay valid. Canonical stack
    // order (basemap < satellite < alerts < radar) is reproduced by adding
    // satellites, then alerts, then radar frames, in creation order each.
    function restoreRuntimeStyle(url) {
        appliedStyleUrl = url;
        var i, entry;
        for (i = 0; i < frameRegistry.length; i++) {
            entry = frameRegistry[i];
            if (entry.kind === 'satellite') addFrameEntry(entry);
        }
        if (alertsEntry) { addAlertsLayers(); registerAlertsPopup(); }
        for (i = 0; i < frameRegistry.length; i++) {
            entry = frameRegistry[i];
            if (entry.kind !== 'satellite') addFrameEntry(entry);
        }
        assertAlertsPosition();
        swapInProgress = false;
        console.log('LibreWXR maplibre: basemap style swapped,', frameRegistry.length, 'runtime layer(s) restored');
        if (desiredStyleUrl !== appliedStyleUrl) swapToStyle(desiredStyleUrl);
    }

    // The adapter object is assigned to a var so methods can re-invoke
    // themselves from deferred ('load'-event) callbacks - a bare-name call
    // inside an object-literal method would be a ReferenceError.
    var adapter = {
        // MapLibre cannot add sources/layers until the style has loaded, so
        // the engine defers its boot until this fires. Also set the styleReady
        // latch here (in both the already-loaded and the deferred branch) so
        // every mutation gate agrees on when the style is usable.
        onMapReady: function (cb) {
            if (map.isStyleLoaded()) {
                styleReady = true;
                cb();
            } else {
                map.once('load', function () {
                    styleReady = true;
                    cb();
                });
            }
        },

        createMap: function (containerId, view) {
            maxZoom = view.maxZoom;
            map = new maplibregl.Map({
                container: containerId,
                style: baseMapStyleUrls.dark,
                center: [view.lon, view.lat],
                zoom: view.zoom,
                maxZoom: view.maxZoom,
                // Allow tile fetches from file:// origin (CORS with null origin)
                transformRequest: function (url) {
                    return { url: url, credentials: 'omit' };
                }
            });
            map.addControl(new maplibregl.NavigationControl());
            // The one-shot 'load' event fires when the initial style is usable.
            // After that, styleReady stays true permanently, so the mutation
            // gates below pass immediately even while tiles keep loading.
            map.once('load', function () { styleReady = true; });
            return map;
        },

        setBasemap: function (theme) {
            desiredStyleUrl = baseMapStyleUrls[theme] || baseMapStyleUrls.dark;
            if (!styleReady) {
                // Pre-first-load: defer exactly like the other adapter methods.
                map.once('load', function () { adapter.setBasemap(theme); });
                return;
            }
            if (desiredStyleUrl === appliedStyleUrl && !swapInProgress) return;
            if (swapInProgress) return; // in-flight swap completion chains to desiredStyleUrl
            swapToStyle(desiredStyleUrl);
        },

        createFrameLayer: function (url, kind) {
            var id = 'lv-frame-' + (frameSeq++);
            var sourceId = 'lv-src-' + id;
            var layerId = kind === 'satellite' ? 'lv-layer-sat-' + id : 'lv-layer-radar-' + id;
            var entry = { sourceId: sourceId, layerId: layerId, kind: kind, url: url, maxZoom: maxZoom, opacity: 0 };
            frameRegistry.push(entry);
            // The engine defers frame creation until the style has loaded, but
            // guard anyway: addSource/addLayer would throw "Style is not done
            // loading" if this ever runs early. The handle (ids) must still be
            // returned synchronously for the engine's layer bookkeeping, so
            // only the map mutation is deferred - re-invoking the whole method
            // would mint a second, orphaned source/layer.
            var addToMap = function () {
                map.addSource(sourceId, {
                    type: 'raster',
                    tiles: [entry.url],
                    tileSize: 256,
                    maxzoom: entry.maxZoom
                });
                // beforeId keeps satellite frames below radar frames: satellite
                // inserts above the basemap/alerts, radar appends at the top.
                var beforeId = kind === 'satellite' ? firstNonBasemapLayerId() : undefined;
                map.addLayer({
                    id: layerId,
                    type: 'raster',
                    source: sourceId,
                    paint: {
                        'raster-opacity': entry.opacity, // created hidden; the engine fades in on ready
                        'raster-opacity-transition': { duration: 0, delay: 0 },
                        'raster-fade-duration': 0
                    }
                }, beforeId);
                // A radar frame added later lands on top of everything; pull
                // the alerts back up directly beneath it so they never end up
                // hidden under a raster layer. Satellite frames stay below
                // alerts, so only radar-kind frames re-assert.
                if (kind !== 'satellite') assertAlertsPosition();
            };
            if (!styleReady) map.once('load', addToMap);
            else if (!swapInProgress) addToMap();
            // While a style swap is in flight the map takes no mutations;
            // restoreRuntimeStyle() materializes this entry from the registry.
            return { sourceId: sourceId, layerId: layerId };
        },

        onLayerReady: function (handle, cb) {
            var done = false;
            var waiter = null;

            function cleanup() {
                map.off('sourcedata', onData);
                map.off('error', onError);
                if (waiter) clearTimeout(waiter);
            }
            function finish() {
                if (done) return;
                done = true;
                cleanup();
                cb();
            }
            function onData(e) {
                if (e.sourceId === handle.sourceId && map.isSourceLoaded(handle.sourceId)) finish();
            }
            function onError(e) {
                // Tile/source failures arrive as map-level 'error' events.
                // Treat any error tied to our source as "settled" so the
                // animation does not hang on a dead tile host.
                if (e.sourceId === handle.sourceId || (e.error && e.error.sourceId === handle.sourceId)) {
                    // Surface the failure in the browser console - the engine
                    // settles on this event, so without this a dead tile host
                    // is silent (MapLibre just renders empty tiles).
                    console.warn('[librewxr] map tile/source error:', e.error ? (e.error.message || e.error) : e);
                    finish();
                }
            }
            // Leak-safe: the sourcedata listener must be removed on success,
            // on error, AND on timeout - a never-loading source must not leave
            // a handler behind.
            map.on('sourcedata', onData);
            map.on('error', onError);
            waiter = setTimeout(finish, 25000);
            // Catch-up: if the source finished loading before we attached
            // (e.g. raced with 'idle'), don't wait for an event that will
            // never come.
            if (map.isSourceLoaded(handle.sourceId)) finish();
        },

        setFrameOpacity: function (handle, v) {
            for (var i = 0; i < frameRegistry.length; i++) {
                if (frameRegistry[i].layerId === handle.layerId) frameRegistry[i].opacity = v;
            }
            if (map.getLayer(handle.layerId)) {
                map.setPaintProperty(handle.layerId, 'raster-opacity', v);
            }
        },

        destroyFrameLayer: function (handle) {
            if (!handle) return;
            for (var i = 0; i < frameRegistry.length; i++) {
                if (frameRegistry[i].layerId === handle.layerId) { frameRegistry.splice(i, 1); break; }
            }
            if (map.getLayer(handle.layerId)) map.removeLayer(handle.layerId);
            if (map.getSource(handle.sourceId)) map.removeSource(handle.sourceId);
        },

        setAlertsOverlay: function (geojsonOrNull, styleFn) {
            // addSource/addLayer throw "Style is not done loading" if the
            // style has not finished loading yet (same guard as setBasemap).
            // Defer until the style is ready, then re-run with the captured
            // arguments so an early toggle click can't silently drop the
            // overlay. Gate on the one-shot styleReady latch, NOT on
            // isStyleLoaded(): in MapLibre GL v6 isStyleLoaded() goes false
            // whenever ANY tile source is mid-fetch (nearly always on a live
            // radar map), and map.once('load') never fires again once 'load'
            // has passed - so an isStyleLoaded()-based deferral can be a
            // permanent silence. styleReady flips true at the initial 'load'
            // and stays true, so once the map is up this guard passes
            // immediately and the overlay renders synchronously.
            try {
                if (!styleReady) {
                    var gj = geojsonOrNull, sf = styleFn;
                    map.once('load', function () {
                        adapter.setAlertsOverlay(gj, sf);
                    });
                    return;
                }

                if (map.getLayer('lv-alerts-fill')) map.removeLayer('lv-alerts-fill');
                if (map.getLayer('lv-alerts-line')) map.removeLayer('lv-alerts-line');
                if (map.getSource('lv-alerts')) map.removeSource('lv-alerts');
                if (popupHandler) {
                    map.off('click', popupHandler);
                    popupHandler = null;
                }
                if (!geojsonOrNull) {
                    alertsEntry = null;
                    return;
                }

                var features = [];
                var list = geojsonOrNull.features || [];
                for (var i = 0; i < list.length; i++) {
                    var f = list[i];
                    if (!f.geometry) continue; // "polygon or null" contract: skip nulls
                    features.push({ type: 'Feature', geometry: f.geometry, properties: f.properties });
                }

                alertsEntry = { geojson: { type: 'FeatureCollection', features: features } };
                // A style swap is in flight: the map takes no mutations now;
                // restoreRuntimeStyle() materializes the overlay afterwards.
                if (swapInProgress) return;

                map.addSource('lv-alerts', {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: features }
                });

                addAlertsLayers();
                // Two addLayer calls anchored on the same beforeId reverse their
                // relative order (line ends up below fill), so re-assert the
                // intended stack: fill directly below line, both directly below
                // the first radar frame layer.
                assertAlertsPosition();
                console.log('LibreWXR alerts: added overlay,', features.length, 'renderable features');

                // Probe rendered-vs-source feature counts once the map goes
                // idle. rendered > 0 means the layer draws; rendered == 0 with
                // source > 0 means a projection/filter/render issue; source == 0
                // means the geojson never made it into the source tiles.
                map.once('idle', function () {
                    try {
                        var rendered = map.queryRenderedFeatures({ layers: ['lv-alerts-fill'] });
                        var inSource = map.querySourceFeatures('lv-alerts');
                        console.log('LibreWXR alerts: probe - rendered features in viewport:', rendered.length, ', source features:', inSource.length);
                    } catch (e) { console.error('LibreWXR alerts: probe failed', e); }
                });

                // Click popup: the engine pre-bakes popup HTML into
                // properties.__popup before the overlay is handed over.
                registerAlertsPopup();
            } catch (e) {
                console.warn('LibreWXR alerts: setAlertsOverlay threw at some step', e);
                throw e;
            }
        },

        flyTo: function (lat, lon, zoom) {
            map.flyTo({ center: [lon, lat], zoom: zoom });
        },

        onViewportChange: function (cb) {
            map.on('moveend', cb);
        },

        getBounds: function () {
            var b = map.getBounds();
            return { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
        }
    };
    return adapter;
};

// === VIEWER ===
LibreWXR.createViewer({
    apiSources: LVR_API_SOURCES,
    apiFixed: LVR_API_FIXED,
    alertsFileWarning: window.location.protocol === 'file:',
    view: { lat: 39.8283, lon: -98.5795, zoom: 4, maxZoom: 12 },
    nowMarker: true,        // red line on the scrubber marking the current wall-clock time
    nowMarkerLabel: true    // small current-time label above the marker
}, new MaplibreAdapter());
</script>

</body>
</html>
