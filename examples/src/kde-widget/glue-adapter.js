// ==========================================================================
// Glue script 2: Leaflet adapter (adapted from examples/src/leaflet.html.js).
// Implements the viewer-core.js adapter contract with Leaflet 1.x primitives,
// wired to the widget's query-param basemap selection.
// ==========================================================================
var WidgetLeafletAdapter = function () {
  var map = null;
  var maxZoom = 12;
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

      // Widget location pin: marker for the location configured in the widget.
      L.marker([view.lat, view.lon], { interactive: false }).addTo(map);

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
      // WIDGET PATCH: resolve via the widget's BG_LIST / FALLBACK_BG instead of
      // a fixed per-theme tile layer, so QML-picked backgrounds are honored.
      var entry = resolveBg(BG_CURRENT, theme);
      if (currentBaseMap) map.removeLayer(currentBaseMap);
      currentBaseMap = L.tileLayer(entry.url, {
        attribution: entry.attribution,
        maxNativeZoom: entry.maxZoom || 19,
        maxZoom: 19
      });
      currentBaseMap.addTo(map);
      currentBaseMap.bringToBack();
      // /WIDGET PATCH
    },

    createFrameLayer: function (url, kind) {
      var pane = kind === 'satellite' ? 'lv-satellite-pane' : 'lv-radar-pane';
      var layer = new L.TileLayer(url, {
        tileSize: 256,
        opacity: 0, // created hidden; the engine fades in on ready
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
          if (!layer._tiles || !layer._tiles[key]) return;
          if (done) done(null, tile);
          layer.off('remove', abortTile);
        };
        var onError = function () {
          tile.removeEventListener('load', onLoad);
          tile.removeEventListener('error', onError);
          if (aborted) return;
          console.warn('[librewxr] tile load failed:', url);
          if (!layer._tiles || !layer._tiles[key]) return;
          if (done) done(new Error('Tile load failed'), tile);
          layer.off('remove', abortTile);
        };
        var abortTile = function () {
          aborted = true;
          layer.off('remove', abortTile);
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
      layer.addTo(map);
      return layer;
    },

    onLayerReady: function (handle, cb) {
      var settled = false;
      var waiter = setTimeout(finish, 25000);
      function finish() {
        if (settled) return;
        settled = true;
        clearTimeout(waiter);
        handle.off('load', finish);
        handle.off('tileerror', finish);
        setTimeout(cb, 0);
      }
      handle.on('load', finish);
      handle.on('tileerror', finish);
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
      if (alertLayers && alertLayers.length) {
        for (var i = 0; i < alertLayers.length; i++) map.removeLayer(alertLayers[i]);
        alertLayers = [];
      }
      if (!geojsonOrNull || !geojsonOrNull.features) return;

      var features = geojsonOrNull.features.slice();
      var sevOrder = { emergency: 5, extreme: 4, severe: 3, moderate: 2, minor: 1 };
      features.sort(function (a, b) {
        var sa = sevOrder[(a.properties && a.properties.severity || '').toLowerCase()] || 0;
        var sb = sevOrder[(b.properties && b.properties.severity || '').toLowerCase()] || 0;
        return sa - sb;
      });

      var pathsByUri = {};
      for (var i = 0; i < features.length; i++) {
        var feature = features[i];
        if (!feature.geometry) continue;
        var sev = (feature.properties && feature.properties.severity || 'unknown').toLowerCase();
        var zIdx = sevOrder[sev] ? sevOrder[sev] * 200 + 200 : 300;
        var fc = { type: 'FeatureCollection', features: [feature] };
        var layer = L.geoJSON(fc, {
          pane: 'lv-alerts-pane',
          style: function (f) { return styleFn(f); },
          onEachFeature: function (f, lyr) {
            if (f.properties && f.properties.__popup) {
              lyr.bindPopup(f.properties.__popup, {
                maxWidth: 320,
                autoPan: true,
                autoPanPadding: [20, 20],
                closeOnClick: true
              });
            }
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
      map.flyTo([lat, lon], (zoom == null) ? map.getZoom() : zoom);
    },

    onViewportChange: function (cb) {
      map.on('moveend', function () {
        // Keep the engine callback (alerts refilter + background preload).
        if (cb) cb();
        // WIDGET PATCH: report the viewport zoom to QML via the page title.
        // (No "bg:" reports are emitted, per the widget contract.)
        document.title = 'zoom:' + map.getZoom();
        // /WIDGET PATCH
      });
    },

    getBounds: function () {
      var b = map.getBounds();
      return { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
    },

    // WIDGET PATCH: expose the Leaflet map for fixViewport().
    getMap: function () { return map; }
    // /WIDGET PATCH
  };
};
