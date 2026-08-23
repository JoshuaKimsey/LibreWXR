// ==========================================================================
// Glue script 1: query-param parsing + widget globals (BEFORE the engine).
// ==========================================================================
var PARAMS = new URLSearchParams(window.location.search);

function paramNum(name, fallback) {
  var v = parseFloat(PARAMS.get(name));
  return isNaN(v) ? fallback : v;
}

// Contract defaults: lat/lon (0), zoom (7 clamped 3..12), server, hour12,
// locale, strings, bglist, bg, layer, color (rounded 0..12), arrows, theme.
var LAT = paramNum('lat', 0);
var LON = paramNum('lon', 0);
var INIT_ZOOM = Math.max(3, Math.min(12, paramNum('zoom', 7)));
var API_BASE = PARAMS.get('server') || 'https://api.librewxr.net';
var HOUR12 = PARAMS.get('hour12') === '1';
var SYS_LOCALE = PARAMS.get('locale') || undefined;

var STRINGS = {};
try { STRINGS = JSON.parse(PARAMS.get('strings') || '{}'); } catch (e) {}

function tr(key, fallback) {
  return STRINGS[key] || fallback;
}

var BG_LIST = [];
try { BG_LIST = JSON.parse(PARAMS.get('bglist') || '[]'); } catch (e) {}

// "auto" resolves against the current map theme; an explicit id stays put.
function bgEntry(id) {
  for (var i = 0; i < BG_LIST.length; i++) {
    if (BG_LIST[i].id === id) return BG_LIST[i];
  }
  return null;
}

// Fallback tiles for the (unexpected) case where QML sent no choices at all.
var FALLBACK_BG = {
  dark: {
    id: 'auto',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a> | <a href="https://librewxr.net/">LibreWXR</a>'
  },
  light: {
    id: 'auto',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors | <a href="https://librewxr.net/">LibreWXR</a>'
  }
};

function resolveBg(id, theme) {
  if (id === 'auto') {
    var themed = bgEntry(theme === 'dark' ? 'carto-darkmatter' : 'osm-standard');
    return themed || FALLBACK_BG[theme] || FALLBACK_BG.dark;
  }
  return bgEntry(id) || FALLBACK_BG[theme] || FALLBACK_BG.dark;
}

var ARROWS_ON = PARAMS.get('arrows') === '1';
var ACTIVE_LAYER = PARAMS.get('layer') || 'radar';   // 'radar' | 'satellite' | 'both'
var CELLS_INIT = PARAMS.get('cells') || '';
var ALERTS_INIT = PARAMS.get('alerts') === '1';
var CURRENT_COLOR = Math.max(0, Math.min(12, Math.round(paramNum('color', 10))));
var ACTIVE_THEME = PARAMS.get('theme') === 'light' ? 'light' : 'dark';
var BG_CURRENT = PARAMS.get('bg') || 'auto';
