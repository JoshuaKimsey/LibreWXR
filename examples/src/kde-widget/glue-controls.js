// ==========================================================================
// Glue script 3: create the viewer and expose the QML-facing window.* API.
// ==========================================================================
var manualArrowsTouched = false;
var _widgetAdapter = new WidgetLeafletAdapter();

var config = {
  apiFixed: API_BASE,
  strings: STRINGS,
  view: { lat: LAT, lon: LON, zoom: INIT_ZOOM, maxZoom: 12 },
  layerMode: ACTIVE_LAYER,                      // 'radar' | 'satellite' | 'both'
  colorScheme: CURRENT_COLOR,                   // clamped 0..12
  arrows: (ARROWS_ON ? (ACTIVE_THEME === 'dark' ? 'light' : 'dark') : ''),
  theme: ACTIVE_THEME,                          // 'dark' | 'light'
  locateMode: 'view',
  nowMarker: true,
  nowMarkerLabel: true,
  locale: SYS_LOCALE,
  hour12: HOUR12,
  onThemeChange: function (theme) {
    // WIDGET PATCH: keep the arrows dropdown following the active theme unless
    // the user has manually overridden it.
    if (ARROWS_ON && !manualArrowsTouched) {
      var a = document.getElementById('lv-arrows');
      if (a) {
        var v = (theme === 'dark' ? 'light' : 'dark');
        if (a.value !== v) { a.value = v; a.dispatchEvent(new Event('change')); }
      }
    }
    // /WIDGET PATCH
  }
};

// === WIDGET DROPDOWN SHIM ===
// Native <select> popups are broken inside the plasmoid's QtWebEngine view
// (they open as a white box and close instantly). Keep the real selects in the
// DOM as the state/event backbone but never let them receive clicks; render a
// custom div-based dropdown mirror for each and drive select.value + dispatch
// change events from the mirror instead.
function installDropdown(selectId) {
  var select = document.getElementById(selectId);
  if (!select) return;

  // Wrap the (hidden, layout-anchored) select and build the custom UI.
  var wrap = document.createElement('span');
  wrap.className = 'dd-wrap';
  select.parentNode.insertBefore(wrap, select);
  wrap.appendChild(select);

  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'dd-btn';
  var label = document.createElement('span');
  label.className = 'dd-label';
  var caret = document.createElement('span');
  caret.className = 'dd-caret';
  caret.textContent = '\u25BC';
  btn.appendChild(label);
  btn.appendChild(caret);
  wrap.appendChild(btn);

  var menu = document.createElement('div');
  menu.className = 'dd-menu';
  wrap.ddMenu = menu;
  // Portal the menu to <body>: the toolbar is an overflow-x:auto scroll
  // container that would clip an absolutely-positioned child, while a
  // fixed-position body child floats above the map (viewport-relative).
  document.body.appendChild(menu);

  var shimSync = false;

  function renderMenu() {
    menu.innerHTML = '';
    var i;
    for (i = 0; i < select.options.length; i++) {
      var opt = select.options[i];
      var row = document.createElement('div');
      row.className = 'dd-option';
      row.textContent = opt.text;
      row.dataset.value = opt.value;
      if (opt.selected) row.classList.add('selected');
      (function (row, opt) {
        row.addEventListener('click', function () {
          shimSync = true;
          select.value = opt.value;
          if (selectId === 'lv-arrows') {
            manualArrowsTouched = true;
          }
          select.dispatchEvent(new Event('change', { bubbles: true }));
          closeAll();
        });
      })(row, opt);
      menu.appendChild(row);
    }
    syncLabel();
    updateVisibility();
  }

  function syncLabel() {
    if (select.selectedOptions && select.selectedOptions[0]) {
      label.textContent = select.selectedOptions[0].text;
    } else {
      label.textContent = '';
    }
    var i;
    for (i = 0; i < menu.children.length; i++) {
      var row = menu.children[i];
      row.classList.toggle('selected', row.dataset.value === select.value);
    }
  }

  function updateVisibility() {
    wrap.classList.toggle('dd-hidden', select.style.display === 'none');
  }

  function openMenu() {
    closeAll();
    positionMenu(menu, btn);
  }

  function closeMenu() {
    menu.classList.remove('open');
  }

  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    if (menu.classList.contains('open')) {
      closeMenu();
    } else {
      openMenu();
    }
  });

  select.addEventListener('change', function () {
    syncLabel();
    updateVisibility();
  }, true);

  var observer = new MutationObserver(function () {
    renderMenu();
  });
  observer.observe(select, { childList: true, subtree: true });

  document.addEventListener('lvselect:sync', function () {
    syncLabel();
    renderMenu();
    updateVisibility();
  });

  document.addEventListener('click', function (e) {
    var t = e.target;
    /* WIDGET PATCH: lv-bg base-map menu (bgMenu, class dd-menu) must not be
       closed here when clicking inside it - its row clicks call closeAll()
       themselves, and clicks on the #lv-bg button are toggled by the
       button's own capture-phase listener. */
    if (t && t.closest && (t.closest('.dd-menu') === bgMenu || t.closest('#lv-bg'))) return;
    /* /WIDGET PATCH */
    var clickedWrap = t && t.closest ? t.closest('.dd-wrap') : null;
    var i;
    var open = document.querySelectorAll('.dd-menu.open');
    for (i = 0; i < open.length; i++) {
      // The clicked wrap's own menu is left open so the button handler can
      // toggle it; every other open menu closes.
      if (clickedWrap && open[i] === clickedWrap.ddMenu) continue;
      open[i].classList.remove('open');
    }
  }, true);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeAll();
  });

  // scroll does not bubble - capture catches toolbar horizontal scrolls that
  // would otherwise detach the fixed-position menu from its button.
  document.addEventListener('scroll', function (e) {
    if (e.target && e.target.closest && e.target.closest('.dd-menu')) return;
    closeAll();
  }, true);

  window.addEventListener('resize', function () {
    closeAll();
  });

  renderMenu();
  syncLabel();
  updateVisibility();
}

installDropdown('lv-layermode');
installDropdown('lv-scheme');
installDropdown('lv-arrows');
installDropdown('lv-cells');
installDropdown('lv-format');
installDropdown('lv-tilesize');

// Shared dropdown helpers (extracted from installDropdown so the base-map
// menu reuses the exact same positioning / closing behaviour).
function positionMenu(menu, btn) {
  menu.classList.add('open');
  var r = btn.getBoundingClientRect();
  menu.style.minWidth = r.width + 'px';
  menu.style.left = r.left + 'px';
  menu.style.top = (r.bottom + 2) + 'px';
  if (r.bottom + 2 + menu.offsetHeight > window.innerHeight) {
    menu.style.top = Math.max(2, r.top - 2 - menu.offsetHeight) + 'px';
  }
  if (r.left + menu.offsetWidth > window.innerWidth) {
    menu.style.left = Math.max(0, window.innerWidth - menu.offsetWidth - 4) + 'px';
  }
}

function closeAll() {
  var menus = document.querySelectorAll('.dd-menu.open');
  var i;
  for (i = 0; i < menus.length; i++) menus[i].classList.remove('open');
}

// === BASE-MAP PICKER (restored widget feature) ===
var bgMenu = document.createElement('div');
bgMenu.className = 'dd-menu';
document.body.appendChild(bgMenu);

function renderBgMenu() {
  var wrap = document.getElementById('lv-bg-wrap');
  if (!BG_LIST.length) {
    if (wrap) wrap.style.display = 'none';
    return;
  }
  if (wrap) wrap.style.display = '';
  bgMenu.innerHTML = '';
  var i;
  for (i = 0; i < BG_LIST.length; i++) {
    var entry = BG_LIST[i];
    var row = document.createElement('div');
    row.className = 'dd-option';
    row.textContent = entry.label;
    row.dataset.value = entry.id;
    if (entry.id === BG_CURRENT) row.classList.add('selected');
    (function (row, entry) {
      row.addEventListener('click', function () {
        BG_CURRENT = entry.id;
        var theme = document.body.getAttribute('data-theme') || 'dark';
        if (_widgetAdapter && typeof _widgetAdapter.setBasemap === 'function') {
          _widgetAdapter.setBasemap(theme);
        }
        document.title = 'bg:' + entry.id;
        closeAll();
        renderBgMenu();
      });
    })(row, entry);
    bgMenu.appendChild(row);
  }
  // Keep the button label in sync: label of the entry matching BG_CURRENT,
  // "Automatic" fallback.
  var label = document.querySelector('#lv-bg .dd-label');
  if (label) {
    var cur = null;
    for (i = 0; i < BG_LIST.length; i++) {
      if (BG_LIST[i].id === BG_CURRENT) { cur = BG_LIST[i]; break; }
    }
    label.textContent = cur ? cur.label : 'Automatic';
  }
}

var bgBtn = document.getElementById('lv-bg');
if (bgBtn) {
  // Capture phase: stopPropagation here prevents the document capture
  // closer from closing the menu first, so the toggle works both ways.
  bgBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    if (bgMenu.classList.contains('open')) {
      closeAll();
    } else {
      renderBgMenu();
      positionMenu(bgMenu, bgBtn);
    }
  }, true);
}

renderBgMenu();
// Because the engine never receives trusted clicks on the selects anymore, the
// glue's setColorScheme MutationObserver still works - it observes the select's
// options, and the select remains in the DOM.

window.LibreWXR.createViewer(config, _widgetAdapter);

// Track user-initiated arrows changes (ignore programmatic dispatches).
var lvArrows = document.getElementById('lv-arrows');
if (lvArrows) {
  lvArrows.addEventListener('change', function (e) {
    if (e && e.isTrusted) manualArrowsTouched = true;
  });
}

// === QML-FACING WINDOW API ===
window.setLayerMode = function (mode) {
  if (mode !== 'radar' && mode !== 'satellite' && mode !== 'both') return;
  var el = document.getElementById('lv-layermode');
  if (!el || el.value === mode) return;
  el.value = mode;
  el.dispatchEvent(new Event('change'));
};

window.setColorScheme = function (n) {
  n = Math.max(0, Math.min(12, Math.round(Number(n))));
  var el = document.getElementById('lv-scheme');
  if (!el) return;
  var found = false;
  for (var i = 0; i < el.options.length; i++) {
    if (Number(el.options[i].value) === n) { found = true; break; }
  }
  if (found) {
    if (Number(el.value) !== n) { el.value = String(n); el.dispatchEvent(new Event('change')); }
  } else {
    // Scheme options not loaded yet - stash the request and apply when the
    // catalog populates them (the engine populates #lv-scheme on load).
    window.__pendingScheme = n;
    if (!window.__schemeObserver) {
      window.__schemeObserver = new MutationObserver(function () {
        var sel = document.getElementById('lv-scheme');
        if (!sel || window.__pendingScheme == null) return;
        var want = window.__pendingScheme;
        var has = false;
        for (var i = 0; i < sel.options.length; i++) {
          if (Number(sel.options[i].value) === want) { has = true; break; }
        }
        if (has) {
          window.__pendingScheme = null;
          if (Number(sel.value) !== want) { sel.value = String(want); sel.dispatchEvent(new Event('change')); }
        }
      });
      window.__schemeObserver.observe(el, { childList: true, subtree: true });
    }
  }
};

window.setArrows = function (on) {
  ARROWS_ON = !!on;
  manualArrowsTouched = false;
  var el = document.getElementById('lv-arrows');
  if (!el) return;
  var theme = document.body.getAttribute('data-theme') || 'dark';
  var v = on ? (theme === 'dark' ? 'light' : 'dark') : '';
  if (el.value !== v) { el.value = v; el.dispatchEvent(new Event('change')); }
};

window.setTheme = function (theme) {
  if (theme !== 'light' && theme !== 'dark') return;   // invalid
  if (document.body.getAttribute('data-theme') === theme) return; // idempotent
  var btn = document.getElementById('lv-theme');
  if (btn) btn.click(); // toggles the engine theme (dark <-> light)
};

window.setBackground = function (id) {
  if (id !== 'auto') {
    var found = false;
    for (var i = 0; i < BG_LIST.length; i++) {
      if (BG_LIST[i].id === id) { found = true; break; }
    }
    if (!found) return; // invalid id
  }
  BG_CURRENT = id;
  var theme = document.body.getAttribute('data-theme') || 'dark';
  if (_widgetAdapter && typeof _widgetAdapter.setBasemap === 'function') {
    _widgetAdapter.setBasemap(theme);
  }
  renderBgMenu(); // keep the picker's active mark in sync with config-driven changes
};

window.fixViewport = function () {
  var m = _widgetAdapter ? _widgetAdapter.getMap() : null;
  if (!m) return;
  // Reproduce the old cure for the plasmoid's WebEngineView: recalculate the
  // container size and do an invisible 1px pan-and-back, which forces Leaflet
  // to reset its view and Chromium to repaint the damaged surface.
  m.invalidateSize(false);
  m.panBy([1, 0], { animate: false });
  m.panBy([-1, 0], { animate: false });
  // Timeout rearm: the engine's moveend restarts a quiet background preload,
  // which the cancelling 1px nudge (never a real view change) safely ignores.
  setTimeout(function () { m.invalidateSize(false); }, 400);
};

// === VIEWPORT FIX WIRING (map lifecycle in the ~380px embedded webview) ===
if (typeof ResizeObserver !== 'undefined') {
  var _roTimer = null;
  new ResizeObserver(function () {
    if (_roTimer) clearTimeout(_roTimer);
    _roTimer = setTimeout(window.fixViewport, 120);
  }).observe(document.getElementById('lv-map'));
}
window.addEventListener('load', function () {
  setTimeout(window.fixViewport, 250);
  setTimeout(window.fixViewport, 1200);
});

// === BUILD PROVENANCE ===
// Generated content - rebuild with: python examples/src/build.py --kde-widget (LibreWRX examples sources).
