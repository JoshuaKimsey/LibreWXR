# Lightning Implementation Plan (GOES GLM Phase 1)

Status: revised draft, 2026-10-08 (second revision — maintainer rulings incorporated). Companion to the lightning sections in docs/source-survey.md (GOES GLM = Lightning Tier 1). This document is the living decision log for the feature; the rulings section encodes the maintainer's answers so later sessions don't re-derive them. **Phase 1 landed 2026-10-09 and Phase 2 landed 2026-10-09 (Phase 1 + Phase 2 both landed; Phase 3 remains locked) — see "Phase 1 as-landed (2026-10-09)" and "Phase 2 as-landed (2026-10-09)" below; the sections above are the historical plan of record and the config block has been superseded.**

## Maintainer rulings (recorded 2026-10-08)

1. **Architecture: alerts-pattern only.** Lightning lives as data/-side infra (`data/lightning_fetcher.py` + `data/lightning_store.py`), never as a `sources/` package or a provider kind — even if FengYun or EUMETSAT point pipelines are added later. Those become additional decoder classes inside the fetcher module, or a future provider-kind promotion if the count ever justifies it.
2. **Refresh: own clock-aligned 5-minute fetch loop + shared disk artifact.** The pipeline fetches on its own clock-aligned background task every `LIBREWXR_LIGHTNING_FETCH_INTERVAL` (default 300 s; same *shape* as the alerts loop but deliberately its own knob, so disabling alerts never disables lightning and vice versa) and writes a self-contained lightning artifact under `<cache_dir>/lightning/`. Render workers read that artifact directly from disk. Lightning has **no section in `state.json`** — zero added snapshot writes. Radar overlay, MCP, and (Phase 2) REST all consume the same artifact: one fetch, no double-decode, 5-minute freshness everywhere.
3. **Watermark: yes** (per-satellite resume file beside the artifact). **Saturation: no data-side cap.** Every flash within the window is stored so MCP/REST always see the full picture. The only caps are (a) the store's RAM circuit-breaker ceiling (oldest-first trim by time, warn-once, normally unreachable) and (b) a purely presentational draw cap per tile (`LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE`) that picks strongest-by-energy glyphs for outbreak-dense tiles — it removes nothing from the store or API.
4. **Animation: yes, via sparse per-frame slot buckets.** Each flash is assigned to the 10-minute time slot it belongs to; the overlay for a frame draws exactly that slot's strikes — they appear in the frames they happened in, at constant brightness, with no age taper, and vanish as newer frames take over (drop-out is ruled by MAX_AGE). Replay comes from the overlay cache; a full recompute happens only on cache eviction or a cold start.
5. **Storm cells in stdio MCP: gap to fix.** `mcp/context.py` currently builds no storm-cell store for the stdio transport, so stdio agents get empty cell answers. Marked as a companion fix task for the Phase 1 work order; lightning itself ships with full both-transports MCP coverage. Landed 2026-10-09: `src/librewxr/mcp/context.py` now builds the store behind the `storm_cells_enabled` gate (`cleanup_tmp=False`, mirroring `main.py:467-475`), adds it to the `stores` dict and assigns `routes.storm_cell_store`, with regression coverage in `tests/test_mcp_context.py` — the lightning work inherits this exact wiring pattern for its own store.
6. **GLM schema: verify-before-trust.** Flash variable names and units are confirmed against the NCEI C01527 data dictionary and the netCDF `units` attributes in a live pre-flight decode before any wiring is trusted.
7. **Overlay parameter: `?lightning=`** with **icon-kind styles, not light/dark basemap variants**: `""` = off; `1` / `true` / `dots` = small filled dots (default); `bolts` = full bolt icon. The bolt is drawn as a **hand-authored vector polygon**, deliberately NOT a unicode/emoji glyph: server-side PIL rendering of text glyphs depends on installed fonts (no emoji in DejaVu; color-emoji fonts render unreliably or not at all in PIL), and a missing font would blank icons across self-hosted containers. The vector bolt renders identically everywhere, scales with zoom, and is toned (warm core + subtle dark outline) to read on both light and dark basemaps without needing variants.

   The specific candidate character `U+1F5F2` ("🗲", the pictograph bolt) was considered and rejected as the *drawn* overlay glyph: it lives in the Miscellaneous Symbols and Pictographs block, where server-side rendering quality is entirely a function of which fonts the server ships — and the project container (`python:3.12-slim` in the Dockerfile) installs no fonts at all, so a text-glyph bolt would be the project's first font dependency. Color-emoji fonts additionally render as fixed-palette bitmap strikes that PIL cannot recolor or cleanly scale down to tile-sized glyphs, and on any host whose fonts lack the codepoint PIL draws `.notdef` tofu boxes at every strike position instead of failing visibly. The drawn bolt therefore remains a hand-authored vector polygon — deliberately shaped to visually match `U+1F5F2`'s outline — while the literal character remains fine for client-side surfaces only (docs, README, librewxr-site, example UI text), where the visitor's browser font renders it.

8. **MAX_AGE default: 30 minutes** — matches the standard "wait 30 minutes after the last thunder" guidance, and covers three synthetic 10-minute slots. MCP `minutes=` clamps to this.
9. **`/v2/lightning` REST endpoint: yes (Phase 2).** No-args returns the full held strike window; `lat`+`lon` (+ optional `radius_km`) returns strikes within a distance; `bbox` returns all strikes in an area. The modes are not redundant: they share one underlying array filter at identical cost, and point/radius versus rectangle genuinely different use cases (personal safety vs region listing) — the same split `/v2/alerts` already serves with its `lat`/`lon` + `bbox` parameters.
10. **No age fade:** constant brightness within a slot; no taper (superseded by ruling 4's frame semantics).
11. **Attribution: README + librewxr-site only** (the site lives in the gitignored `librewxr-site/` directory — a manual companion edit at ship time). NOT in the example frontends, consistent with current practice of not attributing radar/non-map data there. Credit text: "Lightning: NOAA GOES-R Geostationary Lightning Mapper (GLM)".

## Source facts (from the survey — anchors, not discoveries)

- Buckets: anonymous NOAA NODD buckets `noaa-goes18` (137W, GOES-West) and `noaa-goes19` (75.2W, GOES-East), prefix `GLM-L2-LCFA/{YYYY}/{DDD}/{HH}/`. Files: `OR_GLM-L2-LCFA_G{nn}_s{start}_e{end}_c{created}.nc`, one per 20-second window, ~0.2-0.6 MB, landing ~20-40 s after the window ends. A `noaa-goes16` bucket exists but GOES-16 is on-orbit standby — never a primary. Mesoscale GLM variants are not ingested.
- Flash arrays live inside the netCDF `flash` group; Phase 1 consumes `flash_lat`, `flash_lon`, `flash_energy`, `flash_time_offset_of_effect_time`, filters on `flash_quality_flag`, and honors the file's `units` attribute for energy (pre-flight: verify exact names against the NCEI C01527 data dictionary — do not trust this plan's names blindly).
- License: US public domain (NODD). Attribution requested-not-required; LibreWXR modifies (reprojects, styles), so a courtesy credit line goes in README + librewxr-site per ruling 11.
- Operational hazards (all bounded): GLM file sizes balloon during severe CONUS outbreaks (handled per ruling 3 — no data loss, draw-cap only); GOES-19 suffered a full-transmission anomaly 2026-07-15/16 with GLM among the last products restored — a satellite bucket going stale is a degraded hemisphere handled like a radar missed-scan, never a crash. Coverage roughly ±54 degrees latitude per satellite (Americas/Atlantic/Pacific); the 80E-170E observation gap is a survey Phases 2/3 topic, not a Phase 1 problem.

## Architecture

### Data/-side modules (alerts precedent)

- `data/lightning_fetcher.py` — S3 listing, per-satellite watermarking, netCDF flash decode, QC, and the on-disk artifact write. Runs its own clock-aligned background loop (default every 300 s), pipeline-owned.
- `data/lightning_store.py` — `LightningStore`: sparse structured point arrays kept in pipeline memory, trimmed to `MAX_AGE`, with `save_snapshot()` / `load_snapshot()` around the on-disk artifact. **No state.json section** — the artifact IS the cross-process handoff, exactly the shared-tile-store/precip-mask pattern.
- A tiny `_run_lightning()` hook is not needed: the loop is separate, so the fetcher/cycle wiring changes are confined to `data_pipeline.py` (construct store, start loop, close on shutdown) — nothing in fetcher.py.

### Fetch loop (background task; both satellites in parallel)

1. Per configured satellite (`lightning_glm_satellites`, default `goes18,goes19`): anonymous S3 listing `?list-type=2&prefix=GLM-L2-LCFA/{YYYY}/{DDD}/&start-after=<watermark-last-key>` — hour rollover is just extra inherited keys in the same dated prefix; no clock math in steady state.
2. For each newer key `..._s{start}_e{end}_c{created}.nc` whose window-end time `e` is past the satellite watermark: plain HTTPS GET wrapped with `retry_sync` (data/retry.py), which treats a just-not-yet-published/transient miss as None-return rather than retrying forever; re-tail the listing from the last key afterward so nothing is skipped.
3. Decode with h5py: open the `flash` group arrays only (`event`/`group` arrays never materialized — Phase 1 consumes flashes only).
4. QC per ruling 3: drop flagged flashes (`flash_quality_flag` non-zero) and invalid coordinates; honor the `units` attribute. **No energy-thinning, no per-cycle cap.**
5. Merge across satellites, concatenate onto the store, trim to `[now - MAX_AGE .. now]` (oldest-first ring; the store ceiling is a RAM circuit-breaker with a warn-once log, not a data policy), then write refreshes atomically to the artifact.
6. Advance the per-satellite watermark.

Watermark persistence: per satellite `{"bucket": ..., "last_key": ..., "last_e": ts}` in `<cache_dir>/lightning/watermark.json`, atomic tmp+os.replace after each successful batch, so pipeline restarts resume rather than re-decoding hours.

Staleness handling: if a satellite's newest published key is older than ~3 fetch intervals, mark it stale in /health, warn once, keep its last-known points until they age out of MAX_AGE — no crash, no exception path.

### LightningStore (data/lightning_store.py)

- Points as one structured numpy array trimmed by time order; no memmaps needed at point-sparse scale:

```python
_POINT_DTYPE = np.dtype([
    ("time_s", "int64"),        # UTC epoch seconds
    ("lat", "float32"),
    ("lon", "float32"),
    ("energy", "float32"),      # as-read from netCDF, units recorded in meta
    ("satellite", "int8"),      # 18 / 19
])
```

- Ceiling `LIGHTNING_MAX_POINTS` (default 500k) — a pure RAM brake that cannot be disabled by config; normal 30-minute volume is tens of thousands of points (~80k at ~45 flashes/s global).
- API:
  - `replace_points(points, last_seen_s)` — atomic in-memory swap; version = write unix timestamp (monotonic at 5-minute cadence).
  - `save_snapshot(path)` / `load_snapshot(path)` — compressed npz (+ meta JSON: version, window bounds, units) with atomic tmp+os.replace; workers call `load_snapshot()` when the on-disk version changes, stat-checking per request (cheap).
  - `points_in(lat0, lat1, lon0, lon1, minutes=None)`, `points_for_tile(...)` — numpy boolean masks; microseconds at point-sparse scale.
  - Phase 2: `points_for_slot(slot_ts)` — 10-minute slot bucketing for the animation ring (see Phase 2).
- Worker footprint: artifact decode at load (a few MB decompressed into structured arrays per worker); no grids ever.

### Renderer (tiles/renderer.py)

- `present_tile(...)` gains keyword-only `lightning_style: str = ""` and `flash_points: np.ndarray | None = None`, mirroring `cell_style` / `cells_by_region` (renderer.py:335-353).
- New `_draw_lightning(img, geom, flash_points, style)` — PIL RGBA overlay + `Image.alpha_composite`, the `_draw_storm_cells` pattern (renderer.py:1339). Per flash: project lat/lon to tile pixels (same coordinates helpers storm cells use).
  - `dots` (default): small filled circles, size modestly scaled by `log1p(energy)`. (inoperative at joule scale — the landed formula is recorded in Phase 1 as-landed)
  - `bolts`: the vector bolt polygon, same projection, slightly larger footprint; two-toned to read on any basemap.
  - **No age taper:** constant brightness — strikes belong to the frame/slot they happened in (ruling 4).
  - Draw cap: when a tile's flash count exceeds `LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE`, sort by energy and draw the strongest K (presentational ruling 3; log at DEBUG). Most tiles have zero flashes and skip the layer entirely.
- Phase 1 attaches the overlay only on the newest analysis frame (routes.py:1199-1208 rule, same as storm cells). Phase 2 extends attachment to past frames via per-slot buckets.

### Routes plumbing (api/routes.py)

- `lightning: str = Query("")` on the radar tile endpoint beside `cells`/`arrows` (routes.py:1015-1052): `""` off; `1` / `true` / `dots` -> dots; `bolts` -> bolt style. Unknown values fall back to off with a warning.
- Workers hand the overlay its points from the artifact-backed store (loaded once, refreshed on version change); the tile handler narrows to the tile's lat/lon bbox via `points_for_tile`.
- Overlay cache key folds the store's lightning version into `_shared_overlay_key` (routes.py:790-803 / 1291-1301); the geometry cache key (routes.py:1064) is untouched — cached radar geometry stays shared, non-lightning requests byte-identical.

### Lifespan wiring (`data_pipeline.py` + `main.py`)

- data_pipeline.py: construct `LightningStore`, start the fetcher loop task, close it on teardown (alerts_fetcher start/close precedent, data_pipeline.py:295-315 / 338-345). No `on_cycle_complete` coupling, no state.json stores-dict change.
- main.py render lifespan: construct an empty store pointing at the artifact path, no boot snapshot needed (workers find the artifact by path on first use), assign `routes.lightning_store` and `routes.lightning_enabled` (mirrors routes.py:635-637).
- mcp/context.py (stdio): build the store from the artifact path the same way (context.py:131-136 + assignment at context.py:188-202) — and include the storm-cell fix from ruling 5 as part of the Phase 1 work order.

### MCP (mcp/tools.py + mcp/server.py)

- `get_recent_lightning(lightning_store, lat=None, lon=None, radius_km=100.0, minutes=30, limit=2000) -> list[dict]` — pure store-passing function, the get_storm_cells template (tools.py:197-298). `minutes` clamps to `MAX_AGE` (30). Per-strike shape: `{"lat", "lon", "utc", "energy", "satellite"}`. Degrades to `[]` when the store is None or disabled, never raises.
- One `@mcp.tool` wrapper in `server.py` `_register_tools` reading `routes.lightning_store` lazily at call time (server.py:96-121) — registered once, served by both transports.
- MCP server-card/AI catalog entries updated in mcp/discovery.py.

### `/v2/lightning` REST endpoint (Phase 2, committed per ruling 9)

Shape mirrors `/v2/alerts` (routes.py:1674-1760) and `/v2/storm-cells` (routes.py:1767-1821, incl. the deferred-import pattern reusing the MCP query function):
- No args: all strikes in the held window (MAX_AGE), points as GeoJSON-style features with `{lat, lon, utc, energy, satellite}` properties.
- `lat` + `lon` (+ `radius_km`, default 25): within-distance strikes (haversine refine).
- `bbox` (west,south,east,north): strikes inside the rectangle.
- `minutes` (clamped to MAX_AGE) and `limit` optional on every mode.
- Response-size honesty: the no-args mode during a saturated outbreak can be large (~100k+ strikes); document that streamers should prefer bbox/radius + minutes, and add `limit`; verify whether gzip already applies to project responses during implementation rather than assuming.
- 503 when lightning is disabled; models added to api/models.py (`LightningResponse` family).

### Config block (config.py, after the storm-cells fields near line 516) (superseded by Phase 1 as-landed below)

```python
# Lightning (GOES GLM flash points)
lightning_enabled: bool = True                    # LIBREWXR_LIGHTNING_ENABLED
lightning_glm_satellites: str = "goes18,goes19"   # LIBREWXR_GLM_SATELLITES
lightning_fetch_interval: int = 300               # LIBREWXR_LIGHTNING_FETCH_INTERVAL
lightning_max_age_minutes: int = 30               # LIBREWXR_LIGHTNING_MAX_AGE
lightning_max_draw_per_tile: int = 1000           # LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE
```

- No per-cycle saturation config on purpose (ruling 3). The store ceiling is an internal constant, not a knob.

### /health

- Additive `lightning` section following the alerts/storm-cells shape (routes.py:648-665): enabled, points held, artifact version/window bounds, last_updated, per-satellite watermark ages, last loop's warn state (satellite staleness, ceiling warnings). Workers report what they loaded; the pipeline loop exposes fetch-side truth.

### Companion fix: storm cells in stdio MCP

- Add `storm_cell_store` to the stdio context build (`mcp/context.py`, its store dict + routes assignment) so stdio agents receive real cell answers. Own small task + test in the Phase 1 work order; do not silently bundle it into the lightning diffs beyond the shared invocation plumbing.

### Tests (auto-async; marker `lightning`)

- pyproject.toml: add `lightning` to the markers list.
- `tests/test_lightning.py` — decoder units against a synthetic tiny GLM netCDF written by h5py in-fixture (never hit real S3 in CI; the fetcher's HTTP surface is injected): variable mapping + QC filtering; watermark advance/resume; store trim + ring behavior at the ceiling (warn-once); artifact save/load round-trip incl. version monotonicity; slot bucketing when Phase 2 lands.
- `tests/test_lightning_render.py` — `_draw_lightning` present-time draw: dots and bolts styles, draw-cap behavior, with/without points (mirrors test_storm_cell_render.py).
- `tests/test_api_lightning.py` — param parsing (off/dots/bolts; unknown values), overlay key moving with the store version, tile bytes unchanged when the feature is disabled (mirrors test_api_storm_cells.py).
- `tests/test_mcp_lightning.py` — tool path: radius/minutes clamps/limits, degraded-empty, both-transports registration (mirrors test_mcp_storm_cells.py).

### Docs to update when this ships

- AGENTS.md: lightning bullets under data/ + Configuration section listing the new env vars.
- docs/configuration-reference.md: the config block above with defaults.
- docs/lightning.md: dedicated user guide (added 2026-10-09).
- Attribution: README + the gitignored librewxr-site (manual companion edit), per ruling 11 — NOT example frontends.
- This plan remains the living decision log; update rulings as phases land (the satellite-implementation-plan.md convention).

## Phase 2 (committed once Phase 1 lands) - as-landed 2026-10-09

**Landed 2026-10-09 - see "Phase 2 as-landed (2026-10-09)" below.** The numbered items below are the original commitment; item 3 was rejected by maintainer ruling (noted inline).

1. `/v2/lightning` REST endpoint as specified above (ruling 9).
2. Past-frame lightning animation: per-frame slot buckets via `points_for_slot` — the "ring" is ~12 sparse point arrays (~1 MB each, ~12 MB total), NOT an 80 MB raster ring; present_tile draws slot `ts`'s strikes for any requested frame; per-frame overlay versions freeze at creation so replay in the timeline is cache-hit, and only cache eviction or a cold start recomputes. Verify during implementation: whether the shared tile store persists overlay bytes across restarts so even a cold start can warm from disk rather than recompute — nice-to-have, not load-bearing.
3. ~~Optional additional styles (e.g. `heat` blobs) — presentational only, not data changes.~~ **REJECTED by maintainer ruling 2026-10-09: lightning serves specific strikes, never generalized areas of activity - do not implement, do not leave it as an open option.**

## Phase 3 (locked until access or rule changes)

- MTG LI flash-area *imagery* overlay via EUMETView (`mtg_fd:li_afa`, anonymous WMS/WCS; survey Lightning Tier 2) — a separate imagery-layer path, not the point store; reuses the parked satellite-integration branch's anonymous WCS client pattern.
- MTG L2 per-flash points — only if EUMETSAT ships an anonymous carrier (WIS2 watch) or the no-API-keys rule is relaxed.
- FY-4C LMI points — parked on NSMC access; the decoder-class slot exists when it matters.
- NWP litoti proxy gap fill (IFS lightning diagnostics; survey Lightning Tier 2) — only becomes cheap once the Open-Meteo mirror is verified to carry the parameter.

## Footprint budget

- Download: ~30-50 MB/hour at the 5-minute cadence (two satellites), a handful of listings + ~120 GETs/hour — small next to MRMS/OPERA.
- Pipeline CPU: ~1-3 s per 5-minute loop (h5py reading point arrays), no image decode.
- RAM: pipeline store a few MB; each render worker a few MB of loaded points; no raster memmaps.
- Render workers: nothing added to geometry cache keys, no compute for absent tiles; `?lightning=` tiles pay only a small PIL draw.
- Extra wins vs the earlier revision: no state.json growth, no snapshot cadence coupling, and Phase 2 animation reduced from ~80 MB of rasters to ~12 MB of sparse slot buckets.

## Open implementation notes (verification items, not maintainer decisions)

1. GLM variable names/units — confirmed live in pre-flight against NCEI C01527 and the netCDF attrs (ruling 6).
2. Whether shared overflow-byte persistence covers overlay variants across restarts (Phase 2 item 2 verify note).
3. **Resolved (2026-10-09).** Gzip did NOT apply to `/v2/*` responses. A pure-ASGI `JsonGZipMiddleware` in `main.py` now gzips `application/json` responses >= 1 KiB, only when the client advertises `Accept-Encoding: gzip` (see Phase 2 as-landed).

## Phase 1 as-landed (2026-10-09)

- **Config redesign (supersedes the Config block above).** Units are seconds across the timing knobs. Satellite selection became a per-family boolean, `lightning_noaa_enabled` (maintainer ruling 2026-10-09: **never cherry-pick satellites; families only** — EUMETSAT/FengYun booleans land with their decoders; satellite identity is internal, never config). The five landed env vars:
  - `LIBREWXR_LIGHTNING_ENABLED=true` — master toggle (tile overlay + MCP tool).
  - `LIBREWXR_LIGHTNING_NOAA_ENABLED=true` — per-family boolean for the NOAA GOES GLM flash family.
  - `LIBREWXR_LIGHTNING_FETCH_INTERVAL=300` — seconds, clock-aligned.
  - `LIBREWXR_LIGHTNING_MAX_AGE=1800` — seconds (30 min); clamps MCP `minutes=`.
  - `LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE=1000` — presentational strongest-by-energy cap; the store always holds every flash in the window.
- **Overlay.** `?lightning=` takes `""` off, `1`/`true`/`dots` small energy-scaled dots (default) (radius `1.5 + 0.6 * log10(max(energy, 1e-15) / 1e-15)` px, clamped to 1.5-4.0), `bolts` the hand-authored vector bolt polygon (ruling 7). Constant brightness, no age fade; **newest analysis frame only** (Phase 2 adds per-frame animation via 10-minute slot buckets - landed, see Phase 2 as-landed). The overlay cache key folds the store version; the geometry cache is untouched. Unknown values fall back to off **silently** — this supersedes the "with a warning" wording above, matching the `cells`/`arrows` precedent.
- **Pre-flight decode findings (ruling 6, satisfied 2026-10-09 against live `noaa-goes18`/`noaa-goes19` files).**
  - Files are FLAT netCDF (there is no `flash` group).
  - The plan's `flash_time_offset_of_effect_time` does **not** exist; the landed decoder uses `flash_time_offset_of_first_event`.
  - Energy/time/quality variables are int16 storage with `_Unsigned="true"` — view as uint16, then apply the `scale_factor`/`add_offset` attrs (which are 1-element arrays).
  - Time base is scalar `product_time` with units `"seconds since 2000-01-01 12:00:00"` (unix epoch 946728000, parsed from the units string — the plan's 946684800 figure was an arithmetic error, corrected).
  - The flat DAY-prefix S3 listing truncates at 1000 keys (~5.5 h stale); the fetcher lists the day prefix with start-after pagination.
  - Footprint ~360 files/hour across both satellites (corrects the plan's ~120).
- **Store artifact.** A single `current.npz` carries both points and meta (version + per-satellite status) — one atomic replace, so there is no torn two-file read; resume watermarks live in `watermark.json`. The store ceiling is `MAX_POINTS=500k` with a warn-once trim; no state.json section, the artifact is the cross-process handoff (`maybe_reload()` mtime-stat on demand from render workers and the stdio MCP transport).
- **Wiring.** The routes tile handler attaches the overlay on the newest analysis frame via the latest-timestamp helper; `/health` gains the additive `lightning` section (enabled, points, version, window bounds, per-satellite status incl. staleness, ceiling trim total); MCP `get_recent_lightning` on both transports (`lat`/`lon`/`radius_km`/`minutes` clamped to `MAX_AGE`/`limit` newest-first; `[]` when disabled); `mcp/discovery.py` `_DESCRIPTION` shortened to fit the 100-char server-card cap; `routes.mcp_tools` list updated. The storm-cells-in-stdio companion fix (ruling 5) had already landed separately (f346b16).
- **Companion fix.** The `/health` tile-cache overlay classifier now counts overlay keys correctly (`>= 14`-element overlay keys); it had undercounted since the flow/cells version-keying.
- **Tests as-landed.** Marker `lightning` registered. Files split: `tests/test_lightning.py` (store), `tests/test_lightning_fetcher.py` (fetcher + synthetic GLM netCDF fixtures), `tests/test_lightning_render.py` (overlay draw), `tests/test_api_lightning.py` (routes + `/health`), plus lightning cases in `tests/test_mcp_lightning.py` / `tests/test_mcp_context.py` (marker `mcp`). Fetcher tests write synthetic-GLM netCDF fixtures with h5py in-fixture and inject the HTTP surface — **never real S3**.
- **Anchor corrections for future sessions.** `_draw_lightning` lives in `tiles/renderer.py` after `_draw_storm_cells`; the `mcp/context.py` routes assignment is at the routes-write block (~line 223); the routes module singletons are declared at the top of `routes.py` (`storm_cell_store` ~line 86).

## Phase 2 as-landed (2026-10-09)

- **`/v2/lightning` REST endpoint (ruling 9).** A GeoJSON `FeatureCollection` mirroring `/v2/alerts` and `/v2/storm-cells`. No args returns every strike in the held window; `lat` + `lon` (+ `radius_km`, default 25) returns strikes within distance; `bbox=west,south,east,north` returns strikes inside the rectangle (alerts-style 400 validation); `minutes` (clamped to `LIBREWXR_LIGHTNING_MAX_AGE`) and `limit` (default: uncapped full window, newest-first) apply on every mode. Point wins over bbox when both are supplied (mirrors `/v2/alerts` precedence). The bbox mode lives in the SHARED pure query function `mcp.tools.get_recent_lightning`, so the MCP tool gained `bbox` at identical cost; MCP keeps its explicit bounded `limit=2000` default while REST defaults to uncapped. The endpoint defers to the shared function via the storm-cells deferred-import pattern. Models `LightningProperties` / `LightningFeature` / `LightningResponse` in `api/models.py` (GeoJSON Point geometry `[lon, lat]`; properties `utc` / `energy` / `satellite`). Returns `503` when lightning is disabled.
- **JSON gzip middleware (serving-layer companion; resolves open implementation note 3).** A pure-ASGI `JsonGZipMiddleware` in `main.py` gzips ONLY `application/json` responses >= 1 KiB (internal constant `_JSON_GZIP_MIN_BYTES`, deliberately not a config knob), and only when the request advertises `Accept-Encoding: gzip`; it adds `Vary: Accept-Encoding` and never touches PNG/WebP tiles or streaming responses (the first multi-chunk body disables compression for that exchange). Registered before CORS so CORS stays outermost. This resolves the plan's open implementation note 3: gzip did NOT already apply; JSON responses are now gzipped. Motivated by the no-args lightning/alerts modes - a severe-outbreak no-args lightning response can be tens of MB - while the tile hot path stays CPU-free.
- **Per-frame lightning animation (ruling 4).** Every requested frame now draws the strikes that happened during THAT frame's own 10-minute window `(T-600, T]` (internal `_LIGHTNING_SLOT_S = 600`, not config; the lower edge is exclusive so a boundary strike belongs to the earlier frame only, the upper edge inclusive). The Phase 1 newest-frame-only gate is REMOVED - the newest frame now draws its own slot too (previously all held points). Slots older than `LIBREWXR_LIGHTNING_MAX_AGE` age out of the store and render plain. Two design deltas from the plan's Phase 2 sketch, both recorded as improvements:
  - **No slot-ring storage was needed.** The store already holds all points in one time-sorted array, so a slot is a mask query: `points_in` gained an inclusive `until_s`, called with `since_s = T-600+1, until_s = T`. The plan's ~12 sparse arrays (~12 MB) were never materialized.
  - **Overlay cache keys fold a DETERMINISTIC CONTENT FINGERPRINT** of that tile's slot strikes (`"{count}-{newest time_s}"`) instead of the plan's "per-frame overlay versions frozen at creation". Identical bytes whenever recomputed (cache eviction or cold start); a slot that later gains late-fetched strikes re-renders exactly once (frozen-at-creation would permanently miss them); a reload with identical content stays a cache hit (the live store version is deliberately NOT folded in).
  - **Shared-store verify note (plan Phase 2 item 2):** TRUE as-landed - the shared tile store is disk-backed with content-versioned keys, budget-pruned only, so overlay bytes survive restarts and cold starts warm from disk rather than recompute.
- **Heat / generalized-activity styles REJECTED (maintainer ruling 2026-10-09).** Lightning serves specific strikes, never generalized areas of activity. The plan's Phase 2 "Optional additional styles (e.g. heat blobs)" item is struck (see the Phase 2 section above) - not implemented, not left as an open option.
- **No new env vars.** Phase 2 added no configuration surface; the five Phase 1 `LIBREWXR_LIGHTNING_*` vars are unchanged, and the gzip threshold is an internal constant, not a knob.
