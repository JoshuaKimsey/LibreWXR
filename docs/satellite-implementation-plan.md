# Satellite Implementation Plan — GMGSI

> **Status: Phases 1, 1.5, and 2 SHIPPED (2026-05-24).** The `/v2/satellite/...` endpoint serves the GMGSI VIS-over-LW composite live; the legacy IFS-derived synthetic cloud path has been deleted. Phase 4 disk-edge feathering also shipped. Phase 3 (optical-flow interpolation) was **declined** — see the in-text note in that section. The forward-looking language below (e.g. "will be deleted", "Phase 1.5 deletes") is preserved as a record of how the plan was sequenced; treat the per-phase scopes as historical-but-still-accurate descriptions of what landed.

This document was the active plan for shipping real-satellite imagery as a LibreWXR layer. It supersedes an earlier "native per-satellite" plan whose research is preserved on the `satellite-integration` branch (committed `639dfd6`). The pivot reason is at the bottom under [Why not native per-satellite](#why-not-native-per-satellite).

## Table of Contents

- [What we're shipping](#what-were-shipping)
- [GMGSI channel reference](#gmgsi-channel-reference)
- [Source: NOAA GMGSI](#source-noaa-gmgsi)
- [Architecture overview](#architecture-overview)
- [Phase 1 — Foundation + LW ingest (IR baseline)](#phase-1--foundation--lw-ingest-ir-baseline)
- [Phase 1.5 — Delete the IFS-derived synthetic satellite path](#phase-15--delete-the-ifs-derived-synthetic-satellite-path)
- [Phase 2 — VIS ingest + composite renderer](#phase-2--vis-ingest--composite-renderer)
- [Phase 3 — Optical-flow interpolation](#phase-3--optical-flow-interpolation)
- [Phase 4 — Renderer polish + multi-worker verification](#phase-4--renderer-polish--multi-worker-verification)
- [Phase 5 — Optional defense-in-depth](#phase-5--optional-defense-in-depth)
- [Open questions](#open-questions)
- [Why not native per-satellite](#why-not-native-per-satellite)

## What we're shipping

- **One satellite source** covering the populated globe (±73° latitude, all longitudes), composited by NESDIS from GOES-East, GOES-West, Meteosat-10, Meteosat-9, and Himawari-9.
- **Two channels ingested**: longwave IR (`LW`) and visible (`VIS`).
- **One user-facing tile endpoint**: the existing Rain Viewer-compatible `/v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}`, now backed by a **VIS-over-LW composite** with reflectance-driven alpha. The URL pattern, the catalog field (`satellite.infrared`), and downstream Rain Viewer client compatibility all stay identical to what's already shipping.
- **Apparent 10-min cadence** via optical-flow interpolation on a 1-hour native cadence.
- **Hour-fresh** data: ~35 min from observation to publication, then ingested next fetch cycle.

What we're explicitly **not** shipping:

- **WV (water vapor)** — fascinating to forecasters, off-purpose for a cloud viewer. Trivial to add later (~30 LOC subclass) if there's ever a use case.
- **SW (shortwave IR)** — niche (fog / wildfire detection); not visually compelling for general use.
- **SSR (solar surface radiation)** — derived photovoltaic product, not satellite imagery.
- **Native ≤2 km per-satellite resolution.** GMGSI is 8 km globally. At z<10 (typical weather-map zooms) this is invisible; at z≥10 over CONUS/Europe individual storm cells appear softer than the per-satellite path would render. See [Why not native per-satellite](#why-not-native-per-satellite).
- **Separate channel endpoints.** LW and VIS are ingested for the composite renderer's use only — not exposed as their own tile URLs. Keeps the API surface identical to today's Rain Viewer compatibility shape.

### Replacing the IFS-derived synthetic satellite layer

LibreWXR currently backs `/v2/satellite/...` with an IFS-derived synthetic cloud image (a model render that looks IR-like). GMGSI is real satellite data and strictly better for "what's the sky doing right now" — the only thing the synthetic layer offers that GMGSI can't is *forecast-time clouds* (the model can render ahead of the current time), which isn't useful for a satellite cloud viewer.

**The synthetic satellite path gets deleted, not preserved as a fallback.** Sequence:

1. **Phase 1** lands the GMGSI ingest + endpoint backing alongside the existing synthetic code. Both functional during this phase, gated by `LIBREWXR_GMGSI_ENABLED` (default true).
2. **Phase 1.5** (small cleanup commit after Phase 1 live-verification) deletes the synthetic satellite code: `data/cloud_grid.py`, `data/cloud_cache.py`, the IFS-derived rendering path in `tiles/satellite_renderer.py`, and any cloud-cover-only config keys. Verify no other consumers depend on the IFS cloud-cover data before deletion (IFS *precipitation* stays — that's a separate IFS data product and remains the global base layer of the NWP chain).
3. **From Phase 2 onwards** the renderer is GMGSI-only.

When `LIBREWXR_GMGSI_ENABLED=false`, the satellite endpoint returns 503 and the catalog `satellite.infrared` array is empty `[]` — same pattern as `LIBREWXR_RADAR_ENABLED=false`. No fallback to synthetic data. If you want satellite, turn GMGSI on.

## GMGSI channel reference

This section documents what every GMGSI channel actually shows, so the "we shipped LW + VIS only" decision is recoverable and adding a channel later doesn't require re-deriving the rationale.

### LW — Longwave Infrared (~12 µm) — SHIPPED

The "IR window" channel. The atmosphere is mostly transparent at 12 µm, so the satellite sees thermal emission from whatever is topmost in the column.

- **Cold = high cloud tops** (especially deep convection — looks bright white in renders)
- **Warm = bare ground or ocean surface** when no clouds intervene
- **Works 24/7** — thermal emission doesn't care about sunlight

This is the workhorse of every TV weather graphic. If you ship one IR layer, you ship LW.

### VIS — Visible (0.6 µm) — SHIPPED

Pure reflected sunlight. The same image your eyes would see from orbit.

- **Daytime only** — VIS is black on the night side because there's no sun to reflect
- **Highest spatial detail** of any GMGSI channel
- **Clouds appear bright** because of high albedo; ocean is dark, land is medium

Combined with LW via reflectance-driven alpha, gives a smooth day/night composite where VIS detail fades naturally into IR at the terminator — no sun-angle math required (the physics does it for us).

### WV — Water Vapor (~6.7 µm) — NOT SHIPPED

Sees **mid-to-upper tropospheric moisture**, not clouds. The atmosphere is *opaque* at this wavelength because water vapor absorbs it, so the satellite sees whatever altitude the moisture is at: dry air → it sees lower (warmer), moist air → it sees higher (colder).

- Reveals jet stream positions (dry/moist boundaries)
- Atmospheric river plumes
- Upper-level circulation patterns
- Synoptic-scale storm system flow

Indispensable for forecasters; confusing for general consumers without context. Off-scope for a cloud viewer. To add later: a `GMGSIWVSource` subclass + a dedicated tile endpoint at `/v2/satellite/wv/...` with its own colormap (yellow→orange→red dry-to-moist convention).

### SW — Shortwave Infrared (~3.8 µm) — NOT SHIPPED

A multi-purpose oddball channel. During the day it's a mix of reflected solar + thermal emission; at night it's pure thermal at a different wavelength than LW.

- Fog / low cloud detection (low clouds look different than at LW)
- Wildfire hot-spot detection (active fires emit strongly at 3.8 µm and "bloom" white)
- Snow/ice vs cloud differentiation
- Day microphysics products in multi-channel combinations

Not visually compelling on its own. Worth adding only if a specific feature requests it (e.g. "wildfire layer").

### SSR — Solar Surface Radiation — OUT OF SCOPE

Not satellite imagery at all. A *derived product* modeling solar irradiance reaching the surface, used for photovoltaic forecasting and agriculture. Different audience entirely (energy industry, not weather).

## Source: NOAA GMGSI

**Bucket**: `s3://noaa-gmgsi-pds/` (anonymous, NOAA Open Data Dissemination Program). Same access pattern as the existing GOES/Himawari paths the project already uses on the parked branch.

**Layout**:

```
noaa-gmgsi-pds/
├── GMGSI_LW/{YYYY}/{MM}/{DD}/{HH}/GLOBCOMPLIR_v3r0_blend_s{YYYYMMDDHHMMSS}_e{...}_c{...}.nc
├── GMGSI_VIS/...
├── GMGSI_WV/...   (deferred)
├── GMGSI_SW/...   (deferred)
└── GMGSI_SSR/...  (out of scope)
```

One file per hour per channel, ~7.5 MB compressed. Archive depth: 2021 onwards (5 years).

**File structure** (verified against a live LW file 2026-05-23 23:00Z):

| Property | Value |
|---|---|
| Grid | 3000 yc × 5000 xc, regular equirectangular |
| Resolution | 0.0722° (~8 km at equator) |
| Lat coverage | -72.74° to +72.72° |
| Lon coverage | -179.93° to +180° |
| Format | NetCDF4 / CF-1.8 |
| Data variable | `data` (float32, but pre-encoded 0–255 per long_name) |
| Coordinates | 2D `lat`(yc,xc) + `lon`(yc,xc), with values constant along the off-axis (effectively 1D) |
| Quality flag | `dqf` (float32, same shape) |
| Composited from | G19, G18, Meteosat-10, Meteosat-9, H-9 |

**Latency**: ~35 minutes from observation window start to file creation timestamp.

**Encoding direction**: the `data` variable is labelled `0-255 Brightness Temperature`. For LW, live sample stats show cold=high (p5=70, median=126, p95=191) consistent with the convention we already use for IR. For VIS, bright=high by the same convention. No inversion needed for either; store as-is.

## Architecture overview

The single-source nature of GMGSI eliminates large amounts of plumbing that the per-satellite path would have needed:

| Concern | Per-satellite path | GMGSI path |
|---|---|---|
| Geo→latlon reprojection | ~800 LOC in `_reproject_geos_to_latlon` | Not needed |
| Multi-source seam blending | `coverage_quality` argmax per pixel | Not needed |
| Per-satellite priority dispatch | Sorted source list, fallback logic | Not needed |
| Encoding direction handling | `encoded_invert` ClassVar, per-source decisions | Already inverted upstream |
| Source family base class | `GeostationaryChannelBase` + `Tiled…Base` + `WCS…Base` | One source class, channel ClassVar |
| Per-source HDF5 segfault risk | 7 paths × failure rate | 1 path × failure rate (× 2 channels) |
| Public API endpoint count | 1 (already shipping) | 1 (unchanged — composite backs it) |

What we **keep** from the existing project shape:

- The Rain Viewer-compatible `/v2/satellite/...` endpoint (URL, params, caching headers all unchanged)
- Pipeline / render-worker split (already shipped in main)
- Memmap-backed frame storage so render workers can read without re-decoding
- Optical-flow interpolation infrastructure (`data/nwp_interpolation.py`) reused for hourly→10-min smoothing
- Snapshot/state.json plumbing for cross-worker handoff

### File layout on `gmgsi-satellite`

```
src/librewxr/sources/satellite/
└── gmgsi/
    ├── __init__.py        # satellite_provider(settings, cache_dir) → list[SatelliteContribution]
    ├── source.py          # GMGSISource base + GMGSILWSource + GMGSIVISSource
    └── frames.py          # GMGSIFrame dataclass + GMGSIFrameStore (memmap ring buffer)
src/librewxr/tiles/
└── satellite_renderer.py  # MODIFIED: composite path (VIS-over-LW with reflectance alpha);
                           # IFS-derived path kept as fallback when GMGSI disabled
src/librewxr/sources/
└── _base.py               # NEW: SatelliteContribution dataclass + SatelliteSource Protocol
```

We're **not** introducing a `_shared/` base class for satellites. With one source family and two channels (LW + VIS), a single `GMGSISource` base + two thin subclasses is enough.

### Storage encoding

GMGSI's 0–255 encoding maps directly onto our existing satellite-frame convention:

- `encoded == 0` → no data (outside the disk, or quality-flag rejection)
- `encoded ∈ [1, 255]` → channel value, cold=high for LW, bright=high for VIS

No transform on ingest. Memmap as `np.uint8` after a `data > 0` & `dqf == 0` mask.

### Composite renderer logic

Inside `satellite_renderer.py`, when both LW and VIS frames are available for the requested timestamp:

```
lw_grid   = sample LW for tile bounds         # always present (24/7)
vis_grid  = sample VIS for tile bounds        # zero on night side
rgba_lw   = apply LW colormap (cold-white, warm-transparent)
rgba_vis  = apply VIS grayscale palette
alpha_vis = vis_grid / 255.0                  # natural day/night mask, no sun math
output    = alpha_blend(rgba_vis, rgba_lw, alpha=alpha_vis)
```

The terminator (sunrise/sunset line) emerges automatically because reflectance fades smoothly there. Optional gamma curve on `alpha_vis` if the crossfade feels too abrupt — verify visually.

When LW is available but VIS isn't (e.g. VIS channel disabled or ingest lag), render LW alone with the LW colormap. When neither is available, the endpoint returns 503 — there is no synthetic fallback (see [Replacing the IFS-derived synthetic satellite layer](#replacing-the-ifs-derived-synthetic-satellite-layer)).

### Config keys

```bash
# Master switch for GMGSI as the backing source for /v2/satellite/...
# When false, the endpoint falls back to the existing IFS-derived synthetic path.
LIBREWXR_GMGSI_ENABLED=true

# Per-channel ingest toggles. Disabling VIS while LW stays on
# degrades the composite to LW-only.
LIBREWXR_GMGSI_LW_ENABLED=true
LIBREWXR_GMGSI_VIS_ENABLED=true

# Frame retention (hours) — defaults sized for ~12 h smooth animation
LIBREWXR_GMGSI_RETENTION_HOURS=12

# Toggle optical-flow interpolation (hourly → 10-min); leave true unless debugging
LIBREWXR_GMGSI_INTERPOLATION=true
```

All toggles default to enabled per the project's "new sources ship turned on" convention.

## Phase 1 — Foundation + LW ingest (IR baseline)

Goal: LW (longwave IR) ingested, decoded, stored. The `/v2/satellite/...` endpoint switches from IFS-derived synthetic to LW-only when `LIBREWXR_GMGSI_ENABLED=true`. No VIS, no composite, no interpolation yet.

**Scope:**

1. `src/librewxr/sources/_base.py` — add `SatelliteContribution` dataclass and `SatelliteSource` Protocol mirroring the `NWPContribution`/`NWPGrid` pattern.
2. `src/librewxr/sources/satellite/gmgsi/frames.py` — `GMGSIFrame` dataclass (timestamp, channel, encoded uint8 grid, 1-D lat/lon vectors since the grid is regular) + `GMGSIFrameStore` ring buffer.
3. `src/librewxr/sources/satellite/gmgsi/source.py` — `GMGSISource` base class with:
   - `channel: ClassVar[str]` (`"LW"`, `"VIS"`)
   - `s3_product_path: ClassVar[str]` (`"GMGSI_LW"`, `"GMGSI_VIS"`)
   - `friendly_name: ClassVar[str]` (`"GMGSI LW"`, `"GMGSI VIS"`)
   - `_list_recent_keys(window_start, window_end)` — walk `{prefix}/{YYYY}/{MM}/{DD}/{HH}/` for the window, sort by `s{}` token, dedupe to one file per hour.
   - `_fetch_and_decode_slot(key)` — download to tmp file, open with `xr.open_dataset(engine="netcdf4")`, mask with `dqf`, cast to uint8.
   - `fetch(history_seconds, horizon_seconds)` — orchestrator, writes new frames to the store.
   - `GMGSILWSource(GMGSISource)` — concrete LW subclass.
4. `src/librewxr/sources/satellite/gmgsi/__init__.py` — `satellite_provider(settings, cache_dir)` returns a list with the LW contribution.
5. `src/librewxr/sources/__init__.py` — add a `collect_satellite_contributions(settings, cache_dir)` walker analogous to `collect_nwp_contributions`.
6. `src/librewxr/config.py` — `gmgsi_enabled`, `gmgsi_lw_enabled`, `gmgsi_retention_hours` settings.
7. `.env.example` — document the new toggles.
8. `src/librewxr/data_pipeline.py` — invoke `collect_satellite_contributions`, register each source, schedule its fetch in the cycle.
9. `src/librewxr/tiles/satellite_renderer.py` — add a GMGSI rendering path. When GMGSI enabled and an LW frame exists for the requested timestamp, render with the LW colormap; otherwise fall back to the existing IFS-derived path.
10. `src/librewxr/api/routes.py` — extend `satellite_tile` to dispatch between GMGSI and IFS-derived sources. Update the catalog `satellite.infrared` array to source timestamps from GMGSI when enabled.
11. `src/librewxr/main.py` — lifespan registers the GMGSI store(s).

**Tests:**

- `tests/test_satellite_gmgsi.py` — listing, decoding, encoding direction, mask handling, store ring-buffer behaviour, contribution shape.
- Extend `tests/test_sources_discovery.py` to exercise the satellite collector with the master toggle.

**Ship criteria:**

- LW frames ingest live from `s3://noaa-gmgsi-pds/GMGSI_LW/...`
- `/v2/satellite/...` serves a clean LW IR composite over the full globe when `LIBREWXR_GMGSI_ENABLED=true`
- `LIBREWXR_GMGSI_ENABLED=false` falls back to the existing IFS-derived synthetic path (temporary — Phase 1.5 deletes the synthetic path entirely)
- `/health` exposes the GMGSI fetch state alongside existing sources
- Full suite green

**Out of scope (deferred to later phases):**

- VIS ingest
- Composite renderer (VIS over LW)
- Optical-flow interpolation
- Color scheme polish

## Phase 1.5 — Delete the IFS-derived synthetic satellite path

Goal: rip out the now-superseded synthetic satellite renderer. Small cleanup commit, lands as soon as Phase 1 is live-verified.

**Scope:**

1. Delete `src/librewxr/data/cloud_grid.py` and `src/librewxr/data/cloud_cache.py`.
2. Remove the IFS-derived rendering path from `src/librewxr/tiles/satellite_renderer.py` — the renderer now only knows about GMGSI.
3. Remove the IFS cloud-cover fetch from `src/librewxr/data_pipeline.py` (the IFS *precipitation* fetch stays — that's a different IFS product used by the NWP chain).
4. Remove cloud-cover-only config keys (e.g. `cloud_cache_dir`, `cloud_*` settings) from `src/librewxr/config.py` and `.env.example`.
5. Update `src/librewxr/api/routes.py`: when `LIBREWXR_GMGSI_ENABLED=false`, return 503 instead of dispatching to the synthetic renderer; the catalog `satellite.infrared` array is empty `[]` in that case.
6. Remove cloud-cover snapshot keys from `src/librewxr/data/master_state.py`.
7. Delete or update tests that exercise the synthetic satellite path.
8. Update `README.md`, `docs/configuration-reference.md`, and `docs/web-integration-guide.md` to reflect the simpler post-cleanup story (one renderer, GMGSI-or-nothing).

**Pre-flight check:**

Before deleting `cloud_grid.py` / `cloud_cache.py`, grep the codebase for imports to confirm only the synthetic satellite renderer consumes them. The IFS provider in `world/ifs/__init__.py` is a separate code path (precipitation), independent of cloud cover.

**Ship criteria:**

- `git grep cloud_grid` returns zero hits outside the deleted files
- Full suite green
- `/v2/satellite/...` with `LIBREWXR_GMGSI_ENABLED=true` renders as before
- `/v2/satellite/...` with `LIBREWXR_GMGSI_ENABLED=false` returns 503
- IFS precipitation continues to feed the NWP chain unchanged

## Phase 2 — VIS ingest + composite renderer

Goal: the `/v2/satellite/...` endpoint serves the production-quality VIS-over-LW composite with smooth day/night transition via reflectance-driven alpha.

**Scope:**

1. `GMGSIVISSource(GMGSISource)` in `gmgsi/source.py` — channel=`"VIS"`, product=`"GMGSI_VIS"`. Same shape as `GMGSILWSource`.
2. `LIBREWXR_GMGSI_VIS_ENABLED` config key + `.env.example` entry.
3. `gmgsi/__init__.py` registers both LW and VIS contributions.
4. `tiles/satellite_renderer.py` — implement the composite path:
   - Sample both LW and VIS grids for the requested tile bounds + timestamp.
   - Apply LW colormap (cold-white, warm-transparent) to LW.
   - Apply VIS grayscale palette to VIS.
   - Blend with `alpha = vis_value / 255.0` (optional gamma curve, verify visually).
   - Degrade gracefully: VIS missing → LW-only render; both missing → 503 (no synthetic fallback after Phase 1.5).
5. Cache key includes both source frame timestamps so a render is invalidated correctly when either side updates.

**Tests:**

- Composite math: a synthetic LW + VIS pair produces the expected alpha blend; pure-night (VIS=0) renders identical to LW-only; pure-day (VIS=255) renders identical to VIS-only.
- Renderer falls back to LW-only when VIS is unavailable.
- Endpoint returns 503 when GMGSI is disabled (no synthetic fallback).

**Ship criteria:**

- The composite renders smoothly across the terminator (qualitative — eyeball on the live deployment)
- Night side shows LW IR detail; day side shows VIS detail; sunrise/sunset crossfades are not visually jarring
- No measurable latency impact on tile rendering compared to LW-only

## Phase 3 — Optical-flow interpolation

Goal: hourly frames feel like 10-min animation in the UI.

**Approach**: reuse `src/librewxr/data/nwp_interpolation.py`. It already does Farneback optical flow on consecutive hourly IFS frames and emits 10-min interpolations. The grid shape, dtype, and cadence assumptions match GMGSI almost exactly — we just need a thin adapter.

**Scope:**

1. New `GMGSIInterpolator` class wrapping `nwp_interpolation` for the satellite-frame format.
2. `LIBREWXR_GMGSI_INTERPOLATION` config toggle (default True).
3. Renderer samples interpolated frames when available, falls back to the latest native frame at slot boundaries.
4. Tests: round-trip interpolation produces 5 in-between frames between two consecutive hourlies; toggling the setting falls back to native frames.

**Ship criteria:**

- Animation in the UI between satellite frames looks smooth (qualitative — eyeball on the live deployment)
- `LIBREWXR_GMGSI_INTERPOLATION=false` skips interpolation cleanly
- No measurable latency impact on tile rendering

**Risk:** Farneback flow on satellite imagery may produce artefacts at the terminator (day/night sweep across the disk) for VIS, because the underlying VIS field has a sharp sunlit/dark boundary that translates across the hour. LW shouldn't care because brightness temperature is continuous. Mitigation: interpolate LW and VIS independently; if VIS terminator artefacts are visible, disable interpolation on VIS only (LW interpolation alone still smooths the night side).

## Phase 4 — Renderer polish + multi-worker verification

Goal: production-quality output on the rack.

**Scope:**

1. **Color schemes** in `src/librewxr/colors/schemes.py`:
   - LW IR: grayscale with cold=white, warm=transparent (matches every weather app convention)
   - VIS: grayscale natural for the overlay layer
2. **Smoothing at high zoom**: 8 km native resolution is blocky at z≥10. Add a Gaussian blur scaled by zoom level (mirrors the radar renderer's `_compute_blur_radius` pattern in `tiles/renderer.py`).
3. **Coverage map**: update `scripts/generate_coverage_map.py` to draw the GMGSI footprint (±73° band) on the satellite coverage map. Regenerate `docs/coverage-map-satellite.png`.
4. **Multi-worker deployment test**: run the pipeline + 32 render workers on the rack with `LIBREWXR_RADAR_ENABLED=false` and `LIBREWXR_REGIONAL_NWP_ENABLED=false` to isolate satellite-only behaviour. Confirm:
   - Pipeline fetches LW + VIS per cycle without crashes
   - Render workers serve composite tiles from the memmaps with no decode in their process
   - Memory stays within `LIBREWXR_PIPELINE_MEMORY:-12G` and `LIBREWXR_RENDER_MEMORY:-18G` budgets
   - State.json updates surface cleanly to render workers
5. **Live-verification checklist**:
   - Composite renders smoothly across the terminator at all zoom levels
   - LW night-side detail visible behind transparent VIS
   - The ±73° polar gaps are clean (no visual artefacts at the boundary)
   - High-zoom (z≥10) tiles are smoothed but not over-blurred

**Ship criteria:**

- Composite renders correctly on the rack at all zoom levels (1–12)
- 24-hour soak test with no pipeline restarts
- README updated to describe the satellite layer (now real, not synthetic)
- `docs/web-integration-guide.md` confirms the unchanged `/v2/satellite/...` endpoint behaviour

## Phase 5 — Optional defense-in-depth

Goal: insurance against the HDF5 segfault class diagnosed on the per-satellite path.

GMGSI gives us a 7× reduction in HDF5 surface area (1 file/hour × 2 channels instead of dozens of per-source NetCDFs every cycle), so the failure rate should be tolerable without further work. But if a segfault is ever observed in production:

**Scope (if needed):**

1. New `src/librewxr/sources/_shared/hdf5_decode_pool.py` — singleton `multiprocessing.Pool(context="spawn", processes=2, maxtasksperchild=50)`.
2. Refactor `_fetch_and_decode_slot` so the decode body is a picklable module-level function.
3. Submit decodes via `pool.apply_async`; catch `BrokenProcessPool` and skip slot on segfault.
4. `LIBREWXR_HDF5_DECODE_WORKERS` env knob (default 2).

The same pool should also wrap the existing NWP grid decode paths in `world/ifs/`, `regional/north_america/usa/nwp/hrrr/`, etc. — they all touch HDF5 too.

**Ship criteria:**

- Deliberately truncated NetCDF triggers a clean "decode worker died, skipping slot" warning instead of taking down the pipeline.
- Pipeline keeps running across N consecutive bad inputs.

This is a separate, optional commit on top of Phases 1–4. Defer until the failure rate justifies it.

## Open questions

1. **Filename parsing.** GMGSI filenames carry three timestamps (`s` / `e` / `c` for start / end / creation). The `s` timestamp is the observation window start. Should we floor frames to the hour (cleanest for animation) or preserve the `s` token verbatim (slightly more precise but harder to align)? *Tentative answer: floor to hour for simplicity; revisit if alignment with other layers needs the precision.*

2. **Gamma curve on VIS alpha.** Linear `alpha = vis / 255.0` may make the terminator feel too abrupt (VIS drops fast near sunset). A mild `alpha = (vis / 255.0) ** 0.7` softens the crossfade. Worth verifying visually in Phase 2; pick the curve empirically.

3. **Memmap layout.** Should each channel get its own memmap directory under `{cache_dir}/gmgsi/{channel}/`, or one combined directory? *Tentative answer: per-channel directories, mirrors the existing per-source NWP cache layout for parallel structure.*

4. **What happens when the `dqf` mask covers a meaningful fraction of the grid?** Likely we just store the masked pixels as 0 and let the renderer make them transparent (falling through to the LW base or IFS fallback). Verify empirically once Phase 1 is live.

5. **Subprocess isolation timing.** Ship Phase 5 proactively, or wait for an observed segfault? *Tentative answer: wait. Ship-and-monitor.*


## Why not native per-satellite

The previous `satellite-integration` branch built per-satellite ingest for GOES-East, GOES-West, Himawari-9, and four Meteosat variants. By the time it accumulated 17 commits + uncommitted Meteosat WCS scaffolding, two distinct HDF5 segfault classes had surfaced (partial-tile-set and accumulated-libhdf5-state-corruption) and the remaining Phase 5C work (Meteosat live verification) hadn't completed.

The pivot is motivated by:

- **Single-source simplicity**: ~2000 LOC across 7 sources → ~250 LOC, one source family.
- **Pre-composited seams**: NESDIS already handled the GOES-E/Meteosat, Meteosat/Himawari, and Himawari/GOES-W blends; we don't.
- **One HDF5 path**: dramatically lower segfault surface area.
- **No reprojection**: GMGSI ships on a regular equirectangular grid; no geos→latlon code.
- **Pre-encoded 0–255**: storage convention matches with no transform.
- **No new API endpoints**: backs the existing `/v2/satellite/...` endpoint, preserving Rain Viewer compatibility unchanged.

The trade-offs are real: hourly cadence (mitigated by optical-flow interpolation), 8 km vs 2 km native resolution (invisible at typical web map zooms), ~35 min publication latency vs ~10–20 min per-satellite (acceptable for a non-nowcast layer).

The `satellite-integration` branch is preserved (not pushed) as a documented "what native per-satellite looks like" reference, in case the GMGSI trade-offs ever become unacceptable.
