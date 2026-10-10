# Changelog

All notable changes to LibreWXR, newest first. This file is generated from the
published GitHub releases and must not be edited by hand - regenerate it with
`.venv/bin/python scripts/generate_changelog.py` after publishing a release,
and commit the result alongside the version bump.

---

## [v0.3.1 - NWP disk-churn fixes and dini_only profile](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.3.1)

2026-10-10

The headline of this release is the second half of the disk-churn fix for issue #36. v0.2.0 stopped byte-identical rewrites and moved regenerable stores to RAM - this release fixes what was left: the regional NWP fetch loop rewriting boundary frames every 10-minute cycle, the IFS mirror poll firing 144 times a day, and multi-megabyte state.json snapshots written uncompressed.

### NWP boundary churn fixes (#36)

- The fetch range no longer over-fetches one hour past the window while eviction cuts one hour inside it, so boundary natives (DMI DINI, ICON-EU, WRF-SMN) are no longer unlinked every cycle and re-downloaded + re-rewritten the next - a DINI-heavy deployment was sustaining ~52 grid-file writes (~152 MB) per cycle from this alone ([4f01afe](https://github.com/JoshuaKimsey/LibreWXR/commit/4f01afe765b9149944dbc7b1ad108c52348688ac))
- An output-presence gate skips Farneback interpolation entirely when the stored leads are already at the stored cadence - robust to partial publishes, failed-then-retried fetches, and warm restarts ([4f01afe](https://github.com/JoshuaKimsey/LibreWXR/commit/4f01afe765b9149944dbc7b1ad108c52348688ac))
- Fixed a warm-restart bug: after a pipeline restart, mid-life NWP runs could never extend their horizon (accumulator state is memory-only and the recursive prev-step fetch early-returned on existing frames) - missing accums are now rebuilt via an accum-only download without touching resident frames ([4f01afe](https://github.com/JoshuaKimsey/LibreWXR/commit/4f01afe765b9149944dbc7b1ad108c52348688ac))

### Leaner IFS polling (#36)

- ECMWF IFS no longer reads the Open-Meteo mirror's latest.json unconditionally every fetch cycle: a local window-coverage gate skips the poll when nothing new can be fetched (~144 polls/day down to roughly 24-48), with a 1-hour fallback re-poll so newly completed runs are noticed promptly ([0ae64a7](https://github.com/JoshuaKimsey/LibreWXR/commit/0ae64a759f6e71de3be8524293b5f47e3d14301d))

### Gzip state.json snapshots (#36)

- The state snapshot is now gzip-compressed on disk (magic-byte sniffed on load; legacy plaintext files still load). Alert-heavy deployments were writing ~28 MB of JSON every cycle; gzip cuts that several-fold. Anything reading state.json externally will see gzip magic bytes now ([1afb34a](https://github.com/JoshuaKimsey/LibreWXR/commit/1afb34ae42bd8014a098e3e3671f2cb672c5f928))

### New: dini_only EU NWP profile (#36)

- `LIBREWXR_EU_NWP_PROFILE=dini_only` runs DMI DINI (2 km) ahead of IFS without ICON-EU: DINI inside its footprint, IFS everywhere else (Iberia, southern Italy, the Balkans, eastern Europe). The profile matrix is now complete: `ifs`, `icon_eu_only`, `dini_only`, `dini_with_icon_eu` ([93591c5](https://github.com/JoshuaKimsey/LibreWXR/commit/93591c5215115abfbed2a53d680fec748a8253a1))

### Other

- New README hero image: Hurricane Isaias moving onshore ([8b93790](https://github.com/JoshuaKimsey/LibreWXR/commit/8b93790))

Upgrading: no configuration or migration changes - pull, rebuild, restart (docker compose and the auto-spawn bare-metal mode both restart the pipeline and renderers together, which the gzip state.json change wants). Expected effect for the deployment that reported the churn: from ~36 GB/day down to roughly ~9-10 GB/day, with the remaining writes being genuinely new data (new DINI/IFS steps and slots, RRQPE scans, the FrameStore history).

---

## [v0.3.0](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.3.0)

2026-10-09

## v0.3.0 — GOES GLM lightning: strike overlay, REST + MCP queries, per-frame animation

The headline of this release is end-to-end NOAA GOES GLM lightning: flash points ingested from GOES-East + GOES-West on their own 5-minute loop, overlaid on radar tiles frame by frame, and queryable from the REST API and the MCP server.

### New: lightning everywhere

- `?lightning=` radar-tile overlay — `dots` (energy-scaled) or `bolts` (hand-authored vector glyphs), drawn on every requested frame's own 10-minute window, so timeline playback replays the storm strike by strike ([8d54dcf](https://github.com/JoshuaKimsey/LibreWRX/commit/8d54dcf), [01a88a0](https://github.com/JoshuaKimsey/LibreWRX/commit/01a88a0), [ee1b509](https://github.com/JoshuaKimsey/LibreWRX/commit/ee1b509))
- `GET /v2/lightning` GeoJSON endpoint — the full held window, point + radius, or bbox, with `minutes` and `limit` ([eac1a78](https://github.com/JoshuaKimsey/LibreWRX/commit/eac1a78))
- MCP `get_recent_lightning` tool on both HTTP and stdio transports, with bbox support ([08b9866](https://github.com/JoshuaKimsey/LibreWRX/commit/08b9866), [eac1a78](https://github.com/JoshuaKimsey/LibreWRX/commit/eac1a78))
- Example viewers get a Lightning toolbar toggle, and the front-page hero shows live strikes ([55b7bae](https://github.com/JoshuaKimsey/LibreWRX/commit/55b7bae))
- New env vars (`LIBREWXR_LIGHTNING_ENABLED`, `LIBREWXR_LIGHTNING_NOAA_ENABLED`, `LIBREWXR_LIGHTNING_FETCH_INTERVAL`, `LIBREWXR_LIGHTNING_MAX_AGE`, `LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE`), all on by default — see `docs/lightning.md` (the new dedicated guide) and `docs/configuration-reference.md`

### Other improvements

- Large JSON responses (>= 1 KiB) are now gzipped when the client advertises gzip — the PNG/WebP tile hot path and streaming responses are never touched ([6a9c226](https://github.com/JoshuaKimsey/LibreWRX/commit/6a9c226))
- Fixed `/health` tile-cache overlay entry counting, which had undercounted since the flow/cells version-keying ([99de54e](https://github.com/JoshuaKimsey/LibreWRX/commit/99de54e))
- Fixed storm cells in the stdio MCP context — stdio agents previously got empty cell answers ([f346b16](https://github.com/JoshuaKimsey/LibreWRX/commit/f346b16))

### Docs

- New dedicated lightning guide (`docs/lightning.md`), a full MCP tool reference for `get_recent_lightning`, web-integration and Rain Viewer migration coverage, the source survey's GOES GLM entry marked Implemented, sizing notes, and a README overhaul ([d269f54](https://github.com/JoshuaKimsey/LibreWRX/commit/d269f54), [c6fb790](https://github.com/JoshuaKimsey/LibreWRX/commit/c6fb790), [06b54b9](https://github.com/JoshuaKimsey/LibreWRX/commit/06b54b9), [855bd1f](https://github.com/JoshuaKimsey/LibreWRX/commit/855bd1f))
- `LIBREWXR_VOLATILE_CACHE_DIR` override-file wiring documented (#36) ([8dda07f](https://github.com/JoshuaKimsey/LibreWRX/commit/8dda07f))
- Removed the stale legacy `CLAUDE.md` ([e987ec9](https://github.com/JoshuaKimsey/LibreWRX/commit/e987ec9))

Upgrading: no configuration or migration changes — pull, rebuild, restart. All lightning defaults are on; disable with `LIBREWXR_LIGHTNING_ENABLED=false`. The first fetch cycle backfills ~35 minutes of GLM history before settling into the 5-minute rhythm.

---

## [v0.2.0](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.2.0)

2026-10-08

## v0.2.0 — Disk-churn mitigation and RAM-backed regenerated stores

The headline of this release is a major cut in per-cycle disk writes
(issue #36), plus a new opt-in option that moves regenerable data to RAM.

### New: `LIBREWXR_VOLATILE_CACHE_DIR`

Optional RAM-backed (tmpfs) directory for stores that are fully
regenerated every fetch cycle: nowcast frames + optical-flow fields,
per-timestamp precip masks, storm cells, and the RRQPE source scan
cache. A single-region deployment measured ~420 MB of writes per
10-minute fetch cycle (~60 GB/day, ~22 TB/year on a consumer NVMe) —
the majority from exactly these stores, none of which are ever read
back across restarts. Point it at a tmpfs path shared by the pipeline
and all render workers and that traffic moves off the SSD.

Unset (default) = current behavior, everything under
`LIBREWXR_CACHE_DIR`. See `.env.example` (section 10) and the
configuration reference for wiring and sizing (~1 GB single-region,
~1–1.5 GB all regions; tmpfs pages count against container memory).

### Performance and fixes

- Byte-identical data is no longer rewritten to disk every cycle:
  `FrameStore` merges skip unchanged regions (no write, no version
  bump, no tile-cache invalidation), boot-restore reuses files in
  place, carry-forward hardlinks instead of copying, and precip masks
  skip unchanged timestamps.
- ECMWF IFS interpolation is now memoized like the regional NWP
  models: an unchanged hourly set skips re-interpolation entirely, and
  re-interpolated synthetics that come out byte-identical are reused
  rather than rewritten.
- Fixed the RRQPE refresh throttle skipping the first fetch on fresh
  boots.

### Docs

README refresh (badges, hero image, demo animation, Rain Viewer
free-tier comparison), deployment-guide consolidation,
`LIBREWXR_VOLATILE_CACHE_DIR` documented across the configuration
reference, `.env.example`, and the self-host sizing guide.

---

## [v0.1.1](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.1.1)

2026-10-03

Patch release: no configuration or migration changes. Upgrade by pulling and restarting.

What changed

- Perf: the ECMWF IFS fetcher no longer re-downloads every hourly timestep on the hourly window slide ([6a42a12](https://github.com/JoshuaKimsey/LibreWXR/commit/6a42a1252f5af86392f53aeb6e7f332697edf8ed)). It now fetches only newly valid times, merges them over the stored timeline, re-synthesizes just the new bracket, and leaves retained memmap files untouched - cutting S3 downloads and regrids from ~5 to 1 per slide, Farneback synthetics from 20 to 5, and memmap writes from ~50 to ~12. Render-worker behavior and the JSON API are unchanged.
- CI: MCP test modules skip cleanly when fastmcp is absent, and CI installs .[dev,mcp] so they still run there ([6886491](https://github.com/JoshuaKimsey/LibreWXR/commit/6886491d717bf54a862a561cf85176911987b7b1)).
- Tests: fixed a cache_dir-dependent health cluster test that only failed with an ambient .env set ([1cb7f5e](https://github.com/JoshuaKimsey/LibreWXR/commit/1cb7f5edc47835d356be554f795f704590bbfb46)) - test-only.

Upgrading: git pull (or docker compose pull) and restart. Still on single mode? See the v0.1.0 notes or docs/single-mode-migration.md first.

---

## [v0.1.0](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.1.0)

2026-09-29

A legacy COMPOSE_PROFILES=single (Docker) or LIBREWXR_MODE=single (bare metal) is detected automatically: the single compose profile now starts the same pipeline + renderer services, and the app maps the legacy setting to 1 render worker with legacy single-sized caches, with a one-time startup warning. Bare-metal python -m librewxr.main auto-spawns the data pipeline as a child process.
What changed
- Single mode removed; the combined fetch+render lifespan is deleted (c2b01a1)
- TileWarmer removed — render workers rely on the empty-tile fast path + per-worker LRU
- LIBREWXR_CACHE_DIR falls back to a per-host tempdir with a warning when unset
- LIBREWXR_WARMER_THREADS renamed LIBREWXR_RENDER_THREADS (old name still accepted)
- LIBREWXR_WARM_OVERVIEW_ZOOM / _REGIONAL removed
- Fix: stale httpx.DecodeError reference in the alerts retry path (be28eea)
- CI: the test suite now runs on every push and PR (c57975e)
Small boxes: set LIBREWXR_WORKERS=1 or 2 explicitly — the multi default is 16 workers aimed at larger hosts.
Migrating: docs/single-mode-migration.md (https://github.com/JoshuaKimsey/LibreWXR/blob/main/docs/single-mode-migration.md) · Staying on single mode: pin to v0.0.1 (also tagged single-mode-final), no further updates · Background: Discussion #37

---

## [v0.0.1 - The last release with Single mode](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.0.1)

2026-09-29

This is the final LibreWXR version that includes the legacy single deployment mode (one container/process doing both data fetching and tile serving). Starting with v0.1.0, LibreWXR always runs as a pipeline + render-worker pair and single mode is removed.
What this means for you
- Staying on single mode: pin your checkout to this tag — `git fetch --tags && git checkout v0.0.1` (or `single-mode-final`, same commit). Note: this tag receives no further updates or fixes.
- Upgrading past this version: in most cases nothing breaks. A legacy `COMPOSE_PROFILES=single` (Docker) or `LIBREWXR_MODE=single` (bare metal) is detected automatically: the single compose profile now starts the same pipeline + renderer pair, and the app maps the legacy setting to 1 render worker with legacy single-sized caches, with a one-time startup warning. Bare-metal users keep the same one-command workflow — `python -m librewxr.main` now spawns the data pipeline as a child process. For a small box, set `LIBREWXR_WORKERS=1` or `2` explicitly.
- Full migration guide: docs/single-mode-migration.md (https://github.com/JoshuaKimsey/LibreWXR/blob/main/docs/single-mode-migration.md) — what you gain (fetch/render GIL isolation, crash isolation, cluster /health) and what's different (the single-mode tile warmer is gone).
- Background: Discussion #37.
