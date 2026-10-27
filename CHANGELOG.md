# Changelog

All notable changes to LibreWXR, newest first. This file is generated from the
published GitHub releases and must not be edited by hand - regenerate it with
`.venv/bin/python scripts/generate_changelog.py` after publishing a release,
and commit the result alongside the version bump.

---

## [v0.6.1 - Invalid CAP polygon crash fix](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.6.1)

2026-10-27

Fixes a production crash where a self-intersecting (bow-tie) CAP alert polygon - observed from the Indonesian BMKG feed - raised GEOSException (TopologyException: side location conflict) at the group unary_union in the /v2/alerts endpoint, returning HTTP 500 for every request while the offending alert sat in the store.

Highlights:

- New shared ensure_valid_polygon() repair helper: make_valid with a buffer(0) fallback, matching the geocode-table repair convention, and a fast no-op path for already-valid geometry.
- Ingest repair for explicit CAP <polygon> rings, merged multi-area footprints, and NWS zone-geometry unions.
- Store invariant: AlertsStore repairs polygons on both entry points (replace_all and __setstate__), so snapshots written by older revisions - including already-poisoned live ones - heal on render-worker load without waiting for a fetch cycle.
- Defensive guards in /v2/alerts (union and simplify) and the MCP alerts radius tool: invalid geometry degrades to a repaired footprint or a geometry-less alert instead of raising.
- Bow-tie regression tests across the endpoint, ingest, snapshot load, and the MCP tool.

Deployments pulling published images: set LIBREWXR_IMAGE_TAG=0.6.1 and restart; no configuration changes are needed.

---

## [v0.6.0 — Container image distribution](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.6.0)

2026-10-24

The headline of this release is container image distribution (#44): LibreWXR now publishes multi-arch images to GitHub Container Registry for every release, and self-hosting moves from git-pull + source rebuild to `docker compose pull`. No image publishing existed before 0.6.0; images ship routinely from here on, so the publish pipeline gets exercised by every future release before 1.0.

### Published images (#44)

- New release-tagged workflow `.github/workflows/docker-publish.yml`: builds on native runners (`ubuntu-24.04` for amd64, `ubuntu-24.04-arm` for arm64 - no QEMU emulation), each arch pushes by digest, and a merge job assembles the multi-arch manifest list. A full run takes ~2 minutes; `workflow_dispatch` runs publish only a `sha-*` tag, which is how the pipeline was validated before the release.
- Images publish to `ghcr.io/joshuakimsey/librewxr`, anonymously pullable. Tags: exact semver with the `v` stripped (`0.6.0`), `latest` on release tags, and `sha-<short>` per build. The workflow's `GITHUB_TOKEN` with `packages: write` is the only auth - no stored secrets.
- The Dockerfile gains standard OCI labels (title, description, source, url, license AGPL-3.0-or-later) and now ships the LICENSE file; the image is otherwise unchanged (same `python:3.12-slim` base, no compilation on either arch).

### Pull-based deployment and updates

- `docker-compose.yml` carries `image: ghcr.io/joshuakimsey/librewxr:${LIBREWXR_IMAGE_TAG:-latest}` on both services alongside `build: .` - pulling published images and building from a checkout coexist. Pin a release with `LIBREWXR_IMAGE_TAG=0.6.0` in `.env` (defaults to `latest`; compose interpolation only, config.py never reads it).
- New `scripts/update.sh` is the image-based update helper: fast-forwards the checkout for the compose file, reconciles `.env` against `.env.example` (append-only, `.env.bak` first, existing values never modified), then `docker compose pull` + `up -d`. Same opt-in sentinel, flock and `--dry-run`/`--env` flags as the legacy script; `--build` keeps a source-rebuild path.
- `scripts/auto-update.sh` remains for source-based deployments and fixes a latent idempotence bug: a prose-only mention of `LIBREWXR_WARMER_THREADS` in `.env.example` counted as missing on every run and re-appended an empty banner block.

### Docs

- README quickstart leads with the pull flow (clone for the compose file and `.env.example`, `docker compose pull`, `up -d`) with a `docker pull` one-liner; build-from-source moves to its own subsection. The auto-update section documents `scripts/update.sh` with the same cron example and sentinel opt-in.
- Self-hosting docs swept: `docs/configuration-reference.md` gains a `LIBREWXR_IMAGE_TAG` section and pull commands in the deployment prose and production recipe, `docs/single-mode-migration.md` step 3 is pull-based, `docs/self-host-sizing.md` cross-links the pull flow, and `docs/mcp-server.md` notes the image ships the `[mcp]` extra.

### Upgrading

Image deployments: `docker compose pull` && `docker compose up -d`. Cron users: switch scheduled updates to `scripts/update.sh` (same sentinel opt-in); `scripts/auto-update.sh` still works for source builds. Building from a checkout is unchanged (`docker compose up -d --build`). No config keys renamed; one new compose-only var (`LIBREWXR_IMAGE_TAG`). Response schemas unchanged.

---

## [v0.5.0 — WMO alert ingest fix, national geocode tables, SIGBUS guard](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.5.0)

2026-10-23

The headline is a fix for the WMO CAP alert ingest, which had gone silently dark: severeweather.wmo.int retired its legacy per-agency RSS proxy, so every alert source beyond the NWS came back empty while the logs looked healthy. Europe, Germany and Australia are back, alert-code resolution moved into a pluggable registry, and the pipeline can no longer die of a bus error when the volatile cache dir runs out of room (#43).

### WMO CAP alert ingest fix

- severeweather.wmo.int retired the per-agency `v2/cap-alerts/{sid}/rss.xml` proxy (404 for 48 of 49 agencies) and the fetcher ignored the per-agency `capAlertFeed` field, so WMO alerts silently fell to zero while the summary line kept counting attempts. Ingest now follows the declared feeds (comma-split lists included), parses RSS 2.0 and Atom, and matches hub ids with `urn:oid:` normalization in both substring directions, with a bounded fallback for agencies whose id scheme never matches (INMET).
- Agencies whose `sourceId` language suffix drifts from the hub (`de-dwd-en` vs `de-dwd-de`, likewise `ir-irimo`, `kz-kazhydromet`, `ly-nmc`, `th-tmd`) are recovered by (country, authority) family matching.
- Failures are visible: failing feed URLs are counted and warned with the agency list, the summary line reports ok/total feed URLs, and `/health` `ingest_ok` degrades only when every attempted feed URL fails. Verified live against the hub: 0 -> ~1,500 WMO alerts across 53 agencies.

### National geocode tables (pluggable registry)

- CAP geocode resolution moves into `alert_geocodes.py` as a `valueName`-routed `GeocodeProvider` registry: adding a nation is a data drop, not parser surgery.
- MeteoAlarm serves `NUTS2` alongside `EMMA_ID`/`NUTS3` (unlocks most Belgian warnings that were silently dropped) and invalid table polygons are repaired with `buffer(0)` (Croatia HR804/HR806).
- New tables: DWD Warngebiete WFS GeoJSON (Germany, 11,647 warn cells at municipality resolution) and BoM AAC shapefiles (Australia, 958 areas) fetched over anonymous FTP - the only scheme BoM publishes; HTTPS answers 403. New runtime dependency `pyshp>=2.3` (pure-Python shapefile reader, imported lazily by the BoM loader only).
- China (CMA) stays excluded: its CPEAS township-level codes have no openly licensed public geometry source; the rationale is documented at the exclusion.

### /v2/alerts: one entry per CAP info

- Multi-area CAP documents now store as one entry per `<info>` block with a single merged geometry (`Polygon` or assembled `MultiPolygon`) instead of one entry per area: a DWD warning listing a hundred warn cells is one alert with multi-part geometry, not a hundred.
- The response schema and property list are unchanged. `regions` now typically carries one semicolon-joined areaDesc string per feature (the shape WMO feeds natively publish), and MultiPolygon features are more frequent - they were already possible for NWS and WMO multi-area alerts. Library-based map consumers (Leaflet, MapLibre, OpenLayers) handle this natively.
- Geocodes fully inside an explicit area polygon are pruned, so mixed polygon + geocode alerts keep full coverage without duplicated geometry.

### Pipeline SIGBUS guard (#43)

- An undersized volatile dir (Docker default 64 MiB container `/dev/shm`) killed the pipeline with an uncatchable `Fatal Python error: Bus error` at the memmap write: numpy allocates the backing file lazily, so the file looked full-size at 29,250,000 bytes while the page faults hit ENOSPC and the kernel raised SIGBUS. All memmap writers (RRQPE, the NWP grids, the frame store, the radar disk cache and GMGSI) now pre-flight the filesystem for space and write through unique pid+uuid tmp names, raising an actionable error naming the path and the fixes instead of dying by signal. The deterministic tmp names were the same truncation-under-overlapping-writers class the per-cycle stores already fixed after a production incident.
- The RRQPE scan store rehydrates from disk at startup, so a restart no longer re-downloads the whole 15-16 slot window (~5.3 GB).
- `docker-compose.yml` sets `shm_size: "2g"` on pipeline and renderer (tmpfs pages count against the service memory limit; override higher for global volatile footprints).

### Docs

- New `docs/alerts.md`: the ingest pipeline, the geocode-table registry, entry semantics, REST/MCP semantics and attribution credits. Stale alerts mentions fixed across README, the MCP doc, the web-integration guide, the configuration reference and the self-host sizing notes (2 GB volatile headroom; host `df` does not describe the container shm).

### Upgrading

Pull, rebuild, restart (`docker compose up -d --build`, or `python -m librewxr.main` on bare metal). First pipeline start downloads ~67 MB of geocode tables once (MeteoAlarm ~33 MB, DWD ~12 MB, BoM ~22 MB over plain FTP - make sure outbound FTP is allowed; if it is not, Australia stays empty with one warning while everything else keeps working). API consumers: feature granularity and the `regions` string shape change as described above (response schemas unchanged). No config keys were renamed; volatile-cache users should check shm sizing against docs/self-host-sizing.md.

---

## [v0.4.0 - RRQPE decode fix, true-rate MCP sampling, Docker port pin](https://github.com/JoshuaKimsey/LibreWXR/releases/tag/v0.4.0)

2026-10-19

The headline of this release is a correctness fix for the RRQPE global radar band: rain rates were being decoded at one tenth of reality (#41). The map looked close to radar anyway because a +6 dBZ display offset was quietly absorbing the error, so the decode fix lands together with a recalibrated offset, and MCP point sampling now reports true values.

### RRQPE decode fix (#41)

- netCDF4 applies `scale_factor` on read, and the decoder multiplied by it a second time, so every RRQPE rate decoded at one tenth of the file's value. The synthetic test fixture was double-packing on write, which cancelled the bug in tests. The decoder now reads raw packed int16 and unpacks exactly once, and the fixture stores the real-file convention ([651f705](https://github.com/JoshuaKimsey/LibreWXR/commit/651f705))
- New regression test pins the decode contract: a file in the real packed convention must decode to a direct encode of its true rate, and fails under the old double unpack ([651f705](https://github.com/JoshuaKimsey/LibreWXR/commit/651f705))

### RRQPE display recalibration

- Default `LIBREWXR_RRQPE_DBZ_OFFSET` moves from +6 to -7: with decoding fixed, a single-scene comparison against MRMS MergedReflectivityQCComposite put zero offset about 13 dB hot; -7 realigns the RRQPE band with radar ([ba9032a](https://github.com/JoshuaKimsey/LibreWXR/commit/ba9032a))
- Provisional: a multi-scene recalibration against MRMS/OPERA is planned, so this default may shift again in a future release

### MCP: true-rate point sampling

- `get_precip_nowcast` now undoes the decode-time display shift for RRQPE-served radar samples, so reported `dbz` and `rate_mmh` are the file's true values. Previously they read about 0.37x reality for RRQPE points ([2f03c68](https://github.com/JoshuaKimsey/LibreWXR/commit/2f03c68))
- The NWP fallback branch is intentionally uncorrected: chain samples are weighted blends of multiple sources with no single winner. Documented in `docs/mcp-server.md` ([2f03c68](https://github.com/JoshuaKimsey/LibreWXR/commit/2f03c68))

### Docker: LIBREWXR_PORT only moves the published host port (#42)

- The renderer's in-container bind is pinned to 8080 (compose `environment:` overrides `env_file:`), so the published `${LIBREWXR_PORT:-8080}:8080` mapping always lines up. Previously any non-default port dead-ended the published mapping: the container bound the custom port while compose still pointed at 8080 ([3a3ed08](https://github.com/JoshuaKimsey/LibreWXR/commit/3a3ed08))
- On bare metal `LIBREWXR_PORT` remains the bind port

### Memory: RRQPE row-strip decoding (#40)

- RRQPE/DQF decode 64 rows at a time instead of materialising full-globe float64 and float32 copies: peak RSS per GLB file drops from ~2.4 GB to ~0.5 GB at the default downsample, output bit-identical ([29821f9](https://github.com/JoshuaKimsey/LibreWXR/commit/29821f9))
- Variable shapes are captured under the HDF5 lock in one pass and beyond-3D variables are rejected cleanly ([df1ebfc](https://github.com/JoshuaKimsey/LibreWXR/commit/df1ebfc))

### Other

- README: newly identified users added ([89feb62](https://github.com/JoshuaKimsey/LibreWXR/commit/89feb62), [1de18af](https://github.com/JoshuaKimsey/LibreWXR/commit/1de18af))

Upgrading: pull, rebuild, restart (`docker compose up -d --build`, or `python -m librewxr.main` on bare metal). Docker users who set `LIBREWXR_PORT`: it now controls only the published host port and the container always binds 8080. If you hand-tuned `LIBREWXR_RRQPE_DBZ_OFFSET` against the old decode, reset it to the new default (-7). MCP consumers: `dbz` / `rate_mmh` values change for RRQPE-served points (response schemas unchanged).

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
