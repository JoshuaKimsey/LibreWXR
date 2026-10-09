# Single-Mode Removal and Migration Guide

LibreWXR used to ship two deployment shapes: `single` (one process doing
fetch + render) and `multi` (a data pipeline plus render workers). Single
mode was removed in v0.1.0; v0.0.1 is the last release that contains it.
LibreWXR now always runs as a pipeline + render worker pair; what used to
be single mode is just that architecture with one render worker.

This page explains what changed, what to do, and how to keep true single
mode if you really want it.

## What changed and why

- **One architecture.** A data pipeline process
  (`python -m librewxr.data_pipeline`) fetches all radar / NWP /
  satellite / alerts / lightning data and writes a shared `state.json`
  snapshot. One or more render workers (`python -m librewxr.main`)
  memmap that snapshot and serve tiles.
- **Single mode is gone as of v0.1.0.** The old single-container path is
  deleted, and the `librewxr` compose service that ran it is removed. The
  `pipeline` and `renderer` services are now both tagged
  `profiles: ["multi", "single"]`, so a legacy `COMPOSE_PROFILES=single`
  starts the same pair.
- **Compatibility alias.** `LIBREWXR_MODE=single` or
  `COMPOSE_PROFILES=single` still works. `mode` always resolves to
  `"multi"`; a `single` token applies the "legacy-single" defaults
  profile (1 render worker, single-sized caches) and logs a startup
  warning pointing here.
- **Motivation.** Single mode was a legacy leftover. Multi mode with one
  worker covers the same low-end hardware niche, with better fetch /
  render GIL isolation: a fetch crash no longer takes the tile server
  down with it, and `/health` aggregates the cluster. The community
  validated parity on small deployments
  ([Discussion #37](https://github.com/JoshuaKimsey/LibreWXR/discussions/37);
  context in [Issue #36](https://github.com/JoshuaKimsey/LibreWXR/issues/36)).

## TL;DR for Docker users

Nothing breaks on pull. `COMPOSE_PROFILES=single` keeps working through
the compatibility alias, but it now starts the two-service pair instead
of one container. You should still migrate when convenient:

1. In `.env`, change `COMPOSE_PROFILES=single` to `COMPOSE_PROFILES=multi`.
2. Set `LIBREWXR_WORKERS` explicitly for your box. The multi default is
   16 (sized for an 80-core rack); for a small host use `1` or `2`.

```bash
# .env
COMPOSE_PROFILES=multi
LIBREWXR_WORKERS=1        # or 2 on a small multi-core box
```

3. Run `docker compose up -d --build`.

If you leave `COMPOSE_PROFILES=single`, you get the exact same pair with
1 worker and the legacy-single cache sizes, but the app logs a warning
telling you to switch.

## Bare-metal / manual users

The same command works:

```bash
python -m librewxr.main
```

With no flags, `main.py` now auto-spawns the data pipeline as a child
process and runs this process as a render worker, defaulting to 1
uvicorn worker unless `LIBREWXR_WORKERS` is explicitly set. The child's
output shares the parent's console. On shutdown the pipeline is
terminated with the server.

`LIBREWXR_RENDER_ONLY=1` is now only for dedicated render workers that
attach to an already-running pipeline (the Docker `renderer` service sets
it). Do not set it for the one-command bare-metal/dev path.

Set `LIBREWXR_CACHE_DIR` for a stable cache location:

```bash
export LIBREWXR_CACHE_DIR=/path/to/librewxr-cache
python -m librewxr.main
```

When `LIBREWXR_CACHE_DIR` is unset, the app falls back to a per-host
tempdir (`<tmp>/librewxr-cache`) and logs a one-time warning. That works,
but the cache is not stable across reboots and, in Docker, both services
must point at the same shared directory. In Docker the compose file sets
it automatically.

## What you lose

- **Tile pre-warming.** The `TileWarmer` was single-mode-only and is
  deleted along with `LIBREWXR_WARM_OVERVIEW_ZOOM` and
  `LIBREWXR_WARM_OVERVIEW_ZOOM_REGIONAL`. The first hit on a cold tile is
  now slightly slower. In steady state this is covered by the empty-tile
  fast path and the per-worker LRU caches, so repeat requests and
  animation playback stay fast.

## What you gain

- **GIL isolation.** Fetching / decode and tile rendering run in separate
  processes, so a hung or crashing fetch no longer stalls or kills the
  tile server. Render workers keep serving from the last good snapshot.
- **GIL-free rendering.** Adding render workers scales tile throughput
  across cores instead of serializing the render path through one
  process.
- **Cluster health.** `/health` aggregates per-worker pulses (RSS, tile
  cache, coord cache, request counters, cgroup memory) into a top-level
  `cluster` section.
- **A shared encoded-tile store.** Render workers share encoded tile
  bytes on the cache volume, so one worker's encode serves the fleet.

## Removed and renamed settings

| Setting | Status | Replacement |
|---|---|---|
| `LIBREWXR_WARM_OVERVIEW_ZOOM` | Removed | none - overview tiles are served cold and cached per worker (the empty-tile fast path makes cold precip-empty tiles cheap) |
| `LIBREWXR_WARM_OVERVIEW_ZOOM_REGIONAL` | Removed | none |
| `LIBREWXR_WARMER_THREADS` | Renamed | `LIBREWXR_RENDER_THREADS` (the old name is still accepted as an alias) |
| `LIBREWXR_MODE=single` | Legacy alias | `LIBREWXR_MODE=multi` (with an explicit `LIBREWXR_WORKERS`) |
| `COMPOSE_PROFILES=single` | Legacy alias | `COMPOSE_PROFILES=multi` |
| `LIBREWXR_MEMORY` | Removed | `LIBREWXR_PIPELINE_MEMORY` + `LIBREWXR_RENDER_MEMORY` |

`LIBREWXR_RENDER_THREADS` sizes the per-render-worker compute pool (the
thread pool that runs per-tile geometry computes). Multi default: 4.
Legacy-single default: 0, meaning auto (one fewer than the available
cores).

`LIBREWXR_CACHE_DIR` is no longer hard-required. When unset the app uses
the per-host tempdir fallback described above.

## Legacy-single defaults profile

A `single` token in `LIBREWXR_MODE` / `COMPOSE_PROFILES` selects this
profile. Otherwise the multi defaults apply.

| Setting | Legacy-single | Multi |
|---|---|---|
| `LIBREWXR_WORKERS` | 1 | 16 |
| `LIBREWXR_TILE_CACHE_MB` | 200 | 128 |
| `LIBREWXR_COORD_CACHE_SIZE` | 2048 | 512 |
| `LIBREWXR_COORD_STORE_MB` | 4096 | 8192 |
| `LIBREWXR_WARM_COORD_ZOOM` | 4 | -1 (no eager warm) |
| `LIBREWXR_RENDER_THREADS` | 0 (auto) | 4 |

Each value is still overridable: setting one explicitly wins over the
profile default.

## Want true single mode permanently?

Pin to the git tag `single-mode-final` or the release tag `v0.0.1`, the
last commit before single mode was removed in v0.1.0 and the last release
that contains it. Both tags point at the same commit, so either name works
for the checkout:

```bash
git fetch --tags
git checkout single-mode-final
```

That tag receives no further updates - no bug fixes, no new sources, no
security patches. Use it only if you need the old one-process shape and
accept that it is frozen.
