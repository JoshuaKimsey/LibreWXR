# Shared Coordinate-Array Store - Design Notes

**Status:** accepted 2026-08-06 - Option B, implementation in progress.
**Context:** conclusion of the 2026 performance program (commits `85cca22`, `5da587c`, `0e84698`). Everything else from the audit shipped; this was the single item deliberately held back because it is a cache redesign, not a mechanical fix.

## The problem

Every tile render needs projection arrays: for a given (region, zoom, x, y, tile_size, pad), which region-grid pixels each tile pixel maps to. These live in six per-process `@lru_cache` functions in `src/librewxr/tiles/coordinates.py` (`region_pixel_indices`, `region_pixel_indices_padded`, `region_pixel_indices_fractional`, `tile_pixel_latlons`, etc.).

Each entry is ~550 KB (paired int32/float32 arrays at 256^2 or padded 272^2). In multi mode:

- 16 render workers each compute and hold IDENTICAL arrays in private heap memory.
- Worst case at `LIBREWXR_COORD_CACHE_SIZE=1024`: 6 caches x 1024 entries x ~550 KB ~= 3.4 GB per worker -> ~54 GB of the 80 GB renderer cgroup holding 16 copies of the same data.
- Every worker re-derives the same projections at boot (the Tier 1 warm-up made boots correct, but each worker still computes its own copy).
- This RAM sits outside the tile-cache byte budget and silently competes with the cache that actually drives hit rates.

Current observability: per-worker usage is surfaced as `coord_caches_mb` in `/health`.

### Key-space correction (2026-08-06 review)

The sizing above understated the key space. Verified against code:

- `tile_size` is exactly 256 or 512: all three tile endpoints quantize via `tile_size = 512 if size >= 512 else 256` (`src/librewxr/api/routes.py`).
- `pad` is dynamic, not fixed: `pad = int(blur_radius * 3)` per tile in `compute_tile_geometry` (`src/librewxr/tiles/renderer.py:153-158`), so `pad` takes values in `{0} or ~3..48`.

Consequences:

- (a) 512px entries are ~2 MiB, ~4x the `_CACHE_ENTRY_BYTES` estimate, so `/health`'s `coord_caches_mb` underreports on workers serving 512px tiles.
- (b) `warm_coordinate_caches` warms pad=8 / tile_size=256 keys, which barely overlap the request-path key distribution - fixed as part of this work.

## What a shared store would do

Move the arrays onto the shared volume (`/data/cache`) as read-only memmaps - the same pattern every other store already uses (FrameStore, NowcastStore, precip masks, and the coverage/feather mask persistence added in `0e84698`). Compute once, store once physically; all 16 workers map the same pages.

Payoff, in order of importance:

1. **RAM dedup** - one physical copy instead of 16; freed cgroup headroom funds bigger tile caches (the larger lever on average latency).
2. **CPU dedup** - one worker's worth of projection math instead of 16x at boot and on cold keys (projection is trig-heavy for LAEA/tmerc regions like OPERA).
3. **Faster worker boots/restarts** - warming becomes an mmap instead of a compute pass.

Key enabler: coordinate arrays are **pure functions of static inputs** (region definitions never change between fetch cycles). The store needs NO per-cycle invalidation and no state.json integration. The only rebuild trigger is a code/region-definition change, handled with a version/parameter signature - exactly like `mask_signature` in `src/librewxr/data/coverage.py`.

## The design fork (why this was deferred)

> Resolved 2026-08-06: Option B accepted. This section kept for the trade-off record.

Coverage-mask sharing was easy because masks are a fixed, tiny set (13 regions x 2 arrays, built once at boot). The coordinate key space is **open-ended**: (region x z x x,y x tile_size x pad) has millions of potential keys at high zooms. Cannot precompute everything, hence two options:

### Option A - bounded warm-set pack (recommended first step)

Precompute arrays for a bounded, high-traffic set: every tile overlapping enabled regions up to zoom 6-7 (the same set `LIBREWXR_WARM_COORD_ZOOM` warms today).

- Writer: pipeline (or first single-mode boot) after coverage masks; atomic tmp+`os.replace` per file.
- Layout: content-addressed files (key encoded in filename) + a small JSON manifest (key -> file, shape, dtype) + signature/version stamp.
- Reader: render workers memmap the pack read-only at startup; keys outside the pack fall back to the existing per-worker LRU (unchanged code path).
- Effort: a few hundred lines across `tiles/coordinates.py`, `data_pipeline.py`, `main.py`. Mirrors the mask-persistence pattern closely.
- Limitation: dedups only the overview zooms; high-zoom hot tiles (traffic-dependent) stay per-worker.

### Option B - full shared on-demand store (the real redesign)

Any worker that computes an array publishes it to a shared, append-only, size-capped store; the other 15 workers hit it on their next request for the same tile. Converges to "each unique tile computed once, globally."

- Hazards to design for: concurrent publishers (atomic renames solve), partial-write visibility (same), eviction racing readers holding open memmaps (safe on Linux - unlinked inodes stay alive until last reader, same trick FrameStore relies on), eviction policy (size-capped dir, prune by generation or atime), and the general new failure surface.
- Effort: multi-day, with real concurrency testing.
- Only worth it if worker count or RAM footprint ever needs to shrink hard.

## Accepted design (2026-08-06)

- **Decision**: Option B chosen. Rationale: ends the 16-worker RAM and CPU duplication outright; the reviewer pass found no blockers in the lock-free design.
- **Store format**: content-addressed entries under `<cache_dir>/coord/<sha1[:2]>/<sha1>.npy`; sha1 input is `{signature}|{kind}|{region}|{z}|{x}|{y}|{tile_size}|{pad}`. Each entry is ONE `.npy` file stacking the array pair as shape `(2, R, C)`, written/read via `np.lib.format.open_memmap` (self-describing header; `mode="r"` maps are non-writeable, preserving the existing immutability contract). No manifest lookup on the hot path.
- **Signature**: SHA256 over format version + ALGO_VERSION + canonical enabled-region definitions (mirrors `mask_signature` in `src/librewxr/data/coverage.py`), folded into every entry hash - code/region changes namespace all entries, stale files become unreachable garbage that the pruner removes first (oldest mtime). ALGO_VERSION carries a "bump on any projection-math change" discipline comment.
- **Publish/open/prune**: pid-unique tmp + `os.replace`, skip-if-exists, best-effort (the project's standard lock-free idiom; concurrent publishers write byte-identical content). Open validates header shape/dtype and that st_size covers the declared data extent; unlink + recompute on corruption/truncation; transient OSError falls through to compute WITHOUT unlinking. Prune: orphan `*.tmp` older than 1h swept; over budget -> unlink oldest-mtime-first down to 90% of budget (FIFO by publish time; atime deliberately unused - noatime mounts). Prune ownership: pipeline process per fetch cycle (multi), fetch-cycle hook (single, subject to the 30s `on_cycle_complete` debounce - budget is a soft cap). Render workers never prune.
- **Integration**: the six public functions in `src/librewxr/tiles/coordinates.py` keep their signatures; the existing `lru_cache` becomes a per-worker handle cache holding views onto shared pages (bounded by `coord_cache_size`); miss path = store open -> compute + publish. After a publish the wrapper re-opens the entry and returns the shared read-only memmap views rather than the freshly computed heap arrays, so the handle cache pins file-backed pages - not anonymous heap - even for self-published keys; the heap arrays remain only as the fallback when publish or re-open fails. Store gate (`coord_store_enabled` + cache_dir set) evaluated at call time - degrades to current behavior. `_compute_blur_radius` moves from `renderer.py` into `coordinates.py` so the warmer derives request-accurate pads; warm covers 256px only, 512px publishes on demand. No state.json integration (entries are pure functions). 0-15s warm jitter in `_render_only_lifespan`, multi-only.
- **Config**: `LIBREWXR_COORD_STORE_ENABLED` (default true - kill switch), `LIBREWXR_COORD_STORE_MB` (0 = mode default: single 1024, multi 8192; settable via .env like every other knob, restart to apply).
- **Observability**: `/health` gains `coord_store_mb` (shared on-disk), entry count, and store hit/publish rates; per-worker lru stats relabeled as handle cache. Shared clean pages are already excluded from the cgroup decision metric in `memory.py` - the store is memory-pressure-neutral.
- **Failure modes**: all degrade to today's behavior - no cache_dir -> off; unwritable/full disk -> compute; corrupt -> self-heal; crash mid-publish -> orphan tmp swept; cold-key herd -> up to 16 identical computes converging on the first `os.replace`.

## Relevant references

- `src/librewxr/data/coord_store.py` - the shared on-disk store (new; the implementation)
- `src/librewxr/tiles/coordinates.py` - the six LRU caches, entry-size estimates in `coord_cache_stats` (~lines 489-502)
- `src/librewxr/data/coverage.py` - mask persistence pattern to copy (`mask_signature`, `save_masks`, `load_masks`, `persist_masks_in_background`), shipped in `0e84698`
- `src/librewxr/data/store.py` - atomic memmap write pattern (tmp -> flush -> os.replace), ~lines 62-75
- `src/librewxr/main.py` - `_render_only_lifespan` warm-up call (Tier 1) and mask load-first wiring (Tier 2)
- Config knobs: `LIBREWXR_COORD_CACHE_SIZE` (multi default 512, currently 1024 in production), `LIBREWXR_WARM_COORD_ZOOM` (6)
- `/health` field: `coord_caches_mb`
