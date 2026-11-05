# Network Split Mode

Run the data pipeline on one server and render workers on others, with an
object store (or any shared location `fsspec` can reach) carrying the
handoff between them. This is the "network split" from the roadmap.

## How it works

In the standard deployment the pipeline and the render workers share one
cache directory: the pipeline writes memmap files, names them in
`state.json`, and the render workers poll that file and memory-map what it
names. Network split mode keeps exactly that contract on each host and
moves the files between hosts:

```
pipeline host                       object store                  render host
───────────────                     ─────────────                 ──────────────
data_pipeline  ──publish──▶  <prefix>/radar/..., nowcast/...  ◀──pull──  snapshot_pull
  (state.json)               <prefix>/snapshots/<n>.json.gz               (state.json)
                             <prefix>/manifest.json                        render workers
                             <prefix>/lightning/current.npz                (memmap, as usual)
```

- **Publish (pipeline, `LIBREWXR_SNAPSHOT_SYNC_URL`).** After every
  `state.json` write the pipeline uploads the files the snapshot names
  that the store does not have yet (frame files are immutable per
  timestamp, so a cycle uploads roughly one new radar frame per region,
  the regenerated nowcast, and NWP or satellite grids when a run
  changed), then the snapshot under a unique name, then one
  `manifest.json` listing the snapshot and its files with sizes and
  mtimes. The manifest is the single object readers key off, so a
  reader never sees a snapshot whose files are still uploading. The
  publish reads its own hard links to the files (made beside them, no
  copy), so the pipeline can regenerate the nowcast or evict a frame
  meanwhile without the upload losing the content the snapshot named. Files a
  snapshot dropped are deleted after the next publish, the same
  one-cycle grace the pipeline's local eviction keeps, so a reader
  mid-pull can finish. The lightning artifact, which lives outside
  `state.json`, is uploaded whenever the pipeline writes it.
- **Pull (render host, `python -m librewxr.snapshot_pull`).** One puller
  per host polls the manifest, downloads the listed files it lacks into
  `LIBREWXR_CACHE_DIR` under the same relative paths, rewrites the
  pipeline's absolute directories in the snapshot to the local cache,
  and only then writes `state.json`. The render workers on that host
  (`LIBREWXR_RENDER_ONLY=1`, sharing the cache directory) pick it up
  exactly as they would from a local pipeline. The puller also does the
  housekeeping the pipeline would do on that host: pruning the shared
  coordinate and tile stores, page-cache priming of fresh frames, and
  removing files no snapshot names (again one cycle late).

What travels: `state.json`, `radar/`, `nowcast/` (and flow fields), the
NWP grid directories, `gmgsi/`, `mask/`, `storm_cells/`, `lightning/`,
`palettes/`, `coverage_masks/`. What stays per host: `coord/`,
`tiles_shared/`, `workers/` (render-side caches and heartbeats) and the
pipeline's own `rrqpe/` scan cache and `alerts/` geocode tables (alerts
ride inside `state.json`). Files written to `LIBREWXR_VOLATILE_CACHE_DIR`
on the pipeline host land under `<cache_dir>/volatile/` on render hosts.

## Configuration

Pipeline host (`.env` of the pipeline service):

```bash
LIBREWXR_SNAPSHOT_SYNC_URL=s3://my-bucket/librewxr
# S3-compatible stores (MinIO, OVH, ...); credentials from AWS_* variables
LIBREWXR_SNAPSHOT_SYNC_ENDPOINT_URL=https://s3.example.com
LIBREWXR_SNAPSHOT_SYNC_WORKERS=8
```

Render host (`.env` of the puller and the render workers):

```bash
LIBREWXR_SNAPSHOT_SYNC_URL=s3://my-bucket/librewxr
LIBREWXR_SNAPSHOT_SYNC_ENDPOINT_URL=https://s3.example.com
LIBREWXR_SNAPSHOT_PULL_INTERVAL=10
LIBREWXR_CACHE_DIR=/data/cache
LIBREWXR_RENDER_ONLY=1          # for the render workers
```

Keep the source toggles (regions, NWP sources, satellite, lightning) the
same on both sides: render workers construct the stores they expect from
their own settings and apply what the snapshot carries. `file:///path`
works as the URL too, for a shared or local disk and for tests.

With Docker Compose, `COMPOSE_PROFILES=render-node` starts the `puller`
and `renderer-node` services (no pipeline) from the same compose file.

## Start order on a render host

Start the puller before the render workers, as the `render-node` compose
profile does. A render worker waits for `state.json` to exist and then
applies whatever it names, and the puller only writes `state.json` after
every file of that snapshot is in place, so a fresh host is always
consistent. A worker that starts against a `state.json` left over from an
earlier run (files since removed) drops the stores it cannot open and
only recovers them when their part of a later snapshot changes; restart
the worker after the puller has caught up in that case.

## What to expect

- **Lag.** A render host serves a cycle once the pipeline has uploaded
  it and the puller has downloaded it: with the default full-region
  pipeline that is about 1 GB of raw memmaps per 10-minute cycle (the
  nowcast frames dominate, since they are regenerated every cycle), so
  tens of seconds on a fast link. The manifest's `written_at` and the
  puller log show the lag.
- **Measured against OVH object storage** (full-region pipeline, 32
  parallel transfers): a whole snapshot of 1,433 files published in
  121 s and pulled into an empty render cache in 89 s; a publish with
  nothing new takes about 20 s (stat, link and manifest round trips).
  Deletes go out in batches of 100 keys, since a 1,000-key
  multi-object delete timed out there; a failed batch is retried on
  the next publish.
- **Bandwidth.** Roughly 1 GB per cycle per render host at full regions,
  about 150 GB a day. The memmaps are mostly zeros; compressing them in
  transit is a possible follow-up.
- **Failure.** A failed publish leaves the previous manifest in place and
  a render host keeps serving its last snapshot; a failed pull leaves the
  local cache as it was and retries on the next poll. A render host that
  has pulled once serves from local disk whether or not the pipeline is
  reachable.
- **Local damage.** On every poll the puller checks that the files of the
  applied snapshot are present at their published size, and downloads any
  that are missing or truncated (and `state.json`, if that is gone). A
  same-size corruption is not detected. A render worker that already
  dropped a store over a missing file still recovers as described under
  [Start order](#start-order-on-a-render-host).
- **One publisher per store.** Point exactly one pipeline at a store URL.
  Two publishers would silently interleave their snapshots in one series
  and expire each other's files.
- **Same contract.** Nothing in the render path changes: workers still
  memory-map local files named by a local `state.json`.
