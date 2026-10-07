# LibreWXR Self-Hosting Sizing Guide

> **These are rough estimates, not guarantees.** The figures below are
> derived from measured telemetry on one production reference
> deployment, not from load-testing across a range of workloads. Actual
> requirements vary with traffic volume and pattern, the enabled radar
> regions and NWP sources, tile sizes, and client behaviour.
>
> Measure with your own workload. `/health` is the source of truth - its
> top-level `cluster` section reports `cluster.memory.container` with
> the anon / file-page-cache split and `cluster.requests.hit_rate` for
> cache effectiveness. Anon is the OOM-relevant number - the allocation
> you actually have to provision; file/page-cache is kernel-reclaimable
> and shrinks itself under pressure.

## Reference Workload

The tiers below are sized against one reference workload: the full
feature set - all radar regions (`LIBREWXR_ENABLED_REGIONS=ALL`), the
complete regional NWP chain plus IFS, GMGSI satellite, nowcast, and
weather alerts - serving real public traffic at ~15 requests/s average
(~1.3M requests/day) with a long-tail request distribution.

## Deployment Shape

LibreWXR has one shape now: a data pipeline plus N render workers. Size
the worker count to the box:

- **Small / personal:** the same architecture with 1-2 render workers
  (`LIBREWXR_WORKERS=1` or `2`), one or two radar regions, a few NWP
  sources, behind a caching proxy. This covers the old single-process
  niche without putting fetch and render in the same process.
- **Public-facing:** a pipeline sidecar plus N render workers, one per
  physical core. Render workers each map the shared state and bypass the
  Python GIL on the tile-render path. See
  [configuration-reference.md](configuration-reference.md) for the exact
  settings, and [single-mode-migration.md](single-mode-migration.md) if
  you are coming from the removed one-process deployment.

## Spec Tiers

| Tier | vCPU | RAM | Disk | Render workers |
|---|---|---|---|---|
| Minimum (public) | 8 vCPU | 16 GiB | 50 GiB SSD | 4-8 |
| Recommended (public production) | 16 vCPU | 32 GiB | 50-100 GiB SSD | 8-12 |
| Heavy / no CDN | 32 vCPU | 64 GiB | 100 GiB SSD | up to 16 (the c7i.8xlarge's 32 vCPUs are hyperthreaded - 16 physical cores; follow the physical-core rule) |

The **minimum** tier is the floor for public traffic - below it, 1-2
render workers behind a proxy is the honest shape. The **recommended**
tier serves the reference workload with most render capacity idle; that
idle capacity is the headroom that absorbs storms, flash crowds, and the
cold tile long tail. The **heavy** tier is for deployments that must absorb
tile traffic without a CDN in front. Scale vertically: the designed
shape is one pipeline writer plus N render workers per host - not
multi-node.

## Frame Memory per Region

Each enabled radar region keeps `LIBREWXR_MAX_FRAMES` frames (default 12)
in a memory-mapped ring buffer. Approximate per-frame memory:

| Code | Region | Source | RAM per frame |
|---|---|---|---|
| `USCOMP` | Continental US | NCEP MRMS (IEM fallback) | ~63 MB |
| `AKCOMP` | Alaska | NCEP MRMS (IEM fallback) | ~6 MB |
| `HICOMP` | Hawaii | NCEP MRMS (IEM fallback) | ~3.4 MB |
| `PRCOMP` | Puerto Rico | NCEP MRMS (IEM fallback) | ~1 MB |
| `GUCOMP` | Guam | NCEP MRMS (IEM fallback) | ~1 MB |
| `CACOMP` | Canada | MSC GeoMet (MRMS blending) | ~6 MB |
| `SVCOMP` | El Salvador + neighbours | MARN/SNET (San Andrés, 120 km) | <1 MB |
| `OPERA` | Europe (27 countries) | EUMETNET OPERA | ~16 MB |
| `ITCOMP` | Italy + neighbours | DPC Radar (24-radar national composite) | ~2 MB |
| `TWCOMP` | Taiwan + W. Pacific buffer | CWA QPESUMS (7-radar composite) | ~1 MB |
| `JPCOMP` | Japan archipelago | JMA HRPN (20 radars + AMeDAS gauge correction) | ~4 MB |
| `MYPENINSULAR` | Peninsular Malaysia + Singapore + N. Sumatra | MET Malaysia (12-radar composite) | <1 MB |
| `MYEAST` | East Malaysia (Borneo) + Brunei | MET Malaysia (12-radar composite) | <1 MB |
| `PHCOMP` | Philippines (Luzon, Visayas, Mindanao) | PAGASA PANAHON (9-radar mosaic) | ~4 MB |
| `RRQPE` | Global 60S-70N band (always-on) | NOAA Enterprise Rain Rate GLB-5 | ~29 MB |

Multiply by `LIBREWXR_MAX_FRAMES` (default 12) for the frame-store budget;
add NWP grids, satellite frames, nowcast data, and the tile cache on top.
`/health` is the source of truth for actual usage (see the note at the top
of this guide).

## AWS Instance Mapping

Indicative only - instance families, generations, availability, and
pricing change; verify the current generation and price in your target
region before committing.

The workload's CPU:RAM ratio is ~2 GiB per vCPU, which matches AWS's
compute-optimized c-family:

- Minimum (public): **c7i.2xlarge** (8 vCPU / 16 GiB)
- Recommended (public production): **c7i.4xlarge** (16 vCPU / 32 GiB)
- Heavy / no CDN: **c7i.8xlarge** (32 vCPU / 64 GiB)

Do not size for memory-optimized families (r6i and similar). Since the
shared coordinate store shipped, the workload is CPU-bound on renders,
not RAM-bound - extra RAM beyond the sizing below just becomes page
cache.

Stick to **x86_64**. ARM/Graviton instances are cheaper, but the
scientific-Python stack (numpy, OpenCV) is unverified there - test
before committing.

## Three Things That Matter More Than Raw Size

Instance size is the least interesting part of capacity planning. These
three levers move real capacity more than any vCPU count.

### 1. Put a CDN in front

Tiles are immutable per timestamp and ETag'd, so public tile traffic is
massively duplicate - every client asks for the same tiles at the same
timestamps. A CDN (CloudFront or Cloudflare) absorbs that duplication at
the edge; your origin then only renders each unique tile once per frame
cycle. This is the biggest lever by far - a well-cached deployment
serves orders of magnitude more requests than the table above suggests.

### 2. Buy CPU, not RAM

Size memory as roughly **1 GiB of anon per render worker** plus a
**~13 GiB fixed base** for the pipeline and shared data stores. Anything
beyond that just becomes page cache - kernel-reclaimable and rarely
useful. When in doubt, buy more workers, not more RAM.

### 3. Match workers to physical cores

Set `LIBREWXR_WORKERS` to the physical core count. On hyperthreaded
cloud instances that is **vCPUs / 2** - two render workers sharing one
physical core only contend for the same execution units.

## Memory Model

- **Anon memory** is what you must provision: it is the OOM-relevant
  allocation that cgroup limits and instance RAM gate on.
- **File / page cache** is a kernel-managed bonus. It shrinks itself
  under memory pressure, but `docker stats` reports it as container
  memory anyway - so do not size from `docker stats`; size from the
  anon figure in `cluster.memory.container`.
- Extra RAM beyond the anon budget buys nothing except cache that the
  kernel can reclaim on demand.

**Disk:** expect ~20-30 GiB for caches. Note that the coordinate store's
soft cap - `LIBREWXR_COORD_STORE_MB` - is a **disk** budget, not RAM
(default 8192 in multi mode; lower it on small disks - it prunes once
per fetch cycle).

**RRQPE memory:** the RRQPE observed-precipitation layer (NOAA
Enterprise Rain Rate) is now a radar region, so its frames live in the
FrameStore like any other region: 12 past frames × ~29 MB at the
default downsample (`LIBREWXR_RRQPE_DOWNSAMPLE=2`, 0.04° stored) is
~350 MB, plus ~175 MB of nowcast-extrapolated frames (6 × ~29 MB) and
transient warp buffers comparable to USCOMP's existing peaks.  Net add
versus the old NWP-source form is roughly the nowcast half.  Factor 4
(0.08°) cuts a frame to ~7 MB; factor 1 (native 0.02°) is ~117 MB.

## Configuration Knobs

| Setting | Role |
|---|---|
| `LIBREWXR_WORKERS` | Render worker count - set to physical cores, see above. |
| `LIBREWXR_TILE_CACHE_MB` | Tile cache per worker; the 128 MB multi-mode default is fine for these tiers. |
| `LIBREWXR_NWP_FETCH_CONCURRENCY` | Parallel NWP grid decodes in the pipeline; drives decode-time RAM bursts. |
| `LIBREWXR_COORD_STORE_MB` | Disk budget for the shared coordinate store (not RAM). |
| `LIBREWXR_CACHE_DIR` | Shared cache directory; put it on SSD. Recommended - unset falls back to a per-host tempdir with a warning. |
| `LIBREWXR_PIPELINE_MEMORY` | Memory limit for the pipeline container; `12G` compose default. |
| `LIBREWXR_RENDER_MEMORY` | Memory limit for the renderer container; `18G` compose default. |

The compose defaults sum to ~30 GiB of limits (12G pipeline + 18G renderer) - scale these down via the two `LIBREWXR_*_MEMORY` vars on smaller hosts; the 16 GiB minimum tier assumes they have been reduced.

## Bandwidth

Tiles run ~50 KB each as WebP. At the reference workload's ~15 req/s
average that is roughly **6 Mbps** of sustained egress (a bit more with
protocol overhead) - plan for bursts several times higher (storms make
people refresh, and animations re-request the same timestamped tiles).
A CDN absorbs the duplication at the edge, so origin egress usually stays
far below the headline number.

_Note: These rough specs were calculated by Kimi K3 in OpenCode. They may not be perfectly accurate, but it does roughly estimate what my own server capacity would be at those levels from my experience. Your results may vary based on what features are enabled or tweaked. - J.Kimsey_
