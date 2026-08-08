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

## Deployment Shape: Single vs Multi

- **Single mode** is right for personal or small-scope deployments: one
  or two radar regions, a few NWP sources, behind a caching proxy.
- **Single mode is not recommended for public-facing instances.** The
  Python GIL serializes the render path - concurrent cache-miss renders
  queue behind each other in a single process. The narrow exception is a
  genuinely small-scope public deployment: a small area, a few features,
  light traffic.
- **Public-facing means multi mode:** a pipeline sidecar plus N render
  workers (`COMPOSE_PROFILES=multi`, `LIBREWXR_RENDER_ONLY=1` on the
  workers). See [configuration-reference.md](configuration-reference.md)
  for the exact settings.

## Spec Tiers

| Tier | Shape | vCPU | RAM | Disk | Render workers |
|---|---|---|---|---|---|
| Minimum (public) | multi | 8 vCPU | 16 GiB | 50 GiB SSD | 4-8 |
| Recommended (public production) | multi | 16 vCPU | 32 GiB | 50-100 GiB SSD | 8-12 |
| Heavy / no CDN | multi | 32 vCPU | 64 GiB | 100 GiB SSD | up to 32 |

The **minimum** tier is the floor for public traffic - below it, single
mode behind a proxy is the honest shape. The **recommended** tier serves
the reference workload with most render capacity idle; that idle
capacity is the headroom that absorbs storms, flash crowds, and the cold
tile long tail. The **heavy** tier is for deployments that must absorb
tile traffic without a CDN in front. Scale vertically: the designed
shape is one pipeline writer plus N render workers per host - not
multi-node.

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

## Configuration Knobs

| Setting | Role |
|---|---|
| `LIBREWXR_WORKERS` | Render worker count - set to physical cores, see above. |
| `LIBREWXR_TILE_CACHE_MB` | Tile cache per worker; the 128 MB multi-mode default is fine for these tiers. |
| `LIBREWXR_NWP_FETCH_CONCURRENCY` | Parallel NWP grid decodes in the pipeline; drives decode-time RAM bursts. |
| `LIBREWXR_COORD_STORE_MB` | Disk budget for the shared coordinate store (not RAM). |
| `LIBREWXR_CACHE_DIR` | Shared cache directory - required for multi mode. Put it on SSD. |

## Bandwidth

Tiles run ~50 KB each as WebP. At the reference workload's ~15 req/s
average that is roughly **10 Mbps** of sustained egress - plan for
bursts several times higher (storms make people refresh, and animations
re-request the same timestamped tiles). A CDN absorbs the duplication at
the edge, so origin egress usually stays far below the headline number.

_Note: These rough specs were calculated by Kimi K3 in OpenCode. They may not be perfectly accurate, but it does roughly estimate what my own server capacity would be at those levels from my experience. Your results may vary based on what features are enabled or tweaked. - J.Kimsey_
