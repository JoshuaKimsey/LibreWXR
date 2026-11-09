# Health Dashboard and Prometheus Metrics

LibreWXR ships two read-only diagnostic surfaces on top of its existing
`/health` endpoint: a self-hosted operational dashboard page and a
Prometheus metrics exporter. Both render the exact same payload, so the
numbers in the browser and the numbers in your monitoring system can
never disagree.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Disabled endpoints](#disabled-endpoints)
- [Prometheus metrics](#prometheus-metrics)
- [Scrape guidance](#scrape-guidance)
- [Per-worker semantics](#per-worker-semantics)
- [Authentication](#authentication)
- [See also](#see-also)

## Overview

`GET /dashboard` serves a single self-contained HTML page -- one asset
packaged with the server, no telemetry. Its only external requests are
the hot-tile map's MapLibre GL library (unpkg.com) and the OpenFreeMap
dark basemap (tiles.openfreemap.org); when those are unreachable (an
air-gapped or firewalled deployment) the map automatically falls back to
a built-in offline canvas, and the rest of the page makes no external
requests. It is served verbatim from memory and polls `/health` from the
browser (default every 10 seconds; the page header selector offers 5 /
10 / 30 / 60 seconds or manual-only) to render frame counts, per-region
radar state, NWP grid availability, tile-cache and memory breakdowns,
the per-worker cluster aggregation, alerts, storm cells, lightning, and
the WebSocket channel state. Map data is (c) OpenStreetMap contributors,
served by OpenFreeMap; the map is rendered with MapLibre GL and the
required attribution is shown in the map control. Because the asset is
packaged, the page itself always loads on an air-gapped deployment and
costs nothing per request beyond sending the bytes; only the optional
interactive basemap needs egress.

`GET /metrics` serves the Prometheus text-exposition (version 0.0.4) of
that same health payload. It needs no `prometheus_client` dependency:
the exposition is rendered by a pure function over the `/health` dict,
so every metric family maps one-to-one onto a health field and adding a
health section never silently leaves the exporter behind (dynamic NWP
grids, for instance, are discovered from the payload's own top-level
keys).

Both routes are served by every render worker, exactly like `/health`:
there is no separate metrics port or process.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LIBREWXR_DASHBOARD_ENABLED` | `true` | Mount the dashboard page at `LIBREWXR_DASHBOARD_PATH`. When `false`, no route is mounted at all. |
| `LIBREWXR_DASHBOARD_PATH` | `/dashboard` | URL path the dashboard page is served on. Read at import time, so a change needs a restart. |
| `LIBREWXR_METRICS_ENABLED` | `true` | Mount the Prometheus exporter at `LIBREWXR_METRICS_PATH`. When `false`, no route is mounted at all. |
| `LIBREWXR_METRICS_PATH` | `/metrics` | URL path the Prometheus exporter is served on. Read at import time, so a change needs a restart. |

Paths are read from settings when the app is built, the same way
`LIBREWXR_WEBSOCKET_PATH` and `LIBREWXR_MCP_PATH` are. If your reverse
proxy already owns `/metrics` (a common pattern when node_exporter is
fronted by the same vhost), set `LIBREWXR_METRICS_PATH` to something
like `/internal/metrics` and restrict access there.

## Disabled endpoints

`LIBREWXR_DASHBOARD_ENABLED=false` and `LIBREWXR_METRICS_ENABLED=false`
leave no route behind at all: a request is a plain `404` (the same
semantics as the WebSocket channel). Nothing else changes -- `/health`
keeps answering, and the two endpoints are independent of each other.

Neither endpoint is authenticated by LibreWXR itself. See
[Authentication](#authentication) for the reverse-proxy recipes that
keep them private.

## Prometheus metrics

Every exported metric is a **gauge**. Several of them expose counters
(tile requests, coord-cache hits, fast-path totals, lightning trims), but
they are snapshotted as gauges because they are per-worker process
counters: they reset when that worker restarts and they are served by
whichever worker answered the scrape. Use `rate()` / `increase()` on a
gauge at your own risk -- for cluster-wide rates, prefer the `cluster.*`
metrics, which the serving worker aggregates from all its peers'
pulses.

Any metric whose source field is absent or `null` is omitted from the
exposition (Prometheus treats a missing series as stale). A section in
its disabled shape (`{"enabled": false}`) emits **only** its `_enabled 0`
gauge.

Dynamic metric names: NWP grids are exported under a `grid` label
(`librewxr_nwp_grid_*{grid="hrrr_grid"}`). The grid name comes from the
health payload's top-level key, which follows the source slug, so a new
NWP source is exported with no change to the exporter.

| Metric | Labels | `/health` source |
|---|---|---|
| `librewxr_health_status` | | `status` (`1` when `"ok"`, else `0`) |
| `librewxr_uptime_seconds` | | `uptime_seconds` |
| `librewxr_version_info` | `version` | `version` (value is always `1`) |
| `librewxr_color_schemes_info` | `version`, `source`, `last_status` | `color_schemes` (value is always `1`) |
| `librewxr_frames_count` | | `frames.count` |
| `librewxr_frames_max` | | `frames.max` |
| `librewxr_frames_latest_age_seconds` | | `frames.latest_age_seconds` |
| `librewxr_frames_per_region` | `region` | `frames.per_region` |
| `librewxr_tile_cache_entries` | | `tile_cache.entries` |
| `librewxr_tile_cache_used_mb` | | `tile_cache.used_mb` |
| `librewxr_tile_cache_max_mb` | | `tile_cache.max_mb` |
| `librewxr_tile_cache_kind_entries` | `kind` (`geometry`, `present`, `overlay`, `window`, `satellite`) | `tile_cache.<kind>_entries` |
| `librewxr_tile_cache_kind_bytes` | `kind` | `tile_cache.<kind>_bytes` |
| `librewxr_memory_resident_mb` | | `memory.resident_mb` |
| `librewxr_memory_limit_mb` | | `memory.limit_mb` |
| `librewxr_memory_usage_pct` | | `memory.usage_pct` |
| `librewxr_memory_breakdown_mb` | `component` | `memory.breakdown.*_mb` (the `_mb` suffix is stripped; non-MB keys like `coord_store_entries` are skipped) |
| `librewxr_nwp_grid_loaded` | `grid` | `<grid>.loaded` |
| `librewxr_nwp_grid_enabled` | `grid` | `<grid>.enabled` (only when the grid block carries it) |
| `librewxr_nwp_grid_frames` | `grid` | `<grid>.frames` (chain-source shape) |
| `librewxr_nwp_grid_timesteps` | `grid` | `<grid>.timesteps` (IFS shape) |
| `librewxr_nowcast_enabled` | | `nowcast.enabled` |
| `librewxr_nowcast_flows` | | `nowcast.flows` |
| `librewxr_nowcast_nwp_flow` | | `nowcast.nwp_flow` |
| `librewxr_nowcast_frames_count` | | `nowcast.count` |
| `librewxr_satellite_enabled` | | `satellite.enabled` |
| `librewxr_satellite_channel_loaded` | `channel` | `satellite.channels.<channel>.loaded` |
| `librewxr_satellite_channel_frames` | `channel` | `satellite.channels.<channel>.frames` |
| `librewxr_enabled_regions_count` | | `enabled_regions` (length) |
| `librewxr_enabled_regions_info` | `regions` (comma-joined) | `enabled_regions` (value is always `1`) |
| `librewxr_sources_info` | `na_source`, `ca_source`, `msc_blending` | `sources` (value is always `1`; `msc_blending` is only present once blending state is known) |
| `librewxr_radar_cache_enabled` | | `radar_cache.enabled` |
| `librewxr_radar_cache_files` | | `radar_cache.files` |
| `librewxr_radar_cache_used_mb` | | `radar_cache.used_mb` |
| `librewxr_coord_cache_entries` | `cache` | `coord_caches.caches.<cache>.entries` |
| `librewxr_coord_cache_hits` | `cache` | `coord_caches.caches.<cache>.hits` |
| `librewxr_coord_cache_misses` | `cache` | `coord_caches.caches.<cache>.misses` |
| `librewxr_coord_cache_hit_ratio` | `cache` | `coord_caches.caches.<cache>.hit_ratio` (omitted when `null`) |
| `librewxr_coord_store_hits` | | `coord_caches.store.hits` |
| `librewxr_coord_store_misses` | | `coord_caches.store.misses` |
| `librewxr_coord_store_publishes` | | `coord_caches.store.publishes` |
| `librewxr_coord_store_entries` | | `coord_caches.store.entries` |
| `librewxr_coord_store_bytes` | | `coord_caches.store.bytes` |
| `librewxr_cluster_workers_reporting` | | `cluster.workers_reporting` |
| `librewxr_cluster_workers_rss_mb` | `stat` (`sum`, `min`, `max`) | `cluster.memory.workers_rss_mb` |
| `librewxr_cluster_container_mb` | `kind` (`anon`, `file`, `shmem`) | `cluster.memory.container.<kind>_mb` |
| `librewxr_cluster_container_limit_mb` | | `cluster.memory.container.limit_mb` |
| `librewxr_cluster_tile_cache_entries` | | `cluster.tile_cache.entries` |
| `librewxr_cluster_tile_cache_used_mb` | | `cluster.tile_cache.used_mb` |
| `librewxr_cluster_coord_cache_entries` | `cache` | `cluster.coord.caches.<cache>.entries` |
| `librewxr_cluster_coord_cache_hits` | `cache` | `cluster.coord.caches.<cache>.hits` |
| `librewxr_cluster_coord_cache_misses` | `cache` | `cluster.coord.caches.<cache>.misses` |
| `librewxr_cluster_coord_cache_hit_ratio` | `cache` | `cluster.coord.caches.<cache>.hit_ratio` (omitted when `null`) |
| `librewxr_cluster_coord_store_hits` | | `cluster.coord.store.hits` |
| `librewxr_cluster_coord_store_misses` | | `cluster.coord.store.misses` |
| `librewxr_cluster_coord_store_publishes` | | `cluster.coord.store.publishes` |
| `librewxr_cluster_coord_store_entries` | | `cluster.coord.store.entries` |
| `librewxr_cluster_coord_store_bytes` | | `cluster.coord.store.bytes` |
| `librewxr_cluster_requests_total` | | `cluster.requests.total_requests` |
| `librewxr_cluster_cache_hits` | | `cluster.requests.cache_hits` |
| `librewxr_cluster_cache_misses` | | `cluster.requests.cache_misses` |
| `librewxr_cluster_fast_path_total` | | `cluster.requests.fast_path_total` |
| `librewxr_cluster_hot_tiles` | | `cluster.requests.hot_tiles` (summed; may double-count a tile two workers served) |
| `librewxr_cluster_hit_rate` | | `cluster.requests.hit_rate` (recomputed from summed hits/misses) |
| `librewxr_cluster_avg_request_ms` | | `cluster.tile_latency.avg_request_ms` |
| `librewxr_cluster_avg_compute_ms` | | `cluster.tile_latency.avg_compute_ms` |
| `librewxr_cluster_avg_present_ms` | | `cluster.tile_latency.avg_present_ms` |
| `librewxr_tile_requests_enabled` | | `tile_requests.enabled` |
| `librewxr_tile_requests_tracked_tiles` | | `tile_requests.tracked_tiles` |
| `librewxr_tile_requests_total` | | `tile_requests.total_requests` |
| `librewxr_tile_requests_hot_tiles` | | `tile_requests.hot_tiles` |
| `librewxr_tile_requests_cache_hits` | | `tile_requests.cache.hits` |
| `librewxr_tile_requests_cache_misses` | | `tile_requests.cache.misses` |
| `librewxr_tile_requests_cache_hit_rate` | | `tile_requests.cache.hit_rate` |
| `librewxr_tile_requests_avg_request_ms` | | `tile_requests.latency.avg_request_ms` |
| `librewxr_tile_requests_avg_compute_ms` | | `tile_requests.latency.avg_compute_ms` |
| `librewxr_tile_requests_avg_present_ms` | | `tile_requests.latency.avg_present_ms` |
| `librewxr_tile_requests_zoom_tiles` | `zoom` | `tile_requests.by_zoom.<z>.tiles` |
| `librewxr_tile_requests_zoom_requests` | `zoom` | `tile_requests.by_zoom.<z>.requests` |
| `librewxr_tile_requests_top` | `z`, `x`, `y` | `tile_requests.top[]` (`count` as the value) |
| `librewxr_tile_requests_fast_path_total` | | `tile_requests.fast_path.total` |
| `librewxr_tile_requests_fast_path` | `reason` | `tile_requests.fast_path.by_reason` |
| `librewxr_alerts_enabled` | | `alerts.enabled` |
| `librewxr_alerts_count` | | `alerts.count` |
| `librewxr_alerts_last_updated` | | `alerts.last_updated` (epoch seconds) |
| `librewxr_alerts_ingest_ok` | | `alerts.ingest_ok` |
| `librewxr_mcp_enabled` | | `mcp.enabled` |
| `librewxr_mcp_mounted` | | `mcp.mounted` |
| `librewxr_storm_cells_enabled` | | `storm_cells.enabled` |
| `librewxr_storm_cells_count` | | `storm_cells.count` |
| `librewxr_storm_cells_last_updated` | | `storm_cells.last_updated` (epoch seconds) |
| `librewxr_storm_cells_per_region` | `region` | `storm_cells.per_region` |
| `librewxr_lightning_enabled` | | `lightning.enabled` |
| `librewxr_lightning_points` | | `lightning.points` |
| `librewxr_lightning_version` | | `lightning.version` |
| `librewxr_lightning_ceiling_trimmed_total` | | `lightning.ceiling_trimmed_total` |
| `librewxr_websocket_enabled` | | `websocket.enabled` |
| `librewxr_websocket_connections` | | `websocket.connections` |
| `librewxr_websocket_topic_subscribers` | `topic` | `websocket.topics` |

`nwp_chain` is deliberately not exported (it is a static ordered list of
source names; the dashboard renders it, and a scrape gains nothing from
an info metric that only changes on redeploy).

## Scrape guidance

Building the health payload is moderately expensive: it takes the frame
store lock to enumerate timestamps and region keys, walks the tile cache
to split entries by kind, reads every NWP grid's byte count, and scans
the per-worker pulse files under the cache dir for the cluster
aggregation. It is cheap enough for human-paced polling and far too
expensive for sub-second scraping.

- Scrape every **15-60 seconds**. Over that cadence the `cluster.*`
  series are stable and the cost is invisible next to tile serving.
- Never scrape below 1 second, and do not point multiple Prometheus
  replicas at the same worker (every scrape is a full payload build).
- A 404 for `/metrics` means `LIBREWXR_METRICS_ENABLED=false` (or that
  the reverse proxy is not forwarding the path), not a server fault.
- The response is `text/plain; version=0.0.4; charset=utf-8`. It is not
  gzipped by the JSON gzip middleware (that middleware only compresses
  `application/json`), so put it behind a proxy or scrape interval that
  suits your network.

A minimal scrape config:

```yaml
scrape_configs:
  - job_name: librewxr
    metrics_path: /metrics
    scrape_interval: 30s
    static_configs:
      - targets: ["librewxr.example.com:8080"]
```

Behind a load balancer, one scrape hits one render worker, so the
`librewxr_tile_requests_*` and `librewxr_tile_cache_kind_*` series only
describe that worker. Scrape each worker individually (or rely on the
`cluster.*` aggregate) if you need full coverage.

## Per-worker semantics

In the multi-worker architecture every render worker answers `/health`
and `/metrics` from its own process state, plus the cluster aggregation
it derives from the shared worker pulses:

- `librewxr_tile_requests_*` (`top`, `by_zoom`, cache hit/miss counters,
  latency averages, fast-path reasons) and the per-kind
  `librewxr_tile_cache_kind_*` split describe **the serving worker
  only**. Behind a load balancer these move depending on which worker
  answered.
- `librewxr_memory_*` (except `cluster.*`), `librewxr_coord_cache_*`,
  `librewxr_nowcast_*`, `librewxr_satellite_*`, `librewxr_alerts_*`,
  `librewxr_storm_cells_*`, `librewxr_lightning_*`, and
  `librewxr_websocket_*` are likewise per-worker snapshots.
- `librewxr_cluster_*` metrics are the **multi-worker aggregate**: the
  serving worker scans the tiny per-worker pulse files under the shared
  cache dir (mtime-filtered, lock-free) and unions in its own live
  payload. Cluster sums (RSS, tile-cache bytes, request counters) are
  additive; hit rates and latency averages are recomputed from the sums,
  so idle workers do not drag the ratios around. `hot_tiles` is a sum
  and may double-count a tile that several workers served.
- When `cluster` is `null` (no shared cache dir, or the pulse scan
  failed; the failure is logged and `/health` still answers), **no**
  `librewxr_cluster_*` series are exported.
- Frame counts, memory, NWP and satellite state come from the shared
  snapshot on render-only workers, so those are effectively
  cluster-wide already -- only inherently per-process counters differ.

## Authentication

LibreWXR deliberately ships **no built-in authentication**: the dashboard
and the metrics endpoint are ordinary read-only routes, and the project
leaves access control to the reverse proxy that already terminates TLS
for the deployment. The dashboard asset itself issues no requests other
than to its own `/health` and the hot-tile map's MapLibre GL library
(unpkg.com) and OpenFreeMap basemap (tiles.openfreemap.org); everything
else is self-contained, and it sends no telemetry.

Scrapers and browsers must then be pointed at the proxy (or at a
loopback-bound LibreWXR instance), not at the public API host. A few
common patterns, all assuming LibreWXR listens on `127.0.0.1:8080`:

nginx (`auth_basic`):

```nginx
location /dashboard/ {
    auth_basic           "LibreWXR";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass           http://127.0.0.1:8080;
}

location = /metrics {
    auth_basic           "LibreWXR";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass           http://127.0.0.1:8080;
}
```

Caddy (`basic_auth`):

```caddyfile
handle /metrics {
    basic_auth {
        prometheus $2a$14$<bcrypt-hash>
    }
    reverse_proxy 127.0.0.1:8080
}

handle /dashboard* {
    basic_auth {
        operator $2a$14$<bcrypt-hash>
    }
    reverse_proxy 127.0.0.1:8080
}
```

Traefik (`basicAuth` middleware):

```yaml
http:
  middlewares:
    librewxr-auth:
      basicAuth:
        users:
          - "prometheus:$apr1$<htpasswd-hash>"
  routers:
    librewxr-metrics:
      rule: "Host(`librewxr.example.com`) && PathPrefix(`/metrics`)"
      middlewares: ["librewxr-auth"]
      service: librewxr
```

If you would rather not expose the endpoints at all, the simplest option
is to leave the routes enabled for localhost monitoring and have the
reverse proxy never forward `/dashboard` or `/metrics` (or set
`LIBREWXR_METRICS_ENABLED=false` and scrape `/health` yourself). The
dashboard is a static asset that can also be fetched once and hosted
anywhere -- point it at a publicly reachable `/health` if that suits
your setup.

## See also

- [`docs/configuration-reference.md`](configuration-reference.md#monitoring-dashboard--prometheus) -- the four monitoring variables in the full settings reference.
- [`docs/mcp-server.md`](mcp-server.md) -- the MCP endpoints, which report their mount state through `/health`.
- [`docs/web-integration-guide.md`](web-integration-guide.md) -- the Rain Viewer-compatible HTTP API and the WebSocket push channel.
