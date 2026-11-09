# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Prometheus exporter for the ``/health`` payload.

``render_prometheus`` is a pure function turning the dict returned by
``routes.build_health_payload`` into Prometheus text-exposition 0.0.4, so
the metric surface tracks ``/health`` with no second collection path and
no ``prometheus_client`` dependency.  Every emitted metric is a gauge;
the exposed counters (tile requests, coord-cache hits, lightning trims)
are per-worker snapshots and reset when the serving worker restarts.

The router is only included in the app when ``settings.metrics_enabled``
is true, so a disabled endpoint has no route at all (a plain 404).
"""

from fastapi import APIRouter
from fastapi.responses import Response

from librewxr.api import routes
from librewxr.config import settings

router = APIRouter()

# Top-level /health keys with a fixed meaning.  Any OTHER top-level key is
# an NWP grid block (one per ``nwp_grids`` slug, e.g. ``hrrr_grid`` /
# ``ecmwf_grid``), so adding an NWP source needs no edit here.
_FIXED_HEALTH_KEYS = frozenset({
    "status",
    "uptime_seconds",
    "version",
    "color_schemes",
    "cluster",
    "memory",
    "frames",
    "tile_cache",
    "nwp_chain",
    "nowcast",
    "satellite",
    "enabled_regions",
    "sources",
    "radar_cache",
    "coord_caches",
    "tile_requests",
    "alerts",
    "mcp",
    "storm_cells",
    "lightning",
    "websocket",
})

_TILE_CACHE_KINDS = ("geometry", "present", "overlay", "window", "satellite")


def _dict(value: object) -> dict:
    """Return ``value`` when it is a dict, else an empty one."""
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list:
    """Return ``value`` when it is a list, else an empty one."""
    return value if isinstance(value, list) else []


def _format_value(value: object) -> str | None:
    """Render a sample value; None means "skip this metric".

    Booleans become ``1``/``0`` (``True`` must not serialize as ``True``).
    Non-numeric values are skipped rather than emitted as invalid text.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _escape_label_value(value: object) -> str:
    """Escape a label value per the Prometheus text-format spec."""
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _format_labels(labels: dict | None) -> str:
    """Render a ``{k="v",...}`` label set (empty string when there are none)."""
    if not labels:
        return ""
    parts = [
        f'{name}="{_escape_label_value(value)}"'
        for name, value in labels.items()
    ]
    return "{" + ",".join(parts) + "}"


class _Families:
    """Ordered metric families, with HELP/TYPE emitted once per family.

    Samples are buffered per family so every family's lines stay grouped
    (the text format forbids interleaving two families' samples even
    though collection here walks the payload section by section).
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._help: dict[str, str] = {}
        self._samples: dict[str, list[str]] = {}

    def add(
        self,
        name: str,
        value: object,
        help_text: str,
        labels: dict | None = None,
    ) -> None:
        rendered = _format_value(value)
        if rendered is None:
            return
        if name not in self._samples:
            self._order.append(name)
            self._help[name] = help_text
            self._samples[name] = []
        self._samples[name].append(f"{name}{_format_labels(labels)} {rendered}")

    def render(self) -> str:
        lines: list[str] = []
        for name in self._order:
            lines.append(f"# HELP {name} {self._help[name]}")
            lines.append(f"# TYPE {name} gauge")
            lines.extend(self._samples[name])
        return "\n".join(lines) + "\n" if lines else ""


def render_prometheus(payload: dict) -> str:
    """Render a ``/health`` payload as Prometheus text-exposition 0.0.4.

    Robust by design: a missing key, a null value, a disabled section, or
    a payload of the wrong shape skips the affected metrics instead of
    raising, so a scrape never 500s.
    """
    families = _Families()
    try:
        _collect(families, _dict(payload))
    except Exception:
        # Malformed input yields a partial scrape rather than a failed one.
        pass
    return families.render()


def _collect(out: _Families, payload: dict) -> None:
    """Add every metric family derived from ``payload`` to ``out``."""
    # --- status / uptime / version -------------------------------------
    out.add(
        "librewxr_health_status",
        payload.get("status") == "ok",
        "Server health: 1 = ok, 0 = degraded",
    )
    out.add(
        "librewxr_uptime_seconds",
        payload.get("uptime_seconds"),
        "Seconds since the serving worker started",
    )
    version = payload.get("version")
    if version is not None:
        out.add(
            "librewxr_version_info",
            1,
            "Running LibreWXR version (constant 1, version is a label)",
            {"version": version},
        )

    # --- color schemes --------------------------------------------------
    palette = payload.get("color_schemes")
    if isinstance(palette, dict):
        out.add(
            "librewxr_color_schemes_info",
            1,
            "Applied color-scheme table (constant 1, details are labels)",
            {
                "version": palette.get("version"),
                "source": palette.get("source"),
                "last_status": palette.get("last_status"),
            },
        )

    # --- radar frames ---------------------------------------------------
    frames = _dict(payload.get("frames"))
    if frames:
        out.add(
            "librewxr_frames_count",
            frames.get("count"),
            "Radar frames held in the frame store",
        )
        out.add(
            "librewxr_frames_max",
            frames.get("max"),
            "Configured maximum past radar frames (LIBREWXR_MAX_FRAMES)",
        )
        out.add(
            "librewxr_frames_latest_age_seconds",
            frames.get("latest_age_seconds"),
            "Age in seconds of the newest radar frame",
        )
        for region, count in _dict(frames.get("per_region")).items():
            out.add(
                "librewxr_frames_per_region",
                count,
                "Radar frames held per region",
                {"region": region},
            )

    # --- tile cache -----------------------------------------------------
    tile_cache = _dict(payload.get("tile_cache"))
    if tile_cache:
        out.add(
            "librewxr_tile_cache_entries",
            tile_cache.get("entries"),
            "Tile cache entries (geometry records plus encoded variants)",
        )
        out.add(
            "librewxr_tile_cache_used_mb",
            tile_cache.get("used_mb"),
            "Tile cache memory in use (MB)",
        )
        out.add(
            "librewxr_tile_cache_max_mb",
            tile_cache.get("max_mb"),
            "Tile cache byte budget (MB)",
        )
        for kind in _TILE_CACHE_KINDS:
            out.add(
                "librewxr_tile_cache_kind_entries",
                tile_cache.get(f"{kind}_entries"),
                "Tile cache entries by entry kind",
                {"kind": kind},
            )
            out.add(
                "librewxr_tile_cache_kind_bytes",
                tile_cache.get(f"{kind}_bytes"),
                "Tile cache bytes by entry kind",
                {"kind": kind},
            )

    # --- memory ---------------------------------------------------------
    memory = _dict(payload.get("memory"))
    if memory:
        out.add(
            "librewxr_memory_resident_mb",
            memory.get("resident_mb"),
            "Serving worker resident set size (MB)",
        )
        out.add(
            "librewxr_memory_limit_mb",
            memory.get("limit_mb"),
            "Effective container memory limit (MB)",
        )
        out.add(
            "librewxr_memory_usage_pct",
            memory.get("usage_pct"),
            "RSS as a percentage of the effective memory limit",
        )
        for component, value in _dict(memory.get("breakdown")).items():
            if not component.endswith("_mb"):
                continue
            out.add(
                "librewxr_memory_breakdown_mb",
                value,
                "Memory breakdown by component (MB)",
                {"component": component[:-3]},
            )

    # --- NWP grids (one dynamic top-level key per grid) ------------------
    for grid, block in payload.items():
        if grid in _FIXED_HEALTH_KEYS:
            continue
        block = _dict(block)
        if not block:
            continue
        out.add(
            "librewxr_nwp_grid_loaded",
            block.get("loaded"),
            "NWP grid holds usable data (1 = loaded)",
            {"grid": grid},
        )
        if "enabled" in block:
            out.add(
                "librewxr_nwp_grid_enabled",
                block.get("enabled"),
                "NWP grid source enabled (1 = on)",
                {"grid": grid},
            )
        if "frames" in block:
            out.add(
                "librewxr_nwp_grid_frames",
                block.get("frames"),
                "NWP grid frame count (chain-source shape)",
                {"grid": grid},
            )
        if "timesteps" in block:
            out.add(
                "librewxr_nwp_grid_timesteps",
                block.get("timesteps"),
                "NWP grid timestep count (IFS shape)",
                {"grid": grid},
            )

    # ``nwp_chain`` (the ordered source-name list) is intentionally not
    # exported: the dashboard renders it, and a scrape gains nothing from
    # a label-only info metric that changes only on redeploy.

    # --- nowcast --------------------------------------------------------
    nowcast = _dict(payload.get("nowcast"))
    if nowcast:
        out.add(
            "librewxr_nowcast_enabled",
            nowcast.get("enabled"),
            "Nowcast generation enabled (1 = on)",
        )
        out.add(
            "librewxr_nowcast_flows",
            nowcast.get("flows"),
            "Per-region optical-flow fields held",
        )
        out.add(
            "librewxr_nowcast_nwp_flow",
            nowcast.get("nwp_flow"),
            "Composite NWP flow raster available (1 = yes)",
        )
        out.add(
            "librewxr_nowcast_frames_count",
            nowcast.get("count"),
            "Nowcast forecast frames held",
        )

    # --- satellite ------------------------------------------------------
    satellite = _dict(payload.get("satellite"))
    if satellite:
        out.add(
            "librewxr_satellite_enabled",
            satellite.get("enabled"),
            "Satellite layer enabled (1 = on)",
        )
        for channel, info in _dict(satellite.get("channels")).items():
            info = _dict(info)
            out.add(
                "librewxr_satellite_channel_loaded",
                info.get("loaded"),
                "Satellite channel has frames (1 = loaded)",
                {"channel": channel},
            )
            out.add(
                "librewxr_satellite_channel_frames",
                info.get("frames"),
                "Satellite channel frame count",
                {"channel": channel},
            )

    # --- enabled regions ------------------------------------------------
    regions = payload.get("enabled_regions")
    if isinstance(regions, list):
        out.add(
            "librewxr_enabled_regions_count",
            len(regions),
            "Number of enabled radar regions",
        )
        if regions:
            out.add(
                "librewxr_enabled_regions_info",
                1,
                "Enabled radar region names (constant 1, regions is a label)",
                {"regions": ",".join(str(r) for r in regions)},
            )

    # --- radar sources --------------------------------------------------
    sources = payload.get("sources")
    if isinstance(sources, dict):
        labels = {
            "na_source": sources.get("na_source"),
            "ca_source": sources.get("ca_source"),
        }
        blending = sources.get("cacomp_msc_blending")
        if blending is not None:
            labels["msc_blending"] = "true" if blending else "false"
        out.add(
            "librewxr_sources_info",
            1,
            "Configured radar source selection (constant 1)",
            labels,
        )

    # --- persistent radar cache -----------------------------------------
    radar_cache = payload.get("radar_cache")
    if isinstance(radar_cache, dict):
        out.add(
            "librewxr_radar_cache_enabled",
            radar_cache.get("enabled"),
            "Persistent radar disk cache enabled (1 = on)",
        )
        if radar_cache.get("enabled"):
            out.add(
                "librewxr_radar_cache_files",
                radar_cache.get("files"),
                "Files in the persistent radar cache",
            )
            out.add(
                "librewxr_radar_cache_used_mb",
                radar_cache.get("used_mb"),
                "Persistent radar cache footprint (MB)",
            )

    # --- coordinate caches (per worker) ---------------------------------
    coord = payload.get("coord_caches")
    if isinstance(coord, dict):
        for name, info in _dict(coord.get("caches")).items():
            info = _dict(info)
            out.add(
                "librewxr_coord_cache_entries",
                info.get("entries"),
                "Coordinate cache entries",
                {"cache": name},
            )
            out.add(
                "librewxr_coord_cache_hits",
                info.get("hits"),
                "Coordinate cache hits",
                {"cache": name},
            )
            out.add(
                "librewxr_coord_cache_misses",
                info.get("misses"),
                "Coordinate cache misses",
                {"cache": name},
            )
            out.add(
                "librewxr_coord_cache_hit_ratio",
                info.get("hit_ratio"),
                "Coordinate cache hit ratio",
                {"cache": name},
            )
        store = coord.get("store")
        if isinstance(store, dict):
            out.add(
                "librewxr_coord_store_hits",
                store.get("hits"),
                "Shared coord-store hits",
            )
            out.add(
                "librewxr_coord_store_misses",
                store.get("misses"),
                "Shared coord-store misses",
            )
            out.add(
                "librewxr_coord_store_publishes",
                store.get("publishes"),
                "Shared coord-store publishes",
            )
            out.add(
                "librewxr_coord_store_entries",
                store.get("entries"),
                "Shared coord-store entries",
            )
            out.add(
                "librewxr_coord_store_bytes",
                store.get("bytes"),
                "Shared coord-store footprint (bytes)",
            )

    # --- cluster aggregation (multi-worker) ------------------------------
    cluster = payload.get("cluster")
    if isinstance(cluster, dict):
        out.add(
            "librewxr_cluster_workers_reporting",
            cluster.get("workers_reporting"),
            "Render workers reporting in the cluster aggregation",
        )
        cluster_memory = _dict(cluster.get("memory"))
        for stat, value in _dict(cluster_memory.get("workers_rss_mb")).items():
            out.add(
                "librewxr_cluster_workers_rss_mb",
                value,
                "Per-worker RSS across the cluster (MB)",
                {"stat": stat},
            )
        for stat, value in _dict(cluster.get("workers_pss_mb")).items():
            out.add(
                "librewxr_cluster_workers_pss_mb",
                value,
                "Per-worker PSS across the cluster (MB)",
                {"stat": stat},
            )
        # The per-worker list (``cluster.workers``) is deliberately not
        # exported: pid labels churn on every restart, so a per-pid series
        # would leave a permanent stale-series trail for no analytic gain.
        pipeline = _dict(cluster.get("pipeline"))
        out.add(
            "librewxr_cluster_pipeline_reporting",
            pipeline.get("reporting"),
            "Fetch pipeline pulse is fresh in the cluster aggregation (1 = yes)",
        )
        if pipeline.get("reporting"):
            out.add(
                "librewxr_cluster_pipeline_rss_mb",
                pipeline.get("rss_mb"),
                "Data pipeline resident set size (MB)",
            )
            out.add(
                "librewxr_cluster_pipeline_pss_mb",
                pipeline.get("pss_mb"),
                "Data pipeline proportional set size (MB)",
            )
        container = cluster_memory.get("container")
        if isinstance(container, dict):
            for kind in ("anon", "file", "shmem", "current"):
                out.add(
                    "librewxr_cluster_container_mb",
                    container.get(f"{kind}_mb"),
                    "Container cgroup memory (MB); kind=current is the "
                    "kernel's own usage figure including page cache and "
                    "kernel memory",
                    {"kind": kind},
                )
            out.add(
                "librewxr_cluster_container_limit_mb",
                container.get("limit_mb"),
                "Container cgroup memory limit (MB)",
            )
        cluster_tile_cache = _dict(cluster.get("tile_cache"))
        out.add(
            "librewxr_cluster_tile_cache_entries",
            cluster_tile_cache.get("entries"),
            "Tile cache entries summed across the cluster",
        )
        out.add(
            "librewxr_cluster_tile_cache_used_mb",
            cluster_tile_cache.get("used_mb"),
            "Tile cache bytes summed across the cluster (MB)",
        )
        cluster_coord = _dict(cluster.get("coord"))
        for name, info in _dict(cluster_coord.get("caches")).items():
            info = _dict(info)
            out.add(
                "librewxr_cluster_coord_cache_entries",
                info.get("entries"),
                "Coordinate cache entries summed across the cluster",
                {"cache": name},
            )
            out.add(
                "librewxr_cluster_coord_cache_hits",
                info.get("hits"),
                "Coordinate cache hits summed across the cluster",
                {"cache": name},
            )
            out.add(
                "librewxr_cluster_coord_cache_misses",
                info.get("misses"),
                "Coordinate cache misses summed across the cluster",
                {"cache": name},
            )
            out.add(
                "librewxr_cluster_coord_cache_hit_ratio",
                info.get("hit_ratio"),
                "Cluster coordinate cache hit ratio (from summed hits/misses)",
                {"cache": name},
            )
        store = cluster_coord.get("store")
        if isinstance(store, dict):
            out.add(
                "librewxr_cluster_coord_store_hits",
                store.get("hits"),
                "Shared coord-store hits summed across the cluster",
            )
            out.add(
                "librewxr_cluster_coord_store_misses",
                store.get("misses"),
                "Shared coord-store misses summed across the cluster",
            )
            out.add(
                "librewxr_cluster_coord_store_publishes",
                store.get("publishes"),
                "Shared coord-store publishes summed across the cluster",
            )
            out.add(
                "librewxr_cluster_coord_store_entries",
                store.get("entries"),
                "Shared coord-store entries (global on-disk store)",
            )
            out.add(
                "librewxr_cluster_coord_store_bytes",
                store.get("bytes"),
                "Shared coord-store footprint (bytes, global on-disk store)",
            )
        requests = _dict(cluster.get("requests"))
        out.add(
            "librewxr_cluster_requests_total",
            requests.get("total_requests"),
            "Tracked tile requests summed across the cluster",
        )
        out.add(
            "librewxr_cluster_cache_hits",
            requests.get("cache_hits"),
            "Tracked tile cache hits summed across the cluster",
        )
        out.add(
            "librewxr_cluster_cache_misses",
            requests.get("cache_misses"),
            "Tracked tile cache misses summed across the cluster",
        )
        out.add(
            "librewxr_cluster_fast_path_total",
            requests.get("fast_path_total"),
            "Fast-path tile responses summed across the cluster",
        )
        out.add(
            "librewxr_cluster_hot_tiles",
            requests.get("hot_tiles"),
            "Hot tracked tiles summed across the cluster (may double-count)",
        )
        out.add(
            "librewxr_cluster_hit_rate",
            requests.get("hit_rate"),
            "Cluster tile cache hit rate (from summed hits/misses)",
        )
        tile_latency = _dict(cluster.get("tile_latency"))
        out.add(
            "librewxr_cluster_avg_request_ms",
            tile_latency.get("avg_request_ms"),
            "Mean tracked request latency across the cluster (ms)",
        )
        out.add(
            "librewxr_cluster_avg_compute_ms",
            tile_latency.get("avg_compute_ms"),
            "Mean tile geometry compute time across the cluster (ms)",
        )
        out.add(
            "librewxr_cluster_avg_present_ms",
            tile_latency.get("avg_present_ms"),
            "Mean tile present/encode time across the cluster (ms)",
        )

    # --- tile request tracking (serving worker) --------------------------
    tracker = payload.get("tile_requests")
    if isinstance(tracker, dict):
        out.add(
            "librewxr_tile_requests_enabled",
            tracker.get("enabled"),
            "Tile request tracking enabled (1 = on)",
        )
        if tracker.get("enabled"):
            out.add(
                "librewxr_tile_requests_tracked_tiles",
                tracker.get("tracked_tiles"),
                "Distinct tiles tracked by the serving worker",
            )
            out.add(
                "librewxr_tile_requests_total",
                tracker.get("total_requests"),
                "Tracked tile requests handled by the serving worker",
            )
            out.add(
                "librewxr_tile_requests_hot_tiles",
                tracker.get("hot_tiles"),
                "Tracked tiles at or above the hot threshold (serving worker)",
            )
            cache = _dict(tracker.get("cache"))
            out.add(
                "librewxr_tile_requests_cache_hits",
                cache.get("hits"),
                "Tracked tile cache hits (serving worker)",
            )
            out.add(
                "librewxr_tile_requests_cache_misses",
                cache.get("misses"),
                "Tracked tile cache misses (serving worker)",
            )
            out.add(
                "librewxr_tile_requests_cache_hit_rate",
                cache.get("hit_rate"),
                "Tracked tile cache hit rate (serving worker)",
            )
            latency = _dict(tracker.get("latency"))
            out.add(
                "librewxr_tile_requests_avg_request_ms",
                latency.get("avg_request_ms"),
                "Mean tracked request latency (ms, serving worker)",
            )
            out.add(
                "librewxr_tile_requests_avg_compute_ms",
                latency.get("avg_compute_ms"),
                "Mean tile geometry compute time (ms, serving worker)",
            )
            out.add(
                "librewxr_tile_requests_avg_present_ms",
                latency.get("avg_present_ms"),
                "Mean tile present/encode time (ms, serving worker)",
            )
            for zoom, info in _dict(tracker.get("by_zoom")).items():
                info = _dict(info)
                out.add(
                    "librewxr_tile_requests_zoom_tiles",
                    info.get("tiles"),
                    "Distinct tracked tiles by zoom",
                    {"zoom": str(zoom)},
                )
                out.add(
                    "librewxr_tile_requests_zoom_requests",
                    info.get("requests"),
                    "Tracked requests by zoom",
                    {"zoom": str(zoom)},
                )
            for entry in _list(tracker.get("top")):
                entry = _dict(entry)
                out.add(
                    "librewxr_tile_requests_top",
                    entry.get("count"),
                    "Most-requested tracked tiles",
                    {
                        "z": entry.get("z"),
                        "x": entry.get("x"),
                        "y": entry.get("y"),
                    },
                )
            fast_path = _dict(tracker.get("fast_path"))
            out.add(
                "librewxr_tile_requests_fast_path_total",
                fast_path.get("total"),
                "Tiles served by a fast path (empty / transparent)",
            )
            for reason, count in _dict(fast_path.get("by_reason")).items():
                out.add(
                    "librewxr_tile_requests_fast_path",
                    count,
                    "Fast-path tile responses by reason",
                    {"reason": reason},
                )

    # --- alerts ---------------------------------------------------------
    alerts = payload.get("alerts")
    if isinstance(alerts, dict):
        out.add(
            "librewxr_alerts_enabled",
            alerts.get("enabled"),
            "Weather alerts enabled (1 = on)",
        )
        if alerts.get("enabled"):
            out.add(
                "librewxr_alerts_count",
                alerts.get("count"),
                "Held weather alerts",
            )
            out.add(
                "librewxr_alerts_last_updated",
                alerts.get("last_updated"),
                "Unix timestamp of the last successful alerts ingest",
            )
            out.add(
                "librewxr_alerts_ingest_ok",
                alerts.get("ingest_ok"),
                "Last alerts ingest succeeded (1 = yes)",
            )

    # --- MCP ------------------------------------------------------------
    mcp = payload.get("mcp")
    if isinstance(mcp, dict):
        out.add(
            "librewxr_mcp_enabled",
            mcp.get("enabled"),
            "MCP HTTP transport enabled (1 = on)",
        )
        if mcp.get("enabled"):
            out.add(
                "librewxr_mcp_mounted",
                mcp.get("mounted"),
                "MCP HTTP transport mounted in the app (1 = yes)",
            )

    # --- storm cells ----------------------------------------------------
    storm_cells = payload.get("storm_cells")
    if isinstance(storm_cells, dict):
        out.add(
            "librewxr_storm_cells_enabled",
            storm_cells.get("enabled"),
            "Storm-cell detection enabled (1 = on)",
        )
        if storm_cells.get("enabled"):
            out.add(
                "librewxr_storm_cells_count",
                storm_cells.get("count"),
                "Detected storm cells held",
            )
            out.add(
                "librewxr_storm_cells_last_updated",
                storm_cells.get("last_updated"),
                "Unix timestamp of the last storm-cell detection pass",
            )
            for region, count in _dict(storm_cells.get("per_region")).items():
                out.add(
                    "librewxr_storm_cells_per_region",
                    count,
                    "Detected storm cells per region",
                    {"region": region},
                )

    # --- lightning ------------------------------------------------------
    lightning = payload.get("lightning")
    if isinstance(lightning, dict):
        out.add(
            "librewxr_lightning_enabled",
            lightning.get("enabled"),
            "Lightning layer enabled (1 = on)",
        )
        if lightning.get("enabled"):
            out.add(
                "librewxr_lightning_points",
                lightning.get("points"),
                "Flash points held inside the retention window",
            )
            out.add(
                "librewxr_lightning_version",
                lightning.get("version"),
                "Lightning artifact version (changes on every ingest write)",
            )
            out.add(
                "librewxr_lightning_ceiling_trimmed_total",
                lightning.get("ceiling_trimmed_total"),
                "Flashes dropped because of the store RAM ceiling (serving worker)",
            )

    # --- websocket ------------------------------------------------------
    websocket = payload.get("websocket")
    if isinstance(websocket, dict):
        out.add(
            "librewxr_websocket_enabled",
            websocket.get("enabled"),
            "WebSocket push channel enabled (1 = on)",
        )
        if websocket.get("enabled"):
            out.add(
                "librewxr_websocket_connections",
                websocket.get("connections"),
                "Live WebSocket connections on the serving worker",
            )
            for topic, count in _dict(websocket.get("topics")).items():
                out.add(
                    "librewxr_websocket_topic_subscribers",
                    count,
                    "WebSocket subscribers per topic (serving worker)",
                    {"topic": topic},
                )


@router.get(settings.metrics_path)
async def metrics() -> Response:
    """Serve the ``/health`` snapshot as Prometheus text exposition."""
    payload = await routes.build_health_payload()
    return Response(
        content=render_prometheus(payload),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
