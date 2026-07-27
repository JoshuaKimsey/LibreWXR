# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""FastMCP app with HTTP and stdio transport support.

Provides:
- ``build_mcp_http_app()`` -- factory for the HTTP transport ASGI app
  (mounted inside the FastAPI app in Phase C).
- ``main()`` -- console entry point for the stdio transport (runs the
  ``build_stdio_lifespan`` from ``context.py`` to pull data from the
  pipeline's ``state.json``).

Tool wrappers read ``routes.*`` lazily (at call time, not at module
load) so they work regardless of which lifespan set the singletons up.
"""

import logging

from fastmcp import FastMCP

from librewxr.config import settings
from librewxr.api import routes
from librewxr.api.models import AlertsResponse
from librewxr.mcp import tools

logger = logging.getLogger(__name__)


def _register_tools(mcp: FastMCP) -> None:
    """Register MCP tool wrappers that read ``routes.*`` at call time."""

    @mcp.tool(name="get_precip_nowcast")
    async def _precip_nowcast_tool(
        lat: float,
        lon: float,
        minutes: int = 60,
    ) -> list[dict]:
        """Get a precipitation nowcast for a geographic point.

        Returns a list of future frames (up to ``minutes`` minutes ahead),
        each with timestamp, dBZ, rain rate (mm/h), data source
        (``radar``/``nwp``/``none``), blend weight, and coverage
        (``in_range``/``out_of_range``).  Falls back to NWP when the point
        is outside radar coverage; returns an empty list when nowcast is
        disabled.

        Args:
            lat: Query latitude in degrees (-90 to 90).
            lon: Query longitude in degrees (-180 to 180).
            minutes: Forecast horizon in minutes (1-60, default 60).
        """
        return await tools.get_precip_nowcast(
            routes.nwp_chain,
            routes.nowcast_store,
            routes.enabled_regions or [],
            lat,
            lon,
            minutes,
        )

    @mcp.tool(name="get_active_alerts")
    async def _active_alerts_tool(
        lat: float,
        lon: float,
        radius_km: float = 25.0,
        severity: str | None = None,
    ) -> dict:
        """Get active weather alerts within a radius of a geographic point.

        Returns a GeoJSON FeatureCollection of WMO CAP alerts, enriched
        with US NWS point alerts for US locations.  Filter by severity.
        Returns an empty FeatureCollection when alerts are disabled or
        no alerts match; never raises.

        Args:
            lat: Query latitude in degrees (-90 to 90).
            lon: Query longitude in degrees (-180 to 180).
            radius_km: Search radius in kilometres (default 25.0).
            severity: Optional severity filter -- one of
                ``"Extreme"``, ``"Severe"``, ``"Moderate"``, ``"Minor"``.
        """
        result: AlertsResponse = await tools.get_active_alerts(
            routes.alerts_store,
            routes.alerts_enabled,
            lat,
            lon,
            radius_km,
            severity,
        )
        return result.model_dump()


def build_mcp_http_app():
    """Build the FastMCP instance for the HTTP transport.

    No custom lifespan -- the HTTP path's lifespan comes from
    ``combine_lifespans(librewxr_lifespan, mcp_app.lifespan)`` wired in
    Phase C, not from the FastMCP instance itself.  Tools read
    ``routes.*`` at call time.

    The sub-app serves MCP at its root (``path="/"``) so that mounting it
    at ``settings.mcp_path`` (e.g. ``/mcp``) puts the endpoint at the
    final URL ``/mcp`` -- NOT ``/mcp/mcp``.  See the FastMCP Lifespans
    doc pattern (``mcp.http_app(path="/")`` + ``app.mount("/mcp", ...)``).
    """
    mcp = FastMCP("librewxr-mcp")
    _register_tools(mcp)
    return mcp.http_app(path="/")


def main() -> None:
    """Console entry point for the stdio transport (``librewxr-mcp``).

    Requires ``LIBREWXR_CACHE_DIR`` to point at the shared volume where
    the data pipeline (or single-mode server) writes ``state.json``.
    """
    from librewxr.mcp.context import build_stdio_lifespan

    if not settings.cache_dir:
        raise SystemExit(
            "LIBREWXR_CACHE_DIR must be set for the stdio MCP transport -- "
            "it's the shared directory the data pipeline (or single-mode "
            "server) writes state.json into."
        )
    mcp = FastMCP("librewxr-mcp", lifespan=build_stdio_lifespan)
    _register_tools(mcp)
    logger.info("Starting librewxr-mcp stdio transport")
    mcp.run(transport="stdio")
