# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Integration smoke tests for the MCP HTTP transport.

Tests that ``combine_lifespans`` correctly wires both the app lifespan and
the MCP lifespan, that tool wrappers read ``routes.*`` singletons at call
time (not at module load), and that the mount is present.  Each test builds
a minimal FastAPI app that mirrors the Phase-C wiring pattern from
``main.py``.

The FastMCP ``Client`` connects in-process via ``FastMCPTransport``, so
it does not send HTTP requests through the mounted ASGI sub-app.  The
test instead validates the three critical concerns:

1.  ``combine_lifespans(app_lifespan, mcp_app.lifespan)`` — the combined
    lifespan is entered manually and both sub-lifespans run.
2.  The tools callable via ``Client`` read ``routes.*`` that were set by
    ``app_lifespan``, proving the lazy-read contract.
3.  The ``app.mount("/mcp", mcp_app)`` is present in the route table and
    responds (via GET redirect) without a 404.
"""

import pytest
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp import FastMCP, Client
from fastmcp.utilities.lifespan import combine_lifespans

from librewxr.api import routes
from librewxr.mcp.server import _register_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _save_restore_routes_state():
    """Save and restore routes module-level state to prevent cross-test pollution."""
    saved = {
        "nwp_chain": routes.nwp_chain,
        "nowcast_store": routes.nowcast_store,
        "enabled_regions": routes.enabled_regions,
        "alerts_store": routes.alerts_store,
        "alerts_enabled": routes.alerts_enabled,
    }
    yield
    for key, val in saved.items():
        setattr(routes, key, val)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_app_and_mcp(setup_routes):
    """Build a minimal FastAPI app with combined lifespan + MCP mount.

    Parameters
    ----------
    setup_routes : callable
        A zero-argument callable that sets ``routes.*`` singletons to the
        desired state.  It is called inside ``app_lifespan``, before the
        combined lifespan yields, so that tool wrappers (which read
        ``routes.*`` lazily) see the correct state.

    Returns
    -------
    tuple[FastAPI, FastMCP, Callable]
        ``(app, mcp, combined_lifespan)`` — the FastAPI application, the
        FastMCP instance (for the in-process Client transport), and the
        combined lifespan context manager factory so the caller can enter
        it manually (httpx ASGI transport does not send lifespan events).
    """

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        setup_routes()
        yield
        # teardown — reset routes.* to safe defaults
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    mcp = FastMCP("test-librewxr-mcp")
    _register_tools(mcp)
    # Mirror the production pattern: sub-app serves MCP at root, mount adds the prefix.
    mcp_app = mcp.http_app(path="/")
    combined = combine_lifespans(app_lifespan, mcp_app.lifespan)
    app = FastAPI(lifespan=combined)
    app.mount("/mcp", mcp_app)
    return app, mcp, combined


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.mcp
async def test_get_precip_nowcast_over_http_with_no_data():
    """Call ``get_precip_nowcast`` when nowcast_store is None — must return [].

    Exercises lifespan combining (the app lifespan sets ``routes.*`` to
    degraded-empty) and proves the tool wrapper reads ``routes.*`` at
    call time.
    """

    def _setup():
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    app, mcp, combined = _build_app_and_mcp(_setup)

    async with combined(app):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_precip_nowcast",
                {"lat": 40.0, "lon": -100.0, "minutes": 60},
            )
            assert result.data == [], f"Expected empty list, got {result.data!r}"


@pytest.mark.mcp
async def test_get_active_alerts_over_http_disabled():
    """Call ``get_active_alerts`` when alerts are disabled — must return empty FC.

    Proves the alerts tool path and lifespan combining work correctly.
    """

    def _setup():
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    app, mcp, combined = _build_app_and_mcp(_setup)

    async with combined(app):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_active_alerts",
                {"lat": 40.0, "lon": -100.0},
            )
            expected = {"type": "FeatureCollection", "features": []}
            assert result.data == expected, (
                f"Expected empty FeatureCollection, got {result.data!r}"
            )


@pytest.mark.mcp
async def test_mcp_mount_route_present():
    """Verify the ``/mcp`` mount exists in the FastAPI app's route table.

    This is a belt-and-braces check alongside the client-based tool tests.
    """

    def _setup():
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    app, _mcp, _combined = _build_app_and_mcp(_setup)

    # Check the app router for the Mount route
    mount_paths = [r.path for r in app.routes if getattr(r, "path", None) == "/mcp"]
    assert len(mount_paths) == 1, (
        f"Expected exactly one Mount route at /mcp, found {mount_paths}"
    )

    # Also verify via HTTP — GET /mcp returns a redirect (307), not 404,
    # proving the mount is alive even without entering a lifespan.
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/mcp", follow_redirects=False)
        assert resp.status_code != 404, (
            "Expected /mcp mount to exist, got 404"
        )


@pytest.mark.mcp
async def test_mcp_endpoint_callable_over_real_http():
    """POST a real MCP ``initialize`` JSON-RPC to ``/mcp`` -- must respond, not 404.

    This is the regression test for the ``/mcp/mcp`` path-doubling bug:
    with ``http_app(path="/")`` + ``app.mount("/mcp", ...)``, the MCP
    endpoint lives at ``/mcp``.  An earlier version used
    ``http_app(path="/mcp")`` which produced ``/mcp/mcp`` and made every
    HTTP MCP client 404 on the documented path.
    """
    from httpx import ASGITransport, AsyncClient

    def _setup():
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    app, _mcp, combined = _build_app_and_mcp(_setup)

    transport = ASGITransport(app=app)
    # The MCP lifespan must be entered before the sub-app will answer
    # (otherwise the StreamableHTTPSessionManager task group isn't initialized).
    async with combined(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # MCP Streamable HTTP initialize handshake.
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "regression-test", "version": "0.1"},
                },
            }
            # NOTE: Starlette's ``Mount`` redirects ``POST /mcp`` (no trailing
            # slash) with a 307 to ``/mcp/``, so we POST to ``/mcp/`` to reach
            # the actual MCP endpoint.  The docs may need a trailing-slash note.
            resp = await ac.post(
                "/mcp/",
                json=init_payload,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code != 404, (
                f"Expected MCP endpoint at /mcp/, got 404 -- path doubling regressed. "
                f"Body: {resp.text[:200]!r}"
            )
            # The MCP initialize response should be 200 or 202 (the latter if the
            # server responds via SSE).  Either way it must contain a JSON-RPC
            # envelope with the matching request id, OR be an SSE stream that
            # carries one.
            assert resp.status_code in (200, 202), (
                f"Unexpected status {resp.status_code} from /mcp/ MCP initialize. "
                f"Body: {resp.text[:200]!r}"
            )


@pytest.mark.mcp
async def test_mcp_doubled_path_is_404():
    """The doubled path ``/mcp/mcp`` must 404 -- proves the path-doubling bug is fixed.

    Before the fix, ``http_app(path="/mcp")`` + ``app.mount("/mcp", ...)``
    made the MCP endpoint live at ``/mcp/mcp`` (invisible to the existing
    ``Client(mcp)`` tests because they bypass the mount).  With the fix,
    ``/mcp/mcp`` does NOT exist and HTTP MCP clients pointing at the
    documented ``/mcp`` URL succeed instead.
    """
    from httpx import ASGITransport, AsyncClient

    def _setup():
        routes.nwp_chain = None
        routes.nowcast_store = None
        routes.enabled_regions = []
        routes.alerts_store = None
        routes.alerts_enabled = False

    app, _mcp, combined = _build_app_and_mcp(_setup)

    transport = ASGITransport(app=app)
    async with combined(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "regression-test", "version": "0.1"},
                },
            }
            resp = await ac.post(
                "/mcp/mcp",
                json=init_payload,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code == 404, (
                f"Expected /mcp/mcp to be 404 (doubling fixed), got "
                f"{resp.status_code}. Body: {resp.text[:200]!r}"
            )
