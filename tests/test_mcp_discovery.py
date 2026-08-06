# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP discovery metadata (SEP-2127 server card + AI Catalog).

Mirrors the minimal-app fixture pattern from ``test_mcp_http_mount.py``:
a small FastAPI app with the FastMCP sub-app mounted at ``/mcp`` and the
``routes.router`` included (the catalog endpoint lives there, as in
``main.py``).  Uses the production ``build_mcp_http_app()`` so the tests
exercise the real ``custom_route`` registration path through the mount.
The server-card / catalog endpoints do not touch the MCP session manager,
so the combined lifespan is not entered for them (see the ``client``
fixture); the initialize-based tests enter the FastMCP lifespan inside
the test body instead.
"""

import importlib
import json
from importlib.metadata import PackageNotFoundError

import pytest

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from librewxr.api import routes
from librewxr.mcp.server import build_mcp_http_app

SERVER_CARD_SCHEMA = (
    "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json"
)
SERVER_CARD_NAME = "io.github.joshuakimsey/librewxr-mcp"
MCP_TOOL_NAMES = ["get_precip_nowcast", "get_active_alerts", "get_storm_cells"]

_INIT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _initialize_payload(req_id: int = 1) -> dict:
    """Minimal MCP ``initialize`` JSON-RPC request."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-discovery-test", "version": "0.1"},
        },
    }


def _sse_json_rpc_result(body: str, req_id: int) -> dict:
    """Pull the JSON-RPC ``result`` for ``req_id`` out of an SSE response body.

    The production transport answers with ``text/event-stream`` (SSE);
    each event carries ``data: <json-rpc-message>``.  Returns the ``result``
    object of the matching response (raises on error/missing).
    """
    for block in body.split("\n\n"):
        data_lines = [
            line[len("data: "):]
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        message = json.loads("".join(data_lines))
        if message.get("id") == req_id:
            if "error" in message:
                raise AssertionError(f"JSON-RPC error for id {req_id}: {message['error']}")
            return message["result"]
    raise AssertionError(f"No JSON-RPC response with id {req_id} in SSE body: {body!r}")


@pytest.fixture(autouse=True)
def _save_restore_routes_state():
    """Save and restore routes module-level state to prevent cross-test pollution."""
    saved = {
        "mcp_mounted": routes.mcp_mounted,
        "mcp_path": routes.mcp_path,
    }
    yield
    for key, val in saved.items():
        setattr(routes, key, val)


def _build_app_and_mcp():
    """Build a minimal FastAPI app mirroring main.py's MCP mount wiring."""
    mcp_app = build_mcp_http_app()
    app = FastAPI()
    app.include_router(routes.router)
    app.mount("/mcp", mcp_app)
    return app


@pytest.fixture
async def client():
    """httpx ASGI client against the minimal wired app.

    ``routes.mcp_mounted`` is set to mirror main.py's post-mount wiring
    (restored by the autouse save/restore fixture).  Neither endpoint
    under test touches the FastMCP session manager, so the combined
    lifespan is deliberately NOT entered -- exiting a FastMCP session
    manager task group from an async fixture teardown breaks anyio's
    cancel-scope task check.
    """
    routes.mcp_mounted = True
    routes.mcp_path = "/mcp"
    app = _build_app_and_mcp()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.mcp
async def test_server_card_served_through_mount(client):
    """``GET /mcp/server-card`` returns a valid SEP-2127 (draft) card.

    Verifies the custom route registered on the FastMCP instance is
    reachable through the parent ``/mcp`` mount with the expected media
    type, identity fields, remote URL, and response headers.
    """
    resp = await client.get("/mcp/server-card")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(
        "application/mcp-server-card+json"
    )
    data = resp.json()
    assert data["$schema"] == SERVER_CARD_SCHEMA
    assert data["name"] == SERVER_CARD_NAME
    assert data["version"], "version must be non-empty"
    assert data["description"], "description must be present"
    assert len(data["description"]) <= 100
    assert data["remotes"][0]["type"] == "streamable-http"
    assert data["remotes"][0]["url"].endswith("/mcp/")
    assert "max-age=3600" in resp.headers["cache-control"]
    assert resp.headers["etag"]
    assert resp.headers["access-control-allow-origin"] == "*"


@pytest.mark.mcp
async def test_server_card_conditional_get_304(client):
    """Re-requesting with the returned ETag yields a 304 with an empty body."""
    first = await client.get("/mcp/server-card")
    assert first.status_code == 200
    etag = first.headers["etag"]

    resp = await client.get("/mcp/server-card", headers={"If-None-Match": etag})
    assert resp.status_code == 304
    assert resp.content == b""
    assert resp.headers["etag"] == etag
    assert "max-age=3600" in resp.headers["cache-control"]
    assert resp.headers["access-control-allow-origin"] == "*"


@pytest.mark.mcp
async def test_ai_catalog_endpoint(client):
    """``GET /.well-known/ai-catalog.json`` points at the MCP server card."""
    resp = await client.get("/.well-known/ai-catalog.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/ai-catalog+json")
    data = resp.json()
    assert data["specVersion"] == "1.0"
    assert data["entries"][0]["type"] == "application/mcp-server-card+json"
    assert data["entries"][0]["url"].endswith("/mcp/server-card")
    assert data["entries"][0]["identifier"].startswith("urn:air:")


@pytest.mark.mcp
async def test_ai_catalog_404_when_mcp_unavailable(client, monkeypatch):
    """Catalog 404s when MCP is not mounted (e.g. build failed at startup)."""
    monkeypatch.setattr(routes, "mcp_mounted", False)
    resp = await client.get("/.well-known/ai-catalog.json")
    assert resp.status_code == 404


@pytest.mark.mcp
async def test_mcp_stateless_initialize_and_tools_list():
    """Stateless regression: initialize then tools/list with NO session id.

    ``build_mcp_http_app()`` serves stateless HTTP (``stateless_http=True``):
    every request gets a fresh transport with no ``Mcp-Session-Id`` and no
    in-memory session to lose.  This directly exercises the multi-worker
    bug class -- in multi mode a client's next request lands on a
    different render worker, and with per-process sessions that used to
    fail with ``-32600 Session not found``.  Both requests here
    deliberately omit the ``Mcp-Session-Id`` header.
    """
    mcp_app = build_mcp_http_app()
    app = FastAPI()
    app.mount("/mcp", mcp_app)
    transport = ASGITransport(app=app)
    # Even stateless mode needs the session-manager task group the FastMCP
    # lifespan starts, so enter it in-test (a fixture teardown would trip
    # anyio's cancel-scope task check).
    async with mcp_app.lifespan(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/mcp/", json=_initialize_payload(1), headers=_INIT_HEADERS)
            assert resp.status_code in (200, 202), (
                f"initialize failed: {resp.status_code} {resp.text[:200]!r}"
            )
            assert "mcp-session-id" not in resp.headers, (
                "stateless mode must not mint a session id; multi-worker "
                "deployments cannot share one anyway"
            )
            resp = await ac.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                headers=_INIT_HEADERS,
            )
            assert resp.status_code in (200, 202), (
                f"tools/list failed: {resp.status_code} {resp.text[:200]!r}"
            )
            result = _sse_json_rpc_result(resp.text, 2)
            tool_names = [tool["name"] for tool in result["tools"]]
            assert tool_names == MCP_TOOL_NAMES, (
                f"Expected tools {MCP_TOOL_NAMES}, got {tool_names}"
            )


@pytest.mark.mcp
async def test_mcp_initialize_server_info_matches_server_card():
    """initialize's ``serverInfo`` must match what the server card advertises.

    SEP-2127's consistency clause forbids the ``initialize`` response and
    the server card from contradicting each other.  Previously FastMCP's
    own library version leaked into ``serverInfo.version`` while the card
    advertised the ``librewxr`` package version.
    """
    try:
        expected_version = importlib.metadata.version("librewxr")
    except PackageNotFoundError:
        expected_version = "0.1.0"
    mcp_app = build_mcp_http_app()
    app = FastAPI()
    app.mount("/mcp", mcp_app)
    transport = ASGITransport(app=app)
    async with mcp_app.lifespan(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/mcp/", json=_initialize_payload(1), headers=_INIT_HEADERS)
            assert resp.status_code in (200, 202), (
                f"initialize failed: {resp.status_code} {resp.text[:200]!r}"
            )
            server_info = _sse_json_rpc_result(resp.text, 1)["serverInfo"]
            assert server_info["name"] == "librewxr-mcp", (
                f"Unexpected serverInfo.name: {server_info['name']!r}"
            )
            assert server_info["version"] == expected_version, (
                f"serverInfo.version {server_info['version']!r} contradicts the "
                f"server card version {expected_version!r}"
            )
