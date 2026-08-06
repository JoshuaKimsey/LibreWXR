# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP discovery metadata (SEP-2127 server card + AI Catalog).

Mirrors the minimal-app fixture pattern from ``test_mcp_http_mount.py``:
a small FastAPI app with the FastMCP sub-app mounted at ``/mcp`` and the
``routes.router`` included (the catalog endpoint lives there, as in
``main.py``).  Uses the production ``build_mcp_http_app()`` so the tests
exercise the real ``custom_route`` registration path through the mount.
Neither endpoint under test touches the MCP session manager, so the
combined lifespan is not entered (see the ``client`` fixture).
"""

import pytest

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from librewxr.api import routes
from librewxr.mcp.server import build_mcp_http_app

SERVER_CARD_SCHEMA = (
    "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json"
)
SERVER_CARD_NAME = "io.github.joshuakimsey/librewxr-mcp"


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
