# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP discovery metadata: SEP-2127 server card + AI Catalog.

Implements the two self-description documents LibreWXR advertises:

- ``build_server_card()`` / ``server_card_endpoint()`` -- the SEP-2127
  (draft) MCP Server Card served at ``<mcp_path>/server-card``.  Clients
  fetch it to learn the server's remotes and supported protocol versions
  without an ``initialize`` round-trip.
- ``build_ai_catalog()`` -- the AI Catalog (proposal) entry served at
  ``/.well-known/ai-catalog.json``: a directory-of-directories pointer
  that resolves to the server card above.

Both are draft proposals -- neither is a ratified standard, and there is
NO standardized ``/.well-known/mcp.json`` (LibreWXR does not serve one).
Server cards intentionally do not enumerate tools; clients list tools at
runtime via the MCP protocol's ``tools/list``.

Advertised URLs derive from ``settings.public_url`` and
``settings.mcp_path`` at call time (never at import time) so tests can
monkeypatch settings and reverse-proxy deployments advertise their
public URLs.
"""

import hashlib
import importlib
import json
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import Response

from librewxr.config import settings

# SEP-2127 (draft) MCP server-card JSON schema.
_SERVER_CARD_SCHEMA = (
    "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json"
)
_NAME = "io.github.joshuakimsey/librewxr-mcp"
_TITLE = "LibreWXR MCP"
_DESCRIPTION = (
    "Precipitation nowcasts, active weather alerts, and storm-cell "
    "data for any point on Earth."
)
_REPOSITORY_URL = "https://github.com/JoshuaKimsey/LibreWRX"

_SERVER_CARD_MEDIA_TYPE = "application/mcp-server-card+json"
_AI_CATALOG_MEDIA_TYPE = "application/ai-catalog+json"
_CACHE_CONTROL = "public, max-age=3600"


def _package_version() -> str:
    """Return the installed ``librewxr`` distribution version.

    Falls back to ``0.1.0`` (the pyproject version) when the package
    metadata is unavailable, e.g. running from a bare source tree.
    """
    try:
        return _distribution_version("librewxr")
    except PackageNotFoundError:
        return "0.1.0"


def _supported_protocol_versions() -> list[str] | None:
    """Probe the installed ``mcp`` SDK for supported protocol versions.

    Tries ``mcp.types.SUPPORTED_PROTOCOL_VERSIONS`` first, then
    ``mcp.shared.version.SUPPORTED_PROTOCOL_VERSIONS`` (newer SDKs moved
    the export there).  If only a ``LATEST_PROTOCOL_VERSION`` constant is
    available it is wrapped in a single-element list.  Returns ``None``
    when neither import exists so callers can omit the field entirely
    rather than hardcode version strings.
    """
    for module_name, attr in (
        ("mcp.types", "SUPPORTED_PROTOCOL_VERSIONS"),
        ("mcp.shared.version", "SUPPORTED_PROTOCOL_VERSIONS"),
    ):
        try:
            value = getattr(importlib.import_module(module_name), attr)
        except Exception:
            continue
        if isinstance(value, (list, tuple)) and value:
            return [str(v) for v in value]
    for module_name, attr in (
        ("mcp.types", "LATEST_PROTOCOL_VERSION"),
        ("mcp.shared.version", "LATEST_PROTOCOL_VERSION"),
    ):
        try:
            value = getattr(importlib.import_module(module_name), attr)
        except Exception:
            continue
        if value:
            return [str(value)]
    return None


def build_server_card() -> dict:
    """Build the SEP-2127 (draft) MCP server card document.

    Reads ``settings.public_url`` and ``settings.mcp_path`` at call time
    so advertised URLs always reflect the current configuration.
    """
    public_base = settings.public_url.rstrip("/")
    mcp_base = settings.mcp_path.rstrip("/")
    remote: dict = {
        "type": "streamable-http",
        "url": f"{public_base}{mcp_base}/",
    }
    versions = _supported_protocol_versions()
    if versions is not None:
        remote["supportedProtocolVersions"] = versions
    return {
        "$schema": _SERVER_CARD_SCHEMA,
        "name": _NAME,
        "title": _TITLE,
        "description": _DESCRIPTION,
        "version": _package_version(),
        "websiteUrl": public_base,
        "repository": {"source": "github", "url": _REPOSITORY_URL},
        "remotes": [remote],
    }


def _etag_for(body: str) -> str:
    """First 16 hex chars of the SHA-256 of the serialized body."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _card_headers(etag: str) -> dict[str, str]:
    """Unconditional headers for the server-card endpoint."""
    return {
        "Cache-Control": _CACHE_CONTROL,
        "ETag": f'"{etag}"',
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type, If-None-Match",
        "Access-Control-Expose-Headers": "ETag",
    }


async def server_card_endpoint(request: Request) -> Response:
    """Serve the MCP server card with ETag-based conditional GET.

    Usable as a FastMCP ``custom_route`` handler or a plain Starlette
    route.  The body is computed per request (tests monkeypatch
    settings).  An ``If-None-Match`` request header equal to the current
    ETag gets a 304 with an empty body; both the 200 and 304 responses
    carry the same ETag / Cache-Control / CORS headers.
    """
    body = json.dumps(build_server_card())
    etag = _etag_for(body)
    headers = _card_headers(etag)
    if request.headers.get("if-none-match") == f'"{etag}"':
        return Response(status_code=304, headers=headers)
    return Response(
        content=body,
        media_type=_SERVER_CARD_MEDIA_TYPE,
        headers=headers,
    )


def build_ai_catalog() -> dict:
    """Build the AI Catalog (proposal) document pointing at the server card.

    The entry identifier is ``urn:air:<hostname>:mcp:librewxr-mcp`` where
    ``<hostname>`` comes from ``settings.public_url`` (defaulting to
    ``localhost`` when the URL has no hostname).
    """
    public_base = settings.public_url.rstrip("/")
    hostname = urlparse(settings.public_url).hostname or "localhost"
    return {
        "specVersion": "1.0",
        "entries": [
            {
                "identifier": f"urn:air:{hostname}:mcp:librewxr-mcp",
                "type": _SERVER_CARD_MEDIA_TYPE,
                "url": f"{public_base}{settings.mcp_path.rstrip('/')}/server-card",
            }
        ],
    }
