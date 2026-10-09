# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the pure-ASGI ``JsonGZipMiddleware`` in ``librewxr.main``.

A bare ``FastAPI()`` app carries only this middleware plus tiny routes so the
tests exercise the middleware in isolation instead of the full production app
(whose lifespan and stores are irrelevant here).  httpx/TestClient decodes gzip
transparently, so ``response.json()`` matching the original payload proves the
wire content is correct even while the ``content-encoding`` header is visible.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.main import JsonGZipMiddleware

_LARGE_PAYLOAD = [{"i": i, "v": "x" * 20} for i in range(300)]


def _build_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(JsonGZipMiddleware)

    @app.get("/small")
    async def small():
        return {"ok": True}

    @app.get("/large")
    async def large():
        return _LARGE_PAYLOAD

    @app.get("/image")
    async def image():
        # > 1 KiB of arbitrary bytes, but an image media type: must bypass.
        return Response(content=b"\x89PNG" + b"\x00" * 4096, media_type="image/png")

    @app.get("/stream")
    async def stream():
        async def gen():
            yield b"[" + b"x" * 2048
            yield b"x" * 2048 + b"]"

        return StreamingResponse(gen(), media_type="application/json")

    @app.get("/empty")
    async def empty():
        return JSONResponse(content={})

    return TestClient(app)


def test_small_json_below_floor_is_not_compressed():
    client = _build_client()
    response = client.get("/small")
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.json() == {"ok": True}


def test_large_json_is_gzipped():
    client = _build_client()
    response = client.get("/large")
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in response.headers.get("vary", "").lower()
    # httpx decodes gzip transparently; equality proves wire transparency.
    assert response.json() == _LARGE_PAYLOAD
    content_length = response.headers.get("content-length")
    assert content_length is not None
    assert int(content_length) > 0


def test_image_over_floor_is_not_compressed():
    client = _build_client()
    response = client.get("/image")
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.content == b"\x89PNG" + b"\x00" * 4096


def test_streaming_json_is_not_compressed():
    client = _build_client()
    response = client.get("/stream")
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.content == b"[" + b"x" * 2048 + b"x" * 2048 + b"]"


def test_non_gzip_client_is_not_compressed():
    client = _build_client()
    response = client.get("/large", headers={"Accept-Encoding": "identity"})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.json() == _LARGE_PAYLOAD


def test_empty_json_is_not_compressed():
    client = _build_client()
    response = client.get("/empty")
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.json() == {}
