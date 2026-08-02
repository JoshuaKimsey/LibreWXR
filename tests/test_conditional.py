# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from librewxr.api.conditional import (
    compute_etag,
    conditional_response,
    etag_matches,
    parse_if_none_match,
)

# ---------------------------------------------------------------------------
# conditional_response test app
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def root(request: Request):
        return conditional_response(
            request=request,
            body=b"hello",
            etag=compute_etag(b"hello"),
            content_type="text/plain",
            max_age=300,
        )

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_make_app())


# ---------------------------------------------------------------------------
# compute_etag
# ---------------------------------------------------------------------------


class TestComputeEtag:
    def test_compute_etag_deterministic(self):
        tag1 = compute_etag(b"some radar tile bytes")
        tag2 = compute_etag(b"some radar tile bytes")
        assert tag1 == tag2

    def test_compute_etag_different_bytes_differ(self):
        assert compute_etag(b"a") != compute_etag(b"b")

    def test_compute_etag_format(self):
        tag = compute_etag(b"a")
        assert len(tag) == 18
        assert tag.startswith('"') and tag.endswith('"')
        assert not tag.startswith("W/")
        hex_part = tag[1:-1]
        assert len(hex_part) == 16
        assert all(c in "0123456789abcdef" for c in hex_part)


# ---------------------------------------------------------------------------
# parse_if_none_match
# ---------------------------------------------------------------------------


class TestParseIfNoneMatch:
    def test_parse_none_header(self):
        assert parse_if_none_match(None) == (set(), False)

    def test_parse_empty_header(self):
        assert parse_if_none_match("") == (set(), False)

    def test_parse_single_tag(self):
        assert parse_if_none_match('"abc"') == ({'"abc"'}, False)

    def test_parse_comma_list(self):
        assert parse_if_none_match('"abc", "def"') == ({'"abc"', '"def"'}, False)

    def test_parse_wildcard(self):
        assert parse_if_none_match('"abc", *') == (set(), True)

    def test_parse_bare_wildcard(self):
        assert parse_if_none_match("*") == (set(), True)

    def test_parse_strips_weak_prefix(self):
        assert parse_if_none_match('W/"abc"') == ({'"abc"'}, False)

    def test_parse_drops_empty_tokens(self):
        assert parse_if_none_match('"abc", , "def",') == ({'"abc"', '"def"'}, False)


# ---------------------------------------------------------------------------
# etag_matches
# ---------------------------------------------------------------------------


class TestEtagMatches:
    def test_match_wildcard(self):
        assert etag_matches('"abc"', parse_if_none_match("*")) is True

    def test_match_exact(self):
        assert etag_matches('"abc"', parse_if_none_match('"abc"')) is True

    def test_match_weak_vs_strong(self):
        # RFC 7232 If-None-Match: a weak comparison is always used, so a
        # client's W/ tag satisfies a server's strong ETag.
        parsed = parse_if_none_match('W/"abc"')
        assert parsed == ({'"abc"'}, False)
        assert etag_matches('"abc"', parsed) is True

    def test_no_match(self):
        assert etag_matches('"abc"', parse_if_none_match('"def"')) is False


# ---------------------------------------------------------------------------
# conditional_response (via TestClient)
# ---------------------------------------------------------------------------


class TestConditionalResponse:
    def test_conditional_200_without_header(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.content == b"hello"
        assert resp.headers.get("etag") == compute_etag(b"hello")
        assert resp.headers.get("cache-control") == "public, max-age=300"

    def test_conditional_304_on_match(self, client):
        first = client.get("/")
        etag = first.headers["etag"]
        assert first.status_code == 200
        resp = client.get("/", headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control") == "public, max-age=300"

    def test_conditional_304_on_star(self, client):
        resp = client.get("/", headers={"If-None-Match": "*"})
        assert resp.status_code == 304
        assert resp.content == b""

    def test_conditional_200_on_mismatch(self, client):
        resp = client.get("/", headers={"If-None-Match": '"deadbeefdeadbeef"'})
        assert resp.status_code == 200
        assert resp.content == b"hello"

    def test_conditional_304_has_no_body(self, client):
        first = client.get("/")
        resp = client.get(
            "/", headers={"If-None-Match": first.headers["etag"]}
        )
        assert resp.status_code == 304
        assert resp.content == b""
        # httpx/TestClient omits Content-Length entirely on 304 responses
        # (verified locally on this machine: the header is absent, not 0
        # and not "5"), so assert it is absent rather than a specific value.
        assert resp.headers.get("content-length") is None


# ---------------------------------------------------------------------------
# WebP encoding determinism (guards the strong ETag)
# ---------------------------------------------------------------------------

_webp_skip_reason = "webp unavailable"


def _load_webp_encoder():
    """Return the renderer's image encoder or raise SkipTest.

    ``_encode_image`` is underscore-prefixed but importable directly, and
    calling it needs nothing beyond a PIL image + format string, so no
    ``present_tile`` fallback is required.  It reads
    ``settings.webp_quality`` (default 65, lossy) from ``librewxr.config``.
    """
    if not pytest.importorskip("PIL"):
        pytest.skip("PIL unavailable")
    try:
        from PIL import features

        if not features.check("webp"):
            pytest.skip("webp unavailable (Pillow built without webp support)")
        from librewxr.tiles.renderer import _encode_image
    except (ImportError, TypeError) as exc:  # pragma: no cover - defensive
        pytest.skip(f"renderer encoder not importable: {exc}")
    return _encode_image


def test_webp_encoding_is_deterministic():
    encode = _load_webp_encoder()
    if not hasattr(encode, "__call__"):  # pragma: no cover - defensive
        pytest.skip("encoder is not callable with the expected signature")
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL unavailable")

    # Small fixed RGBA image with a couple of distinct pixel values.
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[0, 0] = (255, 0, 0, 255)
    arr[2:5, 2:5] = (0, 0, 255, 255)
    arr[6, 6] = (0, 255, 0, 128)
    img = Image.fromarray(arr, mode="RGBA")

    first = encode(img, "webp")
    second = encode(img, "webp")
    assert first == second, (
        "webp encoding must be byte-for-byte deterministic or a strong "
        "ETag computed on the encoded bytes would flap between requests"
    )
