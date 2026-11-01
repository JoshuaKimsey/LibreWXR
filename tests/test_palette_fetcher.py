# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the pipeline-owned external color-palette resolver.

HTTP is mocked at the ``httpx.AsyncClient`` layer with ``MockTransport``;
the resolver's own retry loop, cap checks and fallback ladder run for
real.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import httpx
import pytest

from librewxr.colors.schemes import packaged_table_bytes, parse_color_table
from librewxr.data import palette_fetcher
from librewxr.data.palette_fetcher import (
    MAX_PALETTE_BYTES,
    palette_artifact_bytes,
    palette_artifact_path,
    resolve_palette,
)

pytestmark = pytest.mark.store


def _table_bytes(names=("Alpha", "Beta")) -> bytes:
    header = ",".join(["dBZ / RGBA", *names])
    colors = tuple(f"#01{idx:02x}0304"[:9] for idx in range(len(names)))
    rows = []
    for i in range(256):
        rows.append(",".join([str(i - 32), *colors]))
    return "\n".join([header, *rows]).encode()


def _version(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:12]


def _settings(source: str) -> SimpleNamespace:
    return SimpleNamespace(color_schemes_file=source)


def _patch_http(monkeypatch, handler) -> None:
    """Route every AsyncClient this module builds through a MockTransport."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(palette_fetcher.httpx, "AsyncClient", factory)
    # Keep transport-retry tests fast.
    monkeypatch.setattr(palette_fetcher, "_RETRY_DELAY_S", 0.0)


@pytest.fixture(autouse=True)
def no_retries(monkeypatch):
    monkeypatch.setattr(palette_fetcher.settings, "download_retries", 0)


def _write_artifact(cache_dir, raw: bytes) -> None:
    path = palette_artifact_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


async def test_url_success_adopts_and_writes_artifact(tmp_path, monkeypatch):
    raw = _table_bytes()
    _patch_http(monkeypatch, lambda request: httpx.Response(200, content=raw))

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["version"] == _version(raw)
    assert section["source"] == "url"
    assert section["last_status"] == "ok"
    assert section["fetched_at"]
    assert palette_artifact_bytes(tmp_path) == raw


async def test_url_unchanged_content_is_noop_write(tmp_path, monkeypatch):
    raw = _table_bytes()
    _write_artifact(tmp_path, raw)
    before = palette_artifact_path(tmp_path).stat().st_mtime_ns
    _patch_http(monkeypatch, lambda request: httpx.Response(200, content=raw))

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["version"] == _version(raw)
    assert palette_artifact_path(tmp_path).stat().st_mtime_ns == before


async def test_non_200_keeps_last_good(tmp_path, monkeypatch):
    good = _table_bytes(names=("Good", "Palette"))
    _write_artifact(tmp_path, good)
    _patch_http(monkeypatch, lambda request: httpx.Response(503))

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["version"] == _version(good)
    assert section["source"] == "url"
    assert section["last_status"] == "fetch_failed"
    assert palette_artifact_bytes(tmp_path) == good


async def test_network_error_keeps_last_good(tmp_path, monkeypatch):
    good = _table_bytes(names=("Still", "Here"))
    _write_artifact(tmp_path, good)

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_http(monkeypatch, handler)

    section = await resolve_palette(
        _settings("http://example.test/palette.csv"), tmp_path,
    )

    assert section["version"] == _version(good)
    assert section["source"] == "url"
    assert section["last_status"] == "fetch_failed"
    assert palette_artifact_bytes(tmp_path) == good


async def test_unreachable_without_artifact_falls_back_to_builtin(
    tmp_path, monkeypatch,
):
    _patch_http(monkeypatch, lambda request: httpx.Response(500))

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    packaged = packaged_table_bytes()
    assert section["version"] == parse_color_table(packaged).version
    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"
    assert palette_artifact_bytes(tmp_path) == packaged


async def test_malformed_body_keeps_last_good(tmp_path, monkeypatch):
    good = _table_bytes(names=("Valid", "Table"))
    _write_artifact(tmp_path, good)
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(200, content=b"# not,a,color table"),
    )

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["version"] == _version(good)
    assert section["last_status"] == "invalid"
    assert palette_artifact_bytes(tmp_path) == good


async def test_malformed_body_without_artifact_republishes_builtin(
    tmp_path, monkeypatch,
):
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(200, content=b"garbage"),
    )

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["source"] == "builtin"
    assert section["last_status"] == "invalid"
    assert palette_artifact_bytes(tmp_path) == packaged_table_bytes()


async def test_content_length_over_cap_fails(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            content=b"x",
            headers={"Content-Length": str(MAX_PALETTE_BYTES + 1)},
        )

    _patch_http(monkeypatch, handler)

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"


async def test_streamed_body_over_cap_fails(tmp_path, monkeypatch):
    # Iterator content => no Content-Length header, so only the running cap
    # can reject it.
    def chunks():
        for _ in range(3):
            yield b"x" * (MAX_PALETTE_BYTES // 2)

    _patch_http(monkeypatch, lambda request: httpx.Response(200, content=chunks()))

    section = await resolve_palette(
        _settings("https://example.test/palette.csv"), tmp_path,
    )

    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"


async def test_file_url_is_rejected_without_network(tmp_path, monkeypatch):
    def handler(request):  # pragma: no cover - must never be called
        raise AssertionError("file:// must not be fetched")

    _patch_http(monkeypatch, handler)

    section = await resolve_palette(
        _settings("file:///tmp/palette.csv"), tmp_path,
    )

    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"


async def test_local_path_mode_reads_file(tmp_path):
    raw = _table_bytes(names=("Path", "Mode"))
    source = tmp_path / "custom.csv"
    source.write_bytes(raw)

    section = await resolve_palette(_settings(str(source)), tmp_path / "cache")

    assert section["version"] == _version(raw)
    assert section["source"] == "path"
    assert section["last_status"] == "ok"
    assert palette_artifact_bytes(tmp_path / "cache") == raw


async def test_local_path_missing_keeps_last_good(tmp_path):
    good = _table_bytes(names=("Cached", "Palette"))
    _write_artifact(tmp_path / "cache", good)

    section = await resolve_palette(
        _settings(str(tmp_path / "does-not-exist.csv")), tmp_path / "cache",
    )

    assert section["version"] == _version(good)
    assert section["source"] == "path"
    assert section["last_status"] == "fetch_failed"


async def test_local_path_over_cap_fails(tmp_path):
    source = tmp_path / "huge.csv"
    source.write_bytes(b"x" * (MAX_PALETTE_BYTES + 1))

    section = await resolve_palette(_settings(str(source)), tmp_path / "cache")

    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"


async def test_builtin_mode_is_stable_packaged_version(tmp_path):
    section = await resolve_palette(_settings(""), tmp_path)

    packaged = packaged_table_bytes()
    assert section["version"] == parse_color_table(packaged).version
    assert section["source"] == "builtin"
    assert section["last_status"] == "ok"
    assert palette_artifact_bytes(tmp_path) == packaged


async def test_section_shape_is_exactly_the_state_contract(tmp_path):
    section = await resolve_palette(_settings(""), tmp_path)
    assert set(section) == {"version", "source", "fetched_at", "last_status"}
    assert len(section["version"]) == 12


async def test_resolve_palette_never_raises(tmp_path, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(palette_fetcher, "_resolve", boom)

    section = await resolve_palette(_settings("https://example.test/x"), tmp_path)

    assert section["source"] == "builtin"
    assert section["last_status"] == "fetch_failed"
    assert section["version"] == parse_color_table(packaged_table_bytes()).version
