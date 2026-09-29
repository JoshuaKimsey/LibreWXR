# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Regression coverage for ``librewxr.data.retry``.

httpx renamed ``DecodeError`` to ``DecodingError`` in 0.28; referencing the
old name raised ``AttributeError`` at runtime (surfacing only when a decode
error actually occurred).  The first test is a literal guard against that
class of stale-name regression.
"""

import re
from pathlib import Path

import httpx
import pytest

from librewxr.data.retry import retry_get

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED = (
    _REPO_ROOT / "src/librewxr/data/retry.py",
    _REPO_ROOT / "src/librewxr/data/alerts_fetcher.py",
)


class _FlakyClient:
    """Minimal async client whose GET always raises ``exc``."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def get(self, url, **kwargs):
        self.calls += 1
        raise self._exc


def test_httpx_exception_names_referenced_exist():
    """Every ``httpx.<...>Error`` named in retry/alerts_fetcher must exist."""
    referenced: dict[str, set[str]] = {}
    for path in _SCANNED:
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"httpx\.([A-Za-z_]\w*)", text):
            if name.endswith("Error"):
                referenced.setdefault(name, set()).add(path.name)

    assert referenced, "expected to find httpx exception references to guard"
    missing = {
        name: sorted(files)
        for name, files in referenced.items()
        if not hasattr(httpx, name)
    }
    assert not missing, f"stale httpx exception names referenced: {missing}"


async def test_retry_get_retries_decoding_error():
    client = _FlakyClient(httpx.DecodingError("truncated response body"))
    result = await retry_get(
        client, "https://example.test/x", retries=2, delay=0.0,
    )
    assert result is None
    assert client.calls == 3


async def test_retry_get_does_not_retry_status_error():
    req = httpx.Request("GET", "https://example.test/x")
    resp = httpx.Response(500, request=req)
    client = _FlakyClient(
        httpx.HTTPStatusError("server error", request=req, response=resp)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await retry_get(client, "https://example.test/x", retries=2, delay=0.0)
    assert client.calls == 1
