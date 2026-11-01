# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Pipeline-owned external color-palette resolution.

``LIBREWXR_COLOR_SCHEMES_FILE`` may point at a local path or an http(s)
URL serving a ``color_table.csv``-shaped file.  Once per fetch cycle the
pipeline resolves it, strictly validates the bytes (dry run - it never
touches the color-scheme globals), publishes the adopted bytes verbatim
to ``<cache_dir>/palettes/current.csv`` and advertises the palette section
in ``state.json``.  Render workers apply it on their state poll loop.

Doctrine (mirrors alerts / lightning):

- The resolved file is the complete truth for the scheme set.
- Malformed content is rejected whole (adopt-or-reject), never partially
  adopted.
- Failure ladder: fresh external -> last-good artifact -> packaged
  built-ins.  A failing source keeps whatever is already being served.
- The palette version is always ``sha256(raw_bytes)[:12]``.
- No conditional GET, no sidecar metadata: one capped GET per cycle and a
  hash/content compare decide whether anything changed.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from librewxr.colors.schemes import packaged_table_bytes, parse_color_table
from librewxr.config import settings

logger = logging.getLogger(__name__)

PALETTE_DIRNAME = "palettes"
PALETTE_ARTIFACT_NAME = "current.csv"
MAX_PALETTE_BYTES = 2 * 1024 * 1024

# Shared User-Agent, matching the alerts fetcher (some upstreams are
# picky about the default httpx UA).
_USER_AGENT = "(LibreWXR, librewxr@localhost)"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_RETRY_DELAY_S = 1.0


class PaletteFetchError(Exception):
    """A palette fetch failed (status, cap, transport, ...)."""


def palette_artifact_path(cache_dir: Path) -> Path:
    """Path of the last-good palette artifact under ``cache_dir``."""
    return Path(cache_dir) / PALETTE_DIRNAME / PALETTE_ARTIFACT_NAME


def palette_artifact_bytes(cache_dir: Path) -> bytes | None:
    """Read the last-good palette artifact; ``None`` when missing/unreadable."""
    try:
        return palette_artifact_path(cache_dir).read_bytes()
    except OSError:
        return None


def _classify_source(value: str) -> str:
    """Classify ``LIBREWXR_COLOR_SCHEMES_FILE`` into builtin / url / path.

    Empty means the packaged built-ins.  ``file://`` URLs are rejected
    explicitly - only plain local paths and http(s) are accepted, so the
    meaning of a path can never depend on URL parsing.
    """
    value = (value or "").strip()
    if not value:
        return "builtin"
    if value.startswith("http://") or value.startswith("https://"):
        return "url"
    if value.startswith("file://"):
        raise ValueError("file:// palette sources are not supported")
    return "path"


def _write_artifact(cache_dir: Path, raw: bytes) -> None:
    """Atomically publish ``raw`` as the last-good artifact (best effort).

    Skips the write when the existing artifact already holds identical
    bytes (hash compare - unchanged content is a no-op).
    """
    path = palette_artifact_path(cache_dir)
    if palette_artifact_bytes(cache_dir) == raw:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Failed to write palette artifact %s: %s", path, e)


async def _stream_capped_get(client: httpx.AsyncClient, url: str) -> bytes:
    """One streaming GET, hard-capped at ``MAX_PALETTE_BYTES``."""
    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise PaletteFetchError(
                f"unexpected HTTP status {response.status_code}"
            )
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                declared_len = int(declared)
            except (TypeError, ValueError):
                declared_len = None
            if declared_len is not None and declared_len > MAX_PALETTE_BYTES:
                raise PaletteFetchError(
                    f"announced Content-Length {declared_len} exceeds "
                    f"{MAX_PALETTE_BYTES} bytes"
                )
        buf = bytearray()
        async for chunk in response.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > MAX_PALETTE_BYTES:
                raise PaletteFetchError(
                    f"body exceeds {MAX_PALETTE_BYTES} bytes"
                )
        return bytes(buf)


async def _fetch_url_bytes(url: str) -> bytes:
    """Fetch the palette URL with a capped stream and transport retries.

    Only transport / decode errors are retried (the server answered
    otherwise, so a retry would not help).  ``retry_get`` is deliberately
    not reused: it buffers the whole body with no size cap.
    """
    retries = max(0, int(getattr(settings, "download_retries", 1)))
    async with httpx.AsyncClient(
        timeout=_TIMEOUT,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        for attempt in range(retries + 1):
            try:
                return await _stream_capped_get(client, url)
            except (httpx.TransportError, httpx.DecodingError) as e:
                if attempt >= retries:
                    raise PaletteFetchError(str(e)) from e
                logger.debug(
                    "palette: transport error, retrying (%d/%d)",
                    attempt + 1, retries,
                )
                await asyncio.sleep(_RETRY_DELAY_S)
    raise PaletteFetchError("palette fetch failed")  # pragma: no cover


def _builtin_section(
    cache_dir: Path, fetched_at: str, last_status: str,
) -> dict:
    """Publish the packaged built-in table and build its section."""
    raw = packaged_table_bytes()
    version = parse_color_table(raw).version
    _write_artifact(cache_dir, raw)
    return {
        "version": version,
        "source": "builtin",
        "fetched_at": fetched_at,
        "last_status": last_status,
    }


async def _resolve(settings, cache_dir: Path, fetched_at: str) -> dict:
    configured = (getattr(settings, "color_schemes_file", "") or "").strip()
    try:
        kind = _classify_source(configured)
    except ValueError as e:
        logger.warning("Color palette source rejected (%s); using built-ins", e)
        return _builtin_section(cache_dir, fetched_at, "fetch_failed")

    raw: bytes | None = None
    status = "ok"
    if kind == "builtin":
        raw = packaged_table_bytes()
    elif kind == "path":
        try:
            path = Path(configured)
            if path.stat().st_size > MAX_PALETTE_BYTES:
                raise PaletteFetchError(
                    f"file exceeds {MAX_PALETTE_BYTES} bytes"
                )
            raw = path.read_bytes()
        except Exception as e:
            logger.warning(
                "Color palette file %s unreadable: %s", configured, e,
            )
            status = "fetch_failed"
    else:  # url
        try:
            raw = await _fetch_url_bytes(configured)
        except Exception as e:
            logger.warning(
                "Color palette URL %s fetch failed: %s", configured, e,
            )
            status = "fetch_failed"

    version = ""
    if raw is not None:
        try:
            version = parse_color_table(raw).version
        except ValueError as e:
            logger.warning(
                "Color palette %s rejected as invalid: %s", configured, e,
            )
            raw = None
            status = "invalid"

    if raw is not None:
        _write_artifact(cache_dir, raw)
        logger.debug(
            "Color palette %s adopted (version %s)", configured, version,
        )
        return {
            "version": version,
            "source": kind,
            "fetched_at": fetched_at,
            "last_status": "ok",
        }

    # Failure ladder: keep the last-good artifact when there is one, else
    # republish the packaged built-ins.
    cached = palette_artifact_bytes(cache_dir)
    if cached is not None:
        try:
            cached_version = parse_color_table(cached).version
        except ValueError:
            logger.warning(
                "Cached color palette artifact is invalid; using packaged built-ins"
            )
        else:
            return {
                "version": cached_version,
                "source": kind,
                "fetched_at": fetched_at,
                "last_status": status,
            }

    return _builtin_section(cache_dir, fetched_at, status)


async def resolve_palette(settings, cache_dir: Path) -> dict:
    """Resolve + validate the palette for this fetch cycle; never raises.

    Returns the ``state.json`` palette section::

        {"version": <12-hex>, "source": "builtin"|"path"|"url",
         "fetched_at": <UTC ISO timestamp>, "last_status": "ok"|...}

    Any unexpected failure degrades to the packaged built-ins with a
    warning rather than breaking the fetch cycle.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        return await _resolve(settings, Path(cache_dir), fetched_at)
    except Exception:
        logger.warning(
            "Palette resolution failed unexpectedly; serving packaged built-ins",
            exc_info=True,
        )
        try:
            return _builtin_section(Path(cache_dir), fetched_at, "fetch_failed")
        except Exception:
            logger.warning("Packaged color table unavailable", exc_info=True)
            return {
                "version": "",
                "source": "builtin",
                "fetched_at": fetched_at,
                "last_status": "fetch_failed",
            }
