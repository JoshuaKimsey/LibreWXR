# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time

import httpx

from librewxr.config import settings

logger = logging.getLogger(__name__)


async def retry_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int | None = None,
    delay: float = 1.0,
    log_name: str = "",
    **kwargs,
) -> httpx.Response | None:
    """Retry an async HTTP GET on transient errors.

    Retries on ``httpx.TransportError`` (connection refused, timeout,
    DNS failure) and ``httpx.DecodeError`` (truncated response body).
    Does **not** retry on ``httpx.HTTPStatusError`` — the server
    responded, retrying won't help.

    Returns the ``httpx.Response`` on success, or ``None`` if all
    attempts fail due to transport/decode errors.
    """
    if retries is None:
        retries = settings.download_retries
    for attempt in range(retries + 1):
        try:
            return await client.get(url, **kwargs)
        except httpx.TransportError:
            if attempt < retries:
                name = log_name or url.split("/")[-1]
                logger.info(
                    "%s: transport error, retrying (%d/%d)",
                    name, attempt + 1, retries,
                )
                await asyncio.sleep(delay)
            else:
                name = log_name or url.split("/")[-1]
                logger.warning(
                    "%s: transport error after %d retries, giving up",
                    name, retries,
                )
        except httpx.DecodeError:
            if attempt < retries:
                name = log_name or url.split("/")[-1]
                logger.info(
                    "%s: decode error, retrying (%d/%d)",
                    name, attempt + 1, retries,
                )
                await asyncio.sleep(delay)
            else:
                name = log_name or url.split("/")[-1]
                logger.warning(
                    "%s: decode error after %d retries, giving up",
                    name, retries,
                )
    return None


def retry_sync(
    fn,
    *args,
    retries: int | None = None,
    delay: float = 1.0,
    log_name: str = "",
    **kwargs,
):
    """Call a synchronous function with retries on transient errors.

    Retries on any ``Exception`` (covers fsspec/S3 transient errors
    and unexpected I/O failures).  ``FileNotFoundError`` is treated as
    a not-yet-published upstream resource rather than a transient
    error: no retry, one INFO log line, no traceback dump.  This is
    the common case for Open-Meteo S3 mirrors where the most-recent
    forecast hours of a run aren't always immediately available.
    Returns the result on success, or ``None`` if all attempts fail.
    """
    if retries is None:
        retries = settings.download_retries
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as e:
            # Upstream just doesn't have this file yet — retrying
            # won't make it appear, and the traceback adds nothing.
            name = log_name or getattr(fn, "__name__", "function")
            logger.info(
                "%s: upstream resource not available yet (%s)", name, e,
            )
            return None
        except Exception:
            if attempt < retries:
                name = log_name or getattr(fn, "__name__", "function")
                logger.info(
                    "%s: error, retrying (%d/%d)",
                    name, attempt + 1, retries,
                    exc_info=True,
                )
                time.sleep(delay)
            else:
                name = log_name or getattr(fn, "__name__", "function")
                logger.warning(
                    "%s: error after %d retries, giving up",
                    name, retries,
                    exc_info=True,
                )
    return None