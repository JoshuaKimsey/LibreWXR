# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging
from datetime import datetime

import httpx
import numpy as np
from PIL import Image

from librewrx.data.regions import RegionDef

logger = logging.getLogger(__name__)


class IEMSource:
    """Iowa Environmental Mesonet NEXRAD composite source.

    Fetches radar composites for any region (USCOMP, AKCOMP, etc.)
    from IEM's live and archive image endpoints.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch live N0Q frame for a region."""
        frame_idx = minutes_ago // 5
        if frame_idx < 0 or frame_idx > 11:
            return None

        url = (
            f"{self._base_url}/data/gis/images/4326"
            f"/{region.live_dir}/n0q_{frame_idx}.png"
        )
        return await self._download_and_parse(url, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch archived N0Q frame for a specific UTC datetime."""
        minute = (dt.minute // 5) * 5
        dt = dt.replace(minute=minute, second=0, microsecond=0)
        path = dt.strftime(
            f"%Y/%m/%d/GIS/{region.archive_dir}/n0q_%Y%m%d%H%M.png"
        )
        url = f"{self._base_url}/archive/data/{path}"
        return await self._download_and_parse(url, region)

    async def _download_and_parse(
        self, url: str, region: RegionDef
    ) -> np.ndarray | None:
        try:
            client = await self._get_client()
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("Failed to fetch %s: HTTP %d", url, resp.status_code)
                return None

            return _parse_n0q_png(resp.content, region)
        except Exception:
            logger.exception("Error fetching %s", url)
            return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_n0q_png(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Parse an IEM N0Q PNG into a raw uint8 numpy array.

    The PNGs are palette-indexed. We extract the raw index values,
    not the RGB colors.
    """
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode == "P":
            arr = np.array(img, dtype=np.uint8)
        else:
            arr = np.array(img.convert("L"), dtype=np.uint8)

        expected = (region.height, region.width)
        if arr.shape != expected:
            logger.warning(
                "Unexpected %s dimensions: %s (expected %s)",
                region.name, arr.shape, expected,
            )
        return arr
    except Exception:
        logger.exception("Failed to parse N0Q PNG for %s", region.name)
        return None
