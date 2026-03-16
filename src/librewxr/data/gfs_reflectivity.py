# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# GFS global grid at 0.25° resolution (same grid as temperature)
WEST = -180.0
EAST = 180.0
NORTH = 90.0
SOUTH = -90.0
PIXEL_SIZE = 0.25
GRID_WIDTH = int((EAST - WEST) / PIXEL_SIZE)    # 1440
GRID_HEIGHT = int((NORTH - SOUTH) / PIXEL_SIZE)  # 720

# WMS parameters for UCAR THREDDS GFS simulated reflectivity at 1000m
COLORSCALE_MAX = 75.0  # dBZ
COLORSCALE_MIN = 0.0
COLORSCALE_RANGE = COLORSCALE_MAX - COLORSCALE_MIN  # 75

THREDDS_WMS_URL = (
    "https://thredds.ucar.edu/thredds/wms/grib/NCEP/GFS/Global_0p25deg/Best"
    "?service=WMS&version=1.3.0&request=GetMap"
    "&LAYERS=Reflectivity_height_above_ground"
    "&CRS=CRS:84"
    f"&BBOX={WEST},{SOUTH},{EAST},{NORTH}"
    f"&WIDTH={GRID_WIDTH}&HEIGHT={GRID_HEIGHT}"
    "&FORMAT=image/png"
    "&ELEVATION=1000.0"
    "&STYLES=default-scalar/seq-Greys"
    f"&COLORSCALERANGE={COLORSCALE_MIN},{COLORSCALE_MAX}"
    "&NUMCOLORBANDS=254"
    "&TRANSPARENT=true"
)


class GFSReflectivityGrid:
    """GFS simulated reflectivity grid for global fallback coverage.

    Fetches GFS simulated reflectivity at 1000m from UCAR THREDDS as a
    greyscale WMS image. Provides low-resolution (~25km) radar-like data
    for areas not covered by any real radar composite.

    Data is stored as uint8 using the same encoding as radar composites:
    pixel = (dBZ + 32) * 2, so it can be fed directly into the color
    scheme and rendering pipeline.
    """

    def __init__(self):
        self.data: np.ndarray | None = None  # uint8, shape (720, 1440)
        self._client: httpx.AsyncClient | None = None

    async def fetch(self) -> bool:
        """Fetch the latest GFS simulated reflectivity from THREDDS."""
        try:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    follow_redirects=True,
                )

            resp = await self._client.get(THREDDS_WMS_URL)
            if resp.status_code != 200:
                logger.warning("GFS reflectivity fetch failed: HTTP %d", resp.status_code)
                return False

            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            arr = np.array(img)

            r_channel = arr[:, :, 0].astype(np.float32)
            alpha = arr[:, :, 3]

            # seq-Greys palette: 0=white (low dBZ) to 255=black (high dBZ)
            dbz = COLORSCALE_MAX - (r_channel / 255.0) * COLORSCALE_RANGE

            # Convert to uint8 using same encoding as radar composites:
            # pixel = clamp((dBZ + 32) * 2, 0, 255)
            result = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)

            # Transparent pixels = no data → 0
            result[alpha == 0] = 0

            # dBZ <= 0 is noise/clear sky → 0
            result[dbz <= 0.0] = 0

            self.data = result

            valid = (alpha > 0) & (dbz > 0)
            logger.info(
                "GFS reflectivity updated: %.1f-%.1f dBZ range, %d pixels with echoes",
                dbz[valid].min() if valid.any() else 0,
                dbz[valid].max() if valid.any() else 0,
                int(valid.sum()),
            )
            return True

        except Exception:
            logger.exception("Error fetching GFS reflectivity data")
            return False

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return uint8 dBZ-encoded values for the given lat/lon arrays.

        Uses the same encoding as radar composites (pixel = (dBZ + 32) * 2)
        so values can be fed directly into the color scheme pipeline.

        Args:
            lat: Latitude array in degrees (any shape).
            lon: Longitude array in degrees (same shape as lat).

        Returns:
            uint8 array of same shape as inputs.
        """
        if self.data is None:
            return np.zeros(lat.shape, dtype=np.uint8)

        # Map lat/lon to GFS grid indices
        # Grid row 0 = NORTH (90°), row 719 = SOUTH (-90°)
        # Grid col 0 = WEST (-180°), col 1439 = EAST (180°)
        row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
        col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)

        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        return self.data[row, col]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
