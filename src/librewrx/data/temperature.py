# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# GFS global temperature grid at 0.25° resolution
WEST = -180.0
EAST = 180.0
NORTH = 90.0
SOUTH = -90.0
TEMP_PIXEL_SIZE = 0.25
GRID_WIDTH = int((EAST - WEST) / TEMP_PIXEL_SIZE)    # 1440
GRID_HEIGHT = int((NORTH - SOUTH) / TEMP_PIXEL_SIZE)  # 720

# WMS parameters for UCAR THREDDS GFS 2m temperature
COLORSCALE_MIN = 220.0  # K (-53°C)
COLORSCALE_MAX = 320.0  # K (+47°C)
COLORSCALE_RANGE = COLORSCALE_MAX - COLORSCALE_MIN  # 100K

THREDDS_WMS_URL = (
    "https://thredds.ucar.edu/thredds/wms/grib/NCEP/GFS/Global_0p25deg/Best"
    "?service=WMS&version=1.3.0&request=GetMap"
    "&LAYERS=Temperature_height_above_ground"
    "&CRS=CRS:84"
    f"&BBOX={WEST},{SOUTH},{EAST},{NORTH}"
    f"&WIDTH={GRID_WIDTH}&HEIGHT={GRID_HEIGHT}"
    "&FORMAT=image/png"
    "&ELEVATION=2.0"
    "&STYLES=default-scalar/seq-Greys"
    f"&COLORSCALERANGE={COLORSCALE_MIN},{COLORSCALE_MAX}"
    "&NUMCOLORBANDS=254"
    "&TRANSPARENT=true"
)

FREEZING_POINT = 273.15  # K


class TemperatureGrid:
    """GFS 2m temperature grid for snow/rain precipitation classification.

    Fetches GFS analysis from UCAR THREDDS as a greyscale WMS image.
    Pixel brightness maps linearly to temperature in Kelvin. Transparent
    pixels indicate no data.

    Uses GFS Global 0.25° for worldwide coverage, replacing the previous
    RTMA CONUS-only source.
    """

    def __init__(self):
        self.data: np.ndarray | None = None  # float32 Kelvin, shape (720, 1440)
        self._client: httpx.AsyncClient | None = None

    async def fetch(self) -> bool:
        """Fetch the latest GFS temperature analysis from THREDDS."""
        try:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    follow_redirects=True,
                )

            resp = await self._client.get(THREDDS_WMS_URL)
            if resp.status_code != 200:
                logger.warning("Temperature fetch failed: HTTP %d", resp.status_code)
                return False

            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            arr = np.array(img)

            r_channel = arr[:, :, 0].astype(np.float32)
            alpha = arr[:, :, 3]

            # seq-Greys palette: 0=white (cold) to 255=black (hot), so invert
            temp_k = COLORSCALE_MAX - (r_channel / 255.0) * COLORSCALE_RANGE

            # No-data pixels default to warm (no snow)
            temp_k[alpha == 0] = 300.0

            self.data = temp_k
            logger.info(
                "Temperature updated (GFS global): %.1f-%.1fK range, %d%% below freezing",
                temp_k[alpha > 0].min() if (alpha > 0).any() else 0,
                temp_k[alpha > 0].max() if (alpha > 0).any() else 0,
                int(100 * np.sum((temp_k <= FREEZING_POINT) & (alpha > 0))
                    / max(1, np.sum(alpha > 0))),
            )
            return True

        except Exception:
            logger.exception("Error fetching temperature data")
            return False

    def get_freezing_mask(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> np.ndarray:
        """Return boolean mask: True where surface temp is at or below freezing.

        Args:
            lat: Latitude array in degrees (any shape).
            lon: Longitude array in degrees (same shape as lat).

        Returns:
            Boolean array of same shape as inputs.
        """
        if self.data is None:
            return np.zeros(lat.shape, dtype=bool)

        # Map lat/lon to GFS grid indices
        # Grid row 0 = NORTH (90°), row 719 = SOUTH (-90°)
        # Grid col 0 = WEST (-180°), col 1439 = EAST (180°)
        temp_row = ((NORTH - lat) / TEMP_PIXEL_SIZE).astype(np.int32)
        temp_col = ((lon - WEST) / TEMP_PIXEL_SIZE).astype(np.int32)

        # Clamp to valid range
        temp_row = np.clip(temp_row, 0, GRID_HEIGHT - 1)
        temp_col = np.clip(temp_col, 0, GRID_WIDTH - 1)

        return self.data[temp_row, temp_col] <= FREEZING_POINT

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
