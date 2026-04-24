# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
from PIL import Image

from librewxr.config import settings
from librewxr.tiles.coordinates import tile_pixel_latlons


def render_satellite_tile(
    cloud_grid,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a satellite-like cloud cover tile from IFS cloud data.

    Composites three cloud layers (high, mid, low) into a semi-transparent
    RGBA tile suitable for overlaying on a base map.  Higher clouds render
    brighter (white) and more opaque, simulating an infrared satellite view
    where cold, high cloud tops appear brightest.

    Args:
        cloud_grid: CloudGrid instance with loaded data.
        z, x, y: Tile coordinates.
        tile_size: 256 or 512.
        timestamp: Unix timestamp to select nearest IFS timestep.
        fmt: "png" or "webp".

    Returns:
        Encoded image bytes.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    high, mid, low = cloud_grid.sample(lat_grid, lon_grid, timestamp)

    # Convert 0-100% to 0.0-1.0 float
    h = high.astype(np.float32) / 100.0
    m = mid.astype(np.float32) / 100.0
    lo = low.astype(np.float32) / 100.0

    # Per-layer opacity weights: high clouds most prominent (IR-like)
    alpha_h = h * 0.80
    alpha_m = m * 0.55
    alpha_l = lo * 0.40

    # Total opacity via over-operator approximation
    total_alpha = 1.0 - (1.0 - alpha_h) * (1.0 - alpha_m) * (1.0 - alpha_l)

    # Weighted brightness: high=white, mid=light gray, low=darker gray
    #   High:  245
    #   Mid:   215
    #   Low:   185
    weight_sum = alpha_h + alpha_m + alpha_l
    has_cloud = weight_sum > 0.001
    brightness = np.where(
        has_cloud,
        (alpha_h * 245.0 + alpha_m * 215.0 + alpha_l * 185.0)
        / np.where(has_cloud, weight_sum, 1.0),
        0.0,
    )

    # Build RGBA
    rgba = np.zeros((*high.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(brightness, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(brightness, 0, 255).astype(np.uint8)
    # Slight cool tint in blue channel
    rgba[..., 2] = np.clip(brightness + 5.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(total_alpha * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    if fmt == "webp":
        q = settings.webp_quality
        if q >= 100:
            img.save(buf, format="WEBP", lossless=True)
        else:
            img.save(buf, format="WEBP", quality=q)
    else:
        img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
