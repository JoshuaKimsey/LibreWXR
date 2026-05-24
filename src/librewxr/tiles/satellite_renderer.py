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


def render_gmgsi_tile(
    source,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    timestamp: int | None = None,
    fmt: str = "png",
) -> bytes:
    """Render a tile from a GMGSI satellite source.

    Phase 1 is LW-only — cold (high cloud tops) → bright + opaque,
    warm (ground / ocean) → transparent.  The encoding is already
    cold=high uint8 directly from NESDIS, so the alpha curve is just
    a normalized version of the encoded value with a mild gamma to
    pull thin cirrus out of the noise floor.

    Phase 2 will introduce the VIS-over-LW composite (alpha = vis/255)
    in a separate `render_gmgsi_composite_tile` function that takes
    both sources; this one stays as the IR-only path.
    """
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    encoded = source.sample(lat_grid, lon_grid, timestamp)

    # Normalize to [0, 1] for both brightness and alpha.  Encoded == 0
    # is the no-data sentinel (outside the disk or dqf-masked); leave
    # it fully transparent.
    norm = encoded.astype(np.float32) / 255.0
    no_data = encoded == 0

    # Gamma curve gently lifts thin clouds so faint features show
    # without darkening the bright cloud tops.  0.7 is the standard
    # GeoColor-style perceptual lift.
    alpha = np.power(np.clip(norm, 0.0, 1.0), 0.7)
    alpha = np.where(no_data, 0.0, alpha)

    # Slight cool tint in the blue channel matches the existing
    # IFS-derived satellite aesthetic; the renderer hasn't visibly
    # changed for users when GMGSI takes over for IFS-cloud.
    brightness = np.clip(norm * 255.0, 0, 255)
    rgba = np.zeros((*encoded.shape, 4), dtype=np.uint8)
    rgba[..., 0] = brightness.astype(np.uint8)
    rgba[..., 1] = brightness.astype(np.uint8)
    rgba[..., 2] = np.clip(brightness + 5.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

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
