# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
from PIL import Image, ImageFilter

from librewxr.colors.schemes import colorize
from librewxr.config import settings
from librewxr.data.regions import RegionDef
from librewxr.tiles.coordinates import (
    overlapping_regions,
    region_pixel_indices,
    region_pixel_indices_fractional,
    region_pixel_indices_padded,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
)


def render_tile(
    frame_regions: dict[str, np.ndarray],
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    color_scheme: int = 2,
    smooth: bool = False,
    snow: bool = False,
    fmt: str = "png",
    temperature_grid=None,
    enabled_regions: list[str] | None = None,
    reflectivity_grid=None,
) -> bytes:
    """Render a single map tile from composite radar data.

    Args:
        frame_regions: dict mapping region name -> uint8 numpy array
        z, x, y: tile coordinates
        tile_size: 256 or 512
        color_scheme: Rain Viewer color scheme ID
        smooth: apply Gaussian blur
        snow: use snow color variant (requires temperature_grid)
        fmt: "png" or "webp"
        temperature_grid: TemperatureGrid for per-pixel snow/rain classification
        enabled_regions: list of enabled region names (for overlap check)
        reflectivity_grid: GFSReflectivityGrid for global fallback coverage

    Returns:
        Encoded image bytes.
    """
    # Find which regions overlap this tile and have data
    regions = overlapping_regions(z, x, y, enabled_regions)
    regions_with_data = [r for r in regions if r.name in frame_regions]

    if not regions_with_data:
        # No radar regions cover this tile — try GFS fallback
        if reflectivity_grid is not None and reflectivity_grid.data is not None:
            return _render_gfs_only_tile(
                reflectivity_grid, z, x, y, tile_size,
                color_scheme, smooth, snow, fmt, temperature_grid,
            )
        return _transparent_tile(tile_size, fmt)

    # Determine blur radius (zoom-scaled) for smooth mode
    blur_radius = 0.0
    if smooth and settings.smooth_radius > 0:
        scale = max(0.0, min(1.0, (z - 3) / 5))
        blur_radius = settings.smooth_radius * scale

    use_blur = smooth and blur_radius >= 0.5
    pad = int(blur_radius * 3) if use_blur else 0

    # Single-region fast path (99%+ of tiles)
    if len(regions_with_data) == 1:
        region = regions_with_data[0]
        values = _sample_region(
            frame_regions[region.name], region, z, x, y, tile_size,
            smooth, use_blur, pad,
        )
    else:
        # Multi-region compositing: layer regions, finest resolution first
        values = _composite_regions(
            frame_regions, regions_with_data, z, x, y, tile_size,
            smooth, use_blur, pad,
        )

    # Fill uncovered pixels from GFS simulated reflectivity
    if reflectivity_grid is not None and reflectivity_grid.data is not None:
        values = _fill_gfs_fallback(
            values, regions, z, x, y, tile_size, pad, reflectivity_grid,
        )

    # Apply noise floor
    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    # Apply color scheme with per-pixel snow/rain selection
    if snow and temperature_grid is not None:
        if pad > 0:
            lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
        else:
            lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
        freezing = temperature_grid.get_freezing_mask(lat_grid, lon_grid)
        rgba_rain = colorize(values, color_scheme, snow=False)
        rgba_snow = colorize(values, color_scheme, snow=True)
        rgba = np.where(freezing[..., np.newaxis], rgba_snow, rgba_rain)
    else:
        rgba = colorize(values, color_scheme, snow=False)

    # Create image
    img = Image.fromarray(rgba, "RGBA")

    if use_blur:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        a = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        r, g, b = rgb.split()
        img = Image.merge("RGBA", (r, g, b, a))

        if pad > 0:
            img = img.crop((pad, pad, pad + tile_size, pad + tile_size))

    return _encode_image(img, fmt)


def _sample_region(
    frame_data: np.ndarray,
    region: RegionDef,
    z: int, x: int, y: int,
    tile_size: int,
    smooth: bool,
    use_blur: bool,
    pad: int,
) -> np.ndarray:
    """Sample pixel values from a single region."""
    if pad > 0:
        row_idx, col_idx = region_pixel_indices_padded(
            region, z, x, y, tile_size, pad
        )
        padded = np.pad(frame_data, ((0, 1), (0, 1)), constant_values=0)
        values = padded[row_idx, col_idx]
    else:
        row_idx, col_idx = region_pixel_indices(region, z, x, y, tile_size)
        if smooth:
            values = _bilinear_sample(frame_data, region, z, x, y, tile_size)
            oob = (row_idx == -1) | (col_idx == -1)
            values[oob] = 0
        else:
            padded = np.pad(frame_data, ((0, 1), (0, 1)), constant_values=0)
            values = padded[row_idx, col_idx]
    return values


def _composite_regions(
    frame_regions: dict[str, np.ndarray],
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int,
    smooth: bool,
    use_blur: bool,
    pad: int,
) -> np.ndarray:
    """Composite values from multiple overlapping regions.

    Regions are processed in order (finest resolution first).
    Later regions fill in zeros left by earlier ones.
    """
    out_size = tile_size + 2 * pad if pad > 0 else tile_size
    values = np.zeros((out_size, out_size), dtype=np.uint8)

    for region in regions:
        data = frame_regions.get(region.name)
        if data is None:
            continue

        if pad > 0:
            row_idx, col_idx = region_pixel_indices_padded(
                region, z, x, y, tile_size, pad
            )
        else:
            row_idx, col_idx = region_pixel_indices(region, z, x, y, tile_size)

        padded = np.pad(data, ((0, 1), (0, 1)), constant_values=0)
        region_values = padded[row_idx, col_idx]

        # Fill zeros in output with this region's data
        fill_mask = (values == 0) & (region_values > 0)
        values[fill_mask] = region_values[fill_mask]

    return values


def render_coverage_tile(
    frame_regions: dict[str, np.ndarray],
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    enabled_regions: list[str] | None = None,
) -> bytes:
    """Render a coverage tile showing where radar data exists."""
    regions = overlapping_regions(z, x, y, enabled_regions)
    regions_with_data = [r for r in regions if r.name in frame_regions]

    if not regions_with_data:
        return _transparent_tile(tile_size, "png")

    # Composite coverage from all regions
    values = np.zeros((tile_size, tile_size), dtype=np.uint8)
    for region in regions_with_data:
        data = frame_regions[region.name]
        row_idx, col_idx = region_pixel_indices(region, z, x, y, tile_size)
        padded = np.pad(data, ((0, 1), (0, 1)), constant_values=0)
        region_values = padded[row_idx, col_idx]
        fill_mask = (values == 0) & (region_values > 0)
        values[fill_mask] = region_values[fill_mask]

    # Coverage: non-zero = white semi-transparent
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    mask = values > 0
    rgba[mask] = [255, 255, 255, 128]

    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, "png")


def _build_coverage_mask(
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int, pad: int,
) -> np.ndarray:
    """Build a boolean mask of pixels covered by any radar region.

    True = this pixel is within at least one radar region's spatial extent
    (regardless of whether there's currently precipitation there).
    """
    out_size = tile_size + 2 * pad if pad > 0 else tile_size
    covered = np.zeros((out_size, out_size), dtype=bool)

    for region in regions:
        if pad > 0:
            row_idx, col_idx = region_pixel_indices_padded(
                region, z, x, y, tile_size, pad
            )
        else:
            row_idx, col_idx = region_pixel_indices(region, z, x, y, tile_size)

        # Pixels with valid indices (not -1) are within this region
        in_bounds = (row_idx != -1) & (col_idx != -1)
        covered |= in_bounds

    return covered


def _fill_gfs_fallback(
    values: np.ndarray,
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int, pad: int,
    reflectivity_grid,
) -> np.ndarray:
    """Fill pixels not covered by any radar region with GFS data."""
    covered = _build_coverage_mask(regions, z, x, y, tile_size, pad)

    # Only fill pixels that are NOT covered by any radar region
    uncovered = ~covered
    if not uncovered.any():
        return values

    # Get lat/lon for the tile pixels
    if pad > 0:
        lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
    else:
        lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)

    gfs_values = reflectivity_grid.sample(lat_grid, lon_grid)

    result = values.copy()
    result[uncovered] = gfs_values[uncovered]
    return result


def _render_gfs_only_tile(
    reflectivity_grid,
    z: int, x: int, y: int,
    tile_size: int,
    color_scheme: int,
    smooth: bool,
    snow: bool,
    fmt: str,
    temperature_grid,
) -> bytes:
    """Render a tile entirely from GFS data (no radar regions overlap)."""
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    values = reflectivity_grid.sample(lat_grid, lon_grid)

    # Apply noise floor
    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    # Apply color scheme with per-pixel snow/rain selection
    if snow and temperature_grid is not None:
        freezing = temperature_grid.get_freezing_mask(lat_grid, lon_grid)
        rgba_rain = colorize(values, color_scheme, snow=False)
        rgba_snow = colorize(values, color_scheme, snow=True)
        rgba = np.where(freezing[..., np.newaxis], rgba_snow, rgba_rain)
    else:
        rgba = colorize(values, color_scheme, snow=False)

    img = Image.fromarray(rgba, "RGBA")
    return _encode_image(img, fmt)


def _bilinear_sample(
    frame_data: np.ndarray, region: RegionDef,
    z: int, x: int, y: int, tile_size: int
) -> np.ndarray:
    """Sample frame data using bilinear interpolation for smooth rendering."""
    row_f, col_f = region_pixel_indices_fractional(region, z, x, y, tile_size)

    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    r1 = np.minimum(r0 + 1, region.height - 1)
    c1 = np.minimum(c0 + 1, region.width - 1)

    dr = row_f - r0
    dc = col_f - c0

    v00 = frame_data[r0, c0].astype(np.float32)
    v01 = frame_data[r0, c1].astype(np.float32)
    v10 = frame_data[r1, c0].astype(np.float32)
    v11 = frame_data[r1, c1].astype(np.float32)

    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)

    interp = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )

    nearest = v00
    result = np.where(any_zero, nearest, interp)

    return np.clip(result + 0.5, 0, 255).astype(np.uint8)


def _transparent_tile(tile_size: int, fmt: str) -> bytes:
    """Return a fully transparent tile."""
    img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
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
