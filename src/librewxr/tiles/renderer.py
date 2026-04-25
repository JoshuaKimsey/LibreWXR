# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from librewxr.colors.schemes import colorize
from librewxr.config import settings
from librewxr.data.coverage import sample_coverage, sample_feather
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
    color_scheme: int = 7,
    smooth: bool = False,
    snow: bool = False,
    fmt: str = "png",
    ecmwf_grid=None,
    enabled_regions: list[str] | None = None,
    frame_timestamp: int | None = None,
    nowcast_blend: float | None = None,
    flow_regions: dict[str, np.ndarray] | None = None,
    ecmwf_flow: np.ndarray | None = None,
    arrow_style: str = "light",
) -> bytes:
    """Render a single map tile from composite radar data.

    Args:
        frame_regions: dict mapping region name -> uint8 numpy array
        z, x, y: tile coordinates
        tile_size: 256 or 512
        color_scheme: Rain Viewer color scheme ID
        smooth: apply Gaussian blur
        snow: use snow color variant (requires ecmwf_grid for classification)
        fmt: "png" or "webp"
        ecmwf_grid: ECMWFGrid for global fallback coverage and snow classification
        enabled_regions: list of enabled region names (for overlap check)
        frame_timestamp: Unix timestamp of the radar frame being rendered
        nowcast_blend: If not None, this is a nowcast frame. Value 0.0–1.0
            indicates how much to trust the extrapolated radar (1.0 = trust
            radar fully, 0.0 = trust IFS fully). The renderer blends
            extrapolated radar with IFS forecast, feathered at coverage
            boundaries.

    Returns:
        Encoded image bytes.
    """
    # Find which regions overlap this tile and have data
    regions = overlapping_regions(z, x, y, enabled_regions)
    regions_with_data = [r for r in regions if r.name in frame_regions]

    if not regions_with_data:
        # No radar regions cover this tile — try ECMWF fallback
        if ecmwf_grid is not None and ecmwf_grid.data is not None:
            return _render_ecmwf_only_tile(
                ecmwf_grid, z, x, y, tile_size,
                color_scheme, smooth, snow, fmt, frame_timestamp,
                ecmwf_flow=ecmwf_flow,
                arrow_style=arrow_style if flow_regions or ecmwf_flow is not None else "",
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

    # Fill uncovered pixels from ECMWF precipitation data.
    # For nowcast frames, blend extrapolated radar with IFS forecast
    # using temporal weight + spatial feathering at coverage boundaries.
    if ecmwf_grid is not None and ecmwf_grid.data is not None:
        if nowcast_blend is not None:
            values = _blend_nowcast(
                values, regions, z, x, y, tile_size, pad, ecmwf_grid,
                frame_timestamp, smooth, nowcast_blend,
            )
        else:
            values = _fill_ecmwf_fallback(
                values, regions, z, x, y, tile_size, pad, ecmwf_grid,
                frame_timestamp, smooth,
            )

    # Apply noise floor
    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    # Apply color scheme with per-pixel snow/rain selection
    if snow and ecmwf_grid is not None:
        if pad > 0:
            lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
        else:
            lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
        is_snow = ecmwf_grid.get_snow_mask(lat_grid, lon_grid, frame_timestamp)
        rgba_rain = colorize(values, color_scheme, snow=False)
        rgba_snow = colorize(values, color_scheme, snow=True)
        rgba = np.where(is_snow[..., np.newaxis], rgba_snow, rgba_rain)
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

    if flow_regions or ecmwf_flow is not None:
        img = _draw_motion_arrows(
            img, flow_regions, frame_regions, regions_with_data,
            z, x, y, tile_size, arrow_style,
            ecmwf_flow=ecmwf_flow,
            ecmwf_grid=ecmwf_grid,
            frame_timestamp=frame_timestamp,
        )

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

    Regions are processed in order (finest resolution first).  Each
    region claims the pixels within its own coverage mask; lower-
    priority regions can only fill pixels that no higher-priority
    region has claimed.  This prevents coarser composites from
    overwriting authoritative "no echo" zeros inside a higher-priority
    region's coverage area — e.g. MSC Canada won't spill light-rain
    returns across the border into NEXRAD-covered Maine.
    """
    out_size = tile_size + 2 * pad if pad > 0 else tile_size
    values = np.zeros((out_size, out_size), dtype=np.uint8)
    # Pixels already authoritatively covered by a higher-priority region.
    claimed = np.zeros((out_size, out_size), dtype=bool)

    # Tile lat/lon grid for coverage-mask lookups (matches the output
    # buffer, including padding when smoothing is enabled).
    if pad > 0:
        tile_lats, tile_lons = tile_pixel_latlons_padded(
            z, x, y, tile_size, pad
        )
    else:
        tile_lats, tile_lons = tile_pixel_latlons(z, x, y, tile_size)

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

        # Fill: only where no higher-priority region has claimed the
        # pixel AND this region actually has data there.
        fill_mask = ~claimed & (region_values > 0)
        values[fill_mask] = region_values[fill_mask]

        # Mark pixels inside this region's coverage as claimed so
        # lower-priority regions can't overwrite them — even the zeros.
        region_coverage = sample_coverage(
            region.name, tile_lats, tile_lons
        )
        claimed |= region_coverage

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


def _fill_ecmwf_fallback(
    values: np.ndarray,
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int, pad: int,
    ecmwf_grid,
    frame_timestamp: int | None = None,
    smooth: bool = False,
) -> np.ndarray:
    """Fill pixels outside radar coverage from ECMWF.

    IEM N0Q and the Nordic / DWD composites all encode pixel value 0
    for both "outside radar range" *and* "clear sky within range", so
    we can't use ``values == 0`` alone — that would make ECMWF bleed
    into legitimately dry areas inside radar coverage. Instead we use
    precomputed station-based coverage masks (see data/coverage.py):
    a pixel is filled only when it has no radar value *and* no region
    whose station circles cover it.
    """
    # Get lat/lon for the tile pixels
    if pad > 0:
        lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
    else:
        lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)

    # Union coverage from every region that overlaps this tile — even
    # regions we don't have a frame for yet, because if a station reaches
    # this tile we still don't want ECMWF overlapping with radar.
    covered = np.zeros(lat_grid.shape, dtype=bool)
    for region in regions:
        covered |= sample_coverage(region.name, lat_grid, lon_grid)

    uncovered = (values == 0) & ~covered
    if not uncovered.any():
        return values

    ecmwf_values = ecmwf_grid.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    result = values.copy()
    result[uncovered] = ecmwf_values[uncovered]
    return result


def _blend_nowcast(
    radar_values: np.ndarray,
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int, pad: int,
    ecmwf_grid,
    frame_timestamp: int | None = None,
    smooth: bool = False,
    blend_weight: float = 1.0,
) -> np.ndarray:
    """Blend extrapolated radar with IFS forecast for nowcast frames.

    Uses a combination of temporal and spatial weighting:

    - **Temporal** (``blend_weight``): 1.0 at T+10 min (trust radar),
      fading to 0.0 at the last nowcast step (trust IFS).
    - **Spatial** (feather mask): 1.0 deep inside radar coverage, fading
      to 0.0 at coverage boundaries to prevent hard seams.

    The effective per-pixel radar weight is ``blend_weight × feather``.
    Outside radar coverage, IFS is used directly (same as past frames).
    """
    if pad > 0:
        lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
    else:
        lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)

    # Sample IFS for ALL pixels (not just uncovered)
    ifs_values = ecmwf_grid.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    # Soften IFS before blending to reduce spatial mismatch artifacts.
    # IFS at 9km resolution has precipitation in slightly different locations
    # than radar — blurring smooths the IFS contribution so the transition
    # looks like a gradual handoff rather than ghosting/doubling.
    ifs_f = ifs_values.astype(np.float32)
    ksize = 5 if tile_size <= 256 else 7
    ifs_f = cv2.GaussianBlur(ifs_f, (ksize, ksize), 0)

    # Build the spatial feather weight: union across all overlapping regions
    feather = np.zeros(lat_grid.shape, dtype=np.float32)
    for region in regions:
        feather = np.maximum(feather, sample_feather(region.name, lat_grid, lon_grid))

    # Per-pixel effective radar weight
    effective_w = blend_weight * feather

    # Blend: extrapolated radar × weight + IFS × (1 − weight)
    radar_f = radar_values.astype(np.float32)
    blended = effective_w * radar_f + (1.0 - effective_w) * ifs_f

    # Don't hallucinate precipitation where neither source has any
    both_zero = (radar_values == 0) & (ifs_values == 0)
    result = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    result[both_zero] = 0

    return result


def _render_ecmwf_only_tile(
    ecmwf_grid,
    z: int, x: int, y: int,
    tile_size: int,
    color_scheme: int,
    smooth: bool,
    snow: bool,
    fmt: str,
    frame_timestamp: int | None = None,
    ecmwf_flow: np.ndarray | None = None,
    arrow_style: str = "",
) -> bytes:
    """Render a tile entirely from ECMWF data (no radar regions overlap)."""
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    values = ecmwf_grid.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    # Apply noise floor
    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    # Apply color scheme with per-pixel snow/rain selection
    if snow:
        is_snow = ecmwf_grid.get_snow_mask(lat_grid, lon_grid, frame_timestamp)
        rgba_rain = colorize(values, color_scheme, snow=False)
        rgba_snow = colorize(values, color_scheme, snow=True)
        rgba = np.where(is_snow[..., np.newaxis], rgba_snow, rgba_rain)
    else:
        rgba = colorize(values, color_scheme, snow=False)

    img = Image.fromarray(rgba, "RGBA")

    if ecmwf_flow is not None and arrow_style:
        img = _draw_motion_arrows(
            img, None, {}, [],
            z, x, y, tile_size, arrow_style,
            ecmwf_flow=ecmwf_flow,
            ecmwf_grid=ecmwf_grid,
            frame_timestamp=frame_timestamp,
        )

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


def _draw_motion_arrows(
    img: Image.Image,
    flow_regions: dict[str, np.ndarray] | None,
    frame_regions: dict[str, np.ndarray],
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int,
    style: str = "light",
    ecmwf_flow: np.ndarray | None = None,
    ecmwf_grid=None,
    frame_timestamp: int | None = None,
) -> Image.Image:
    """Draw precipitation motion vector arrows on the tile.

    Overlays semi-transparent arrows on areas with active precipitation,
    showing storm movement direction and relative speed. Arrows are
    derived from the optical flow field computed between the two most
    recent radar frames, with ECMWF IFS flow as a global fallback
    outside radar coverage.

    ``style`` selects the arrow colour: ``"light"`` for white arrows
    (best on dark maps) or ``"dark"`` for dark arrows (best on light maps).
    """
    # Regions that have both frame data and flow data
    if flow_regions:
        valid_regions = [
            r for r in regions
            if r.name in flow_regions and r.name in frame_regions
        ]
    else:
        valid_regions = []

    has_ecmwf = (
        ecmwf_flow is not None
        and ecmwf_grid is not None
        and ecmwf_grid.data is not None
    )

    if not valid_regions and not has_ecmwf:
        return img

    # Precompute pixel-index arrays for each valid radar region
    region_info = []
    for r in valid_regions:
        row_f, col_f = region_pixel_indices_fractional(r, z, x, y, tile_size)
        row_i, col_i = region_pixel_indices(r, z, x, y, tile_size)
        region_info.append((r, row_f, col_f, row_i, col_i))

    # Precompute lat/lon grid for ECMWF fallback (only if needed)
    ecmwf_latlons = None
    ecmwf_precip = None
    radar_coverage = None
    if has_ecmwf:
        from librewxr.data.ecmwf_grid import GRID_HEIGHT, GRID_WIDTH, NORTH, PIXEL_SIZE, WEST
        ecmwf_latlons = tile_pixel_latlons(z, x, y, tile_size)
        ecmwf_precip = ecmwf_grid.sample(
            ecmwf_latlons[0], ecmwf_latlons[1], frame_timestamp,
        )
        # Precompute radar coverage so we can distinguish "clear sky under
        # radar" from "outside radar coverage" when deciding whether to
        # fall through to ECMWF arrows.
        if region_info:
            lat_grid, lon_grid = ecmwf_latlons
            radar_coverage = np.zeros(lat_grid.shape, dtype=bool)
            for r in regions:
                radar_coverage |= sample_coverage(r.name, lat_grid, lon_grid)

    # Noise floor: arrows should only appear where the rendered tile
    # actually shows precipitation (same threshold used for display).
    noise_threshold = 0
    if settings.noise_floor_dbz > -32:
        noise_threshold = int((settings.noise_floor_dbz + 32) * 2)

    spacing = 32 if tile_size <= 256 else 48
    line_w = 2 if tile_size <= 256 else 3
    arrow_color = (40, 40, 40, 180) if style == "dark" else (255, 255, 255, 160)
    speed_scale = 4.0
    min_len = 5.0
    max_len = spacing * 0.75

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ty in range(spacing // 2, tile_size, spacing):
        for tx in range(spacing // 2, tile_size, spacing):
            arrow_dx = arrow_dy = 0.0
            found = False

            # Try radar regions in priority order (finest resolution first)
            for r, row_f, col_f, row_i, col_i in region_info:
                ri, ci = int(row_i[ty, tx]), int(col_i[ty, tx])
                if ri < 0 or ci < 0:
                    continue  # Outside this region, try next

                frame = frame_regions[r.name]
                if frame[ri, ci] < noise_threshold:
                    # Only claim the pixel if it's within actual radar
                    # coverage (clear sky).  Pixels inside the region's
                    # bounding box but outside station coverage should
                    # fall through to ECMWF.
                    if radar_coverage is None or radar_coverage[ty, tx]:
                        found = True
                        break
                    continue

                flow = flow_regions[r.name]
                rf = min(max(int(row_f[ty, tx]), 0), flow.shape[0] - 1)
                cf = min(max(int(col_f[ty, tx]), 0), flow.shape[1] - 1)

                fx = float(flow[rf, cf, 0])
                fy = float(flow[rf, cf, 1])

                # Local scale: region pixels per tile pixel (finite diff)
                tx1 = min(tx + 1, tile_size - 1)
                ty1 = min(ty + 1, tile_size - 1)
                tx0 = max(tx - 1, 0)
                ty0 = max(ty - 1, 0)
                dcol = (col_f[ty, tx1] - col_f[ty, tx0]) / (tx1 - tx0)
                drow = (row_f[ty1, tx] - row_f[ty0, tx]) / (ty1 - ty0)

                if abs(dcol) < 1e-8 or abs(drow) < 1e-8:
                    found = True
                    break

                raw_dx = fx / dcol
                raw_dy = fy / drow
                raw_len = math.hypot(raw_dx, raw_dy)

                if raw_len < 0.5:
                    found = True
                    break  # Effectively stationary

                target_len = min(max(raw_len * speed_scale, min_len), max_len)
                arrow_dx = raw_dx / raw_len * target_len
                arrow_dy = raw_dy / raw_len * target_len
                found = True
                break  # Used this region for this grid point

            # ECMWF fallback: only if no radar region claimed this pixel
            if not found and has_ecmwf:
                if ecmwf_precip[ty, tx] < noise_threshold:
                    continue  # Below noise floor — not visible on tile

                lat = float(ecmwf_latlons[0][ty, tx])
                lon = float(ecmwf_latlons[1][ty, tx])

                # Convert lat/lon to ECMWF grid indices
                er = (NORTH - lat) / PIXEL_SIZE
                ec = (lon - WEST) / PIXEL_SIZE
                eri = min(max(int(er), 0), GRID_HEIGHT - 1)
                eci = min(max(int(ec), 0), GRID_WIDTH - 1)

                fx = float(ecmwf_flow[eri, eci, 0])
                fy = float(ecmwf_flow[eri, eci, 1])

                # Local scale: ECMWF grid pixels per tile pixel
                # Use lat/lon difference to compute the Jacobian
                tx1 = min(tx + 1, tile_size - 1)
                ty1 = min(ty + 1, tile_size - 1)
                tx0 = max(tx - 1, 0)
                ty0 = max(ty - 1, 0)

                dlat_dy = (ecmwf_latlons[0][ty1, tx] - ecmwf_latlons[0][ty0, tx]) / (ty1 - ty0)
                dlon_dx = (ecmwf_latlons[1][ty, tx1] - ecmwf_latlons[1][ty, tx0]) / (tx1 - tx0)

                # Convert degrees to ECMWF grid pixels
                drow_dy = -dlat_dy / PIXEL_SIZE  # negative: lat decreases as row increases
                dcol_dx = dlon_dx / PIXEL_SIZE

                if abs(dcol_dx) < 1e-8 or abs(drow_dy) < 1e-8:
                    continue

                raw_dx = fx / dcol_dx
                raw_dy = fy / drow_dy
                raw_len = math.hypot(raw_dx, raw_dy)

                if raw_len < 0.5:
                    continue

                target_len = min(max(raw_len * speed_scale, min_len), max_len)
                arrow_dx = raw_dx / raw_len * target_len
                arrow_dy = raw_dy / raw_len * target_len
                found = True

            if not found or (arrow_dx == 0.0 and arrow_dy == 0.0):
                continue

            # Arrow biased toward the tip (60% forward)
            x0 = tx - arrow_dx * 0.4
            y0 = ty - arrow_dy * 0.4
            x1 = tx + arrow_dx * 0.6
            y1 = ty + arrow_dy * 0.6

            # Shaft
            draw.line(
                [(x0, y0), (x1, y1)],
                fill=arrow_color, width=line_w,
            )

            # Arrowhead
            angle = math.atan2(arrow_dy, arrow_dx)
            head_len = max(4.0, min(8.0, math.hypot(arrow_dx, arrow_dy) * 0.35))
            ha = 0.45  # half-angle
            draw.polygon(
                [
                    (x1, y1),
                    (x1 - head_len * math.cos(angle - ha),
                     y1 - head_len * math.sin(angle - ha)),
                    (x1 - head_len * math.cos(angle + ha),
                     y1 - head_len * math.sin(angle + ha)),
                ],
                fill=arrow_color,
            )

    return Image.alpha_composite(img, overlay)


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
