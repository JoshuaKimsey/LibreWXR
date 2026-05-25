# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import math
from dataclasses import dataclass

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
    region_pixel_indices_fractional_padded,
    region_pixel_indices_padded,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
)


# ---------------------------------------------------------------------------
# Cacheable geometry
# ---------------------------------------------------------------------------
# A tile request expands across many independent axes — color scheme (9),
# output format (PNG / WebP), snow flag, smooth flag, arrow style.  The
# expensive 95% of rendering (region sampling, smoothing, NWP blend) is
# identical across all of them; only the final colorize + blur + encode +
# arrow overlay differ.  ``TileGeometry`` is the cacheable intermediate:
# everything you need to produce any variant of the final image, without
# committing to a color scheme or format yet.


@dataclass
class TileGeometry:
    """Pre-presentation tile data, cached between compute and present steps.

    ``values`` holds the post-sample, post-blend, post-noise-floor uint8
    indices into the color LUT.  ``snow_mask`` is only populated when the
    geometry was computed with ``snow=True`` and an NWP chain was
    available — if a later request asks for ``snow=False``, present just
    ignores the mask.  The cache key includes ``snow`` so we never feed a
    ``snow=False`` request an entry that lacks the mask but expects it.
    """

    values: np.ndarray  # uint8 (H, W) where H = W = tile_size + 2*pad
    snow_mask: np.ndarray | None  # bool, same shape; None if snow wasn't requested
    tile_size: int  # final size after blur-crop
    pad: int  # padding on each side; 0 unless blur will be applied
    blur_radius: float  # 0.0 = no blur
    is_transparent: bool = False

    @classmethod
    def transparent(cls, tile_size: int) -> "TileGeometry":
        """Sentinel for tiles with neither radar nor NWP coverage."""
        return cls(
            values=np.empty((0, 0), dtype=np.uint8),
            snow_mask=None,
            tile_size=tile_size,
            pad=0,
            blur_radius=0.0,
            is_transparent=True,
        )

    @property
    def nbytes(self) -> int:
        if self.is_transparent:
            return 64  # account for object overhead
        n = int(self.values.nbytes)
        if self.snow_mask is not None:
            n += int(self.snow_mask.nbytes)
        return max(n, 64)


# ---------------------------------------------------------------------------
# Compute step (cached)
# ---------------------------------------------------------------------------


def compute_tile_geometry(
    frame_regions: dict[str, np.ndarray],
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
    smooth: bool = False,
    snow: bool = False,
    nwp_chain=None,
    enabled_regions: list[str] | None = None,
    frame_timestamp: int | None = None,
    nowcast_blend: float | None = None,
) -> TileGeometry:
    """Compute the cacheable geometry for a tile.

    Performs the expensive work — region sampling, multi-region
    compositing, NWP fill/blend, noise-floor masking, and the snow mask
    (when requested) — and returns a ``TileGeometry`` that any number of
    color schemes / output formats / arrow styles can be rendered from
    via ``present_tile``.
    """
    regions = overlapping_regions(z, x, y, enabled_regions)
    regions_with_data = [r for r in regions if r.name in frame_regions]

    has_nwp = nwp_chain is not None and nwp_chain.has_data()

    if not regions_with_data:
        if has_nwp:
            return _compute_nwp_only_geometry(
                nwp_chain, z, x, y, tile_size, smooth, snow, frame_timestamp,
            )
        return TileGeometry.transparent(tile_size)

    # Determine blur radius from local geometry: scale Gaussian kernel
    # to the number of tile pixels covered by a single region pixel.
    # Uses the highest-priority (finest) region's Jacobian so that mixed
    # coarse + fine tiles size their blur to the resolution that's
    # actually visible at the center.
    blur_radius = _compute_blur_radius(
        regions_with_data[0], z, x, y, tile_size,
    ) if smooth else 0.0

    use_blur = blur_radius >= 0.5
    pad = int(blur_radius * 3) if use_blur else 0

    # Single-region fast path (99%+ of tiles)
    if len(regions_with_data) == 1:
        region = regions_with_data[0]
        values = _sample_region(
            frame_regions[region.name], region, z, x, y, tile_size,
            smooth, use_blur, pad,
        )
    else:
        values = _composite_regions(
            frame_regions, regions_with_data, z, x, y, tile_size,
            smooth, use_blur, pad,
        )

    # Fill uncovered pixels from NWP precipitation data.  For nowcast
    # frames, blend extrapolated radar with NWP using temporal weight +
    # spatial feathering at coverage boundaries.
    if has_nwp:
        if nowcast_blend is not None:
            values = _blend_nowcast(
                values, regions, z, x, y, tile_size, pad, nwp_chain,
                frame_timestamp, smooth, nowcast_blend,
            )
        else:
            values = _fill_ecmwf_fallback(
                values, regions, z, x, y, tile_size, pad, nwp_chain,
                frame_timestamp, smooth,
            )

    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    snow_mask = None
    if snow and nwp_chain is not None:
        if pad > 0:
            lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
        else:
            lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
        snow_mask = nwp_chain.get_snow_mask(lat_grid, lon_grid, frame_timestamp)

    return TileGeometry(
        values=values,
        snow_mask=snow_mask,
        tile_size=tile_size,
        pad=pad,
        blur_radius=blur_radius,
    )


def _compute_nwp_only_geometry(
    nwp_chain,
    z: int, x: int, y: int,
    tile_size: int,
    smooth: bool,
    snow: bool,
    frame_timestamp: int | None,
) -> TileGeometry:
    """Geometry for a tile entirely from NWP (no radar regions overlap)."""
    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    values = nwp_chain.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    snow_mask = None
    if snow:
        snow_mask = nwp_chain.get_snow_mask(lat_grid, lon_grid, frame_timestamp)

    return TileGeometry(
        values=values,
        snow_mask=snow_mask,
        tile_size=tile_size,
        pad=0,
        blur_radius=0.0,
    )


# ---------------------------------------------------------------------------
# Present step (per-request, cheap)
# ---------------------------------------------------------------------------


def present_tile(
    geom: TileGeometry,
    color_scheme: int,
    fmt: str,
    *,
    arrow_style: str = "",
    flow_regions: dict[str, np.ndarray] | None = None,
    frame_regions: dict[str, np.ndarray] | None = None,
    enabled_regions: list[str] | None = None,
    ecmwf_flow: np.ndarray | None = None,
    ecmwf_grid=None,
    frame_timestamp: int | None = None,
    z: int = 0,
    x: int = 0,
    y: int = 0,
) -> bytes:
    """Render a cached ``TileGeometry`` to encoded bytes.

    Does the cheap tail: LUT colorize, optional Gaussian blur + crop,
    optional motion-arrow overlay, image encode.  All of these are
    per-request because they depend on the request's color/format/arrow
    parameters, which are deliberately *not* in the cache key.

    Arrow inputs (``flow_regions``, ``ecmwf_flow``, etc.) are passed
    fresh on each call rather than baked into the geometry so a tile
    that's cached without arrows can still render an arrow variant when
    a later request asks for one.
    """
    if geom.is_transparent:
        return _transparent_tile(geom.tile_size, fmt)

    if geom.snow_mask is not None:
        rgba_rain = colorize(geom.values, color_scheme, snow=False)
        rgba_snow = colorize(geom.values, color_scheme, snow=True)
        rgba = np.where(geom.snow_mask[..., np.newaxis], rgba_snow, rgba_rain)
    else:
        rgba = colorize(geom.values, color_scheme, snow=False)

    img = Image.fromarray(rgba, "RGBA")

    if geom.blur_radius >= 0.5:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=geom.blur_radius))
        a = a.filter(ImageFilter.GaussianBlur(radius=geom.blur_radius))
        r, g, b = rgb.split()
        img = Image.merge("RGBA", (r, g, b, a))

        if geom.pad > 0:
            img = img.crop(
                (geom.pad, geom.pad, geom.pad + geom.tile_size, geom.pad + geom.tile_size)
            )

    if arrow_style and (flow_regions or ecmwf_flow is not None):
        regions = overlapping_regions(z, x, y, enabled_regions)
        if frame_regions:
            regions_with_data = [r for r in regions if r.name in frame_regions]
        else:
            regions_with_data = []
        img = _draw_motion_arrows(
            img, flow_regions, frame_regions or {}, regions_with_data,
            z, x, y, geom.tile_size, arrow_style,
            ecmwf_flow=ecmwf_flow,
            ecmwf_grid=ecmwf_grid,
            frame_timestamp=frame_timestamp,
        )

    return _encode_image(img, fmt)


# ---------------------------------------------------------------------------
# Convenience: render in one call (used by tests + the warmer's startup path)
# ---------------------------------------------------------------------------


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
    nwp_chain=None,
    enabled_regions: list[str] | None = None,
    frame_timestamp: int | None = None,
    nowcast_blend: float | None = None,
    flow_regions: dict[str, np.ndarray] | None = None,
    ecmwf_flow: np.ndarray | None = None,
    arrow_style: str = "light",
) -> bytes:
    """Compute geometry and present it in a single call.

    Convenience wrapper for callers (mostly tests) that don't care about
    the cache layer.  Production code paths call ``compute_tile_geometry``
    and ``present_tile`` separately so the geometry can be cached.
    """
    geom = compute_tile_geometry(
        frame_regions=frame_regions,
        z=z, x=x, y=y,
        tile_size=tile_size,
        smooth=smooth,
        snow=snow,
        nwp_chain=nwp_chain,
        enabled_regions=enabled_regions,
        frame_timestamp=frame_timestamp,
        nowcast_blend=nowcast_blend,
    )
    return present_tile(
        geom,
        color_scheme=color_scheme,
        fmt=fmt,
        arrow_style=arrow_style if (flow_regions or ecmwf_flow is not None) else "",
        flow_regions=flow_regions,
        frame_regions=frame_regions,
        enabled_regions=enabled_regions,
        ecmwf_flow=ecmwf_flow,
        ecmwf_grid=ecmwf_grid,
        frame_timestamp=frame_timestamp,
        z=z, x=x, y=y,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_blur_radius(
    region: RegionDef, z: int, x: int, y: int, tile_size: int
) -> float:
    """Pick a Gaussian blur radius matched to the visible region pixel size.

    Reads the local Jacobian of ``region_pixel_indices_fractional`` at the
    tile centre to find how many tile pixels a single region pixel covers
    (``tile_per_region``).  Blur radius scales as a quarter of that span,
    which is the σ that rounds a single region-pixel "block" at its
    edges without merging it with its neighbours (the visible Gaussian
    width is ~3σ, so a quarter-block σ touches half a block on each side).
    At low zoom the ratio is < 1 and the radius collapses to
    ``smooth_radius`` (baseline); at high zoom on a very coarse source
    growth is capped at ``tile_size / 32`` to keep the kernel from
    smearing unrelated cells together.
    """
    base = settings.smooth_radius
    if base <= 0:
        return 0.0
    row_f, col_f = region_pixel_indices_fractional(region, z, x, y, tile_size)
    cy = cx = tile_size // 2
    drow = abs(float(row_f[cy + 1, cx] - row_f[cy - 1, cx])) / 2.0
    dcol = abs(float(col_f[cy, cx + 1] - col_f[cy, cx - 1])) / 2.0
    if drow < 1e-6 or dcol < 1e-6:
        return base
    tile_per_region = max(1.0 / drow, 1.0 / dcol)
    raw = base * max(1.0, tile_per_region * 0.25)
    return min(raw, tile_size / 32.0)


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
        if smooth:
            values = _bilinear_sample(
                frame_data, region, z, x, y, tile_size, pad=pad,
            )
            oob = (row_idx == -1) | (col_idx == -1)
            values[oob] = 0
        else:
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

        if smooth:
            region_values = _bilinear_sample(
                data, region, z, x, y, tile_size, pad=pad,
            )
            oob = (row_idx == -1) | (col_idx == -1)
            region_values[oob] = 0
        else:
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
    nwp_chain,
    frame_timestamp: int | None = None,
    smooth: bool = False,
) -> np.ndarray:
    """Fill pixels outside radar coverage from NWP fallback.

    IEM N0Q and the Nordic / DWD composites all encode pixel value 0
    for both "outside radar range" *and* "clear sky within range", so
    we can't use ``values == 0`` alone — that would make NWP bleed
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
    # this tile we still don't want NWP overlapping with radar.
    covered = np.zeros(lat_grid.shape, dtype=bool)
    for region in regions:
        covered |= sample_coverage(region.name, lat_grid, lon_grid)

    uncovered = (values == 0) & ~covered
    if not uncovered.any():
        return values

    nwp_values = nwp_chain.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    result = values.copy()
    result[uncovered] = nwp_values[uncovered]
    return result


def _blend_nowcast(
    radar_values: np.ndarray,
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int, pad: int,
    nwp_chain,
    frame_timestamp: int | None = None,
    smooth: bool = False,
    blend_weight: float = 1.0,
) -> np.ndarray:
    """Blend extrapolated radar with NWP forecast for nowcast frames.

    Uses a combination of temporal and spatial weighting:

    - **Temporal** (``blend_weight``): 1.0 at T+10 min (trust radar),
      fading to 0.0 at the last nowcast step (trust NWP).
    - **Spatial** (feather mask): 1.0 deep inside radar coverage, fading
      to 0.0 at coverage boundaries to prevent hard seams.

    The effective per-pixel radar weight is ``blend_weight × feather``.
    Outside radar coverage, NWP is used directly (same as past frames).
    """
    if pad > 0:
        lat_grid, lon_grid = tile_pixel_latlons_padded(z, x, y, tile_size, pad)
    else:
        lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)

    # Sample NWP for ALL pixels (not just uncovered)
    model_values = nwp_chain.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    # Soften the model values before blending to reduce spatial mismatch
    # artifacts where radar and the model disagree on storm position.
    # Tuned for HRRR's 3 km native resolution: storm positions are within
    # ~1-2 cells of radar, so a small kernel is enough.  Outside HRRR's
    # CONUS domain the chain falls back to IFS at 9 km, where this
    # under-blurs slightly — but the feather already handles the spatial
    # transition between sources, and over-blurring kills HRRR's sharpness
    # everywhere else, which is the worse trade-off.
    model_f = model_values.astype(np.float32)
    ksize = 3 if tile_size <= 256 else 5
    model_f = cv2.GaussianBlur(model_f, (ksize, ksize), 0)

    # Build the spatial feather weight: union across all overlapping regions
    feather = np.zeros(lat_grid.shape, dtype=np.float32)
    for region in regions:
        feather = np.maximum(feather, sample_feather(region.name, lat_grid, lon_grid))

    # Per-pixel effective radar weight
    effective_w = blend_weight * feather

    # Blend: extrapolated radar × weight + model × (1 − weight)
    radar_f = radar_values.astype(np.float32)
    blended = effective_w * radar_f + (1.0 - effective_w) * model_f

    # Don't hallucinate precipitation where neither source has any
    both_zero = (radar_values == 0) & (model_values == 0)
    result = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    result[both_zero] = 0

    return result


def _bilinear_sample(
    frame_data: np.ndarray, region: RegionDef,
    z: int, x: int, y: int, tile_size: int,
    pad: int = 0,
) -> np.ndarray:
    """Sample frame data using bilinear interpolation for smooth rendering."""
    if pad > 0:
        row_f, col_f = region_pixel_indices_fractional_padded(
            region, z, x, y, tile_size, pad
        )
    else:
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
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT, GRID_WIDTH, NORTH, PIXEL_SIZE, WEST
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
