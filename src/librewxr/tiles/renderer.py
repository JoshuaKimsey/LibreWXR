# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    tile_bounds,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
)
from librewxr.tiles.png_palette import encode_png

if TYPE_CHECKING:
    from librewxr.data.precip_mask import PrecipMaskStore


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
    # Fast-path outcome label, populated ONLY on transparent geometries so
    # routes.py can classify which empty-tile fast path fired without
    # reaching into compute internals.  ``None`` for real (non-transparent)
    # geometries.  String values are the Tier 1 / Tier 2 (precip-mask)
    # vocabulary in the transparent return sites of
    # ``compute_tile_geometry`` / ``_compute_nwp_only_geometry``.
    fast_path: str | None = None

    @classmethod
    def transparent(cls, tile_size: int, fast_path: str | None = None) -> "TileGeometry":
        """Sentinel for tiles with neither radar nor NWP coverage."""
        return cls(
            values=np.empty((0, 0), dtype=np.uint8),
            snow_mask=None,
            tile_size=tile_size,
            pad=0,
            blur_radius=0.0,
            is_transparent=True,
            fast_path=fast_path,
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
    precip_mask=None,  # PrecipMaskStore | None — multi-mode only
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
                precip_mask,
            )
        return TileGeometry.transparent(tile_size, fast_path="no_regions_no_nwp")

    # Tier 2: pre-sample global precip-mask gate (multi-mode only).  The
    # mask ORs radar regions + all NWP source samples + nowcast regions
    # into one coarse boolean grid per timestamp, so the gate fires for
    # the past-radar path AND the nowcast path together — Tier 3 (the
    # nowcast bbox) is folded in.  Single mode has no mask
    # (``precip_mask is None``) and falls through to the existing Tier 1 /
    # Case A paths unchanged.  Hoisted ahead of the ``_sample_region``
    # calls so clear-sky tiles bail in O(1): the mask includes the radar
    # contribution, so no precip in the bbox guarantees the radar sample
    # is empty and the pre-hoist ``radar_empty`` term was always true.
    if has_nwp and precip_mask is not None:
        if not precip_mask.has_precip_in_bbox(frame_timestamp, tile_bounds(z, x, y)):
            label = "tier2_mask_nowcast" if nowcast_blend is not None else "tier2_mask_past"
            return TileGeometry.transparent(tile_size, fast_path=label)

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

    # Compute the noise-floor threshold ONCE here for the radar-empty
    # predicate.  IMPORTANT: this is NON-MUTATING — do not touch `values`
    # (it may be a view into the memmap frame).  The existing post-NWP
    # noise-floor block below still runs unchanged and does the actual
    # zeroing (with its own copy).
    pixel_threshold = (
        int((settings.noise_floor_dbz + 32) * 2)
        if settings.noise_floor_dbz > -32 else 0
    )
    radar_empty = not (values >= pixel_threshold).any()

    # Case A: no NWP at all and radar sample is empty — transparent,
    # no further work.
    if not has_nwp and radar_empty:
        return TileGeometry.transparent(tile_size, fast_path="case_a_no_nwp_empty_radar")

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

    # Tier 1: post-NWP-fill empty check.  If after fill/blend + noise
    # floor the tile is all-zero (NWP also sampled empty, or nowcast
    # blend produced empty), return transparent BEFORE the snow-mask
    # step (no precip = nothing to phase-classify).
    if not values.any():
        return TileGeometry.transparent(
            tile_size,
            fast_path=(
                "tier1_post_blend" if nowcast_blend is not None else "tier1_post_fill"
            ),
        )

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
    precip_mask=None,  # PrecipMaskStore | None — multi-mode only
) -> TileGeometry:
    """Geometry for a tile entirely from NWP (no radar regions overlap)."""
    # Tier 2: skip the full-grid NWP sample entirely when the global
    # precip mask has no precip in this tile's bbox.  Same mask gate as
    # the past-radar / nowcast paths (multi-mode only; skipped when the
    # mask is absent).
    if precip_mask is not None and not precip_mask.has_precip_in_bbox(
        frame_timestamp, tile_bounds(z, x, y),
    ):
        return TileGeometry.transparent(tile_size, fast_path="tier2_mask_nwp_only")

    lat_grid, lon_grid = tile_pixel_latlons(z, x, y, tile_size)
    values = nwp_chain.sample(
        lat_grid, lon_grid, frame_timestamp, bilinear=smooth,
    )

    if settings.noise_floor_dbz > -32:
        pixel_threshold = int((settings.noise_floor_dbz + 32) * 2)
        values = values.copy()
        values[values < pixel_threshold] = 0

    # Tier 1: post-sample empty check — all-zero after the noise floor
    # means nothing to render, so bail before the snow-mask step.
    if not values.any():
        return TileGeometry.transparent(tile_size, fast_path="tier1_nwp_only_post_sample")

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
    nwp_flow: np.ndarray | None = None,
    nwp_chain=None,
    frame_timestamp: int | None = None,
    z: int = 0,
    x: int = 0,
    y: int = 0,
    cell_style: str = "",
    cells_by_region: dict[str, np.ndarray] | None = None,
    cell_counts: dict[str, int] | None = None,
) -> bytes:
    """Render a cached ``TileGeometry`` to encoded bytes.

    Does the cheap tail: LUT colorize, optional Gaussian blur + crop,
    optional motion-arrow overlay, image encode.  All of these are
    per-request because they depend on the request's color/format/arrow
    parameters, which are deliberately *not* in the cache key.

    Arrow inputs (``flow_regions``, ``nwp_flow``, ``nwp_chain``) are
    passed fresh on each call rather than baked into the geometry so a
    tile that's cached without arrows can still render an arrow variant
    when a later request asks for one.  ``nwp_flow`` is the single
    composite NWP optical-flow raster (one global field, see
    ``NowcastGenerator._compute_nwp_flow_sync``); ``nwp_chain`` is the
    dispatch chain used to gate arrow presence on the chain's own
    precip at the point.
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

    if arrow_style and (flow_regions or nwp_flow is not None):
        regions = overlapping_regions(z, x, y, enabled_regions)
        if frame_regions:
            regions_with_data = [r for r in regions if r.name in frame_regions]
        else:
            regions_with_data = []
        img = _draw_motion_arrows(
            img, flow_regions, frame_regions or {}, regions_with_data,
            z, x, y, geom.tile_size, arrow_style,
            nwp_flow=nwp_flow,
            nwp_chain=nwp_chain,
            frame_timestamp=frame_timestamp,
            # The geometry is still padded when blur is applied; the arrow
            # overlay works in cropped tile coordinates, so slice the pad
            # border off before handing it to the presence gate.
            geom_values=(
                geom.values
                if geom.pad == 0
                else geom.values[
                    geom.pad:geom.pad + geom.tile_size,
                    geom.pad:geom.pad + geom.tile_size,
                ]
            ),
        )

    if cell_style and cells_by_region:
        regions = overlapping_regions(z, x, y, enabled_regions)
        if frame_regions:
            regions_with_data = [r for r in regions if r.name in frame_regions]
        else:
            regions_with_data = []
        img = _draw_storm_cells(
            img, cells_by_region, cell_counts or {}, regions_with_data,
            z, x, y, geom.tile_size, cell_style,
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
    nwp_chain=None,
    enabled_regions: list[str] | None = None,
    frame_timestamp: int | None = None,
    nowcast_blend: float | None = None,
    flow_regions: dict[str, np.ndarray] | None = None,
    nwp_flow: np.ndarray | None = None,
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
        arrow_style=arrow_style if (flow_regions or nwp_flow is not None) else "",
        flow_regions=flow_regions,
        frame_regions=frame_regions,
        enabled_regions=enabled_regions,
        nwp_flow=nwp_flow,
        nwp_chain=nwp_chain,
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


def _gather_clipped(
    frame_data: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray
) -> np.ndarray:
    """Fancy-index ``frame_data`` with every out-of-bounds index mapped to 0.

    Replaces the old ``np.pad(frame_data, ((0, 1), (0, 1)))`` trick, which
    copied the entire (tens-of-MB) region grid per tile just to turn the
    ``-1`` sentinels in the index arrays into the zero padding row/col.
    A clipped gather touches only the tile-sized index arrays: indices
    inside the grid are read normally, and any index that is negative OR
    past the grid edge is zeroed afterwards — byte-identical output, no
    full-frame copy.
    """
    valid = (
        (row_idx >= 0)
        & (col_idx >= 0)
        & (row_idx < frame_data.shape[0])
        & (col_idx < frame_data.shape[1])
    )
    row_c = np.clip(row_idx, 0, frame_data.shape[0] - 1)
    col_c = np.clip(col_idx, 0, frame_data.shape[1] - 1)
    values = frame_data[row_c, col_c]
    values[~valid] = 0
    return values


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
            values = _gather_clipped(frame_data, row_idx, col_idx)
    else:
        row_idx, col_idx = region_pixel_indices(region, z, x, y, tile_size)
        if smooth:
            values = _bilinear_sample(frame_data, region, z, x, y, tile_size)
            oob = (row_idx == -1) | (col_idx == -1)
            values[oob] = 0
        else:
            values = _gather_clipped(frame_data, row_idx, col_idx)
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
            region_values = _gather_clipped(data, row_idx, col_idx)

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
        region_values = _gather_clipped(data, row_idx, col_idx)
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

    # Fractional offsets must stay float32: the four corner values are
    # float32, and without the cast the int32 subtract would promote the
    # whole interpolation to float64.  The final clip + 0.5 -> uint8
    # rounding is unchanged.
    dr = (row_f - r0).astype(np.float32)
    dc = (col_f - c0).astype(np.float32)

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


def _sample_flow_at(
    flow: np.ndarray, row: float, col: float,
    region_h: int, region_w: int,
) -> tuple[float, float]:
    """Sample a per-region flow field at a full-res region pixel.

    Radar flows are stored at the resolution they were computed at
    (longest dim ≤ 1000 px, see ``nowcast._compute_flow_low``), not
    upscaled to the region grid.  Full-res coordinates are mapped into
    the stored grid with the same center mapping ``cv2.resize`` uses
    when it upscales a low-res field (``(p + 0.5) * (small / full) -
    0.5``) and the nearest stored pixel is sampled — equivalent to
    sampling the upscaled field at the same point without materialising
    a full-res copy.  The vector values themselves are already in
    full-res pixel units, so no magnitude scaling is applied.  When the
    field is full-res this degenerates to the legacy
    ``int(row) / int(col)`` sampling.
    """
    fh, fw = flow.shape[0], flow.shape[1]
    if fh == region_h and fw == region_w:
        rf = int(row)
        cf = int(col)
    else:
        rf = int(round((row + 0.5) * fh / region_h - 0.5))
        cf = int(round((col + 0.5) * fw / region_w - 0.5))
    rf = min(max(rf, 0), fh - 1)
    cf = min(max(cf, 0), fw - 1)
    return float(flow[rf, cf, 0]), float(flow[rf, cf, 1])


def _draw_motion_arrows(
    img: Image.Image,
    flow_regions: dict[str, np.ndarray] | None,
    frame_regions: dict[str, np.ndarray],
    regions: list[RegionDef],
    z: int, x: int, y: int,
    tile_size: int,
    style: str = "light",
    nwp_flow: np.ndarray | None = None,
    nwp_chain=None,
    frame_timestamp: int | None = None,
    geom_values: np.ndarray | None = None,
) -> Image.Image:
    """Draw precipitation motion vector arrows on the tile.

    Overlays semi-transparent arrows on areas with active precipitation,
    showing storm movement direction and relative speed. Arrows are
    derived from the optical flow field computed between the two most
    recent radar frames, with a single composite NWP optical-flow raster
    as the global fallback outside radar coverage — reflecting whichever
    regional NWP source is active at each point (HRRR over CONUS,
    ICON-EU over Europe, JMA MSM over Japan, IFS elsewhere) rather than
    IFS alone.

    ``style`` selects the arrow colour: ``"light"`` for white arrows
    (best on dark maps) or ``"dark"`` for dark arrows (best on light maps).

    ``geom_values`` is the tile geometry's post-fill uint8 values, already
    cropped to the tile (pad border sliced off by the caller); it
    replaces a full-grid ``nwp_chain.sample`` that was previously needed
    purely to gate the composite-NWP arrow presence on precip.
    """
    # Regions that have both frame data and flow data
    if flow_regions:
        valid_regions = [
            r for r in regions
            if r.name in flow_regions and r.name in frame_regions
        ]
    else:
        valid_regions = []

    has_nwp = (
        nwp_flow is not None
        and nwp_chain is not None
    )

    if not valid_regions and not has_nwp:
        return img

    # Precompute pixel-index arrays for each valid radar region
    region_info = []
    for r in valid_regions:
        row_f, col_f = region_pixel_indices_fractional(r, z, x, y, tile_size)
        row_i, col_i = region_pixel_indices(r, z, x, y, tile_size)
        region_info.append((r, row_f, col_f, row_i, col_i))

    # Precompute lat/lon grid for composite NWP flow fallback (only if needed)
    nwp_latlons = None
    radar_coverage = None
    if has_nwp:
        from librewxr.data.nowcast import (
            NWP_FLOW_NORTH, NWP_FLOW_SOUTH, NWP_FLOW_WEST,
        )
        nwp_latlons = tile_pixel_latlons(z, x, y, tile_size)
        # Presence gate: the tile geometry's post-fill values at the
        # point, above the noise floor.  The geometry already carries the
        # NWP fill (see compute_tile_geometry), so we don't re-sample the
        # chain over the full grid just to check for precip.  This keeps
        # the "HRRR precip present but IFS dry" fix where the old
        # IFS-hardcoded path suppressed arrows.
        # Derive raster pixel size from the loaded array's shape so a
        # resolution-knob change between cycles can't drift out of sync.
        nwp_res = (NWP_FLOW_NORTH - NWP_FLOW_SOUTH) / (nwp_flow.shape[0] - 1)
        # Precompute radar coverage so we can distinguish "clear sky
        # under radar" from "outside radar coverage" when deciding
        # whether to fall through to the composite NWP arrows.  Per the
        # user's chosen behavior, radar coverage suppresses NWP arrows
        # even when the radar flow is zero/stationary.
        if region_info:
            lat_grid, lon_grid = nwp_latlons
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
                    # fall through to the composite NWP arrows.
                    if radar_coverage is None or radar_coverage[ty, tx]:
                        found = True
                        break
                    continue

                flow = flow_regions[r.name]
                # Flow is stored at reduced resolution (≤ 1000 px target
                # dim); sample it at the tile pixel's full-res region
                # coordinates via the resize center mapping.
                fx, fy = _sample_flow_at(
                    flow, row_f[ty, tx], col_f[ty, tx],
                    r.height, r.width,
                )

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

            # Composite NWP flow fallback: only if no radar region claimed
            # this pixel (either no radar data here, or the radar frame
            # says "dry" outside coverage).
            if not found and has_nwp:
                if geom_values is None or geom_values[ty, tx] < noise_threshold:
                    continue  # Below noise floor — not visible on tile

                lat = float(nwp_latlons[0][ty, tx])
                lon = float(nwp_latlons[1][ty, tx])

                # Convert lat/lon to composite raster indices
                nr = (NWP_FLOW_NORTH - lat) / nwp_res
                nc = (lon - NWP_FLOW_WEST) / nwp_res
                nri = min(max(int(nr), 0), nwp_flow.shape[0] - 1)
                nci = min(max(int(nc), 0), nwp_flow.shape[1] - 1)

                fx = float(nwp_flow[nri, nci, 0])
                fy = float(nwp_flow[nri, nci, 1])

                # Local scale: composite raster pixels per tile pixel.
                # Use lat/lon difference to compute the Jacobian.
                tx1 = min(tx + 1, tile_size - 1)
                ty1 = min(ty + 1, tile_size - 1)
                tx0 = max(tx - 1, 0)
                ty0 = max(ty - 1, 0)

                dlat_dy = (nwp_latlons[0][ty1, tx] - nwp_latlons[0][ty0, tx]) / (ty1 - ty0)
                dlon_dx = (nwp_latlons[1][ty, tx1] - nwp_latlons[1][ty, tx0]) / (tx1 - tx0)

                # Convert degrees to composite raster pixels
                drow_dy = -dlat_dy / nwp_res  # negative: lat decreases as row increases
                dcol_dx = dlon_dx / nwp_res

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


def _draw_storm_cells(
    img: Image.Image,
    cells_by_region: dict[str, np.ndarray],
    cell_counts: dict[str, int],
    valid_regions: list,
    z: int,
    x: int,
    y: int,
    tile_size: int,
    style: str,
) -> Image.Image:
    """Draw storm-cell markers (filled circles) + motion arrows on the tile.

    For each detected cell whose centroid falls within the tile's coverage
    of its region, draws a filled circle sized by area_km2 and (when motion
    data is available) a motion-vector arrow from the optical flow at the
    cell's centroid.  The centroid -> tile-pixel mapping uses a
    nearest-neighbor search on the precomputed ``region_pixel_indices_fractional``
    grid, which handles all projections (latlon, laea, tmerc) without
    needing an inverse projection.
    """
    if not cells_by_region or not valid_regions:
        return img

    cell_color = (40, 40, 40, 200) if style == "dark" else (255, 255, 255, 200)
    arrow_color = (40, 40, 40, 230) if style == "dark" else (255, 255, 255, 230)
    line_w = 3 if tile_size <= 256 else 4

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for r in valid_regions:
        cells = cells_by_region.get(r.name)
        if cells is None or len(cells) == 0:
            continue
        count = cell_counts.get(r.name, 0)
        if count == 0:
            continue

        # Precompute the region-pixel -> tile-pixel mapping for this region+tile.
        # row_f[ty, tx] = fractional region-row for tile-pixel (ty, tx).
        # row_i[ty, tx] = integer region-row, or -1 if outside the region.
        row_f, col_f = region_pixel_indices_fractional(r, z, x, y, tile_size)
        row_i, col_i = region_pixel_indices(r, z, x, y, tile_size)
        valid_mask = row_i >= 0
        if not valid_mask.any():
            continue

        # Precompute the valid region-pixel range for a quick cell-in-tile check.
        valid_rows = row_f[valid_mask]
        valid_cols = col_f[valid_mask]
        row_min, row_max = float(valid_rows.min()), float(valid_rows.max())
        col_min, col_max = float(valid_cols.min()), float(valid_cols.max())

        for i in range(count):
            cell = cells[i]
            cr = float(cell["centroid_row"])
            cc = float(cell["centroid_col"])

            # Quick bounds check: is the cell within the tile's region coverage?
            # The +-2 padding accounts for sub-pixel rounding at the tile edges.
            if not (row_min - 2 <= cr <= row_max + 2 and col_min - 2 <= cc <= col_max + 2):
                continue

            # Nearest-neighbor: find the tile pixel whose region-pixel coords
            # are closest to the cell's centroid.  This is projection-agnostic
            # because row_f/col_f already encode the forward projection.
            d2 = np.where(valid_mask, (row_f - cr) ** 2 + (col_f - cc) ** 2, np.inf)
            flat_idx = int(d2.argmin())
            ty, tx = divmod(flat_idx, tile_size)

            # Circle radius scaled by area (log scale -- area spans orders of
            # magnitude from ~25 km^2 single cells to ~10000 km^2 MCSs).
            area = float(cell["area_km2"])
            radius = max(3.0, min(12.0, 3.0 + math.log10(max(area, 1.0)) * 3.0))

            # Draw filled circle at the centroid.
            draw.ellipse(
                [(tx - radius, ty - radius), (tx + radius, ty + radius)],
                fill=cell_color,
                outline=cell_color,
            )

            # Draw motion arrow if speed is available and non-zero.
            speed = float(cell["motion_speed_kmh"])
            if not math.isnan(speed) and speed > 0:
                dx = float(cell["motion_dx_px"])
                dy = float(cell["motion_dy_px"])
                raw_len = math.hypot(dx, dy)
                if raw_len > 1e-9:
                    speed_scale = 6.0
                    min_len = 8.0
                    max_len = 28.0
                    target_len = min(max(raw_len * speed_scale, min_len), max_len)
                    adx = dx / raw_len * target_len
                    ady = dy / raw_len * target_len

                    # Arrow biased toward the tip (60% forward) -- same as
                    # _draw_motion_arrows.
                    x0 = tx - adx * 0.4
                    y0 = ty - ady * 0.4
                    x1 = tx + adx * 0.6
                    y1 = ty + ady * 0.6
                    draw.line([(x0, y0), (x1, y1)], fill=arrow_color, width=line_w)
                    angle = math.atan2(ady, adx)
                    head_len = max(4.0, min(8.0, math.hypot(adx, ady) * 0.35))
                    ha = 0.45
                    draw.polygon([
                        (x1, y1),
                        (x1 - head_len * math.cos(angle - ha),
                         y1 - head_len * math.sin(angle - ha)),
                        (x1 - head_len * math.cos(angle + ha),
                         y1 - head_len * math.sin(angle + ha)),
                    ], fill=arrow_color)

    return Image.alpha_composite(img, overlay)


# Cached fully-transparent tile bytes.  Deterministic per
# (tile_size, fmt, webp_quality) and populated lazily on first use; the
# worst-case race (two threads encoding the same constant once) is
# harmless, so no lock is needed.
_TRANSPARENT_TILE_BYTES: dict[tuple[int, str, int], bytes] = {}


def _transparent_tile(tile_size: int, fmt: str) -> bytes:
    """Return a fully transparent tile (cached constant bytes)."""
    key = (tile_size, fmt, settings.webp_quality)
    cached = _TRANSPARENT_TILE_BYTES.get(key)
    if cached is None:
        img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        cached = _encode_image(img, fmt)
        _TRANSPARENT_TILE_BYTES[key] = cached
    return cached


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    """Encode a PIL image to bytes."""
    if fmt == "webp":
        buf = io.BytesIO()
        q = settings.webp_quality
        if q >= 100:
            img.save(buf, format="WEBP", lossless=True)
        else:
            img.save(buf, format="WEBP", quality=q)
        return buf.getvalue()
    # PNG: adaptive lossless — exact 8-bit palette when the tile has few
    # enough unique colors, otherwise plain 32-bit RGBA (see png_palette).
    return encode_png(img)
