# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Lat/lon-centered window stitching.

Assembles an arbitrary ``extent`` x ``extent`` canvas (expressed in
global pixel coordinates, centered on a lat/lon point) from standard
integer-tile ``TileGeometry`` records, plus a synchronous RGBA variant
for coverage tiles.  Self-contained except for ``TileGeometry``.
"""
import asyncio
from collections import namedtuple
from collections.abc import Awaitable, Callable, Mapping

import numpy as np

from .renderer import TileGeometry

# One span = one rectangle of source tile copied onto the window canvas.
# tx,ty = tile indices (tx already mod-wrapped); win_x,win_y = offset in
# the window canvas; tile_x,tile_y = offset within the source tile's core
# rect; w,h = copy run length.
TileSpan = namedtuple("TileSpan", "tx ty win_x win_y tile_x tile_y w h")


def _x_runs(
    px0: int, extent: int, tile_size: int, n_tiles: int,
) -> list[tuple[int, int, int, int]]:
    """Run-length decomposition of the wrapped X axis.

    Returns (tx, win_x, tile_x, w) tuples: wrapped tile column, window
    x-offset, x-offset within the source tile's core rect, and copy
    width.  ``px0`` may be any int; ``extent`` never exceeds the world
    width by caller contract.
    """
    runs: list[tuple[int, int, int, int]] = []
    win_x = 0
    col = px0 // tile_size
    tile_x = px0 % tile_size
    while win_x < extent:
        w = min(extent - win_x, tile_size - tile_x)
        runs.append((col % n_tiles, win_x, tile_x, w))
        win_x += w
        col += 1
        tile_x = 0
    return runs


def _y_runs(
    py0: int, extent: int, tile_size: int, world_rows: int,
) -> list[tuple[int, int, int, int]]:
    """Run-length decomposition of the (clamped) Y axis.

    Returns (ty, win_y, tile_y, h) tuples: tile row, window y-offset,
    y-offset within the source tile's core rect, and copy height.  Rows
    outside [0, world_rows) are defensively clipped rather than raising.
    """
    row0 = max(py0, 0)
    row1 = min(py0 + extent, world_rows)
    if row0 >= row1:
        return []
    runs: list[tuple[int, int, int, int]] = []
    win_y = row0 - py0
    ty = row0 // tile_size
    tile_y = row0 % tile_size
    remaining = row1 - row0
    while remaining > 0:
        h = min(remaining, tile_size - tile_y)
        runs.append((ty, win_y, tile_y, h))
        remaining -= h
        win_y += h
        ty += 1
        tile_y = 0
    return runs


def covered_tiles(
    z: int, px0: int, py0: int, extent: int, tile_size: int,
) -> list[TileSpan]:
    """Spans of integer tiles covering the global-pixel rect [px0, px0+extent) x [py0, py0+extent).
    world = (2**z) * tile_size. X WRAPS: tx = ((px0 + i) // tile_size) % (2**z) (px0 may be any int;
    the rect never exceeds world in width because extent <= world by caller contract).
    Y is pre-clamped by callers: 0 <= py0 and py0 + extent <= world; defensively clip rows outside
    [0, world) rather than raising.
    Each TileSpan: tx,ty = tile indices (tx already mod-wrapped); win_x,win_y = offset in the window
    canvas; tile_x,tile_y = offset within the source tile's core rect; w,h = copy run length.
    """
    world = (2 ** z) * tile_size
    n_tiles = 2 ** z
    spans: list[TileSpan] = []
    for ty, win_y, tile_y, h in _y_runs(py0, extent, tile_size, world):
        for tx, win_x, tile_x, w in _x_runs(px0, extent, tile_size, n_tiles):
            spans.append(TileSpan(tx, ty, win_x, win_y, tile_x, tile_y, w, h))
    return spans


def stitch_geometries(
    components: Mapping[tuple[int, int], TileGeometry],
    z: int,
    px0: int,
    py0: int,
    extent: int,
    tile_size: int,
) -> TileGeometry:
    """Assemble an extent x extent canvas from component tile geometries.
    - canvas = np.zeros((extent, extent), np.uint8).
    - For each span: geom = components.get((tx, ty)); missing or geom.is_transparent -> leave zeros.
    - Otherwise core = geom.values[geom.pad : geom.pad + tile_size, geom.pad : geom.pad + tile_size]
      (component tile_size MUST equal the tile_size arg; raise ValueError otherwise) and paste
      core[tile_y:tile_y+h, tile_x:tile_x+w] into canvas[win_y:win_y+h, win_x:win_x+w].
    - snow_mask: allocate a zeros(bool) canvas LAZILY on the first component whose snow_mask is not
      None; paste mask cores the same way; missing/transparent/None-mask components contribute False.
      Final snow_mask is None iff every present non-transparent component had None.
    - blur_radius = max(geom.blur_radius over present non-transparent components, default 0.0).
    - If NO present non-transparent components: return TileGeometry.transparent(extent).
    - Else return TileGeometry(values=canvas, snow_mask=..., tile_size=extent, pad=0, blur_radius=max_r).
    """
    canvas = np.zeros((extent, extent), dtype=np.uint8)
    snow_canvas: np.ndarray | None = None
    max_r = 0.0
    present = False
    for span in covered_tiles(z, px0, py0, extent, tile_size):
        geom = components.get((span.tx, span.ty))
        if geom is None or geom.is_transparent:
            continue
        if geom.tile_size != tile_size:
            raise ValueError(
                f"component tile ({span.tx}, {span.ty}) has tile_size "
                f"{geom.tile_size}, expected {tile_size}"
            )
        present = True
        if geom.blur_radius > max_r:
            max_r = geom.blur_radius
        core = geom.values[
            geom.pad:geom.pad + tile_size,
            geom.pad:geom.pad + tile_size,
        ]
        canvas[span.win_y:span.win_y + span.h, span.win_x:span.win_x + span.w] = (
            core[span.tile_y:span.tile_y + span.h, span.tile_x:span.tile_x + span.w]
        )
        if geom.snow_mask is not None:
            if snow_canvas is None:
                snow_canvas = np.zeros((extent, extent), dtype=bool)
            mask_core = geom.snow_mask[
                geom.pad:geom.pad + tile_size,
                geom.pad:geom.pad + tile_size,
            ]
            snow_canvas[span.win_y:span.win_y + span.h, span.win_x:span.win_x + span.w] = (
                mask_core[
                    span.tile_y:span.tile_y + span.h,
                    span.tile_x:span.tile_x + span.w,
                ]
            )
    if not present:
        return TileGeometry.transparent(extent)
    return TileGeometry(
        values=canvas,
        snow_mask=snow_canvas,
        tile_size=extent,
        pad=0,
        blur_radius=max_r,
    )


async def compute_window_geometry(
    get_component: Callable[[int, int], Awaitable[TileGeometry]],
    z: int,
    px0: int,
    py0: int,
    tile_size: int,
) -> TileGeometry:
    """Two-phase orchestrator. get_component is an async callable (tx, ty) -> TileGeometry.
    world = (2**z) * tile_size.
    Phase 1: spans = covered_tiles(z, px0, py0, tile_size, tile_size); fetch all concurrently via
    asyncio.gather; if every component is transparent -> return TileGeometry.transparent(tile_size).
    max_r = max blur_radius; pad = int(max_r * 3)   (mirrors renderer.py:193 exactly)
    Phase 2 (only when max_r >= 0.5 and pad > 0 AND tile_size + 2*pad <= world):
      extent = tile_size + 2*pad; epx0 = (px0 - pad) % world;
      epy0 = min(max(py0 - pad, 0), world - extent)
      fetch any not-yet-fetched expanded-span components concurrently; stitch the expanded canvas;
      return TileGeometry(values=expanded.values, snow_mask=expanded.snow_mask, tile_size=tile_size,
                          pad=pad, blur_radius=max_r)
    Otherwise: stitch the base window and return TileGeometry(values, snow_mask, tile_size=tile_size,
    pad=0, blur_radius=max_r).
    """
    world = (2 ** z) * tile_size

    # Phase 1: base window (extent == tile_size).
    spans = covered_tiles(z, px0, py0, tile_size, tile_size)
    keys = list(dict.fromkeys((s.tx, s.ty) for s in spans))
    fetched = await asyncio.gather(
        *(get_component(tx, ty) for tx, ty in keys)
    )
    components: dict[tuple[int, int], TileGeometry] = dict(zip(keys, fetched))

    if all(g.is_transparent for g in components.values()):
        return TileGeometry.transparent(tile_size)

    max_r = max((g.blur_radius for g in components.values()), default=0.0)
    pad = int(max_r * 3)

    if max_r >= 0.5 and pad > 0 and tile_size + 2 * pad <= world:
        extent = tile_size + 2 * pad
        epx0 = (px0 - pad) % world
        epy0 = min(max(py0 - pad, 0), world - extent)
        extra_keys: list[tuple[int, int]] = []
        seen = set(components)
        for span in covered_tiles(z, epx0, epy0, extent, tile_size):
            key = (span.tx, span.ty)
            if key not in seen:
                seen.add(key)
                extra_keys.append(key)
        if extra_keys:
            extra = await asyncio.gather(
                *(get_component(tx, ty) for tx, ty in extra_keys)
            )
            components.update(zip(extra_keys, extra))
        expanded = stitch_geometries(components, z, epx0, epy0, extent, tile_size)
        return TileGeometry(
            values=expanded.values,
            snow_mask=expanded.snow_mask,
            tile_size=tile_size,
            pad=pad,
            blur_radius=max_r,
        )

    stitched = stitch_geometries(components, z, px0, py0, tile_size, tile_size)
    return TileGeometry(
        values=stitched.values,
        snow_mask=stitched.snow_mask,
        tile_size=tile_size,
        pad=0,
        blur_radius=max_r,
    )


def stitch_coverage(
    components: Mapping[tuple[int, int], np.ndarray | None],
    z: int,
    px0: int,
    py0: int,
    tile_size: int,
) -> np.ndarray:
    """Synchronous RGBA variant (coverage has no blur/pad/snow): canvas zeros((tile_size, tile_size, 4),
    uint8); paste each present (non-None) component's full (tile_size, tile_size, 4) array per span;
    return the canvas.
    """
    canvas = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    for span in covered_tiles(z, px0, py0, tile_size, tile_size):
        comp = components.get((span.tx, span.ty))
        if comp is None:
            continue
        canvas[span.win_y:span.win_y + span.h, span.win_x:span.win_x + span.w] = comp[
            span.tile_y:span.tile_y + span.h,
            span.tile_x:span.tile_x + span.w,
        ]
    return canvas
