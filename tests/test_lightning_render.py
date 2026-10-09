# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the _draw_lightning tile overlay helper.

Verifies that GOES GLM flash points are forward-projected into tile-local
pixels and drawn as either filled energy-scaled dots or the hand-authored
vector bolt polygon (never a text glyph), with the presentational
strongest-by-energy draw cap, and that invalid / off-tile points are
skipped without error.  Mirrors tests/test_storm_cell_render.py.
"""

import logging
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from librewxr.config import settings
from librewxr.tiles.renderer import TileGeometry, _draw_lightning, present_tile

pytestmark = pytest.mark.lightning

_POINT_DTYPE = np.dtype([
    ("time_s", "int64"),
    ("lat", "float32"),
    ("lon", "float32"),
    ("energy", "float32"),
    ("satellite", "int8"),
])

# Marker name mirrors renderer.logger's __name__ so caplog can target it.
_RENDERER_LOGGER = "librewxr.tiles.renderer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_points(coords) -> np.ndarray:
    """Build a structured flash-point array from (lat, lon, energy) tuples."""
    arr = np.zeros(len(coords), dtype=_POINT_DTYPE)
    for i, (lat, lon, energy) in enumerate(coords):
        arr["lat"][i] = lat
        arr["lon"][i] = lon
        arr["energy"][i] = energy
        arr["time_s"][i] = 1_700_000_000 + i * 10
        arr["satellite"][i] = 19
    return arr


def _tile_center_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    """Inverse Web-Mercator of the tile center (the point that projects back
    to tile-local pixel (tile_size/2, tile_size/2))."""
    n = 1 << z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n))))
    return lat, lon


def _blank(size: int = 256) -> Image.Image:
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def _changed_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of pixels the overlay actually painted.

    The base image is fully transparent, so alpha_composite leaves alpha > 0
    exactly where the overlay drew.
    """
    return np.asarray(img)[..., 3] > 0


def _cluster_count(mask: np.ndarray) -> int:
    """Count connected changed-pixel regions (one per glyph)."""
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return int(n) - 1  # exclude the background component


def _opaque_geometry(tile_size: int = 256) -> TileGeometry:
    """A non-transparent geometry so the present tail exercises the overlay."""
    values = np.full((tile_size, tile_size), 5, dtype=np.uint8)
    return TileGeometry(
        values=values,
        snow_mask=None,
        tile_size=tile_size,
        pad=0,
        blur_radius=0.0,
    )


# ---------------------------------------------------------------------------
# Case 1: no points / empty style / unknown style -> unchanged
# ---------------------------------------------------------------------------

def test_no_points_or_empty_style_unchanged():
    img = _blank()
    base = list(img.getdata())

    assert list(_draw_lightning(img, None, 1, 1, 1, 256, "dots").getdata()) == base
    assert list(
        _draw_lightning(img, np.zeros(0, dtype=_POINT_DTYPE), 1, 1, 1, 256, "dots").getdata()
    ) == base

    pts = _make_points([(0.0, 90.0, 1e-12)])
    assert list(_draw_lightning(img, pts, 1, 1, 1, 256, "").getdata()) == base
    assert list(_draw_lightning(img, pts, 1, 1, 1, 256, "sparkles").getdata()) == base


def test_present_tile_defaults_byte_identical():
    """Existing callers that pass no lightning kwargs are unaffected."""
    transparent = TileGeometry.transparent(256)
    assert present_tile(transparent, 0, "png") == present_tile(
        transparent, 0, "png", lightning_style="", flash_points=None
    )

    pts = _make_points([(0.0, 90.0, 1e-12)])
    assert present_tile(transparent, 0, "png") == present_tile(
        transparent, 0, "png", lightning_style="dots", flash_points=pts
    )

    geom = _opaque_geometry()
    assert present_tile(geom, 0, "png") == present_tile(
        geom, 0, "png", lightning_style="", flash_points=None
    )


def test_present_tile_draws_on_real_geometry():
    """A real (non-transparent) geometry actually gets the overlay."""
    lat, lon = _tile_center_latlon(1, 1, 1)
    pts = _make_points([(lat, lon, 1e-12)])
    geom = _opaque_geometry()
    plain = present_tile(geom, 0, "png")
    overlaid = present_tile(geom, 0, "png", lightning_style="dots", flash_points=pts)
    assert overlaid != plain


# ---------------------------------------------------------------------------
# Case 2: dots -- center draws, far-outside is skipped
# ---------------------------------------------------------------------------

def test_dots_center_draws_and_outside_skipped():
    lat, lon = _tile_center_latlon(1, 1, 1)
    img = _blank()
    result = _draw_lightning(
        img, _make_points([(lat, lon, 1e-12)]), 1, 1, 1, 256, "dots"
    )
    assert list(result.getdata()) != list(img.getdata())

    # lon=-170 / lat=-80 maps far off tile z1/x1/y1 -> culled, unchanged.
    outside = _make_points([(-80.0, -170.0, 1e-12)])
    result_out = _draw_lightning(_blank(), outside, 1, 1, 1, 256, "dots")
    assert list(result_out.getdata()) == list(_blank().getdata())


# ---------------------------------------------------------------------------
# Case 3: bolts -- center draws, and differ from dots
# ---------------------------------------------------------------------------

def test_bolts_center_draws_and_differs_from_dots():
    lat, lon = _tile_center_latlon(1, 1, 1)
    pts = _make_points([(lat, lon, 1e-12)])
    dots = _draw_lightning(_blank(), pts, 1, 1, 1, 256, "dots")
    bolts = _draw_lightning(_blank(), pts, 1, 1, 1, 256, "bolts")

    assert list(bolts.getdata()) != list(_blank().getdata())
    assert list(dots.getdata()) != list(bolts.getdata())


# ---------------------------------------------------------------------------
# Case 4: projection sanity -- changed pixels cluster at the tile center
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["dots", "bolts"])
def test_projection_clusters_at_tile_center(style):
    lat, lon = _tile_center_latlon(1, 1, 1)
    result = _draw_lightning(
        _blank(), _make_points([(lat, lon, 4e-12)]), 1, 1, 1, 256, style
    )
    ys, xs = np.nonzero(_changed_mask(result))
    assert len(xs) > 0, "Expected the center flash to paint pixels"
    dist = np.hypot(xs - 128.0, ys - 128.0)
    assert dist.min() <= 8.0, "No changed pixel near the tile center"
    assert dist.max() <= 40.0, "Changed pixels leaked far from the tile center"


# ---------------------------------------------------------------------------
# Case 5: presentational draw cap keeps the strongest-by-energy glyphs
# ---------------------------------------------------------------------------

def test_draw_cap_limits_glyphs(monkeypatch, caplog):
    # Five well-separated points, all inside the z0 world tile.
    coords = [
        (0.0, 0.0, 1e-13),
        (0.0, 30.0, 2e-13),
        (0.0, -30.0, 3e-13),
        (0.0, 60.0, 4e-13),
        (0.0, -60.0, 5e-13),
    ]
    pts = _make_points(coords)

    monkeypatch.setattr(settings, "lightning_max_draw_per_tile", 1000)
    uncapped = _draw_lightning(_blank(), pts, 0, 0, 0, 256, "dots")
    uncapped_mask = _changed_mask(uncapped)
    assert _cluster_count(uncapped_mask) == 5

    monkeypatch.setattr(settings, "lightning_max_draw_per_tile", 2)
    with caplog.at_level(logging.DEBUG, logger=_RENDERER_LOGGER):
        capped = _draw_lightning(_blank(), pts, 0, 0, 0, 256, "dots")
    capped_mask = _changed_mask(capped)

    assert _cluster_count(capped_mask) == 2
    assert int(capped_mask.sum()) < int(uncapped_mask.sum())
    assert any(
        "lightning draw cap" in record.getMessage() for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Case 6: non-finite / invalid coordinates are skipped without error
# ---------------------------------------------------------------------------

def test_invalid_coordinates_skipped():
    pts = _make_points([
        (float("nan"), 0.0, 1e-12),   # non-finite lat
        (0.0, float("inf"), 1e-12),   # non-finite lon
        (89.0, 0.0, 1e-12),           # beyond the Mercator latitude clamp
        (0.0, 200.0, 1e-12),          # |lon| > 180
    ])
    img = _blank()
    result = _draw_lightning(img, pts, 1, 1, 1, 256, "dots")
    assert list(result.getdata()) == list(img.getdata())


def test_energy_scales_dot_radius():
    """Higher energy -> a strictly larger painted footprint (dots)."""
    lat, lon = _tile_center_latlon(0, 0, 0)
    low = _draw_lightning(
        _blank(), _make_points([(lat, lon, 1e-15)]), 0, 0, 0, 256, "dots"
    )
    high = _draw_lightning(
        _blank(), _make_points([(lat, lon, 4e-12)]), 0, 0, 0, 256, "dots"
    )
    assert int(_changed_mask(high).sum()) > int(_changed_mask(low).sum())
