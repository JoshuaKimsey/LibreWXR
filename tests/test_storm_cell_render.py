# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the _draw_storm_cells tile overlay helper.

Verifies that storm-cell centroids are rendered as filled circles
(sized by area_km2) on tile images, with optional motion-vector arrows,
using the nearest-neighbor mapping approach (argmin on
region_pixel_indices_fractional) that avoids any inverse projection.
"""

import math

import numpy as np
import pytest
from PIL import Image

from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.storm_cells import _CELL_DTYPE
from librewxr.tiles.coordinates import region_pixel_indices, region_pixel_indices_fractional
from librewxr.tiles.renderer import _draw_storm_cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_syn_region(name: str = "TEST_CELLS") -> RegionDef:
    """Build a synthetic latlon region covering +/-10 degrees at
    pixel_size=0.01, yielding a 2000x2000 pixel grid."""
    return RegionDef(
        name=name,
        west=-10.0, east=10.0,
        south=-10.0, north=10.0,
        pixel_size=0.01,
        group="TEST",
    )


def _make_cell(
    centroid_row: float = 1000.0,
    centroid_col: float = 1000.0,
    area_km2: float = 100.0,
    max_dbz: float = 48.0,
    motion_dx_px: float = 0.0,
    motion_dy_px: float = 0.0,
    motion_speed_kmh: float = float("nan"),
    motion_heading_deg: float = float("nan"),
) -> np.ndarray:
    """Build a single-cell structured array."""
    arr = np.zeros(1, dtype=_CELL_DTYPE)
    arr["centroid_row"] = centroid_row
    arr["centroid_col"] = centroid_col
    arr["area_px"] = area_km2 / 1.2321  # approximate px count
    arr["area_km2"] = area_km2
    arr["max_dbz"] = max_dbz
    arr["motion_dx_px"] = motion_dx_px
    arr["motion_dy_px"] = motion_dy_px
    arr["motion_speed_kmh"] = motion_speed_kmh
    arr["motion_heading_deg"] = motion_heading_deg
    return arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.storm_cells
def test_draw_storm_cells_empty():
    """No cells => image unchanged."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    original = img.copy()
    region = _make_syn_region()
    result = _draw_storm_cells(
        img, {}, {}, [region],
        0, 0, 0, 256, "light",
    )
    assert list(result.getdata()) == list(original.getdata())


@pytest.mark.storm_cells
def test_draw_storm_cells_basic():
    """One cell at the region center => circle drawn near tile pixel (128,128)
    at z=0 (the single world tile)."""
    region = _make_syn_region("TEST_CELLS_BASIC")
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    cell = _make_cell(centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0)
    result = _draw_storm_cells(
        img, {"TEST_CELLS_BASIC": cell}, {"TEST_CELLS_BASIC": 1}, [region],
        0, 0, 0, 256, "light",
    )
    # Verify the image changed (overlay was drawn)
    assert list(result.getdata()) != list(img.getdata())

    # The cell at region pixel (1000, 1000) on a 2000x2000 region spanning
    # +/-10 degrees means lat=0, lon=0.  At z=0, the single world tile
    # (x=0,y=0) maps (lat=0, lon=0) to tile pixel (128, 128).
    # Check that pixel (128, 128) is non-transparent.
    px = result.getpixel((128, 128))
    assert px[3] > 0, "Centroid pixel should be non-transparent"


@pytest.mark.storm_cells
def test_draw_storm_cells_cell_outside_tile():
    """Cell at a region pixel far from the tile => image unchanged."""
    region = _make_syn_region("TEST_CELLS_OUTSIDE")
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    original = img.copy()
    # Place the cell at region pixel (100, 100) ~ (lat=9, lon=-9), which
    # is far from the tile center at (lat=0, lon=0) for z=0.  With a
    # 2000x2000 region (+/-10 deg), region pixel (100, 100) maps to
    # roughly lat=9, lon=-9.  The z=0 tile covers the whole world, so
    # the cell IS actually on the tile.  Let's instead use a higher zoom
    # where the tile covers only a small area.
    # Use z=3, x=4, y=3 — this tile covers roughly (lon=0..45, lat=0..~30).
    # Place cell at lat=-5, lon=-5 which maps to region pixel ~(1500, 500)
    # and should be outside this tile.
    cell = _make_cell(centroid_row=1500.0, centroid_col=500.0, area_km2=100.0)
    result = _draw_storm_cells(
        img, {"TEST_CELLS_OUTSIDE": cell}, {"TEST_CELLS_OUTSIDE": 1}, [region],
        3, 4, 3, 256, "light",
    )
    assert list(result.getdata()) == list(original.getdata())


@pytest.mark.storm_cells
def test_draw_storm_cells_motion_arrow():
    """Cell with non-NaN motion => arrow drawn (image differs from
    circle-only case)."""
    region = _make_syn_region("TEST_CELLS_MOTION")
    img_no_motion = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    img_with_motion = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    cell_no_motion = _make_cell(
        centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0,
    )
    cell_with_motion = _make_cell(
        centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0,
        motion_dx_px=5.0, motion_dy_px=-3.0,
        motion_speed_kmh=20.0, motion_heading_deg=59.0,
    )

    result_no = _draw_storm_cells(
        img_no_motion, {"TEST_CELLS_MOTION": cell_no_motion},
        {"TEST_CELLS_MOTION": 1}, [region],
        0, 0, 0, 256, "light",
    )
    result_with = _draw_storm_cells(
        img_with_motion, {"TEST_CELLS_MOTION": cell_with_motion},
        {"TEST_CELLS_MOTION": 1}, [region],
        0, 0, 0, 256, "light",
    )

    # The with-motion result should have more non-transparent pixels
    # (the arrow extends beyond the circle).
    no_data = list(result_no.getdata())
    with_data = list(result_with.getdata())
    assert with_data != no_data, "Motion arrow should add pixels beyond the circle"


@pytest.mark.storm_cells
def test_draw_storm_cells_nan_motion_no_arrow():
    """Cell with NaN motion => only the circle is drawn (same as
    zero-motion circle-only case)."""
    region = _make_syn_region("TEST_CELLS_NAN")
    img_zero = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    img_nan = Image.new("RGBA", (256, 256), (0, 0, 0, 0))

    cell_zero_motion = _make_cell(
        centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0,
        motion_dx_px=0.0, motion_dy_px=0.0,
        motion_speed_kmh=0.0, motion_heading_deg=0.0,
    )
    cell_nan_motion = _make_cell(
        centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0,
    )

    result_zero = _draw_storm_cells(
        img_zero, {"TEST_CELLS_NAN": cell_zero_motion},
        {"TEST_CELLS_NAN": 1}, [region],
        0, 0, 0, 256, "light",
    )
    result_nan = _draw_storm_cells(
        img_nan, {"TEST_CELLS_NAN": cell_nan_motion},
        {"TEST_CELLS_NAN": 1}, [region],
        0, 0, 0, 256, "light",
    )

    no_zero = list(result_zero.getdata())
    no_nan = list(result_nan.getdata())
    assert no_nan == no_zero, "NaN-motion cell should produce same overlay as zero-motion cell"


@pytest.mark.storm_cells
def test_draw_storm_cells_dark_style():
    """Dark style => drawn pixels are dark-colored (closer to (40,40,40)
    than to (255,255,255))."""
    region = _make_syn_region("TEST_CELLS_DARK")
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    cell = _make_cell(centroid_row=1000.0, centroid_col=1000.0, area_km2=100.0)
    result = _draw_storm_cells(
        img, {"TEST_CELLS_DARK": cell}, {"TEST_CELLS_DARK": 1}, [region],
        0, 0, 0, 256, "dark",
    )
    # Check pixel at (128, 128) is dark-colored
    px = result.getpixel((128, 128))
    assert px[3] > 0, "Centroid pixel should be non-transparent"
    # RGB should be closer to (40, 40, 40) than to (255, 255, 255)
    r, g, b, a = px
    dist_to_dark = math.sqrt((r - 40) ** 2 + (g - 40) ** 2 + (b - 40) ** 2)
    dist_to_light = math.sqrt((r - 255) ** 2 + (g - 255) ** 2 + (b - 255) ** 2)
    assert dist_to_dark < dist_to_light, (
        f"Dark-style cell pixels should be dark-colored; got rgb=({r},{g},{b})"
    )
