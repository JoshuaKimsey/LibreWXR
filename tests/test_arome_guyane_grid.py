# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-variant smoke tests for AROME Guyane.

Heavy machinery (Z-R, timing, decode orientation, run picking, fetch
loop) is exercised by the AROME Antilles tests against the shared
``AROMEOverseasGrid`` base, so this file only covers the constants
the Guyane subclass is responsible for: domain extent, URL token,
feather sizing, and Protocol conformance.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.arome_guyane

from librewxr.data.nwp_source import NWPSource
from librewxr.sources.regional.south_america.nwp.arome_guyane.grid import (
    AROMEGuyaneGrid,
)


class TestDomain:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Cayenne, FG",       4.93, -52.33, True),
            ("Paramaribo, SR",    5.87, -55.17, True),
            ("Macapá, BR",        0.04, -51.07, False),   # south of LAT_SOUTH=1.05
            ("Manaus, BR",       -3.12, -60.02, False),
            ("Bogotá, CO",        4.71, -74.07, False),
        ],
    )
    def test_known_points(self, name, lat, lon, inside):
        m = AROMEGuyaneGrid.domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_nw_corner(self):
        row, col = AROMEGuyaneGrid.grid_indices(
            np.array([AROMEGuyaneGrid.LAT_NORTH]),
            np.array([AROMEGuyaneGrid.LON_WEST_DEG_E - 360.0]),
        )
        assert abs(row[0]) < 1e-3
        assert abs(col[0]) < 1e-3

    def test_se_corner(self):
        row, col = AROMEGuyaneGrid.grid_indices(
            np.array([AROMEGuyaneGrid.LAT_SOUTH]),
            np.array([AROMEGuyaneGrid.LON_EAST_DEG_E - 360.0]),
        )
        assert abs(row[0] - (AROMEGuyaneGrid.GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - (AROMEGuyaneGrid.GRID_WIDTH - 1)) < 1e-3

    def test_grid_dims_match_decoded_values(self):
        # Sanity check: back-decoded from a real 2026-05-21 GRIB.
        assert AROMEGuyaneGrid.GRID_HEIGHT == 317
        assert AROMEGuyaneGrid.GRID_WIDTH == 419


class TestFeather:
    def test_feather_full_weight_at_centre(self):
        f = AROMEGuyaneGrid.feather_mask(np.array([4.93]), np.array([-52.33]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_feather_zero_outside(self):
        f = AROMEGuyaneGrid.feather_mask(np.array([40.0]), np.array([-3.7]))
        assert f[0] == 0.0


class TestURL:
    def test_url_contains_guyane_token(self):
        run = datetime(2026, 5, 21, 6, tzinfo=timezone.utc)
        url = AROMEGuyaneGrid.file_url(run, 6)
        assert "/arome-om/GUYANE/0025/SP1/" in url
        assert "arome-om-GUYANE__0025__SP1__006H__" in url

    def test_url_uses_ovh_bucket(self):
        run = datetime(2026, 5, 21, 0, tzinfo=timezone.utc)
        url = AROMEGuyaneGrid.file_url(run, 1)
        assert "meteofrance-pnt" in url


class TestProtocol:
    def test_satisfies_nwp_source(self):
        grid = AROMEGuyaneGrid()
        assert isinstance(grid, NWPSource)
        assert grid.supports_snow is False  # tropical
