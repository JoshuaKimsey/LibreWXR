# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-variant smoke tests for AROME Indien (Réunion/Mayotte/Madagascar)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.arome_indien

from librewxr.data.nwp_source import NWPSource
from librewxr.sources.regional.africa.nwp.arome_indien.grid import (
    AROMEIndienGrid,
)


class TestDomain:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Saint-Denis, RE",       -20.88,  55.45, True),
            ("Mamoudzou, YT",         -12.78,  45.23, True),
            ("Antananarivo, MG",      -18.88,  47.51, True),
            ("Moroni, KM",            -11.70,  43.26, True),
            ("Port Louis, MU",        -20.16,  57.50, True),
            ("Dar es Salaam, TZ",      -6.79,  39.21, True),   # near northern edge
            ("Nairobi, KE",            -1.29,  36.82, False),  # north of LAT_NORTH
            ("Perth, AU",             -31.95, 115.86, False),  # east + south of grid
            ("Bangalore, IN",          12.97,  77.59, False),
        ],
    )
    def test_known_points(self, name, lat, lon, inside):
        m = AROMEIndienGrid.domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_nw_corner(self):
        row, col = AROMEIndienGrid.grid_indices(
            np.array([AROMEIndienGrid.LAT_NORTH]),
            np.array([AROMEIndienGrid.LON_WEST_DEG_E]),
        )
        assert abs(row[0]) < 1e-3
        assert abs(col[0]) < 1e-3

    def test_se_corner(self):
        row, col = AROMEIndienGrid.grid_indices(
            np.array([AROMEIndienGrid.LAT_SOUTH]),
            np.array([AROMEIndienGrid.LON_EAST_DEG_E]),
        )
        assert abs(row[0] - (AROMEIndienGrid.GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - (AROMEIndienGrid.GRID_WIDTH - 1)) < 1e-3

    def test_grid_dims_match_decoded_values(self):
        assert AROMEIndienGrid.GRID_HEIGHT == 899
        assert AROMEIndienGrid.GRID_WIDTH == 1395


class TestFeather:
    def test_full_weight_at_reunion(self):
        f = AROMEIndienGrid.feather_mask(np.array([-20.88]), np.array([55.45]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_zero_outside(self):
        f = AROMEIndienGrid.feather_mask(np.array([0.0]), np.array([0.0]))
        assert f[0] == 0.0

    def test_larger_feather_than_antilles(self):
        # INDIEN is the largest AROME-OM grid; its feather should be wider.
        from librewxr.sources.regional.caribbean.nwp.arome_antilles.grid import (
            AROMEAntillesGrid,
        )
        assert AROMEIndienGrid.FEATHER_DISTANCE_CELLS > AROMEAntillesGrid.FEATHER_DISTANCE_CELLS


class TestURL:
    def test_url_contains_indien_token(self):
        run = datetime(2026, 5, 21, 6, tzinfo=timezone.utc)
        url = AROMEIndienGrid.file_url(run, 6)
        assert "/arome-om/INDIEN/0025/SP1/" in url
        assert "arome-om-INDIEN__0025__SP1__006H__" in url


class TestProtocol:
    def test_satisfies_nwp_source(self):
        grid = AROMEIndienGrid()
        assert isinstance(grid, NWPSource)
        assert grid.supports_snow is False  # tropical / sub-tropical
