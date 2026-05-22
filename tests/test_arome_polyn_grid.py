# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-variant smoke tests for AROME Polynésie française."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.arome_polyn

from librewxr.data.nwp_source import NWPSource
from librewxr.sources.regional.oceania.nwp.arome_polyn.grid import (
    AROMEPolynGrid,
)


class TestDomain:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            # Society Islands
            ("Pape'ete (Tahiti)",  -17.54, -149.57, True),
            ("Bora Bora",          -16.50, -151.74, True),
            # Tuamotu Islands
            ("Rangiroa",           -15.10, -147.65, True),
            # Outside the published grid
            ("Marquesas (Nuku Hiva)", -8.92, -140.10, False),  # north of LAT_NORTH
            ("Rapa Nui",          -27.12, -109.36, False),     # east + south of grid
            ("Nuku'alofa, TO",    -21.13, -175.20, False),     # west of LON_WEST
            ("Honolulu, HI",       21.31, -157.86, False),     # north of equator
            ("Sydney, AU",        -33.86,  151.21, False),
        ],
    )
    def test_known_points(self, name, lat, lon, inside):
        m = AROMEPolynGrid.domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_nw_corner(self):
        # NW corner uses lon in [0,360) form (no -360 shift) since the
        # grid sits at 202.5-215.5°E (i.e. western hemisphere when
        # expressed as negative degrees).
        row, col = AROMEPolynGrid.grid_indices(
            np.array([AROMEPolynGrid.LAT_NORTH]),
            np.array([AROMEPolynGrid.LON_WEST_DEG_E]),
        )
        assert abs(row[0]) < 1e-3
        assert abs(col[0]) < 1e-3

    def test_se_corner(self):
        row, col = AROMEPolynGrid.grid_indices(
            np.array([AROMEPolynGrid.LAT_SOUTH]),
            np.array([AROMEPolynGrid.LON_EAST_DEG_E]),
        )
        assert abs(row[0] - (AROMEPolynGrid.GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - (AROMEPolynGrid.GRID_WIDTH - 1)) < 1e-3

    def test_grid_dims_match_decoded_values(self):
        assert AROMEPolynGrid.GRID_HEIGHT == 507
        assert AROMEPolynGrid.GRID_WIDTH == 521

    def test_negative_lon_input_form_works(self):
        # Pape'ete given as -149.57°E (standard form) and 210.43°E
        # (bucket form) should map to the same grid cell.
        r1, c1 = AROMEPolynGrid.grid_indices(np.array([-17.54]), np.array([-149.57]))
        r2, c2 = AROMEPolynGrid.grid_indices(np.array([-17.54]), np.array([210.43]))
        assert r1[0] == pytest.approx(r2[0])
        assert c1[0] == pytest.approx(c2[0])


class TestFeather:
    def test_full_weight_at_papeete(self):
        f = AROMEPolynGrid.feather_mask(np.array([-17.54]), np.array([-149.57]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_zero_outside(self):
        f = AROMEPolynGrid.feather_mask(np.array([0.0]), np.array([0.0]))
        assert f[0] == 0.0


class TestURL:
    def test_url_contains_polyn_token(self):
        run = datetime(2026, 5, 21, 6, tzinfo=timezone.utc)
        url = AROMEPolynGrid.file_url(run, 6)
        assert "/arome-om/POLYN/0025/SP1/" in url
        assert "arome-om-POLYN__0025__SP1__006H__" in url


class TestProtocol:
    def test_satisfies_nwp_source(self):
        grid = AROMEPolynGrid()
        assert isinstance(grid, NWPSource)
        assert grid.supports_snow is False
