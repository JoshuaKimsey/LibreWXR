# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-variant smoke tests for AROME Nouvelle-Calédonie."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.arome_ncaled

from librewxr.data.nwp_source import NWPSource
from librewxr.sources.regional.oceania.nwp.arome_ncaled.grid import (
    AROMENCaledGrid,
)


class TestDomain:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Nouméa, NC",       -22.27, 166.46, True),
            ("Maré, NC",         -21.50, 168.05, True),
            ("Port Vila, VU",    -17.74, 168.32, True),   # Vanuatu, inside east edge
            ("Brisbane, AU",     -27.47, 153.03, False),  # west + south of grid
            ("Suva, FJ",         -18.13, 178.42, False),  # east of LON_EAST
            ("Auckland, NZ",     -36.85, 174.76, False),  # south of LAT_SOUTH
        ],
    )
    def test_known_points(self, name, lat, lon, inside):
        m = AROMENCaledGrid.domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_nw_corner(self):
        row, col = AROMENCaledGrid.grid_indices(
            np.array([AROMENCaledGrid.LAT_NORTH]),
            np.array([AROMENCaledGrid.LON_WEST_DEG_E]),
        )
        assert abs(row[0]) < 1e-3
        assert abs(col[0]) < 1e-3

    def test_se_corner(self):
        row, col = AROMENCaledGrid.grid_indices(
            np.array([AROMENCaledGrid.LAT_SOUTH]),
            np.array([AROMENCaledGrid.LON_EAST_DEG_E]),
        )
        assert abs(row[0] - (AROMENCaledGrid.GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - (AROMENCaledGrid.GRID_WIDTH - 1)) < 1e-3

    def test_grid_dims_match_decoded_values(self):
        assert AROMENCaledGrid.GRID_HEIGHT == 491
        assert AROMENCaledGrid.GRID_WIDTH == 521


class TestFeather:
    def test_full_weight_at_noumea(self):
        f = AROMENCaledGrid.feather_mask(np.array([-22.27]), np.array([166.46]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_zero_outside(self):
        f = AROMENCaledGrid.feather_mask(np.array([0.0]), np.array([0.0]))
        assert f[0] == 0.0


class TestURL:
    def test_url_contains_ncaled_token(self):
        run = datetime(2026, 5, 21, 6, tzinfo=timezone.utc)
        url = AROMENCaledGrid.file_url(run, 6)
        assert "/arome-om/NCALED/0025/SP1/" in url
        assert "arome-om-NCALED__0025__SP1__006H__" in url


class TestProtocol:
    def test_satisfies_nwp_source(self):
        grid = AROMENCaledGrid()
        assert isinstance(grid, NWPSource)
        assert grid.supports_snow is False
