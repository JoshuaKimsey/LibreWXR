# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NWPChain dispatcher.

The Tier 2 precip-bbox fast-path surface (``_probe_domain_bbox`` /
``_bbox_intersects`` / ``has_precip_in_bbox`` / ``_domain_bboxes``) was
removed when the stitched global precip mask (``PrecipMaskStore``)
replaced it — see ``tests/test_precip_mask.py``.  What remains here is
the dispatcher behaviour that is independent of that gate.
"""

import numpy as np

from librewxr.data.nwp_source import NWPChain


class FakeRegionalSource:
    """A regional NWP source: finite projection-only domain."""

    def __init__(self, west, south, east, north):
        self._bbox = (west, south, east, north)

    @property
    def name(self) -> str:
        return "fake_regional"

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        w, s, e, n = self._bbox
        return (
            (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        )

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.domain_mask(lat, lon).astype(np.float32)

    def has_data(self) -> bool:
        return True

    def has_data_at(self, timestamp: int) -> bool:
        return True

    def sample(self, lat, lon, timestamp=None, bilinear=False) -> np.ndarray:
        return np.zeros(lat.shape, dtype=np.uint8)

    @property
    def supports_snow(self) -> bool:
        return False

    def get_snow_mask(self, lat, lon, timestamp=None) -> np.ndarray:
        return np.zeros(lat.shape, dtype=bool)


class TestNWPChain:
    def test_sources_property_unchanged(self):
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        assert len(chain.sources) == 1
        assert chain.sources[0].name == "fake_regional"
