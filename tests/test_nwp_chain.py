# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NWPChain precip-bbox fast-path gate (empty-tile Tier 2).

Covers ``_bbox_intersects`` AABB math, the init-time domain-bbox probe, and
``has_precip_in_bbox`` conservatism: a source without ``precip_bbox`` (all
regional sources), a missing/antimeridian-spanning bbox, or a raising source
must all report "may have precip" so no tile is wrongly fast-pathed.
"""

import numpy as np
import pytest

from librewxr.data.nwp_source import NWPChain


# ---------------------------------------------------------------------------
# Fake sources
# ---------------------------------------------------------------------------


class FakeRegionalSource:
    """A regional NWP source: finite projection-only domain, NO precip_bbox."""

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


class FakeIFSSource(FakeRegionalSource):
    """Global IFS-like source: full-globe domain + optional precip_bbox."""

    def __init__(self, precip_bboxes):
        super().__init__(-180.0, -90.0, 180.0, 90.0)
        self._precip_bboxes = dict(precip_bboxes)

    def precip_bbox(self, timestamp: int):
        return self._precip_bboxes.get(timestamp)


class _RaisingIFSSource(FakeIFSSource):
    def precip_bbox(self, timestamp: int):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# _bbox_intersects
# ---------------------------------------------------------------------------


class TestBBoxIntersects:
    def test_identical_boxes(self):
        box = (-10.0, 20.0, 10.0, 40.0)
        assert NWPChain._bbox_intersects(box, box) is True

    def test_nested_boxes(self):
        outer = (-30.0, -10.0, 30.0, 50.0)
        inner = (-5.0, 0.0, 5.0, 20.0)
        assert NWPChain._bbox_intersects(inner, outer) is True
        assert NWPChain._bbox_intersects(outer, inner) is True

    def test_edge_touching_longitude_is_intersecting(self):
        """a.east == b.west with full lat overlap: formula says True.

        ``a[2] >= b[0]`` (a.east >= b.west) is satisfied by equality and
        ``a[0] <= b[2]`` (a.west <= b.east) trivially holds, so touching
        boxes DO intersect under the plain AABB formula.
        """
        a = (-20.0, 0.0, 0.0, 30.0)
        b = (0.0, 0.0, 20.0, 30.0)
        assert NWPChain._bbox_intersects(a, b) is True

    def test_disjoint_longitude(self):
        a = (-20.0, 0.0, -10.0, 30.0)
        b = (10.0, 0.0, 20.0, 30.0)
        assert NWPChain._bbox_intersects(a, b) is False

    def test_disjoint_latitude(self):
        a = (-20.0, 0.0, 10.0, 10.0)
        b = (-20.0, 40.0, 10.0, 50.0)
        assert NWPChain._bbox_intersects(a, b) is False

    def test_inverted_boxes_are_plain_aabb(self):
        """Antimeridian-style inverted boxes get no special casing.

        Both inputs are guaranteed non-wrapping by the callers, so the
        method is plain AABB math over the given coordinates: an inverted
        box (west > east) that "wraps" in reality is treated as disjoint
        from a normal box near the seam — and even from itself, because
        the west <= east check fails.
        """
        inverted = (170.0, -10.0, -170.0, 10.0)  # west 170 > east -170
        normal = (-5.0, -5.0, 5.0, 5.0)
        # inverted.east(-170) < normal.west(-5) -> disjoint on longitude.
        assert NWPChain._bbox_intersects(inverted, normal) is False
        # Plain AABB: west(170) <= east(-170) fails, so not self-intersecting.
        assert NWPChain._bbox_intersects(inverted, inverted) is False


# ---------------------------------------------------------------------------
# Domain-bbox probe + has_precip_in_bbox
# ---------------------------------------------------------------------------


class TestHasPrecipInBBox:
    def test_domain_bbox_probe_matches_domain(self):
        """Init-time probe approximates a regional source's domain (+-1 deg)."""
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        w, s, e, n = chain._domain_bboxes[0]
        assert w <= -125.0 + 1.0
        assert s <= 25.0 + 1.0
        assert e >= -70.0 - 1.0
        assert n >= 50.0 - 1.0

    def test_no_source_covers_tile_returns_false(self):
        """Regional domain far from the tile bbox -> no precip possible."""
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)  # far from CONUS
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is False

    def test_regional_source_overlapping_is_conservative(self):
        """(p) Regional source (no precip_bbox) covering the tile -> True."""
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        tile_bbox = (-100.0, 30.0, -90.0, 40.0)  # inside CONUS
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_ifs_bbox_does_not_intersect_tile(self):
        """IFS precip bbox far from the tile bbox -> False."""
        chain = NWPChain([FakeIFSSource({1700000000: (-170.0, 30.0, -150.0, 40.0)})])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is False

    def test_ifs_bbox_intersects_tile(self):
        chain = NWPChain([FakeIFSSource({1700000000: (-95.0, 30.0, -85.0, 40.0)})])
        tile_bbox = (-100.0, 30.0, -90.0, 40.0)
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_ifs_missing_timestamp_is_conservative(self):
        """precip_bbox(None/missing) -> assume precip -> True."""
        chain = NWPChain([FakeIFSSource({1700000000: (-95.0, 30.0, -85.0, 40.0)})])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)
        assert chain.has_precip_in_bbox(999999999, tile_bbox) is True

    def test_ifs_bbox_none_is_conservative(self):
        """A stored None bbox (antimeridian / no precip marker) -> True."""
        chain = NWPChain([FakeIFSSource({1700000000: None})])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_ifs_error_is_conservative(self):
        """A raising precip_bbox -> assume precip -> True."""
        chain = NWPChain([_RaisingIFSSource({1700000000: (-95.0, 30.0, -85.0, 40.0)})])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_mixed_chain_regional_wins(self):
        """Regional (conservative True) + IFS (no intersect) -> True."""
        chain = NWPChain([
            FakeIFSSource({1700000000: (-170.0, 30.0, -150.0, 40.0)}),
            FakeRegionalSource(-125.0, 25.0, -70.0, 50.0),
        ])
        tile_bbox = (-100.0, 30.0, -90.0, 40.0)  # IFS bbox misses, regional covers
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_ifs_only_chain_empty_bbox_map(self):
        """Empty precip_bbox map -> missing key -> conservative True."""
        chain = NWPChain([FakeIFSSource({})])
        tile_bbox = (10.0, 10.0, 20.0, 20.0)
        assert chain.has_precip_in_bbox(1700000000, tile_bbox) is True

    def test_sources_property_unchanged(self):
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        assert len(chain.sources) == 1
        assert chain.sources[0].name == "fake_regional"
