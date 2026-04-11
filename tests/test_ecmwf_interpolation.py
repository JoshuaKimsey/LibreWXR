# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for ECMWF IFS optical-flow temporal interpolation."""
import numpy as np
import pytest

from librewxr.data.ecmwf_interpolation import (
    _compute_flow,
    _interpolate_frame,
    _interpolate_snow,
    interpolate_timesteps,
)


# Use small grids for fast tests (full grid is 1801×3600)
H, W = 120, 240


def _make_blob(cy: int, cx: int, radius: int = 20, value: int = 150) -> np.ndarray:
    """Create a test grid with a circular precipitation blob."""
    grid = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.ogrid[0:H, 0:W]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    grid[mask] = value
    return grid


class TestInterpolateTimesteps:
    def test_single_timestep_unchanged(self):
        ts = {1000: (np.zeros((H, W), dtype=np.uint8), np.zeros((H, W), dtype=bool))}
        result = interpolate_timesteps(ts)
        assert list(result.keys()) == [1000]

    def test_creates_intermediate_frames(self):
        precip = np.full((H, W), 100, dtype=np.uint8)
        snow = np.zeros((H, W), dtype=bool)
        ts = {
            0: (precip.copy(), snow.copy()),
            3600: (precip.copy(), snow.copy()),
        }
        result = interpolate_timesteps(ts, interval_seconds=600)
        # 3600 / 600 = 6 slots, endpoints included → 5 intermediates
        assert len(result) == 7
        expected_times = [0, 600, 1200, 1800, 2400, 3000, 3600]
        assert sorted(result.keys()) == expected_times

    def test_preserves_original_frames(self):
        precip0 = np.full((H, W), 80, dtype=np.uint8)
        precip1 = np.full((H, W), 120, dtype=np.uint8)
        snow = np.zeros((H, W), dtype=bool)
        ts = {
            0: (precip0, snow.copy()),
            3600: (precip1, snow.copy()),
        }
        result = interpolate_timesteps(ts, interval_seconds=600)
        # Original frames should be byte-identical
        assert np.array_equal(result[0][0], precip0)
        assert np.array_equal(result[3600][0], precip1)

    def test_skips_already_fine_grained(self):
        precip = np.zeros((H, W), dtype=np.uint8)
        snow = np.zeros((H, W), dtype=bool)
        ts = {
            0: (precip.copy(), snow.copy()),
            600: (precip.copy(), snow.copy()),
        }
        result = interpolate_timesteps(ts, interval_seconds=600)
        # Gap == interval → nothing to interpolate
        assert len(result) == 2

    def test_multiple_pairs(self):
        precip = np.full((H, W), 100, dtype=np.uint8)
        snow = np.zeros((H, W), dtype=bool)
        ts = {
            0: (precip.copy(), snow.copy()),
            3600: (precip.copy(), snow.copy()),
            7200: (precip.copy(), snow.copy()),
        }
        result = interpolate_timesteps(ts, interval_seconds=600)
        # 2 pairs × 5 intermediates + 3 originals = 13
        assert len(result) == 13

    def test_clear_sky_stays_clear(self):
        """No-precipitation frames should produce all-zero intermediates."""
        precip = np.zeros((H, W), dtype=np.uint8)
        snow = np.zeros((H, W), dtype=bool)
        ts = {
            0: (precip.copy(), snow.copy()),
            3600: (precip.copy(), snow.copy()),
        }
        result = interpolate_timesteps(ts, interval_seconds=600)
        for t, (p, s) in result.items():
            assert (p == 0).all(), f"Non-zero precip at t={t}"
            assert not s.any(), f"Snow at t={t}"


class TestComputeFlow:
    def test_static_frame_zero_flow(self):
        """Identical frames should produce near-zero flow."""
        frame = _make_blob(60, 120)
        flow = _compute_flow(frame, frame)
        assert flow.shape == (H, W, 2)
        assert np.abs(flow).max() < 1.0

    def test_moving_blob_nonzero_flow(self):
        """A shifted blob should produce detectable flow."""
        frame0 = _make_blob(60, 100)
        frame1 = _make_blob(60, 130)  # moved 30px right
        flow = _compute_flow(frame0, frame1)
        # Flow in the blob region should point rightward (positive x)
        blob_mask = frame0 > 0
        mean_fx = flow[blob_mask, 0].mean()
        assert mean_fx > 2.0, f"Expected rightward flow, got mean_fx={mean_fx:.2f}"


class TestInterpolateFrame:
    def test_midpoint_has_values(self):
        """Interpolated midpoint should have non-zero pixels where inputs do."""
        frame0 = _make_blob(60, 100, value=200)
        frame1 = _make_blob(60, 130, value=200)
        flow = _compute_flow(frame0, frame1)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        mid = _interpolate_frame(frame0, frame1, flow, 0.5, xs, ys)
        assert mid.dtype == np.uint8
        assert mid.max() > 0
        # Midpoint blob should be roughly between the two positions
        nonzero_cols = np.where(mid.any(axis=0))[0]
        center = nonzero_cols.mean()
        assert 100 < center < 135

    def test_t_near_zero_matches_frame0(self):
        """At t≈0 the result should closely match frame0."""
        frame = _make_blob(60, 120, value=180)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        result = _interpolate_frame(frame, frame, flow, 0.01, xs, ys)
        diff = np.abs(result.astype(np.int16) - frame.astype(np.int16))
        assert diff.max() <= 2

    def test_both_zero_stays_zero(self):
        """Where both frames are zero, interpolation must not create values."""
        frame = np.zeros((H, W), dtype=np.uint8)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        result = _interpolate_frame(frame, frame, flow, 0.5, xs, ys)
        assert (result == 0).all()


class TestInterpolateSnow:
    def test_snow_mask_preserved(self):
        """All-snow frames should produce all-snow intermediates."""
        snow = np.ones((H, W), dtype=bool)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        result = _interpolate_snow(snow, snow, flow, 0.5, xs, ys)
        assert result.dtype == bool
        assert result.all()

    def test_no_snow_stays_clear(self):
        snow = np.zeros((H, W), dtype=bool)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        result = _interpolate_snow(snow, snow, flow, 0.5, xs, ys)
        assert not result.any()
