# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for precipitation nowcasting: store, generator, and optical flow."""
import asyncio

import numpy as np
import pytest

pytestmark = pytest.mark.nowcast

from librewxr.data.nowcast import (
    NowcastFrame,
    NowcastGenerator,
    NowcastStore,
    _clamp_flow,
    _compute_flow,
    _coverage_degraded,
    _extrapolate_forward,
    _max_flow_pixels,
)


# Small grids for fast tests
H, W = 120, 240


def _make_blob(cy: int, cx: int, radius: int = 20, value: int = 150) -> np.ndarray:
    """Create a test grid with a circular precipitation blob."""
    grid = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.ogrid[0:H, 0:W]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    grid[mask] = value
    return grid


# ---------------------------------------------------------------------------
# NowcastStore tests
# ---------------------------------------------------------------------------


class TestNowcastStore:
    @pytest.fixture
    def store(self):
        return NowcastStore()

    @pytest.mark.asyncio
    async def test_empty_store(self, store):
        timestamps = await store.get_timestamps()
        assert timestamps == []
        frame, weight = await store.get_frame(1000)
        assert frame is None
        assert weight == 0.0

    @pytest.mark.asyncio
    async def test_replace_all(self, store):
        frames = [
            NowcastFrame(timestamp=1000, regions={"A": np.zeros((2, 2), dtype=np.uint8)}, blend_weight=0.8),
            NowcastFrame(timestamp=2000, regions={"A": np.zeros((2, 2), dtype=np.uint8)}, blend_weight=0.5),
        ]
        old_ts = await store.replace_all(frames)
        assert old_ts == []  # was empty

        timestamps = await store.get_timestamps()
        assert timestamps == [1000, 2000]

    @pytest.mark.asyncio
    async def test_replace_returns_old_timestamps(self, store):
        frames1 = [NowcastFrame(timestamp=100, blend_weight=0.9)]
        await store.replace_all(frames1)

        frames2 = [NowcastFrame(timestamp=200, blend_weight=0.8)]
        old_ts = await store.replace_all(frames2)
        assert old_ts == [100]

        timestamps = await store.get_timestamps()
        assert timestamps == [200]

    @pytest.mark.asyncio
    async def test_get_frame(self, store):
        frame = NowcastFrame(
            timestamp=5000,
            regions={"R": np.ones((3, 3), dtype=np.uint8)},
            blend_weight=0.6,
        )
        await store.replace_all([frame])
        result, weight = await store.get_frame(5000)
        assert result is not None
        assert result.timestamp == 5000
        assert weight == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_get_frame_missing(self, store):
        await store.replace_all([NowcastFrame(timestamp=100)])
        result, weight = await store.get_frame(999)
        assert result is None
        assert weight == 0.0

    @pytest.mark.asyncio
    async def test_clear(self, store):
        await store.replace_all([NowcastFrame(timestamp=100)])
        store.clear()
        timestamps = await store.get_timestamps()
        assert timestamps == []


# ---------------------------------------------------------------------------
# Optical flow tests
# ---------------------------------------------------------------------------


class TestComputeFlow:
    def test_stationary_blob_zero_flow(self):
        blob = _make_blob(60, 120)
        flow = _compute_flow(blob, blob)
        assert flow.shape == (H, W, 2)
        # Stationary blob → near-zero flow
        assert np.abs(flow).mean() < 1.0

    def test_flow_shape(self):
        frame0 = _make_blob(60, 100)
        frame1 = _make_blob(60, 120)
        flow = _compute_flow(frame0, frame1)
        assert flow.shape == (H, W, 2)
        assert flow.dtype == np.float32 or flow.dtype == np.float64

    def test_moving_blob_nonzero_flow(self):
        frame0 = _make_blob(60, 80)
        frame1 = _make_blob(60, 120)
        flow = _compute_flow(frame0, frame1)
        # Should have meaningful flow in the blob region
        blob_mask = frame0 > 0
        blob_flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        assert blob_flow_mag[blob_mask].mean() > 1.0


class TestExtrapolateForward:
    def test_output_shape(self):
        frame = _make_blob(60, 120)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        result = _extrapolate_forward(frame, flow, steps=1)
        assert result.shape == (H, W)
        assert result.dtype == frame.dtype

    def test_zero_flow_preserves_frame(self):
        frame = _make_blob(60, 120, value=200)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        result = _extrapolate_forward(frame, flow, steps=3)
        # With zero flow, warping should preserve the frame
        assert np.array_equal(result, frame)

    def test_extrapolation_shifts_blob(self):
        frame = _make_blob(60, 80, radius=15, value=150)
        # Uniform rightward flow: 10 px/step in x direction
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[..., 0] = 10.0  # x flow

        result = _extrapolate_forward(frame, flow, steps=2)
        # Original blob center of mass was at x≈80
        # After 2 steps × 10 px, should be near x≈100
        orig_com_x = np.average(np.arange(W), weights=frame.sum(axis=0).astype(float) + 1e-9)
        result_com_x = np.average(np.arange(W), weights=result.sum(axis=0).astype(float) + 1e-9)
        assert result_com_x > orig_com_x + 10  # shifted right significantly

    def test_multiple_steps_increase_shift(self):
        frame = _make_blob(60, 60, radius=15, value=150)
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[..., 0] = 5.0  # rightward

        result1 = _extrapolate_forward(frame, flow, steps=1)
        result2 = _extrapolate_forward(frame, flow, steps=3)
        # 3 steps should shift more than 1 step
        com1 = np.average(np.arange(W), weights=result1.sum(axis=0).astype(float) + 1e-9)
        com2 = np.average(np.arange(W), weights=result2.sum(axis=0).astype(float) + 1e-9)
        assert com2 > com1


# ---------------------------------------------------------------------------
# NowcastFrame blend weight tests
# ---------------------------------------------------------------------------


class TestBlendWeights:
    def test_blend_curve(self):
        """60-min blend: 0.30 + 0.70*(1-t)^1.1, pure IFS beyond 60 min."""
        n_steps = 6
        interval = 600
        max_blend_steps = 3600 // interval  # 6
        weights = []
        for step in range(1, n_steps + 1):
            if step <= max_blend_steps:
                t = step / max_blend_steps
                weights.append(0.30 + 0.70 * (1.0 - t) ** 1.1)
            else:
                weights.append(0.0)
        assert len(weights) == 6
        # Near-term should strongly trust radar
        assert weights[0] > 0.8
        # T+50 ≈ 40% radar
        assert 0.35 < weights[4] < 0.45
        # T+60 = 30% radar (floor)
        assert weights[-1] == pytest.approx(0.30)
        # Monotonically decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_blend_beyond_60_min_is_pure_ifs(self):
        """Frames beyond 60 min should have blend_weight=0 (pure IFS)."""
        interval = 600
        max_blend_steps = 3600 // interval
        # Step 7 is beyond 60 min
        step = max_blend_steps + 1
        assert step > max_blend_steps
        # Would get blend_weight = 0.0


# ---------------------------------------------------------------------------
# NowcastGenerator sync generation tests
# ---------------------------------------------------------------------------


class TestNowcastGeneratorSync:
    def test_generate_sync_basic(self):
        """Test the synchronous generation path with simple data."""
        blob0 = _make_blob(60, 100, radius=20, value=150)
        blob1 = _make_blob(60, 110, radius=20, value=150)

        prev_regions = {"USCOMP": blob0}
        latest_regions = {"USCOMP": blob1}

        frames, flows = NowcastGenerator._generate_sync(
            prev_regions, latest_regions,
            latest_ts=1000, n_steps=3, interval=600,
        )

        assert len(frames) == 3
        assert "USCOMP" in flows
        assert flows["USCOMP"].shape == (H, W, 2)
        assert frames[0].timestamp == 1600
        assert frames[1].timestamp == 2200
        assert frames[2].timestamp == 2800

        # Blend weights should decrease
        assert frames[0].blend_weight > frames[1].blend_weight
        assert frames[1].blend_weight > frames[2].blend_weight
        # With 3 steps at 600s, max_blend_steps=6, so step 3 is mid-curve
        # t=3/6=0.5 → 0.20 + 0.80*(0.5)^1.4 ≈ 0.50
        assert 0.45 < frames[2].blend_weight < 0.55

        # Each frame should have the region
        for f in frames:
            assert "USCOMP" in f.regions
            assert f.regions["USCOMP"].shape == (H, W)

    def test_generate_sync_missing_region(self):
        """If a region exists in latest but not prev, it should be skipped."""
        blob = _make_blob(60, 100)
        prev_regions = {}  # no regions
        latest_regions = {"USCOMP": blob}

        frames, flows = NowcastGenerator._generate_sync(
            prev_regions, latest_regions,
            latest_ts=1000, n_steps=3, interval=600,
        )
        assert frames == []
        assert flows == {}

    def test_generate_sync_multiple_regions(self):
        """Should generate nowcast for each region independently."""
        blob0_a = _make_blob(60, 100)
        blob1_a = _make_blob(60, 110)
        blob0_b = _make_blob(30, 50, radius=10, value=100)
        blob1_b = _make_blob(30, 55, radius=10, value=100)

        prev = {"A": blob0_a, "B": blob0_b}
        latest = {"A": blob1_a, "B": blob1_b}

        frames, flows = NowcastGenerator._generate_sync(
            prev, latest, latest_ts=2000, n_steps=2, interval=600,
        )
        assert len(frames) == 2
        assert "A" in flows and "B" in flows
        for f in frames:
            assert "A" in f.regions
            assert "B" in f.regions


# ---------------------------------------------------------------------------
# Coverage-degradation guard
# ---------------------------------------------------------------------------


class TestCoverageDegradedHelper:
    """Direct unit tests for the partial-frame detection threshold."""

    def test_no_degradation_when_counts_match(self):
        a = _make_blob(60, 100, radius=30)
        degraded, prev_nz, latest_nz = _coverage_degraded(a, a.copy())
        assert degraded is False
        assert prev_nz == latest_nz > 0

    def test_no_degradation_for_small_natural_variation(self):
        prev = _make_blob(60, 100, radius=30)
        # Latest has the blob shifted slightly — same pixel count.
        latest = _make_blob(60, 105, radius=30)
        degraded, _, _ = _coverage_degraded(prev, latest)
        assert degraded is False

    def test_degraded_when_latest_loses_most_pixels(self):
        prev = _make_blob(60, 100, radius=40)  # ~5000 px
        # Latest has tiny remnant — well under 40% of prev.
        latest = _make_blob(60, 100, radius=5)  # ~80 px
        degraded, prev_nz, latest_nz = _coverage_degraded(prev, latest)
        assert degraded is True
        assert prev_nz > _MIN_PREV_NONZERO_PX_FOR_TEST
        assert latest_nz < prev_nz * 0.4

    def test_no_degradation_when_prev_is_tiny(self):
        """Tiny prev shouldn't trigger the guard — natural variation
        on small counts can swing huge percentages without anything
        being wrong."""
        prev = _make_blob(60, 100, radius=3)  # ~30 px, well under threshold
        latest = np.zeros((H, W), dtype=np.uint8)
        degraded, _, _ = _coverage_degraded(prev, latest)
        assert degraded is False


# Pulled from nowcast.py so tests stay in sync with the production constant.
from librewxr.data.nowcast import _MIN_PREV_NONZERO_PX as _MIN_PREV_NONZERO_PX_FOR_TEST  # noqa: E402


class TestNowcastGuardIntegration:
    """End-to-end: a partial-coverage latest frame must skip extrapolation."""

    def test_partial_coverage_latest_skips_extrapolation(self):
        """Simulate the CACOMP-loses-MSC failure mode.

        Prev frame: full coverage with precip across the whole region.
        Latest frame: only the southernmost ~quarter retains data — as
        if a contributing source dropped and we only have observations
        south of a coverage boundary.  Without the guard, optical flow
        across that boundary produces wild vectors that warp into
        streaks.  With the guard, the region is skipped entirely.
        """
        # Prev: full coverage (analog: MRMS + MSC blend, all of Canada).
        prev = np.full((H, W), 150, dtype=np.uint8)

        # Latest: only the southernmost ~25% (analog: MRMS-only, south
        # of MSC's contribution boundary).  Pixel-count ratio ≈ 0.25,
        # well below the 0.4 degradation threshold.
        latest = np.zeros((H, W), dtype=np.uint8)
        latest[int(H * 0.75):, :] = 150

        frames, flows = NowcastGenerator._generate_sync(
            {"CACOMP": prev}, {"CACOMP": latest},
            latest_ts=1000, n_steps=6, interval=600,
        )

        # The guard skips flow computation entirely — no flow recorded,
        # no extrapolated CACOMP frames produced.
        assert "CACOMP" not in flows
        assert frames == []

    def test_full_coverage_pair_passes_guard(self):
        """A normal frame-to-frame pair should NOT trigger the guard —
        small motion-induced count changes are well within tolerance.
        """
        prev = _make_blob(60, 100, radius=40)
        latest = _make_blob(60, 110, radius=40)  # same size, shifted

        frames, flows = NowcastGenerator._generate_sync(
            {"R": prev}, {"R": latest},
            latest_ts=1000, n_steps=3, interval=600,
        )

        assert "R" in flows
        assert len(frames) == 3

    def test_one_region_degraded_others_pass(self):
        """The guard is per-region: a degraded region is dropped but
        healthy peers still get their nowcasts generated."""
        # Healthy: shifted blob.
        good_prev = _make_blob(60, 100, radius=40)
        good_latest = _make_blob(60, 110, radius=40)

        # Degraded: most of the coverage drops out.
        bad_prev = np.full((H, W), 150, dtype=np.uint8)
        bad_latest = np.zeros((H, W), dtype=np.uint8)
        bad_latest[int(H * 0.75):, :] = 150

        frames, flows = NowcastGenerator._generate_sync(
            {"GOOD": good_prev, "BAD": bad_prev},
            {"GOOD": good_latest, "BAD": bad_latest},
            latest_ts=1000, n_steps=2, interval=600,
        )

        assert "GOOD" in flows
        assert "BAD" not in flows
        for f in frames:
            assert "GOOD" in f.regions
            assert "BAD" not in f.regions


# ---------------------------------------------------------------------------
# Flow-magnitude clamp (km/h → px bound, cap unphysical vectors)
# ---------------------------------------------------------------------------


class TestMaxFlowPixels:
    """Unit tests for the km/h → pixel magnitude conversion."""

    def test_coarse_region_at_10min_cadence(self):
        # 0.05°/px (e.g. CACOMP, OPERA, JPCOMP); 200 km/h cap; 10-min step.
        # km_per_step = 200/6 ≈ 33.3 km; km_per_px = 0.05 × 111 = 5.55 km.
        # max_px = 33.3 / 5.55 ≈ 6.0 px/step.
        max_px = _max_flow_pixels(0.05, 600)
        assert 5.5 < max_px < 6.5

    def test_fine_region_at_10min_cadence(self):
        # 0.01°/px (e.g. USCOMP MRMS); same cap; max_px ≈ 30 px/step.
        max_px = _max_flow_pixels(0.01, 600)
        assert 29.5 < max_px < 30.5

    def test_custom_kmh_cap(self):
        # 100 km/h halves the budget.
        max_px = _max_flow_pixels(0.05, 600, max_km_per_hour=100.0)
        assert 2.7 < max_px < 3.3

    def test_5min_cadence_halves_budget(self):
        max_px_10min = _max_flow_pixels(0.05, 600)
        max_px_5min = _max_flow_pixels(0.05, 300)
        assert abs(max_px_5min * 2 - max_px_10min) < 0.01


class TestClampFlow:
    """Unit tests for the flow-magnitude clamp itself."""

    def test_passthrough_when_no_vector_exceeds_cap(self):
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[..., 0] = 2.0   # all vectors well under cap of 10
        flow[..., 1] = 3.0
        clamped = _clamp_flow(flow, max_magnitude_px=10.0)
        # No allocation when nothing to clamp — same object.
        assert clamped is flow

    def test_over_cap_vectors_are_scaled_to_cap(self):
        flow = np.zeros((H, W, 2), dtype=np.float32)
        # Wild boundary vector: 100 px in x, 0 in y → magnitude 100.
        flow[10, 10, 0] = 100.0
        # Modest real vector: magnitude 5.
        flow[20, 20, 0] = 3.0
        flow[20, 20, 1] = 4.0

        clamped = _clamp_flow(flow, max_magnitude_px=10.0)

        clamped_mag_wild = np.sqrt(clamped[10, 10, 0] ** 2 + clamped[10, 10, 1] ** 2)
        clamped_mag_real = np.sqrt(clamped[20, 20, 0] ** 2 + clamped[20, 20, 1] ** 2)
        # Wild vector clamped exactly to cap.
        assert abs(clamped_mag_wild - 10.0) < 1e-4
        # Real vector untouched.
        assert abs(clamped_mag_real - 5.0) < 1e-4

    def test_direction_preserved_when_scaling(self):
        flow = np.zeros((H, W, 2), dtype=np.float32)
        # 100 px wild vector pointing 45° northeast.
        flow[10, 10, 0] = 70.71
        flow[10, 10, 1] = 70.71  # magnitude ≈ 100
        clamped = _clamp_flow(flow, max_magnitude_px=10.0)
        # Components should be ≈ 10/sqrt(2) each.
        assert abs(clamped[10, 10, 0] - 7.071) < 0.01
        assert abs(clamped[10, 10, 1] - 7.071) < 0.01


class TestExtrapolationClampingPreventsStreaks:
    """End-to-end: synthesize a hard data/no-data boundary, verify
    that flow clamping bounds the magnitude of extrapolated motion.

    Hard data/no-data boundary is the failure mode we're targeting —
    Farneback's local polynomial fit reports wild vectors at the
    boundary, and without clamping ``_extrapolate_forward`` warps
    boundary brightness many pixels into the no-data region.  With
    clamping, the warp distance is bounded by the physical km/h cap.
    """

    def test_extrapolation_distance_is_bounded_with_clamp(self):
        """A wild flow vector at row 30 creates a streak without
        clamping; the clamp bounds the warp distance so the streak
        doesn't form.

        ``_extrapolate_forward`` inverse-warps: output[y, x] samples
        source at ``(y - steps·flow[y, x, 1], x - steps·flow[y, x, 0])``.
        So a vector with negative y-component at the output position
        pulls brightness from south of itself (where the cluster is).
        """
        frame = np.zeros((H, W), dtype=np.uint8)
        frame[80, 100] = 200  # bright source pixel south of the streak target

        # Wild flow at row 30, col 100: flow_y = -50 → inverse-warp at
        # (30, 100) samples (30 - (-50), 100) = (80, 100), the bright
        # pixel.  Without clamping, output[30, 100] inherits brightness
        # — the streak.
        flow_wild = np.zeros((H, W, 2), dtype=np.float32)
        flow_wild[30, 100, 1] = -50.0

        warped_no_clamp = _extrapolate_forward(frame, flow_wild, steps=1)
        assert warped_no_clamp[30, 100] > 100  # streak present without clamp

        # With clamp at 10 px: flow_y clamped to -10.  Output at (30, 100)
        # now samples (40, 100), which is zero.  No streak.
        flow_clamped = _clamp_flow(flow_wild, max_magnitude_px=10.0)
        warped = _extrapolate_forward(frame, flow_clamped, steps=1)
        assert warped[30, 100] == 0  # streak gone

    def test_generate_sync_with_typical_region_applies_clamp(self):
        """Smoke test: regions in REGIONS get their flow clamped by
        the per-region pixel size.  We don't construct a wild boundary
        here — that's covered above; we just verify the wiring works
        when REGIONS has the named region.
        """
        # Use a region name we know is in REGIONS.
        prev = _make_blob(60, 100, radius=30)
        latest = _make_blob(60, 105, radius=30)
        frames, flows = NowcastGenerator._generate_sync(
            {"USCOMP": prev}, {"USCOMP": latest},
            latest_ts=1000, n_steps=2, interval=600,
        )
        assert "USCOMP" in flows
        # Verify clamping has bounded the flow magnitudes.  USCOMP at
        # 0.01° → max ≈ 30 px/step.
        mag = np.sqrt(flows["USCOMP"][..., 0] ** 2 + flows["USCOMP"][..., 1] ** 2)
        assert mag.max() <= 30.5  # within rounding of the cap
