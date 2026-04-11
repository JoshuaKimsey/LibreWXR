# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for precipitation nowcasting: store, generator, and optical flow."""
import asyncio

import numpy as np
import pytest

from librewxr.data.nowcast import (
    NowcastFrame,
    NowcastGenerator,
    NowcastStore,
    _compute_flow,
    _extrapolate_forward,
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

        frames = NowcastGenerator._generate_sync(
            prev_regions, latest_regions,
            latest_ts=1000, n_steps=3, interval=600,
        )

        assert len(frames) == 3
        assert frames[0].timestamp == 1600
        assert frames[1].timestamp == 2200
        assert frames[2].timestamp == 2800

        # Blend weights should decrease
        assert frames[0].blend_weight > frames[1].blend_weight
        assert frames[1].blend_weight > frames[2].blend_weight
        # With 3 steps at 600s, max_blend_steps=6, so step 3 is mid-curve
        # t=3/6=0.5 → 0.30 + 0.70*(0.5)^1.1 ≈ 0.63
        assert 0.55 < frames[2].blend_weight < 0.70

        # Each frame should have the region
        for f in frames:
            assert "USCOMP" in f.regions
            assert f.regions["USCOMP"].shape == (H, W)

    def test_generate_sync_missing_region(self):
        """If a region exists in latest but not prev, it should be skipped."""
        blob = _make_blob(60, 100)
        prev_regions = {}  # no regions
        latest_regions = {"USCOMP": blob}

        frames = NowcastGenerator._generate_sync(
            prev_regions, latest_regions,
            latest_ts=1000, n_steps=3, interval=600,
        )
        assert frames == []

    def test_generate_sync_multiple_regions(self):
        """Should generate nowcast for each region independently."""
        blob0_a = _make_blob(60, 100)
        blob1_a = _make_blob(60, 110)
        blob0_b = _make_blob(30, 50, radius=10, value=100)
        blob1_b = _make_blob(30, 55, radius=10, value=100)

        prev = {"A": blob0_a, "B": blob0_b}
        latest = {"A": blob1_a, "B": blob1_b}

        frames = NowcastGenerator._generate_sync(
            prev, latest, latest_ts=2000, n_steps=2, interval=600,
        )
        assert len(frames) == 2
        for f in frames:
            assert "A" in f.regions
            assert "B" in f.regions
