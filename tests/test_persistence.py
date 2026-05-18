# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Round-trip tests for ``__getstate__`` / ``__setstate__`` on all stores.

Phase 1 of the multi-worker rollout adds persistent ``cache_dir`` mode and
JSON-serialisable state snapshots to every store.  These tests verify the
round-trip: dump → ``json.dumps`` → ``json.loads`` → setstate → read back
the same data via the public interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from librewxr.sources.regional.caribbean.nwp.arome_antilles.grid import AROMEAntillesGrid
from librewxr.data.cloud_grid import CloudGrid
from librewxr.sources.regional.europe.nwp.dmi_dini.grid import DMIDiniGrid
from librewxr.sources.world.ifs.grid import ECMWFGrid
from librewxr.sources.regional.north_america.canada.nwp.hrdps.grid import HRDPSGrid
from librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska.grid import HRRRAlaskaGrid
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import HRRRGrid
from librewxr.sources.regional.europe.nwp.icon_eu.grid import ICONEUGrid
from librewxr.data.nowcast import NowcastFrame, NowcastStore
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.sources.regional.south_america.nwp.wrf_smn.grid import WRFSMNGrid

pytestmark = pytest.mark.store


def _roundtrip(state: dict) -> dict:
    """Force the snapshot through JSON to confirm it is serialisable."""
    return json.loads(json.dumps(state))


# ──────────────────────────────────────────────────────────────────────────
# FrameStore — populated round-trip
# ──────────────────────────────────────────────────────────────────────────


class TestFrameStorePersistence:
    @pytest.mark.asyncio
    async def test_roundtrip_preserves_frames(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        producer = FrameStore(max_frames=4, cache_dir=cache)
        arr = np.full((10, 20), 42, dtype=np.uint8)
        await producer.add_frame(RadarFrame(timestamp=1700000000, regions={"REGA": arr}))

        snapshot = _roundtrip(producer.__getstate__())
        consumer = FrameStore(max_frames=4)  # placeholder before restore
        consumer.__setstate__(snapshot)

        timestamps = await consumer.get_timestamps()
        assert timestamps == [1700000000]
        frame = await consumer.get_frame(1700000000)
        assert frame is not None
        assert "REGA" in frame.regions
        np.testing.assert_array_equal(frame.regions["REGA"], arr)

    def test_cache_dir_is_persistent(self, tmp_path: Path) -> None:
        store = FrameStore(cache_dir=tmp_path)
        assert store._persistent is True
        assert store._memmap_dir == tmp_path / "radar"

    def test_no_cache_dir_uses_tempdir(self) -> None:
        store = FrameStore()
        try:
            assert store._persistent is False
        finally:
            store.cleanup()


# ──────────────────────────────────────────────────────────────────────────
# NowcastStore — populated round-trip
# ──────────────────────────────────────────────────────────────────────────


class TestNowcastStorePersistence:
    @pytest.mark.asyncio
    async def test_roundtrip_preserves_frames_and_flows(self, tmp_path: Path) -> None:
        producer = NowcastStore(cache_dir=tmp_path)
        radar = np.full((4, 6), 17, dtype=np.uint8)
        flow = np.full((4, 6, 2), 1.5, dtype=np.float32)
        frame = NowcastFrame(timestamp=1700000600, blend_weight=0.6, regions={"R1": radar})
        await producer.replace_all([frame])
        await producer.replace_flows({"R1": flow})

        snapshot = _roundtrip(producer.__getstate__())
        consumer = NowcastStore()
        consumer.__setstate__(snapshot)

        ts_list = await consumer.get_timestamps()
        assert ts_list == [1700000600]
        nc_frame, weight = await consumer.get_frame(1700000600)
        assert nc_frame is not None
        assert weight == pytest.approx(0.6)
        np.testing.assert_array_equal(nc_frame.regions["R1"], radar)
        flows = await consumer.get_flows()
        np.testing.assert_array_equal(flows["R1"], flow)


# ──────────────────────────────────────────────────────────────────────────
# Empty round-trips for stores that need real fetches to populate
# ──────────────────────────────────────────────────────────────────────────


class TestECMWFGridPersistence:
    def test_empty_roundtrip(self, tmp_path: Path) -> None:
        producer = ECMWFGrid(cache_dir=tmp_path)
        snapshot = _roundtrip(producer.__getstate__())
        consumer = ECMWFGrid(cache_dir=tmp_path)
        consumer.__setstate__(snapshot)
        assert consumer.timestep_count == 0
        assert consumer.flow is None
        assert consumer.reference_time is None

    def test_populated_roundtrip(self, tmp_path: Path) -> None:
        producer = ECMWFGrid(cache_dir=tmp_path)
        precip = np.zeros((1801, 3600), dtype=np.uint8)
        snow = np.zeros((1801, 3600), dtype=bool)
        ts = 1700000000
        producer._timesteps[ts] = (
            producer._to_memmap(f"{ts}_precip", precip),
            producer._to_memmap(f"{ts}_snow", snow),
        )
        producer._sorted_timestamps = [ts]
        producer._reference_time = "2023-11-14T00:00:00Z"

        snapshot = _roundtrip(producer.__getstate__())
        consumer = ECMWFGrid(cache_dir=tmp_path)
        consumer.__setstate__(snapshot)
        assert consumer.timestep_count == 1
        assert consumer.reference_time == "2023-11-14T00:00:00Z"
        # Memory-mapped arrays should be readable
        assert consumer.data is not None
        assert consumer.data.shape == (1801, 3600)


class TestCloudGridPersistence:
    def test_empty_roundtrip(self, tmp_path: Path) -> None:
        producer = CloudGrid(cache_dir=tmp_path)
        snapshot = _roundtrip(producer.__getstate__())
        consumer = CloudGrid(cache_dir=tmp_path)
        consumer.__setstate__(snapshot)
        assert consumer.timestep_count == 0


# ──────────────────────────────────────────────────────────────────────────
# NWP grids — empty + faked-frame round-trips
# ──────────────────────────────────────────────────────────────────────────


def _write_fake_frame(memmap_dir: Path, run_ts: int, lead: int,
                      shape: tuple[int, int]) -> None:
    """Drop a synthetic ``r{run}_l{lead}.dat`` file in the memmap directory."""
    arr = np.zeros(shape, dtype=np.uint8)
    path = memmap_dir / f"r{run_ts}_l{lead}.dat"
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    mm[:] = arr
    mm.flush()
    del mm


@pytest.mark.parametrize(
    "cls,subdir,shape",
    [
        (HRRRGrid, "hrrr", None),  # shape inferred — see below
        (HRRRAlaskaGrid, "hrrr_alaska", None),
        (HRDPSGrid, "hrdps", None),
        (AROMEAntillesGrid, "arome_antilles", None),
        (WRFSMNGrid, "wrf_smn", None),
        (ICONEUGrid, "icon_eu", None),
        (DMIDiniGrid, "dmi_dini", None),
    ],
)
def test_nwp_grid_empty_roundtrip(tmp_path: Path, cls, subdir, shape) -> None:
    producer = cls(cache_dir=tmp_path)
    snapshot = _roundtrip(producer.__getstate__())
    assert snapshot["memmap_dir"].endswith(subdir)
    consumer = cls(cache_dir=tmp_path)
    consumer.__setstate__(snapshot)
    assert consumer.frame_count == 0


@pytest.mark.parametrize(
    "cls,subdir,height_attr,width_attr",
    [
        (HRRRGrid, "hrrr", "HRRR_GRID_HEIGHT", "HRRR_GRID_WIDTH"),
        (HRRRAlaskaGrid, "hrrr_alaska", "HRRR_AK_GRID_HEIGHT", "HRRR_AK_GRID_WIDTH"),
        (ICONEUGrid, "icon_eu", "ICON_EU_GRID_HEIGHT", "ICON_EU_GRID_WIDTH"),
        (DMIDiniGrid, "dmi_dini", "DMI_DINI_GRID_HEIGHT", "DMI_DINI_GRID_WIDTH"),
    ],
)
def test_nwp_grid_populated_roundtrip(
    tmp_path: Path, cls, subdir, height_attr, width_attr,
) -> None:
    """Fake a single cached frame on disk, then verify __setstate__ picks it up."""
    import importlib

    module = importlib.import_module(cls.__module__)
    h = getattr(module, height_attr)
    w = getattr(module, width_attr)

    # Construct once to lay out the directory, then drop a synthetic frame.
    producer = cls(cache_dir=tmp_path)
    run_ts = 1700000000
    lead = 3600
    _write_fake_frame(producer._memmap_dir, run_ts, lead, (h, w))

    snapshot = _roundtrip(producer.__getstate__())
    consumer = cls(cache_dir=tmp_path)
    consumer.__setstate__(snapshot)
    # The on-disk fake frame must be visible after rescan
    assert consumer.frame_count >= 1
    assert (run_ts, lead) in consumer._frames
