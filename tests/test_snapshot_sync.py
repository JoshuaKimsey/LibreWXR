# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Snapshot sync: the pipeline/renderer handoff carried over an artifact store.

Uses a ``file://`` store so the round trip runs offline; the same code
path serves ``s3://`` through fsspec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from librewxr.data.master_state import load_state, snapshot_state, write_state_snapshot
from librewxr.data.snapshot_sync import (
    LIGHTNING_ARTIFACT,
    SnapshotPublisher,
    SnapshotPuller,
    rewrite_directories,
    snapshot_files,
)
from librewxr.data.store import FrameStore, RadarFrame

T0 = 1_793_650_200


def _frame(ts: int, value: int) -> RadarFrame:
    return RadarFrame(timestamp=ts, regions={"GUCOMP": np.full((20, 20), value, np.uint8)})


async def _publish(store: FrameStore, cache: Path, publisher: SnapshotPublisher) -> dict:
    payload = snapshot_state({"frame_store": store})
    write_state_snapshot(payload, cache)
    manifest = publisher.publish(payload)
    assert manifest is not None
    return manifest


@pytest.fixture
def pipeline_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "pipeline"
    (cache / "coverage_masks").mkdir(parents=True)
    (cache / "coverage_masks" / "masks.json").write_text("{}")
    (cache / "lightning").mkdir()
    (cache / "lightning" / "current.npz").write_bytes(b"strikes-v1")
    return cache


async def test_publish_then_pull_rebuilds_the_snapshot_on_another_node(
    tmp_path: Path, pipeline_cache: Path
):
    url = f"file://{tmp_path / 'store'}"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    await store.add_frame(_frame(T0 + 600, 2))
    publisher = SnapshotPublisher(pipeline_cache, url)
    manifest = await _publish(store, pipeline_cache, publisher)
    assert manifest["sequence"] == 1
    assert set(manifest["files"]) == {
        f"radar/{T0}_GUCOMP.dat",
        f"radar/{T0 + 600}_GUCOMP.dat",
        "coverage_masks/masks.json",
        LIGHTNING_ARTIFACT,
    }

    render_cache = tmp_path / "render"
    puller = SnapshotPuller(render_cache, url)
    assert puller.pull()
    assert not puller.pull()  # nothing newer

    # The snapshot now points at the render node's own cache, and a render
    # worker's store can open the frames from it.
    state = load_state(render_cache)
    assert state["stores"]["frame_store"]["memmap_dir"] == str(render_cache / "radar")
    render_store = FrameStore(max_frames=2, cache_dir=render_cache)
    render_store.__setstate__(state["stores"]["frame_store"])
    frame = await render_store.get_frame(T0 + 600)
    assert frame is not None and int(frame.regions["GUCOMP"][0, 0]) == 2
    assert (render_cache / "coverage_masks" / "masks.json").read_text() == "{}"
    assert (render_cache / LIGHTNING_ARTIFACT).read_bytes() == b"strikes-v1"


async def test_only_changed_files_travel_and_evicted_files_go_after_one_cycle(
    tmp_path: Path, pipeline_cache: Path
):
    url = f"file://{tmp_path / 'store'}"
    remote = tmp_path / "store"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    await store.add_frame(_frame(T0 + 600, 2))
    publisher = SnapshotPublisher(pipeline_cache, url)
    await _publish(store, pipeline_cache, publisher)
    render_cache = tmp_path / "render"
    puller = SnapshotPuller(render_cache, url)
    assert puller.pull()

    # A new frame evicts the oldest: one upload, the evicted file kept one cycle
    await store.add_frame(_frame(T0 + 1200, 3))
    manifest = await _publish(store, pipeline_cache, publisher)
    assert manifest["stale"] == [f"radar/{T0}_GUCOMP.dat"]
    assert (remote / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert puller.pull()
    assert (render_cache / "radar" / f"{T0 + 1200}_GUCOMP.dat").exists()
    assert (render_cache / "radar" / f"{T0}_GUCOMP.dat").exists()

    # The cycle after that, it is gone from the store and the render node
    await store.add_frame(_frame(T0 + 1800, 4))
    await _publish(store, pipeline_cache, publisher)
    assert not (remote / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert len(list((remote / "snapshots").iterdir())) == 2
    assert puller.pull()
    assert not (render_cache / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert (render_cache / "radar" / f"{T0 + 600}_GUCOMP.dat").exists()


async def test_lightning_artifact_follows_the_store_between_snapshots(
    tmp_path: Path, pipeline_cache: Path
):
    url = f"file://{tmp_path / 'store'}"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    publisher = SnapshotPublisher(pipeline_cache, url)
    await _publish(store, pipeline_cache, publisher)
    render_cache = tmp_path / "render"
    puller = SnapshotPuller(render_cache, url)
    assert puller.pull()

    (pipeline_cache / LIGHTNING_ARTIFACT).write_bytes(b"strikes-v2-longer")
    assert publisher.publish_lightning()
    assert not puller.pull()  # no new snapshot, but the artifact is refreshed
    assert (render_cache / LIGHTNING_ARTIFACT).read_bytes() == b"strikes-v2-longer"


def test_rewrite_directories_maps_cache_and_volatile_dirs(tmp_path: Path):
    payload = {
        "stores": {
            "frame_store": {"memmap_dir": "/srv/cache/radar", "frames": []},
            "nowcast_store": {"memmap_dir": "/dev/shm/librewxr/nowcast"},
            "gmgsi_lw_grid": {"cache_root": "/srv/cache", "channel": "lw"},
            "ecmwf_grid": {"memmap_dir": "/elsewhere/ecmwf"},
        }
    }
    out = rewrite_directories(payload, "/srv/cache", "/dev/shm/librewxr", tmp_path)
    stores = out["stores"]
    assert stores["frame_store"]["memmap_dir"] == str(tmp_path / "radar")
    assert stores["nowcast_store"]["memmap_dir"] == str(tmp_path / "volatile" / "nowcast")
    assert stores["gmgsi_lw_grid"]["cache_root"] == str(tmp_path)
    assert stores["ecmwf_grid"]["memmap_dir"] == "/elsewhere/ecmwf"
    assert payload["stores"]["frame_store"]["memmap_dir"] == "/srv/cache/radar"


def test_snapshot_files_skips_files_outside_the_cache(tmp_path: Path, caplog):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.dat").write_bytes(b"0")
    payload = {"stores": {"s": {"memmap_dir": str(outside), "frames": [["x.dat", "|u1", [1]]]}}}
    assert snapshot_files(payload, tmp_path / "cache", None) == {}
    assert "outside the cache" in caplog.text
