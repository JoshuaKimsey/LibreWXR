# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Snapshot sync: the pipeline/renderer handoff carried over an artifact store.

Uses a ``file://`` store so the round trip runs offline; the same code
path serves ``s3://`` through fsspec.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from librewxr.data.master_state import load_state, snapshot_state, write_state_snapshot
from librewxr.data.snapshot_sync import (
    LIGHTNING_ARTIFACT,
    MANIFEST_VERSION,
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
    publisher = SnapshotPublisher(pipeline_cache, url, stale_grace_s=0)
    await _publish(store, pipeline_cache, publisher)
    render_cache = tmp_path / "render"
    puller = SnapshotPuller(render_cache, url)
    assert puller.pull()

    # A new frame evicts the oldest: one upload, the evicted file kept until
    # the grace has passed (here: the next publish)
    await store.add_frame(_frame(T0 + 1200, 3))
    manifest = await _publish(store, pipeline_cache, publisher)
    assert list(manifest["stale"]) == [f"radar/{T0}_GUCOMP.dat"]
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


def test_snapshot_files_cover_grid_directories_that_derive_their_file_names(tmp_path: Path):
    """The regional NWP grids snapshot ``[run, lead]`` keys, not file triples,
    so their files must come from the directory the store names."""
    cache = tmp_path / "cache"
    grid_dir = cache / "hrrr"
    grid_dir.mkdir(parents=True)
    (grid_dir / "1793650200_0.dat").write_bytes(b"0")
    (grid_dir / "1793650200_1.dat").write_bytes(b"0")
    (grid_dir / "1793650200_2.dat.123.tmp").write_bytes(b"0")
    payload = {
        "stores": {
            "hrrr_grid": {
                "memmap_dir": str(grid_dir),
                "latest_run_ts": 1793650200,
                "frame_keys": [[1793650200, 0], [1793650200, 1]],
            }
        }
    }
    assert set(snapshot_files(payload, cache, None)) == {
        "hrrr/1793650200_0.dat",
        "hrrr/1793650200_1.dat",
    }


async def test_stale_files_survive_quick_republishes_within_the_grace(
    tmp_path: Path, pipeline_cache: Path
):
    """The pipeline publishes several snapshots per cycle, so a file a reader
    is still downloading must not disappear on the very next publish."""
    url = f"file://{tmp_path / 'store'}"
    remote = tmp_path / "store"
    store = FrameStore(max_frames=1, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    publisher = SnapshotPublisher(pipeline_cache, url, stale_grace_s=3600)
    await _publish(store, pipeline_cache, publisher)
    for k in (1, 2, 3):
        await store.add_frame(_frame(T0 + 600 * k, k))
        manifest = await _publish(store, pipeline_cache, publisher)
    assert (remote / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert set(manifest["stale"]) == {
        f"radar/{T0 + 600 * k}_GUCOMP.dat" for k in (0, 1, 2)
    }
    assert len(manifest["stale_snapshots"]) == 3


async def test_file_removed_during_publish_still_reaches_the_store(
    tmp_path: Path, pipeline_cache: Path, monkeypatch
):
    """The pipeline keeps writing while a publish runs (a nowcast regeneration
    deletes the previous frames); the publish must still upload the content
    the snapshot described, from its own hard link."""
    url = f"file://{tmp_path / 'store'}"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    await store.add_frame(_frame(T0 + 600, 2))
    publisher = SnapshotPublisher(pipeline_cache, url)
    payload = snapshot_state({"frame_store": store})
    doomed = f"radar/{T0}_GUCOMP.dat"
    real_put = publisher._fs.put_file

    def put_file(lpath, rpath, **kwargs):
        if rpath.endswith(doomed):
            (pipeline_cache / doomed).unlink()
        return real_put(lpath, rpath, **kwargs)

    # fsspec caches filesystem instances: patch through monkeypatch so the
    # shared file:// instance is restored for the other tests
    monkeypatch.setattr(publisher._fs, "put_file", put_file)
    manifest = publisher.publish(payload)
    assert manifest is not None
    # The publish read its own hard link, so the deleted original still
    # reached the store with the content the snapshot described
    assert doomed in manifest["files"]
    assert (tmp_path / "store" / doomed).read_bytes() == b"\x01" * 400
    assert f"radar/{T0 + 600}_GUCOMP.dat" in manifest["files"]
    assert not list((pipeline_cache / "radar").glob(".publish-*"))


async def test_file_gone_before_staging_is_left_out_of_the_manifest(
    tmp_path: Path, pipeline_cache: Path, caplog
):
    """A file that is already gone when the publish starts (no link possible)
    must not fail the publish; the next snapshot no longer names it."""
    url = f"file://{tmp_path / 'store'}"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    await store.add_frame(_frame(T0 + 600, 2))
    payload = snapshot_state({"frame_store": store})
    publisher = SnapshotPublisher(pipeline_cache, url)
    original_stage = publisher._stage

    def stage(files, sequence):
        (pipeline_cache / f"radar/{T0}_GUCOMP.dat").unlink()
        return original_stage(files, sequence)

    publisher._stage = stage
    manifest = publisher.publish(payload)
    assert manifest is not None
    assert f"radar/{T0}_GUCOMP.dat" not in manifest["files"]
    assert f"radar/{T0 + 600}_GUCOMP.dat" in manifest["files"]


def test_publisher_removes_link_directories_a_crash_left_behind(
    tmp_path: Path, pipeline_cache: Path
):
    """A publish interrupted mid-upload leaves its hard links behind, pinning
    deleted frames' disk space; the next publisher start must reclaim them."""
    stale = pipeline_cache / "radar" / ".publish-7"
    stale.mkdir(parents=True)
    (stale / "old_frame.dat").write_bytes(b"x")
    SnapshotPublisher(pipeline_cache, f"file://{tmp_path / 'store'}")
    assert not stale.exists()
    assert (pipeline_cache / "radar").is_dir()


async def test_failed_delete_is_retried_on_the_next_publish(
    tmp_path: Path, pipeline_cache: Path, caplog, monkeypatch
):
    """A multi-object delete can time out on a slow store; the files must stay
    listed as stale so a later publish deletes them, never leak."""
    url = f"file://{tmp_path / 'store'}"
    remote = tmp_path / "store"
    store = FrameStore(max_frames=1, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    publisher = SnapshotPublisher(pipeline_cache, url, stale_grace_s=0)
    await _publish(store, pipeline_cache, publisher)
    await store.add_frame(_frame(T0 + 600, 2))
    await _publish(store, pipeline_cache, publisher)

    def failing_rm(paths):
        raise TimeoutError("slow store")

    with monkeypatch.context() as patched:
        patched.setattr(publisher._fs, "rm", failing_rm)
        await store.add_frame(_frame(T0 + 1200, 3))
        manifest = await _publish(store, pipeline_cache, publisher)
    assert f"radar/{T0}_GUCOMP.dat" in manifest["stale"]
    assert (remote / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert "retrying next publish" in caplog.text

    await store.add_frame(_frame(T0 + 1800, 4))
    manifest = await _publish(store, pipeline_cache, publisher)
    assert f"radar/{T0}_GUCOMP.dat" not in manifest["stale"]
    assert not (remote / "radar" / f"{T0}_GUCOMP.dat").exists()


async def test_manifest_of_another_version_is_replaced_not_read(
    tmp_path: Path, pipeline_cache: Path, caplog
):
    """A manifest written by a build with a different layout must not be
    interpreted: the publisher starts over and the puller refuses it."""
    import json

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "sequence": 9, "files": {}, "stale": ["x"]})
    )
    url = f"file://{store_dir}"
    with pytest.raises(ValueError, match="manifest version"):
        SnapshotPuller(tmp_path / "render", url).pull()

    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    manifest = await _publish(store, pipeline_cache, SnapshotPublisher(pipeline_cache, url))
    assert manifest["version"] == MANIFEST_VERSION and manifest["sequence"] == 1
    assert "Ignoring manifest of version 1" in caplog.text
    assert SnapshotPuller(tmp_path / "render", url).pull()


async def test_puller_follows_a_publisher_that_started_a_new_series(
    tmp_path: Path, pipeline_cache: Path
):
    """A publisher that starts over (fresh store, or a manifest of another
    version it ignored) numbers its snapshots from 1 again; a puller that had
    applied a higher sequence must still take them."""
    url = f"file://{tmp_path / 'store'}"
    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    publisher = SnapshotPublisher(pipeline_cache, url)
    for k in (1, 2, 3):
        await store.add_frame(_frame(T0 + 600 * k, k + 1))
        await _publish(store, pipeline_cache, publisher)
    puller = SnapshotPuller(tmp_path / "render", url)
    assert puller.pull() and puller.applied_sequence == 3

    # The store is wiped and a new publisher starts a new series at 1
    shutil.rmtree(tmp_path / "store")
    fresh = SnapshotPublisher(pipeline_cache, url)
    manifest = await _publish(store, pipeline_cache, fresh)
    assert manifest["sequence"] == 1
    assert puller.pull() and puller.applied_sequence == 1
    assert not puller.pull()


async def test_new_series_reclaims_files_an_earlier_series_left(
    tmp_path: Path, pipeline_cache: Path
):
    """A new series (no manifest it can continue, e.g. after a manifest
    version bump) must expire what the store already holds instead of
    leaving it there for ever."""
    store_dir = tmp_path / "store"
    (store_dir / "radar").mkdir(parents=True)
    orphan = store_dir / "radar" / f"{T0 - 6000}_GUCOMP.dat"
    orphan.write_bytes(b"old frame")
    old_snapshot = store_dir / "snapshots" / "1-1.json.gz"
    old_snapshot.parent.mkdir()
    old_snapshot.write_bytes(b"old snapshot")
    url = f"file://{store_dir}"

    store = FrameStore(max_frames=2, cache_dir=pipeline_cache)
    await store.add_frame(_frame(T0, 1))
    publisher = SnapshotPublisher(pipeline_cache, url, stale_grace_s=0)
    manifest = await _publish(store, pipeline_cache, publisher)
    # Kept through the first publish: a reader may still be on the old series
    assert orphan.exists() and old_snapshot.exists()
    assert f"radar/{T0 - 6000}_GUCOMP.dat" in manifest["stale"]
    assert f"radar/{T0}_GUCOMP.dat" not in manifest["stale"]

    await _publish(store, pipeline_cache, publisher)
    assert not orphan.exists() and not old_snapshot.exists()
    assert (store_dir / "radar" / f"{T0}_GUCOMP.dat").exists()
    assert SnapshotPuller(tmp_path / "render", url).pull()
