# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Equality guards for ``FrameStore.add_frame`` (issue #36).

Re-fetch merges and carry-forward rebuild the same frames every cycle;
these tests pin the byte-identity fast paths that keep identical content
from being rewritten to disk: merging an unchanged region, hardlinking a
carry-forward memmap of a sibling slot, and reopening a memmap that is
already the exact target file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from librewxr.data.store import FrameStore, RadarFrame

pytestmark = pytest.mark.store

_TS = 1700000000


def _read_memmap(path: Path, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(np.memmap(path, dtype=np.uint8, mode="r", shape=shape))


class TestMergeEqualityGuard:
    async def test_identical_merge_is_noop(self, tmp_path: Path) -> None:
        """A byte-identical merge writes nothing and does not bump the version."""
        store = FrameStore(max_frames=4, cache_dir=tmp_path / "cache")
        arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
        await store.add_frame(RadarFrame(timestamp=_TS, regions={"A": arr}))

        target = store._memmap_dir / f"{_TS}_A.dat"
        st_before = target.stat()
        version_before = store.frame_version(_TS)

        evicted, merged = await store.add_frame(
            RadarFrame(timestamp=_TS, regions={"A": arr.copy()}),
        )

        assert evicted is None
        assert merged is False
        st_after = target.stat()
        # os.replace always installs a fresh inode, so an unchanged inode is
        # the definitive proof that no write happened.
        assert st_after.st_ino == st_before.st_ino
        assert st_after.st_mtime_ns == st_before.st_mtime_ns
        assert store.frame_version(_TS) == version_before

    async def test_changed_merge_bumps_and_rewrites(self, tmp_path: Path) -> None:
        """A differing merge bumps the version and replaces the file."""
        store = FrameStore(max_frames=4, cache_dir=tmp_path / "cache")
        await store.add_frame(
            RadarFrame(timestamp=_TS, regions={"A": np.zeros((8, 8), np.uint8)}),
        )
        target = store._memmap_dir / f"{_TS}_A.dat"
        ino_before = target.stat().st_ino
        version_before = store.frame_version(_TS)

        changed = np.ones((8, 8), dtype=np.uint8)
        evicted, merged = await store.add_frame(
            RadarFrame(timestamp=_TS, regions={"A": changed}),
        )

        assert evicted is None
        assert merged is True
        assert store.frame_version(_TS) == version_before + 1
        assert target.stat().st_ino != ino_before
        np.testing.assert_array_equal(_read_memmap(target, (8, 8)), changed)
        frame = await store.get_frame(_TS)
        np.testing.assert_array_equal(
            np.asarray(frame.regions["A"]), changed,
        )


class TestCarryForwardHardlink:
    async def test_sibling_memmap_is_hardlinked(self, tmp_path: Path) -> None:
        """Appending a frame from a sibling slot memmap hardlinks, not copies."""
        store = FrameStore(max_frames=4, cache_dir=tmp_path / "cache")
        src_arr = np.full((8, 8), 7, dtype=np.uint8)
        await store.add_frame(RadarFrame(timestamp=_TS, regions={"A": src_arr}))
        src_memmap = (await store.get_frame(_TS)).regions["A"]

        dst_ts = _TS + 600
        evicted, merged = await store.add_frame(
            RadarFrame(timestamp=dst_ts, regions={"A": src_memmap}),
        )

        assert evicted is None
        assert merged is False
        src_path = store._memmap_dir / f"{_TS}_A.dat"
        dst_path = store._memmap_dir / f"{dst_ts}_A.dat"
        assert dst_path.stat().st_ino == src_path.stat().st_ino
        np.testing.assert_array_equal(_read_memmap(dst_path, (8, 8)), src_arr)

    async def test_differing_merge_breaks_link_leaving_source_intact(
        self, tmp_path: Path,
    ) -> None:
        """A rewrite of a hardlinked slot must not touch the source slot."""
        store = FrameStore(max_frames=4, cache_dir=tmp_path / "cache")
        src_arr = np.full((8, 8), 7, dtype=np.uint8)
        await store.add_frame(RadarFrame(timestamp=_TS, regions={"A": src_arr}))
        src_memmap = (await store.get_frame(_TS)).regions["A"]

        dst_ts = _TS + 600
        await store.add_frame(
            RadarFrame(timestamp=dst_ts, regions={"A": src_memmap}),
        )
        src_path = store._memmap_dir / f"{_TS}_A.dat"
        dst_path = store._memmap_dir / f"{dst_ts}_A.dat"
        assert dst_path.stat().st_ino == src_path.stat().st_ino

        changed = np.full((8, 8), 9, dtype=np.uint8)
        _, merged = await store.add_frame(
            RadarFrame(timestamp=dst_ts, regions={"A": changed}),
        )

        assert merged is True
        assert dst_path.stat().st_ino != src_path.stat().st_ino
        # Source slot keeps its original bytes; destination gets the new data.
        np.testing.assert_array_equal(_read_memmap(src_path, (8, 8)), src_arr)
        np.testing.assert_array_equal(_read_memmap(dst_path, (8, 8)), changed)


class TestExactTargetMemmap:
    async def test_target_memmap_is_reopened_not_rewritten(
        self, tmp_path: Path,
    ) -> None:
        """Data already mapped from the target file (boot-restore) is not written."""
        store = FrameStore(max_frames=4, cache_dir=tmp_path / "cache")
        target = store._memmap_dir / f"{_TS}_A.dat"
        expected = np.full((4, 4), 5, dtype=np.uint8)
        mm = np.memmap(target, dtype=np.uint8, mode="w+", shape=(4, 4))
        mm[:] = expected
        mm.flush()
        del mm

        data = np.memmap(target, dtype=np.uint8, mode="r", shape=(4, 4))
        st_before = target.stat()
        evicted, merged = await store.add_frame(
            RadarFrame(timestamp=_TS, regions={"A": data}),
        )

        assert evicted is None
        assert merged is False
        st_after = target.stat()
        assert st_after.st_ino == st_before.st_ino
        assert st_after.st_mtime_ns == st_before.st_mtime_ns
        frame = await store.get_frame(_TS)
        np.testing.assert_array_equal(np.asarray(frame.regions["A"]), expected)
