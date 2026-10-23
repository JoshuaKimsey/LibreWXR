# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the guarded memmap write path (GitHub issue #43).

No network, no real tmpfs: the space probe is monkeypatched where a
shortfall is needed, and the roundtrip uses a tmp_path directory.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from librewxr.data import memmap_guard
from librewxr.data.memmap_guard import atomic_memmap_write, ensure_space


class _Stat:
    """Minimal ``os.statvfs`` result."""

    def __init__(self, free: int, block: int = 4096) -> None:
        self.f_bavail = free
        self.f_frsize = block


class TestEnsureSpace:
    def test_healthy_directory_passes(self, tmp_path):
        ensure_space(tmp_path, 1024, what="test frame")

    def test_low_space_raises_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            memmap_guard.os, "statvfs", lambda path: _Stat(free=1, block=4096)
        )

        with pytest.raises(RuntimeError) as excinfo:
            ensure_space(tmp_path, 10_000_000, what="RRQPE scan 1785130200")

        message = str(excinfo.value)
        # what, absolute directory, free bytes, needed bytes (10 MB * 1.1)
        assert "RRQPE scan 1785130200" in message
        assert str(Path(tmp_path).resolve()) in message
        assert "4096 bytes free" in message
        assert "11000000 bytes needed" in message
        # actionable guidance
        assert "shm_size" in message
        assert "bind a host tmpfs" in message
        assert "volatile cache dir" in message

    def test_unstattable_directory_raises(self, tmp_path, monkeypatch):
        def _boom(path):
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(memmap_guard.os, "statvfs", _boom)

        with pytest.raises(RuntimeError, match="pre-flight"):
            ensure_space(tmp_path, 1024, what="test frame")


class TestAtomicMemmapWrite:
    def test_roundtrip_leaves_no_tmp_residue(self, tmp_path):
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        final = tmp_path / "frame.dat"

        mm = atomic_memmap_write(final, arr, what="test frame")

        try:
            assert isinstance(mm, np.memmap)
            assert mm.mode == "r"
            assert mm.shape == (3, 4)
            assert mm.dtype == np.uint8
            assert np.array_equal(np.asarray(mm), arr)
        finally:
            del mm
        assert final.exists()
        assert final.stat().st_size == arr.nbytes
        assert list(tmp_path.glob("*.tmp")) == []

    def test_tmp_name_carries_pid_and_uuid(self, tmp_path, monkeypatch):
        arr = np.zeros((2, 2), dtype=np.uint8)
        final = tmp_path / "frame.dat"
        seen: list[Path] = []
        real_memmap = np.memmap

        def _spy(path, *args, **kwargs):
            seen.append(Path(path))
            return real_memmap(path, *args, **kwargs)

        monkeypatch.setattr(memmap_guard.np, "memmap", _spy)
        mm = atomic_memmap_write(final, arr, what="test frame")
        del mm

        assert seen, "memmap was never called"
        tmp_used = seen[0]
        assert tmp_used.name.startswith("frame.dat.")
        assert tmp_used.name.endswith(".tmp")
        # <final>.<pid>.<uuid hex>.tmp
        pid_part, uuid_part = tmp_used.name.split(".")[-3:-1]
        assert pid_part == str(os.getpid())
        assert len(uuid_part) == 32
        int(uuid_part, 16)  # hex digest
        assert not tmp_used.exists()  # replaced into place
        assert final.exists()

    def test_no_space_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            memmap_guard.os, "statvfs", lambda path: _Stat(free=1, block=1)
        )
        final = tmp_path / "frame.dat"

        with pytest.raises(RuntimeError, match="not enough space"):
            atomic_memmap_write(
                final, np.zeros((4, 4), dtype=np.uint8), what="test frame"
            )

        assert not final.exists()
        assert list(tmp_path.iterdir()) == []
