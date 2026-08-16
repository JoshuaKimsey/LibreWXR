# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the pipeline page-cache priming (``data/pagecache.py``).

Cover the snapshot-payload walker (memmap_dir + basename pairing,
satellite-style cache_root/channel resolution, noise filtering, dedupe)
and the end-to-end prime path (mtime dedupe, corrupt-record recovery,
missing-file tolerance).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from librewxr.data import pagecache


def test_collect_memmap_files_frame_store():
    payload = {
        "memmap_dir": "/tmp/radar",
        "frames": [
            {
                "timestamp": 1,
                "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]},
            }
        ],
        "max_frames": 12,
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/radar/1_USCOMP.dat"),
    }


def test_collect_memmap_files_nwp():
    payload = {
        "memmap_dir": "/tmp/ecmwf_ifs",
        "timesteps": {
            "1700000000": {
                "precip": ["1700000000_precip.dat", "|u1", [4, 4]],
                "snow": ["1700000000_snow.dat", "|u1", [4, 4]],
            }
        },
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/ecmwf_ifs/1700000000_precip.dat"),
        Path("/tmp/ecmwf_ifs/1700000000_snow.dat"),
    }


def test_collect_memmap_files_ignores_non_memmap_and_dedupes():
    payload = {
        "memmap_dir": "/tmp/radar",
        "shapes": [4, 4],  # shape list: first element is not a str
        "names": ["USCOMP"],  # plain string, not a .dat basename
        "frames": [
            {
                "timestamp": 1,
                "regions": {
                    "USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]],
                    "RRQPE": ["1_RRQPE.dat", "|u1", [4, 4]],
                },
            },
            {
                # Same basename again — must be deduped away.
                "timestamp": 2,
                "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]},
            },
        ],
        "max_frames": 12,
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/radar/1_USCOMP.dat"),
        Path("/tmp/radar/1_RRQPE.dat"),
    }


def test_prime_fresh_memmaps_end_to_end(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    radar_dir = cache_dir / "radar"
    radar_dir.mkdir(parents=True)
    f1 = radar_dir / "1_USCOMP.dat"
    f2 = radar_dir / "2_USCOMP.dat"
    f1.write_bytes(b"x" * 16)
    f2.write_bytes(b"y" * 32)
    payload = {
        "stores": {
            "frame_store": {
                "memmap_dir": str(radar_dir),
                "frames": [
                    {"timestamp": 1, "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]}},
                    {"timestamp": 2, "regions": {"USCOMP": ["2_USCOMP.dat", "|u1", [4, 4]]}},
                ],
                "max_frames": 12,
            }
        }
    }
    opened: list[str] = []
    advised: list[int] = []
    fd_to_path: dict[int, str] = {}
    next_fd = iter(range(100, 200))

    def fake_open(path, flags):
        fd = next(next_fd)
        fd_to_path[fd] = str(path)
        opened.append(str(path))
        return fd

    def fake_fadvise(fd, offset, length, advice):
        advised.append(fd)

    monkeypatch.setattr(pagecache.os, "open", fake_open)
    monkeypatch.setattr(pagecache.os, "close", lambda fd: None)
    monkeypatch.setattr(pagecache.os, "posix_fadvise", fake_fadvise)

    # First call: both files primed once, bytes primed = sum of sizes.
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 16 + 32
    assert set(opened) == {str(f1), str(f2)}
    assert {fd_to_path[fd] for fd in advised} == {str(f1), str(f2)}
    assert (cache_dir / "primed_memmaps.json").exists()

    # Second call with unchanged files: nothing re-primed.
    advised.clear()
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 0
    assert advised == []

    # Touch one file: only it gets re-primed (its own size).
    st = f1.stat()
    os.utime(f1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    advised.clear()
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 16
    assert {fd_to_path[fd] for fd in advised} == {str(f1)}


def test_prime_fresh_memmaps_missing_file_and_corrupt_record(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Corrupt the priming record file first.
    record = cache_dir / "primed_memmaps.json"
    record.write_text("{not valid json")
    payload = {
        "stores": {
            "frame_store": {
                "memmap_dir": str(cache_dir),
                "frames": [
                    {"timestamp": 1, "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]}},
                ],
                "max_frames": 12,
            }
        }
    }
    advised = []
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda *args: advised.append(args))
    # The referenced file does not exist -> skipped, no raise.
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 0
    assert advised == []
    # The corrupt record file is rewritten as valid JSON.
    assert json.loads(record.read_text()) == {}


def test_prime_fresh_memmaps_without_stores(tmp_path, monkeypatch):
    advised = []
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda *args: advised.append(args))
    assert pagecache.prime_fresh_memmaps({"version": 1, "written_at": 0}, tmp_path) == 0
    assert advised == []
