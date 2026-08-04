# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory monitor tests: cgroup stat parsing, fallback, jitter, hysteresis."""
import random
from types import SimpleNamespace

from librewxr.memory import (
    CgroupUsage,
    MemoryMonitor,
    _jittered_thresholds,
    _read_cgroup_memory_usage,
    _read_stat_sum,
)


class _FakeTileCache:
    """Records evict_half / clear calls instead of holding real tiles."""

    def __init__(self):
        self.evict_calls = 0
        self.clear_calls = 0

    def evict_half(self) -> int:
        self.evict_calls += 1
        return 0

    def clear(self) -> None:
        self.clear_calls += 1


def _make_monitor(monkeypatch, limit_mb: int = 1000, check_interval: int = 30):
    """Build a MemoryMonitor with a fake tile cache and coord-clear recorder."""
    tile_cache = _FakeTileCache()
    coord_calls: list[int] = []
    monitor = MemoryMonitor(
        tile_cache,
        lambda: coord_calls.append(1),
        limit_mb,
        check_interval,
    )
    return monitor, tile_cache, coord_calls


def _fake_cgroup_usage(monkeypatch, limit_mb: int = 1000, fraction: float = 0.0):
    """Monkeypatch the cgroup reader to return a fixed usage fraction."""
    limit_bytes = limit_mb * 1024 * 1024
    state = {"fraction": fraction}

    def fake() -> CgroupUsage:
        decision = int(limit_bytes * state["fraction"])
        # Raw cgroup total runs 17 MB higher than the decision metric.
        return CgroupUsage(decision, decision + 17 * 1024 * 1024, "anon+shmem", True)

    monkeypatch.setattr("librewxr.memory._read_cgroup_memory_usage", fake)
    return state


def _fake_cgroup_files(monkeypatch, *, v2_stat, v2_current, v1_stat, v1_usage):
    """Route the cgroup file reads through per-path fakes."""
    v2_stat_path = "/sys/fs/cgroup/memory.stat"
    v2_current_path = "/sys/fs/cgroup/memory.current"

    def fake_stat_sum(path, fields):
        if str(path) == v2_stat_path:
            return v2_stat
        return v1_stat

    def fake_read_int(path):
        if str(path) == v2_current_path:
            return v2_current
        return v1_usage

    monkeypatch.setattr("librewxr.memory._read_stat_sum", fake_stat_sum)
    monkeypatch.setattr("librewxr.memory._read_int", fake_read_int)


# ---------------------------------------------------------------------------
# Fix 1: reclaimable-aware usage (anon + shmem / rss + shmem, with fallback)
# ---------------------------------------------------------------------------


class TestCgroupStatParsing:
    def test_stat_sum_parses_anon_and_shmem(self, tmp_path):
        stat = tmp_path / "memory.stat"
        stat.write_text("anon 1000\nfile 999999999\nshmem 2000\nslab 300\n")
        assert _read_stat_sum(stat, ("anon", "shmem")) == 3000

    def test_stat_sum_returns_none_when_field_missing(self, tmp_path):
        stat = tmp_path / "memory.stat"
        stat.write_text("anon 1000\nfile 999999999\n")
        assert _read_stat_sum(stat, ("anon", "shmem")) is None

    def test_stat_sum_returns_none_when_file_missing(self, tmp_path):
        assert _read_stat_sum(tmp_path / "nope.stat", ("anon", "shmem")) is None

    def test_stat_sum_returns_none_on_malformed_counter(self, tmp_path):
        stat = tmp_path / "memory.stat"
        stat.write_text("anon not-a-number\nshmem 2000\n")
        assert _read_stat_sum(stat, ("anon", "shmem")) is None

    def test_cgroup_v2_uses_anon_plus_shmem_as_decision(self, monkeypatch):
        _fake_cgroup_files(
            monkeypatch,
            v2_stat=3_000, v2_current=9_000, v1_stat=None, v1_usage=None,
        )
        usage = _read_cgroup_memory_usage()
        assert usage is not None
        assert usage.decision_bytes == 3_000
        assert usage.total_bytes == 9_000
        assert usage.label == "anon+shmem"
        assert usage.stat_based is True

    def test_cgroup_v1_uses_rss_plus_shmem_as_decision(self, monkeypatch):
        _fake_cgroup_files(
            monkeypatch,
            v2_stat=None, v2_current=None, v1_stat=5_000, v1_usage=8_000,
        )
        usage = _read_cgroup_memory_usage()
        assert usage is not None
        assert usage.decision_bytes == 5_000
        assert usage.total_bytes == 8_000
        assert usage.label == "rss+shmem"
        assert usage.stat_based is True


class TestCgroupFallback:
    def test_v2_falls_back_to_memory_current_when_stat_unreadable(self, monkeypatch):
        _fake_cgroup_files(
            monkeypatch,
            v2_stat=None, v2_current=9_000, v1_stat=None, v1_usage=None,
        )
        usage = _read_cgroup_memory_usage()
        assert usage is not None
        assert usage.decision_bytes == 9_000
        assert usage.total_bytes == 9_000
        assert usage.label == "cgroup"
        assert usage.stat_based is False

    def test_v1_falls_back_to_usage_file_when_stat_unreadable(self, monkeypatch):
        _fake_cgroup_files(
            monkeypatch,
            v2_stat=None, v2_current=None, v1_stat=None, v1_usage=8_000,
        )
        usage = _read_cgroup_memory_usage()
        assert usage is not None
        assert usage.decision_bytes == 8_000
        assert usage.total_bytes == 8_000
        assert usage.label == "cgroup"
        assert usage.stat_based is False

    def test_returns_none_outside_any_container(self, monkeypatch):
        _fake_cgroup_files(
            monkeypatch,
            v2_stat=None, v2_current=None, v1_stat=None, v1_usage=None,
        )
        assert _read_cgroup_memory_usage() is None

    def test_fallback_does_not_crash_monitor_check(self, monkeypatch):
        # Raw usage fallback: threshold decisions still run, no exception.
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        monkeypatch.setattr(
            "librewxr.memory._read_cgroup_memory_usage",
            lambda: CgroupUsage(
                int(0.95 * 1000 * 1024 * 1024),
                int(0.95 * 1000 * 1024 * 1024),
                "cgroup",
                False,
            ),
        )
        monitor._check()
        monitor._check()
        assert tile_cache.clear_calls == 1

    def test_rss_fallback_outside_container_still_works(self, monkeypatch):
        monitor, tile_cache, coord_calls = _make_monitor(monkeypatch)
        monkeypatch.setattr("librewxr.memory._read_cgroup_memory_usage", lambda: None)
        limit_bytes = 1000 * 1024 * 1024

        class _FakeProc:
            def memory_info(self):
                return SimpleNamespace(rss=int(0.95 * limit_bytes))

        monkeypatch.setattr(monitor, "_process", _FakeProc())
        monitor._check()
        assert tile_cache.clear_calls == 0  # hysteresis: one check not enough
        monitor._check()
        assert tile_cache.clear_calls == 1
        assert coord_calls == [1]
        assert monitor.cgroup_total_mb is None


# ---------------------------------------------------------------------------
# Fix 2: per-worker jitter + two-check hysteresis
# ---------------------------------------------------------------------------


class TestThresholdJitter:
    def test_jitter_bounds_and_ordering_across_many_seeds(self):
        state = random.getstate()
        try:
            for seed in range(500):
                random.seed(seed)
                warn, evict, clear = _jittered_thresholds()
                assert 0.79 <= warn <= 0.81
                assert 0.83 <= evict <= 0.87
                assert 0.88 <= clear <= 0.92
                assert warn < evict < clear
        finally:
            random.setstate(state)

    def test_jitter_is_fixed_for_process_lifetime(self, monkeypatch):
        monitor, _, _ = _make_monitor(monkeypatch)
        thresholds = (
            monitor._warn_threshold,
            monitor._evict_tiles_threshold,
            monitor._evict_all_threshold,
        )
        monitor._check()
        assert (
            monitor._warn_threshold,
            monitor._evict_tiles_threshold,
            monitor._evict_all_threshold,
        ) == thresholds

    def test_base_constants_never_mutated_by_jitter(self):
        import librewxr.memory as memory_module

        base = (memory_module._WARN_THRESHOLD,
                memory_module._EVICT_TILES_THRESHOLD,
                memory_module._EVICT_ALL_THRESHOLD)
        _jittered_thresholds()
        assert (memory_module._WARN_THRESHOLD,
                memory_module._EVICT_TILES_THRESHOLD,
                memory_module._EVICT_ALL_THRESHOLD) == base


class TestHysteresis:
    # Usage fractions are chosen to be in-band for EVERY possible jitter:
    # warn in [0.79, 0.81], evict in [0.83, 0.87], clear in [0.88, 0.92].
    # 0.82 is always warn-only, 0.875 always evict-band, 0.95 always
    # clear-band, 0.70 always below everything.

    def test_single_high_check_does_not_evict(self, monkeypatch):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        _fake_cgroup_usage(monkeypatch, fraction=0.875)
        monitor._check()
        assert tile_cache.evict_calls == 0
        assert tile_cache.clear_calls == 0

    def test_two_consecutive_checks_evict_half(self, monkeypatch):
        monitor, tile_cache, coord_calls = _make_monitor(monkeypatch)
        _fake_cgroup_usage(monkeypatch, fraction=0.875)
        monitor._check()
        monitor._check()
        assert tile_cache.evict_calls == 1
        assert tile_cache.clear_calls == 0
        assert coord_calls == []  # evict-half does not clear coord caches

    def test_below_threshold_check_resets_evict_streak(self, monkeypatch):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        state = _fake_cgroup_usage(monkeypatch, fraction=0.875)
        monitor._check()                      # streak 1
        state["fraction"] = 0.70              # below warn: reset
        monitor._check()
        state["fraction"] = 0.875
        monitor._check()                      # streak 1 again
        assert tile_cache.evict_calls == 0
        monitor._check()                      # streak 2: act
        assert tile_cache.evict_calls == 1

    def test_warn_band_resets_evict_streak(self, monkeypatch):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        state = _fake_cgroup_usage(monkeypatch, fraction=0.875)
        monitor._check()                      # streak 1
        state["fraction"] = 0.82              # warn-only band: reset evict streak
        monitor._check()
        state["fraction"] = 0.875
        monitor._check()                      # streak 1 again
        assert tile_cache.evict_calls == 0
        monitor._check()                      # streak 2: act
        assert tile_cache.evict_calls == 1

    def test_clear_level_requires_two_checks_and_clears_coords(self, monkeypatch):
        monitor, tile_cache, coord_calls = _make_monitor(monkeypatch)
        _fake_cgroup_usage(monkeypatch, fraction=0.95)
        monitor._check()
        assert tile_cache.clear_calls == 0
        monitor._check()
        assert tile_cache.clear_calls == 1
        assert tile_cache.evict_calls == 0
        assert coord_calls == [1]

    def test_clear_streak_resets_when_usage_drops_to_evict_band(self, monkeypatch):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        state = _fake_cgroup_usage(monkeypatch, fraction=0.95)
        monitor._check()                      # clear streak 1
        state["fraction"] = 0.875             # below clear: reset clear streak
        monitor._check()
        state["fraction"] = 0.95
        monitor._check()                      # clear streak 1 again
        assert tile_cache.clear_calls == 0
        monitor._check()                      # clear streak 2: act
        assert tile_cache.clear_calls == 1

    def test_warn_level_logs_on_first_crossing(self, monkeypatch, caplog):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        _fake_cgroup_usage(monkeypatch, fraction=0.82)
        with caplog.at_level("INFO", logger="librewxr.memory"):
            monitor._check()
        assert any("Memory usage elevated" in r.message for r in caplog.records)
        assert tile_cache.evict_calls == 0
        assert tile_cache.clear_calls == 0

    def test_eviction_acts_every_check_once_past_hysteresis(self, monkeypatch):
        monitor, tile_cache, _ = _make_monitor(monkeypatch)
        _fake_cgroup_usage(monkeypatch, fraction=0.875)
        for _ in range(4):
            monitor._check()
        assert tile_cache.evict_calls == 3  # checks 2, 3, 4 all act

    def test_cgroup_total_mb_reported_from_check(self, monkeypatch):
        monitor, _, _ = _make_monitor(monkeypatch)
        assert monitor.cgroup_total_mb is None
        _fake_cgroup_usage(monkeypatch, fraction=0.5)
        monitor._check()
        # decision == 500 MB, raw cgroup total == decision + 17 MB
        assert monitor.cgroup_total_mb == 517
