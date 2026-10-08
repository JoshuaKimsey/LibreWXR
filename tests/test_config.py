# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-mode default resolution for the coordinate-warm setting.

``warm_coord_zoom`` follows the same "0 = mode default" sentinel scheme
as ``workers`` / ``tile_cache_mb`` (see ``config._MODE_DEFAULTS`` and
``Settings._apply_mode_defaults``):

- unset or 0 -> per-mode default (legacy_single: 4, multi: -1 = no eager warm)
- negative   -> warm disabled entirely
- positive   -> force that zoom

Single mode was removed: a legacy ``single`` token now resolves to the
multi architecture with the ``legacy_single`` 1-worker defaults profile
(``legacy_single=True``).  The default for multi render workers is still
no eager warm at boot (coordinate entries load lazily through the shared
on-disk store), which is what the -1 resolution encodes.  Note the
meaning of 0 changed from "disabled" (pre-change) to "mode default".
"""

import tempfile
from pathlib import Path

import pytest

from librewxr.config import Settings, resolve_cache_dir, resolve_volatile_dir


def _fresh_settings(monkeypatch, *, mode, warm_zoom=None):
    """Build Settings from a controlled environment (no repo .env).

    The repo's own ``.env`` pins LIBREWXR_WARM_COORD_ZOOM=6, so tests
    construct fresh Settings with ``_env_file=None`` and drive the value
    purely through process env vars.
    """
    monkeypatch.setenv("LIBREWXR_MODE", mode)
    monkeypatch.delenv("COMPOSE_PROFILES", raising=False)
    if warm_zoom is None:
        monkeypatch.delenv("LIBREWXR_WARM_COORD_ZOOM", raising=False)
    else:
        monkeypatch.setenv("LIBREWXR_WARM_COORD_ZOOM", str(warm_zoom))
    return Settings(_env_file=None)


@pytest.mark.parametrize(
    "mode,expected_legacy,expected_zoom",
    [
        ("single", True, 4),  # legacy single warms to zoom 4 by default
        ("multi", False, -1),  # multi render workers do no eager warm
    ],
)
def test_warm_coord_zoom_mode_default(
    monkeypatch, mode, expected_legacy, expected_zoom,
):
    s = _fresh_settings(monkeypatch, mode=mode)
    # Single mode no longer exists: the legacy "single" token resolves to
    # the multi architecture with the legacy_single defaults profile.
    assert s.mode == "multi"
    assert s.legacy_single is expected_legacy
    assert s.warm_coord_zoom == expected_zoom


def test_warm_coord_zoom_explicit_zero_uses_mode_default(monkeypatch):
    # 0 is the "use mode default" sentinel, not "disabled".
    s = _fresh_settings(monkeypatch, mode="single", warm_zoom=0)
    assert s.legacy_single is True
    assert s.warm_coord_zoom == 4
    s = _fresh_settings(monkeypatch, mode="multi", warm_zoom=0)
    assert s.legacy_single is False
    assert s.warm_coord_zoom == -1


def test_warm_coord_zoom_forced_positive_zoom(monkeypatch):
    # A positive value forces that zoom regardless of profile (e.g.
    # re-enabling the warm in multi).
    s = _fresh_settings(monkeypatch, mode="multi", warm_zoom=4)
    assert s.warm_coord_zoom == 4
    s = _fresh_settings(monkeypatch, mode="single", warm_zoom=8)
    assert s.warm_coord_zoom == 8


def test_warm_coord_zoom_disabled(monkeypatch):
    # Negative disables the warm entirely regardless of profile.
    s = _fresh_settings(monkeypatch, mode="single", warm_zoom=-1)
    assert s.warm_coord_zoom == -1
    s = _fresh_settings(monkeypatch, mode="multi", warm_zoom=-5)
    assert s.warm_coord_zoom == -5


def test_warm_coord_zoom_via_compose_profiles(monkeypatch):
    # COMPOSE_PROFILES drives the mode fallback; the warm resolution
    # follows the resolved mode.
    monkeypatch.delenv("LIBREWXR_MODE", raising=False)
    monkeypatch.setenv("COMPOSE_PROFILES", "multi,manual")
    monkeypatch.delenv("LIBREWXR_WARM_COORD_ZOOM", raising=False)
    s = Settings(_env_file=None)
    assert s.mode == "multi"
    assert s.legacy_single is False
    assert s.warm_coord_zoom == -1


def test_warm_coord_zoom_via_compose_profiles_single(monkeypatch):
    # A legacy "single" token in COMPOSE_PROFILES selects the
    # legacy_single defaults profile without changing the resolved mode.
    monkeypatch.delenv("LIBREWXR_MODE", raising=False)
    monkeypatch.setenv("COMPOSE_PROFILES", "single")
    monkeypatch.delenv("LIBREWXR_WARM_COORD_ZOOM", raising=False)
    s = Settings(_env_file=None)
    assert s.mode == "multi"
    assert s.legacy_single is True
    assert s.warm_coord_zoom == 4


def test_legacy_single_token_emits_warning(monkeypatch, caplog):
    # A one-time migration warning fires when the removed single mode is
    # requested, so the operator knows what happened.
    import logging

    with caplog.at_level(logging.WARNING, logger="librewxr.config"):
        _fresh_settings(monkeypatch, mode="single")
    assert any(
        "single mode was removed" in record.getMessage()
        for record in caplog.records
    )


def test_multi_token_emits_no_legacy_warning(monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="librewxr.config"):
        _fresh_settings(monkeypatch, mode="multi")
    assert not any(
        "single mode was removed" in record.getMessage()
        for record in caplog.records
    )


def test_resolve_cache_dir_uses_configured_path(monkeypatch, tmp_path):
    s = _fresh_settings(monkeypatch, mode="multi")
    target = tmp_path / "cache"
    monkeypatch.setattr(s, "cache_dir", str(target))
    assert resolve_cache_dir(s) == target


def test_resolve_cache_dir_falls_back_to_tempdir(monkeypatch):
    s = _fresh_settings(monkeypatch, mode="multi")
    monkeypatch.setattr(s, "cache_dir", "")
    expected = Path(tempfile.gettempdir()) / "librewxr-cache"
    assert resolve_cache_dir(s) == expected


def test_resolve_volatile_dir_unset_returns_none(monkeypatch):
    s = _fresh_settings(monkeypatch, mode="multi")
    monkeypatch.setattr(s, "volatile_cache_dir", "")
    assert resolve_volatile_dir(s) is None


def test_resolve_volatile_dir_uses_configured_path(monkeypatch, tmp_path):
    s = _fresh_settings(monkeypatch, mode="multi")
    target = tmp_path / "volatile"
    monkeypatch.setattr(s, "volatile_cache_dir", str(target))
    assert resolve_volatile_dir(s) == target
