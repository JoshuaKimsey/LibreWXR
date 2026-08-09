# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-mode default resolution for the coordinate-warm setting.

``warm_coord_zoom`` follows the same "0 = mode default" sentinel scheme
as ``workers`` / ``tile_cache_mb`` / ``warmer_threads`` (see
``config._MODE_DEFAULTS`` and ``Settings._apply_mode_defaults``):

- unset or 0 -> per-mode default (single: 6, multi: -1 = no eager warm)
- negative   -> warm disabled entirely
- positive   -> force that zoom

The default for multi render workers changed: they no longer eager-warm
at boot (coordinate entries load lazily through the shared on-disk
store), which is what the -1 resolution encodes.  Note the meaning of 0
changed from "disabled" (pre-change) to "mode default".
"""

import pytest

from librewxr.config import Settings


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
    "mode,expected",
    [
        ("single", 6),  # single warms to zoom 6 in the background by default
        ("multi", -1),  # multi render workers do no eager warm by default
    ],
)
def test_warm_coord_zoom_mode_default(monkeypatch, mode, expected):
    s = _fresh_settings(monkeypatch, mode=mode)
    assert s.mode == mode
    assert s.warm_coord_zoom == expected


def test_warm_coord_zoom_explicit_zero_uses_mode_default(monkeypatch):
    # 0 is the "use mode default" sentinel, not "disabled".
    s = _fresh_settings(monkeypatch, mode="single", warm_zoom=0)
    assert s.warm_coord_zoom == 6
    s = _fresh_settings(monkeypatch, mode="multi", warm_zoom=0)
    assert s.warm_coord_zoom == -1


def test_warm_coord_zoom_forced_positive_zoom(monkeypatch):
    # A positive value forces that zoom in either mode (e.g. re-enabling
    # the warm in multi).
    s = _fresh_settings(monkeypatch, mode="multi", warm_zoom=4)
    assert s.warm_coord_zoom == 4
    s = _fresh_settings(monkeypatch, mode="single", warm_zoom=8)
    assert s.warm_coord_zoom == 8


def test_warm_coord_zoom_disabled(monkeypatch):
    # Negative disables the warm entirely in either mode.
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
    assert s.warm_coord_zoom == -1
