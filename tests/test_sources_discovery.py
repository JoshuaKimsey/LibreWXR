# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the source-discovery scaffolding (Phase 0 of the sources
refactor — see ``.claude-context/sources-refactor-plan.md``).

The walker, protocols, and contribution dataclasses must be importable
and the wiring into ``data.regions`` / ``data.fetcher`` must not break
anything even when no source packages have been migrated yet.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.sources


def test_iter_source_packages_returns_at_least_the_subtrees():
    """Phase 0 ships two empty subtrees (``world`` and ``regional``) —
    they should both be discovered, but neither should expose any
    provider yet."""
    from librewxr.sources import iter_source_packages

    pkg_names = {mod.__name__ for mod in iter_source_packages()}
    assert "librewxr.sources.world" in pkg_names
    assert "librewxr.sources.regional" in pkg_names


def test_provider_lists_are_empty_in_phase_0():
    from librewxr.sources import NWP_PROVIDERS, RADAR_PROVIDERS

    assert RADAR_PROVIDERS == []
    assert NWP_PROVIDERS == []


def test_protocols_and_contribution_dataclasses_importable():
    from librewxr.sources._base import (
        NWPContribution,
        NWPGrid,
        RadarSource,
        RadarSourceContribution,
    )

    # Smoke-construct the contribution dataclasses with minimal args to
    # confirm the field shapes match the plan.
    radar = RadarSourceContribution(regions=[], instance=None, group="X")  # type: ignore[arg-type]
    assert radar.regions == []
    assert radar.preempts == ()
    assert radar.stations == []

    nwp = NWPContribution(instance=None, priority=10, name="X")  # type: ignore[arg-type]
    assert nwp.priority == 10

    # Protocols themselves should be importable + runtime_checkable.
    assert hasattr(RadarSource, "__protocol_attrs__") or hasattr(
        RadarSource, "_is_runtime_protocol"
    )
    assert hasattr(NWPGrid, "__protocol_attrs__") or hasattr(
        NWPGrid, "_is_runtime_protocol"
    )


def test_regions_module_imports_cleanly_with_discovery_wired():
    """``data.regions._merge_discovered_regions()`` runs at import time;
    in Phase 0 it has nothing to merge but must not raise."""
    from librewxr.data import regions

    # Existing hand-defined regions are still present (proving the
    # merge didn't clobber them).
    assert "USCOMP" in regions.REGIONS
    assert "OPERA" in regions.REGIONS
    assert "MYPENINSULAR" in regions.REGIONS


def test_fetcher_imports_with_empty_registry():
    """RadarFetcher.__init__ now iterates RADAR_PROVIDERS; with the
    registry empty in Phase 0 the loop should be a no-op."""
    from librewxr.data.fetcher import RADAR_PROVIDERS

    assert RADAR_PROVIDERS == []
