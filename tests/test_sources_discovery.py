# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the source-discovery scaffolding (Phase 0+ of the sources
refactor — see ``.claude-context/sources-refactor-plan.md``).

The walker, protocols, and contribution dataclasses must be importable
and the wiring into ``data.regions`` / ``data.fetcher`` must not break
anything as source packages migrate one by one.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.sources


def test_iter_source_packages_returns_at_least_the_subtrees():
    """The two top-level subtrees (``world`` and ``regional``) must both
    be discovered.  As sources migrate, deeper packages join them."""
    from librewxr.sources import iter_source_packages

    pkg_names = {mod.__name__ for mod in iter_source_packages()}
    assert "librewxr.sources.world" in pkg_names
    assert "librewxr.sources.regional" in pkg_names


def test_radar_providers_are_callables():
    """Every entry in ``RADAR_PROVIDERS`` must be a callable so the
    fetcher can invoke it with ``settings`` and get a contribution back
    (or ``None``)."""
    from librewxr.sources import RADAR_PROVIDERS

    assert all(callable(p) for p in RADAR_PROVIDERS)


def test_mmd_provider_is_registered():
    """Phase 1 step 3: MET Malaysia is the first source on the registry
    path.  When ``mmd_enabled`` is True (default) the provider should
    return a contribution covering both Malaysia regions."""
    from librewxr.config import settings
    from librewxr.sources import RADAR_PROVIDERS
    from librewxr.sources._base import RadarSourceContribution

    contribs = [p(settings) for p in RADAR_PROVIDERS]
    mmd = [
        c for c in contribs
        if isinstance(c, RadarSourceContribution) and c.group == "SOUTHEAST_ASIA"
    ]
    assert len(mmd) == 1, "expected exactly one SOUTHEAST_ASIA contribution"
    region_names = {r.name for r in mmd[0].regions}
    assert region_names == {"MYPENINSULAR", "MYEAST"}


def test_protocols_and_contribution_dataclasses_importable():
    from librewxr.sources._base import (
        NWPContribution,
        NWPGrid,
        RadarSource,
        RadarSourceContribution,
        SatelliteContribution,
        SatelliteSource,
    )

    # Smoke-construct the contribution dataclasses with minimal args to
    # confirm the field shapes match the plan.
    radar = RadarSourceContribution(regions=[], instance=None, group="X")  # type: ignore[arg-type]
    assert radar.regions == []
    assert radar.preempts == ()
    assert radar.station_map == {}
    assert radar.range_overrides == {}

    nwp = NWPContribution(instance=None, priority=10, name="X")  # type: ignore[arg-type]
    assert nwp.priority == 10

    sat = SatelliteContribution(instance=None, priority=10, name="X")  # type: ignore[arg-type]
    assert sat.priority == 10
    assert sat.slug is None

    # Protocols themselves should be importable + runtime_checkable.
    for proto in (RadarSource, NWPGrid, SatelliteSource):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(
            proto, "_is_runtime_protocol"
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


def test_fetcher_sees_registry_providers():
    """RadarFetcher.__init__ walks the radar provider registry via
    ``collect_radar_contributions`` to populate ``self._sources``.
    The registry itself must hold callables; check that here so a
    regression in the discovery walker can't silently empty it."""
    from librewxr.sources import RADAR_PROVIDERS

    assert len(RADAR_PROVIDERS) >= 1
    assert all(callable(p) for p in RADAR_PROVIDERS)


class TestNACASourceSplit:
    """The US-side ``na_source`` and Canada-side ``ca_source`` knobs are
    independent.  These tests pin the dispatch wiring across every
    relevant combination so a future regression can't silently put MSC
    in MRMS's CACOMP slot (the bug Phase 1 introduced and Phase 5 fixes).
    """

    @staticmethod
    def _fetcher_with(monkeypatch, na, ca):
        from librewxr.config import settings as S
        from librewxr.data.fetcher import RadarFetcher
        from librewxr.data.store import FrameStore
        from librewxr.tiles.cache import TileCache

        # These tests exercise radar dispatch wiring; pin the global
        # radar toggle on regardless of what the loaded .env says.
        monkeypatch.setattr(S, "radar_enabled", True)
        monkeypatch.setattr(S, "na_source", na)
        monkeypatch.setattr(S, "ca_source", ca)
        store = FrameStore(max_frames=2)
        cache = TileCache(max_mb=10)
        return RadarFetcher(store, cache)

    def test_mrms_with_msc_blend_uses_mrms_as_cacomp_primary(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import (
            MSCCanadaSource,
        )
        from librewxr.sources.regional.north_america.usa.radar.mrms import (
            MRMSCompositeSource,
        )
        f = self._fetcher_with(monkeypatch, "mrms_fallback", "mrms_with_msc_blend")
        # MRMS owns the CACOMP dispatch slot.
        assert isinstance(f._sources["CACOMP"], MRMSCompositeSource)
        # MSC is set up separately as the blend partner.
        assert isinstance(f._cacomp_msc_source, MSCCanadaSource)

    def test_ca_msc_uses_msc_standalone(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import (
            MSCCanadaSource,
        )
        f = self._fetcher_with(monkeypatch, "mrms_fallback", "msc")
        assert isinstance(f._sources["CACOMP"], MSCCanadaSource)
        assert f._cacomp_msc_source is None  # no blend partner needed

    def test_ca_mrms_uses_mrms_only_no_msc(self, monkeypatch):
        from librewxr.sources.regional.north_america.usa.radar.mrms import (
            MRMSCompositeSource,
        )
        f = self._fetcher_with(monkeypatch, "mrms_fallback", "mrms")
        assert isinstance(f._sources["CACOMP"], MRMSCompositeSource)
        assert f._cacomp_msc_source is None  # no MSC fetched at all

    def test_na_iem_ca_mrms_activates_mrms_just_for_cacomp(self, monkeypatch):
        """The decoupling lets ``na_source=iem`` coexist with MRMS for CA."""
        from librewxr.sources.regional.north_america.usa.radar.iem import IEMSource
        from librewxr.sources.regional.north_america.usa.radar.mrms import (
            MRMSCompositeSource,
        )
        f = self._fetcher_with(monkeypatch, "iem", "mrms")
        assert isinstance(f._sources["USCOMP"], IEMSource)
        assert isinstance(f._sources["CACOMP"], MRMSCompositeSource)
        assert f._iem_fallback is None  # no US-side fallback in iem mode

    def test_na_iem_ca_msc_classic_legacy_combination(self, monkeypatch):
        from librewxr.sources.regional.north_america.canada.radar.msc_canada import (
            MSCCanadaSource,
        )
        from librewxr.sources.regional.north_america.usa.radar.iem import IEMSource
        f = self._fetcher_with(monkeypatch, "iem", "msc")
        assert isinstance(f._sources["USCOMP"], IEMSource)
        assert isinstance(f._sources["CACOMP"], MSCCanadaSource)


def test_radar_disabled_returns_empty_contributions(monkeypatch):
    """LIBREWXR_RADAR_ENABLED=false short-circuits every provider call."""
    from unittest.mock import MagicMock

    from librewxr.sources import RADAR_PROVIDERS, collect_radar_contributions

    settings = MagicMock()
    settings.radar_enabled = False
    assert collect_radar_contributions(settings) == []

    # Re-enable and confirm at least one contribution exists (sanity:
    # we didn't accidentally break the enabled path).
    settings.radar_enabled = True
    # Settings used by real providers — wire just enough to get past
    # the provider gates without exercising network calls.
    settings.na_source = "iem"
    settings.ca_source = "msc"
    settings.iem_enabled = True
    settings.msc_canada_enabled = True
    settings.opera_enabled = False
    settings.mmd_enabled = False
    settings.cwa_enabled = False
    settings.marn_enabled = False
    settings.iem_base_url = "https://example.com"
    settings.msc_canada_base_url = "https://example.com"
    contribs = collect_radar_contributions(settings)
    assert len(contribs) >= 1
    assert len(contribs) <= len(RADAR_PROVIDERS)


def test_radar_disabled_skips_coverage_metadata(monkeypatch):
    """When radar is off, the coverage helper returns empty maps too."""
    from unittest.mock import MagicMock

    from librewxr.sources import collect_radar_coverage_metadata

    settings = MagicMock()
    settings.radar_enabled = False
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    assert station_map == {}
    assert range_overrides == {}
    assert coverage_polygons == {}


def test_regional_nwp_disabled_keeps_only_global_sources(tmp_path):
    """LIBREWXR_REGIONAL_NWP_ENABLED=false collapses the chain to the global
    sources.

    Locks in the master-toggle contract for any future regional NWP
    source: as long as it leaves ``NWPContribution.regional`` at its
    default (True), the central collector drops it when the toggle is
    off.  IFS opts out via ``regional=False`` so the global base layer
    keeps running, and RRQPE (global observed satellite data) also opts
    out so it stays ahead of the base layer.
    """
    from librewxr.config import settings as real_settings
    from librewxr.sources import collect_nwp_contributions

    # Baseline: when enabled, more than just the global sources contribute.
    real_settings.regional_nwp_enabled = True
    enabled = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
    assert len(enabled) > 1, "expected regional sources alongside IFS"
    assert any(c.name == "ECMWF IFS" for c in enabled)

    # Toggle off: every regional contribution drops out.
    real_settings.regional_nwp_enabled = False
    try:
        disabled = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
        assert [c.name for c in disabled] == ["NOAA RRQPE", "ECMWF IFS"]
    finally:
        real_settings.regional_nwp_enabled = True


def test_rrqpe_contribution_present_when_enabled(tmp_path):
    """RRQPE registers as the highest-priority chain member when enabled."""
    from librewxr.config import settings as real_settings
    from librewxr.sources import collect_nwp_contributions

    real_settings.rrqpe_enabled = True
    contribs = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
    rrqpe = [c for c in contribs if c.slug == "rrqpe_grid"]
    assert len(rrqpe) == 1
    assert rrqpe[0].priority == 5
    assert rrqpe[0].name == "NOAA RRQPE"
    # Priority 5 sorts ahead of HRRR (10) — first in the chain.
    assert contribs[0].slug == "rrqpe_grid"
    assert contribs[0].priority == 5


def test_rrqpe_dropped_when_disabled(tmp_path):
    """``rrqpe_enabled=False`` drops the contribution entirely."""
    from librewxr.config import settings as real_settings
    from librewxr.sources import collect_nwp_contributions

    real_settings.rrqpe_enabled = False
    try:
        contribs = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
        assert not any(c.slug == "rrqpe_grid" for c in contribs)
    finally:
        real_settings.rrqpe_enabled = True


def test_rrqpe_survives_regional_nwp_disabled(tmp_path):
    """RRQPE is a global observed layer — ``regional_nwp_enabled`` must not
    drop it (``regional=False``), keeping it ahead of IFS."""
    from librewxr.config import settings as real_settings
    from librewxr.sources import collect_nwp_contributions

    real_settings.regional_nwp_enabled = False
    try:
        contribs = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
        assert all(c.regional is False for c in contribs)
        assert [c.slug for c in contribs] == ["rrqpe_grid", "ecmwf_grid"]
    finally:
        real_settings.regional_nwp_enabled = True


def test_satellite_source_slug_uses_override_when_set():
    """Explicit ``slug`` wins over name-derived slug."""
    from librewxr.sources import satellite_source_slug
    from librewxr.sources._base import SatelliteContribution

    sat = SatelliteContribution(
        instance=None, priority=10, name="GMGSI LW", slug="custom_key",  # type: ignore[arg-type]
    )
    assert satellite_source_slug(sat) == "custom_key"


def test_satellite_source_slug_derives_from_name():
    """Name-derived slug: lowercase + non-word → underscore + ``_grid`` suffix."""
    from librewxr.sources import satellite_source_slug
    from librewxr.sources._base import SatelliteContribution

    sat = SatelliteContribution(
        instance=None, priority=10, name="GMGSI LW",  # type: ignore[arg-type]
    )
    assert satellite_source_slug(sat) == "gmgsi_lw_grid"


def test_satellite_providers_are_callables():
    """Every entry in ``SATELLITE_PROVIDERS`` must be callable."""
    from librewxr.sources import SATELLITE_PROVIDERS

    assert all(callable(p) for p in SATELLITE_PROVIDERS)


def test_satellite_disabled_returns_empty_contributions(tmp_path):
    """``satellite_enabled=False`` short-circuits the satellite collector."""
    from unittest.mock import MagicMock

    from librewxr.sources import collect_satellite_contributions

    settings = MagicMock()
    settings.satellite_enabled = False
    contribs = collect_satellite_contributions(settings, cache_dir=tmp_path)
    assert contribs == []


def test_satellite_collector_normalizes_provider_list_returns(tmp_path, monkeypatch):
    """Providers may return one contribution or a list of them.

    Multi-channel sources (GMGSI: LW + VIS) return a list; the collector
    flattens them so callers see a uniform ``list[SatelliteContribution]``.
    """
    from librewxr.sources import _base
    from librewxr.sources import collect_satellite_contributions

    settings = MagicMock_settings()

    lw = _base.SatelliteContribution(instance=_StubSat(), priority=10, name="LW")
    vis = _base.SatelliteContribution(instance=_StubSat(), priority=11, name="VIS")
    wv = _base.SatelliteContribution(instance=_StubSat(), priority=12, name="WV")

    # Provider returning a list — two contributions in one call.
    def provider_multi(_settings, _cache_dir):
        return [lw, vis]

    # Provider returning a single contribution — same call.
    def provider_single(_settings, _cache_dir):
        return wv

    monkeypatch.setattr(
        "librewxr.sources.SATELLITE_PROVIDERS",
        [provider_multi, provider_single],
    )
    contribs = collect_satellite_contributions(settings, cache_dir=tmp_path)
    names = [c.name for c in contribs]
    # Sorted by priority: 10, 11, 12 → LW, VIS, WV
    assert names == ["LW", "VIS", "WV"]


def MagicMock_settings():
    """A MagicMock pre-seeded with the toggle defaults the collector reads."""
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.satellite_enabled = True
    return settings


class _StubSat:
    """Minimal stand-in for a SatelliteSource — only needs to be truthy."""
    name = "stub"
    timestamps: list[int] = []

    async def close(self) -> None: ...
