# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Round-trip tests for the persisted coverage/feather mask cache.

``save_masks`` / ``load_masks`` share the built coverage + feather masks
between processes via read-only memmaps under the cache dir.  These tests
verify the round trip (build -> save -> load into fresh module state ->
identical arrays), the failure modes (missing manifest, signature
mismatch, missing / truncated files), and that the parameter signature
actually tracks the inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from librewxr.data import coverage as cov

pytestmark = pytest.mark.store


@pytest.fixture
def _mask_state():
    """Restore the module-global mask dicts after each test."""
    coverage_before = dict(cov._COVERAGE_MASKS)
    feather_before = dict(cov._FEATHER_MASKS)
    yield
    cov._COVERAGE_MASKS.clear()
    cov._COVERAGE_MASKS.update(coverage_before)
    cov._FEATHER_MASKS.clear()
    cov._FEATHER_MASKS.update(feather_before)


# Tiny synthetic station sets over small real regions (SVCOMP / TWCOMP) so
# the station-circle rasterisation stays cheap.
_STATIONS: dict[str, list[tuple[float, float]]] = {
    "SVCOMP": [(13.7, -89.2), (13.9, -88.9), (14.2, -89.4)],
    "TWCOMP": [(23.5, 120.5), (22.6, 120.3)],
}
_ENABLED = ["SVCOMP", "TWCOMP"]
_OVERRIDES = {"SVCOMP": 120.0}


def _build_and_snapshot():
    """Build masks for the synthetic set; snapshot copies of the arrays."""
    cov.build_coverage_masks(_STATIONS, range_overrides=_OVERRIDES)
    cov.build_feather_masks()
    expected_cov = {
        name: (mask.copy(), west, south, dx, dy)
        for name, (mask, west, south, dx, dy) in cov._COVERAGE_MASKS.items()
    }
    expected_feather = {
        name: (mask.copy(), west, south, dx, dy)
        for name, (mask, west, south, dx, dy) in cov._FEATHER_MASKS.items()
    }
    return expected_cov, expected_feather


def test_save_load_roundtrip(tmp_path: Path, _mask_state):
    """Built masks survive a save -> fresh state -> load cycle byte-for-byte."""
    expected_cov, expected_feather = _build_and_snapshot()
    assert set(expected_cov) == {"SVCOMP", "TWCOMP"}

    cov.save_masks(tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES)

    # Fresh module state: nothing loaded yet.
    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()
    assert cov.load_masks(
        tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    ) is True

    assert set(cov._COVERAGE_MASKS) == {"SVCOMP", "TWCOMP"}
    assert set(cov._FEATHER_MASKS) == {"SVCOMP", "TWCOMP"}
    for name, (exp_mask, exp_west, exp_south, exp_dx, exp_dy) in expected_cov.items():
        loaded_mask, west, south, dx, dy = cov._COVERAGE_MASKS[name]
        np.testing.assert_array_equal(loaded_mask, exp_mask)
        assert (west, south, dx, dy) == (exp_west, exp_south, exp_dx, exp_dy)
        loaded_feather, f_west, f_south, f_dx, f_dy = cov._FEATHER_MASKS[name]
        exp_feather = expected_feather[name][0]
        np.testing.assert_array_equal(loaded_feather, exp_feather)
        assert (f_west, f_south, f_dx, f_dy) == (exp_west, exp_south, exp_dx, exp_dy)


def test_save_load_roundtrip_with_polygon(tmp_path: Path, _mask_state):
    """Polygon-based masks (the JPCOMP-style path) persist and load alike."""
    polygon = {
        "SVCOMP": [
            [(13.0, -90.0), (13.0, -87.5), (14.8, -87.5), (14.8, -90.0)],
        ],
    }
    cov.build_coverage_masks({}, coverage_polygons=polygon)
    cov.build_feather_masks()
    expected = {
        name: mask.copy()
        for name, (mask, _w, _s, _dx, _dy) in cov._COVERAGE_MASKS.items()
    }
    assert set(expected) == {"SVCOMP"}

    cov.save_masks(tmp_path, ["SVCOMP"], {}, coverage_polygons=polygon)
    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()
    assert cov.load_masks(
        tmp_path, ["SVCOMP"], {}, coverage_polygons=polygon,
    ) is True
    np.testing.assert_array_equal(cov._COVERAGE_MASKS["SVCOMP"][0], expected["SVCOMP"])


def test_file_layout(tmp_path: Path, _mask_state):
    """Save writes one .dat per mask kind plus a manifest; no tmp leftovers."""
    _build_and_snapshot()
    cov.save_masks(tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES)

    mask_dir = tmp_path / cov.MASK_CACHE_DIRNAME
    files = sorted(p.name for p in mask_dir.iterdir())
    assert files == [
        "SVCOMP.coverage.dat",
        "SVCOMP.feather.dat",
        "TWCOMP.coverage.dat",
        "TWCOMP.feather.dat",
        "masks.json",
    ]

    manifest = json.loads((mask_dir / "masks.json").read_text())
    assert manifest["format_version"] == cov.MASK_FORMAT_VERSION
    assert manifest["signature"] == cov.mask_signature(
        _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    )
    assert set(manifest["regions"]) == {"SVCOMP", "TWCOMP"}
    sv = manifest["regions"]["SVCOMP"]
    assert sv["coverage"]["dtype"] == "bool"
    assert sv["feather"]["dtype"] == "float32"
    assert sv["coverage"]["shape"] == sv["feather"]["shape"]
    # No leftover tmp files after a clean save.
    assert not list(mask_dir.glob("*.tmp.*"))


def test_load_missing_manifest_returns_false(tmp_path: Path, _mask_state):
    """No masks.json -> False and the dicts stay untouched."""
    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()
    assert cov.load_masks(
        tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    ) is False
    assert cov._COVERAGE_MASKS == {}
    assert cov._FEATHER_MASKS == {}


def test_load_rejects_signature_mismatch(tmp_path: Path, _mask_state):
    """Different build parameters -> signature mismatch -> False, no install."""
    _build_and_snapshot()
    cov.save_masks(tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES)

    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()

    # Same region set, but one station moved — the mask content changes.
    moved = {name: list(stations) for name, stations in _STATIONS.items()}
    moved["SVCOMP"][0] = (13.75, -89.25)
    assert cov.load_masks(
        tmp_path, _ENABLED, moved, range_overrides=_OVERRIDES,
    ) is False
    assert cov._COVERAGE_MASKS == {}
    assert cov._FEATHER_MASKS == {}

    # A different enabled-region set is also a signature change.
    assert cov.load_masks(
        tmp_path, ["SVCOMP"], {"SVCOMP": _STATIONS["SVCOMP"]},
        range_overrides=_OVERRIDES,
    ) is False
    assert cov._COVERAGE_MASKS == {}
    assert cov._FEATHER_MASKS == {}


def test_load_rejects_missing_file(tmp_path: Path, _mask_state):
    """A deleted .dat file fails the load and leaves the dicts untouched."""
    _build_and_snapshot()
    cov.save_masks(tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES)

    (tmp_path / cov.MASK_CACHE_DIRNAME / "SVCOMP.feather.dat").unlink()
    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()
    assert cov.load_masks(
        tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    ) is False
    assert cov._COVERAGE_MASKS == {}
    assert cov._FEATHER_MASKS == {}


def test_load_rejects_truncated_file(tmp_path: Path, _mask_state):
    """A truncated .dat file (e.g. crashed writer) fails the load cleanly."""
    _build_and_snapshot()
    cov.save_masks(tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES)

    feather = tmp_path / cov.MASK_CACHE_DIRNAME / "SVCOMP.feather.dat"
    feather.write_bytes(feather.read_bytes()[:16])
    cov._COVERAGE_MASKS.clear()
    cov._FEATHER_MASKS.clear()
    assert cov.load_masks(
        tmp_path, _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    ) is False
    assert cov._COVERAGE_MASKS == {}
    assert cov._FEATHER_MASKS == {}


def test_signature_tracks_parameters(_mask_state):
    """mask_signature is deterministic and changes with any input."""
    base = cov.mask_signature(_ENABLED, _STATIONS, range_overrides=_OVERRIDES)
    assert base == cov.mask_signature(
        _ENABLED, _STATIONS, range_overrides=_OVERRIDES,
    )

    moved = {name: list(stations) for name, stations in _STATIONS.items()}
    moved["SVCOMP"].append((13.5, -89.1))
    assert cov.mask_signature(_ENABLED, moved, range_overrides=_OVERRIDES) != base

    assert cov.mask_signature(
        ["SVCOMP"], {"SVCOMP": _STATIONS["SVCOMP"]}, range_overrides=_OVERRIDES,
    ) != base
    assert cov.mask_signature(
        _ENABLED, _STATIONS, range_overrides={"SVCOMP": 150.0},
    ) != base
