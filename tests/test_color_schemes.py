# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.tiles

import librewxr.colors.schemes as schemes
from librewxr.colors.schemes import (
    SCHEME_NAMES,
    ParsedTable,
    apply_external_table,
    capture_luts,
    colorize,
    colorize_with_lut,
    current_version,
    get_lut,
    packaged_table_bytes,
    parse_color_table,
)


class TestGetLut:
    def test_lut_shape(self):
        for scheme_id in SCHEME_NAMES:
            lut = get_lut(scheme_id)
            assert lut.shape == (256, 4), f"Scheme {scheme_id} LUT wrong shape"
            assert lut.dtype == np.uint8

    def test_snow_lut_shape(self):
        for scheme_id in SCHEME_NAMES:
            lut = get_lut(scheme_id, snow=True)
            assert lut.shape == (256, 4)

    def test_raw_scheme(self):
        lut = get_lut(255)
        assert lut.shape == (256, 4)
        # Pixel value 0 should be transparent
        assert lut[0, 3] == 0
        # Pixel value 128 should be grayscale 128, opaque
        assert lut[128, 0] == 128
        assert lut[128, 3] == 255

    def test_zero_pixel_transparent(self):
        """Pixel value 0 maps to dBZ -32, which should be transparent for most schemes."""
        for scheme_id in [1, 2, 3, 4, 5]:
            lut = get_lut(scheme_id)
            assert lut[0, 3] == 0, f"Scheme {scheme_id}: pixel 0 should be transparent"

    def test_invalid_scheme_defaults(self):
        """Unknown scheme ID should fall back to Rainbow @ Selex SI."""
        lut = get_lut(99)
        expected = get_lut(7)
        np.testing.assert_array_equal(lut, expected)


class TestColorize:
    def test_basic_colorize(self):
        values = np.array([[0, 128, 255]], dtype=np.uint8)
        result = colorize(values, scheme=1)
        assert result.shape == (1, 3, 4)
        assert result.dtype == np.uint8

    def test_zero_values_transparent(self):
        values = np.zeros((10, 10), dtype=np.uint8)
        result = colorize(values, scheme=2)
        # All alpha should be 0
        assert np.all(result[:, :, 3] == 0)

    def test_vectorized(self):
        """Colorize should work on arbitrary shaped arrays."""
        values = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        result = colorize(values, scheme=6)
        assert result.shape == (100, 100, 4)


class TestConcurrentLazyInit:
    """Reset module-level LUTs and hammer ``get_lut`` from many threads.

    Pre-fix this would intermittently KeyError because ``_load_color_table``
    populated the global dicts entry-by-entry — readers entering between
    the first ``_rain_luts[0] = …`` and the final ``_rain_luts[255] = …``
    saw a truthy-but-incomplete dict and skipped the reload.  The fix
    builds the dicts in locals and atomically reassigns at the end,
    inside a lock that dedupes redundant loads.
    """

    def test_no_keyerror_under_concurrent_first_use(self):
        import concurrent.futures
        import librewxr.colors.schemes as schemes

        # Force a fresh lazy-init for this test.
        schemes._rain_luts = {}
        schemes._snow_luts = {}

        schemes_to_try = list(SCHEME_NAMES.keys()) + [99, 255]  # include invalid + raw

        def worker(scheme: int) -> tuple[int, int]:
            # Each worker hammers both rain + snow paths for one scheme.
            rain = schemes.get_lut(scheme, snow=False)
            snow = schemes.get_lut(scheme, snow=True)
            return rain.shape[0], snow.shape[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(worker, s) for s in schemes_to_try * 8]
            results = [f.result(timeout=10) for f in futures]

        assert all(r == (256, 256) for r in results)


# ---------------------------------------------------------------------------
# External palettes: packaged golden values, strict parsing, apply semantics
# ---------------------------------------------------------------------------

# The registry the packaged CSV derives.  This is the pre-external-palette
# hardcoded literal, unchanged; only the CSV header's decorative column 2
# differs from it ("Original" -> "Rainviewer Original").
EXPECTED_NAMES = {
    0: "Black and White",
    1: "Rainviewer Original",
    2: "Universal Blue",
    3: "Titan",
    4: "The Weather Channel (TWC)",
    5: "Meteored",
    6: "NEXRAD Level III",
    7: "Rainbow @ Selex SI",
    8: "Dark Sky",
    9: "Datameteo Valerio",
    10: "Viper HD",
    11: "MRMS CREF",
    12: "33/40 Max Storm",
    13: "MetService NZ (Dark)",
    14: "Windy",
}

# (scheme, pixel value) -> RGBA, frozen from the pre-change table.  These
# prove the CSV-derived LUTs are byte-identical to the old hardcoded path.
GOLDEN_RAIN = {
    (0, 64): [32, 32, 32, 255],
    (1, 100): [155, 234, 143, 255],
    (2, 128): [255, 238, 0, 255],
    (7, 200): [255, 62, 255, 255],
    (11, 255): [149, 87, 198, 255],
    (14, 33): [0, 0, 0, 0],
}
GOLDEN_SNOW = {
    (7, 100): [123, 235, 255, 255],
}


def _table_from_rows(names, rows) -> bytes:
    header = ",".join(["dBZ / RGBA", *names])
    return "\n".join([header, *[",".join(r) for r in rows]]).encode()


def _valid_rows(rain_colors, snow_colors) -> list[list[str]]:
    rows = []
    for i in range(256):
        colors = rain_colors if i < 128 else snow_colors
        rows.append([str(i - 32), *colors])
    return rows


def _table_bytes(
    names=("Alpha", "Beta"),
    rain_colors=("#01020304", "#11121314"),
    snow_colors=("#05060708", "#15161718"),
) -> bytes:
    return _table_from_rows(
        names, _valid_rows(rain_colors, snow_colors),
    )


@pytest.fixture
def packaged_palette():
    """Restore the packaged table after a test mutates the globals."""
    yield
    apply_external_table(packaged_table_bytes())


class TestPackagedGolden:
    def test_registry_matches_previous_literal(self):
        assert SCHEME_NAMES == EXPECTED_NAMES
        assert parse_color_table(packaged_table_bytes()).names == EXPECTED_NAMES

    def test_registry_populated_at_import(self):
        # scripts import SCHEME_NAMES without ever touching a LUT.
        assert SCHEME_NAMES is schemes.SCHEME_NAMES
        assert len(SCHEME_NAMES) == 15

    def test_routes_holds_the_same_dict_object(self):
        # ``api.routes`` imports SCHEME_NAMES by name; a reload must mutate
        # this exact object, never rebind the module global.
        from librewxr.api import routes

        assert routes.SCHEME_NAMES is schemes.SCHEME_NAMES

    def test_raw_scheme_not_in_registry(self):
        assert 255 not in SCHEME_NAMES

    def test_golden_rain_lut_values(self):
        for (scheme_id, pixel), expected in GOLDEN_RAIN.items():
            assert get_lut(scheme_id)[pixel].tolist() == expected

    def test_golden_snow_lut_values(self):
        for (scheme_id, pixel), expected in GOLDEN_SNOW.items():
            assert get_lut(scheme_id, snow=True)[pixel].tolist() == expected

    def test_packaged_version_is_content_hash(self):
        raw = packaged_table_bytes()
        assert current_version() == hashlib.sha256(raw).hexdigest()[:12]


class TestParseColorTable:
    def test_minimal_table_round_trips(self):
        raw = _table_bytes()
        parsed = parse_color_table(raw)
        assert isinstance(parsed, ParsedTable)
        assert parsed.names == {0: "Alpha", 1: "Beta"}
        assert parsed.version == hashlib.sha256(raw).hexdigest()[:12]
        for lut in list(parsed.rain_luts.values()) + list(parsed.snow_luts.values()):
            assert lut.shape == (256, 4)
            assert lut.dtype == np.uint8
        # Pixel value v reads block row v // 2.
        assert parsed.rain_luts[0][0].tolist() == [1, 2, 3, 4]
        assert parsed.rain_luts[1][0].tolist() == [0x11, 0x12, 0x13, 0x14]
        assert parsed.snow_luts[0][200].tolist() == [5, 6, 7, 8]
        # Raw grayscale entry is synthesized into both dicts, never named.
        assert parsed.rain_luts[255][128].tolist() == [128, 128, 128, 255]
        assert parsed.rain_luts[255][0].tolist() == [0, 0, 0, 0]
        assert parsed.snow_luts[255][128].tolist() == [128, 128, 128, 255]
        assert 255 not in parsed.names

    def test_parse_does_not_mutate_globals(self):
        before_names = dict(SCHEME_NAMES)
        before_version = current_version()
        before_rain = schemes._rain_luts
        parse_color_table(_table_bytes())
        assert SCHEME_NAMES == before_names
        assert current_version() == before_version
        assert schemes._rain_luts is before_rain

    def test_rejects_missing_snow_block(self):
        rows = _valid_rows(
            ("#01020304", "#11121314"), ("#05060708", "#15161718"),
        )[:128]
        with pytest.raises(ValueError):
            parse_color_table(_table_from_rows(("Alpha", "Beta"), rows))

    def test_rejects_wrong_row_count(self):
        rows = _valid_rows(
            ("#01020304", "#11121314"), ("#05060708", "#15161718"),
        )[:-1]
        with pytest.raises(ValueError):
            parse_color_table(_table_from_rows(("Alpha", "Beta"), rows))

    def test_rejects_ragged_rows(self):
        rows = _valid_rows(
            ("#01020304", "#11121314"), ("#05060708", "#15161718"),
        )
        rows[17] = rows[17][:-1]
        with pytest.raises(ValueError):
            parse_color_table(_table_from_rows(("Alpha", "Beta"), rows))

    def test_rejects_bad_hex(self):
        rows = _valid_rows(
            ("#01020304", "#11121314"), ("#05060708", "#15161718"),
        )
        rows[200][1] = "#0102030"
        with pytest.raises(ValueError):
            parse_color_table(_table_from_rows(("Alpha", "Beta"), rows))

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            parse_color_table(
                _table_bytes(
                    names=("Alpha", "   "),
                    rain_colors=("#01020304", "#11121314"),
                    snow_colors=("#05060708", "#15161718"),
                )
            )

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError):
            parse_color_table(
                _table_bytes(
                    names=("Alpha", " Alpha "),
                    rain_colors=("#01020304", "#11121314"),
                    snow_colors=("#05060708", "#15161718"),
                )
            )

    def test_rejects_zero_scheme_columns(self):
        rows = _valid_rows(("#01020304",), "#05060708")
        raw = _table_from_rows((), rows)
        with pytest.raises(ValueError):
            parse_color_table(raw)

    def test_rejects_more_than_255_scheme_columns(self):
        names = tuple(f"Scheme {i}" for i in range(256))
        colors = tuple("#01020304" for _ in range(256))
        with pytest.raises(ValueError):
            parse_color_table(_table_bytes(
                names=names, rain_colors=colors, snow_colors=colors,
            ))

    def test_rejects_undecodable_bytes(self):
        with pytest.raises(ValueError):
            parse_color_table(b"\xff\xfe\x00")


class TestApplyExternalTable:
    def test_applies_custom_two_scheme_table(self, packaged_palette):
        raw = _table_bytes()
        names_obj = SCHEME_NAMES

        version = apply_external_table(raw)

        assert version == hashlib.sha256(raw).hexdigest()[:12]
        assert current_version() == version
        assert SCHEME_NAMES == {0: "Alpha", 1: "Beta"}
        # Same dict object (routes.py holds this exact binding).
        assert SCHEME_NAMES is names_obj
        assert get_lut(0, snow=True)[0].tolist() == [5, 6, 7, 8]
        assert colorize(
            np.array([[0, 128]], dtype=np.uint8), scheme=0,
        ).tolist() == [
            [[1, 2, 3, 4], [1, 2, 3, 4]],
        ]
        # Raw survives as a synthesized entry.
        assert get_lut(255)[128].tolist() == [128, 128, 128, 255]

    def test_malformed_input_leaves_everything_untouched(self, packaged_palette):
        before_names = dict(SCHEME_NAMES)
        before_version = current_version()
        before_rain = schemes._rain_luts
        before_snow = schemes._snow_luts

        with pytest.raises(ValueError):
            apply_external_table(b"not,a,color,table")

        assert SCHEME_NAMES == before_names
        assert current_version() == before_version
        assert schemes._rain_luts is before_rain
        assert schemes._snow_luts is before_snow

    def test_version_changes_across_applies(self, packaged_palette):
        first = current_version()
        apply_external_table(_table_bytes(names=("One", "Two")))
        second = current_version()
        apply_external_table(_table_bytes(names=("Three", "Four")))
        third = current_version()
        assert len({first, second, third}) == 3

    def test_reapplying_identical_bytes_is_stable(self, packaged_palette):
        raw = _table_bytes()
        assert apply_external_table(raw) == apply_external_table(raw)


class TestFallbackRule:
    def test_default_table_falls_back_to_scheme_7(self):
        np.testing.assert_array_equal(get_lut(99), get_lut(7))
        np.testing.assert_array_equal(get_lut(99, snow=True), get_lut(7, snow=True))

    def test_custom_table_without_7_uses_lowest_id(self, packaged_palette):
        apply_external_table(_table_bytes())
        np.testing.assert_array_equal(get_lut(99), get_lut(0))
        np.testing.assert_array_equal(get_lut(99, snow=True), get_lut(0, snow=True))
        assert get_lut(99) is get_lut(0)

    def test_custom_table_with_7_prefers_7(self, packaged_palette):
        names = tuple(f"S{i}" for i in range(9))
        colors = tuple(f"#0102030{i}" for i in range(9))
        snow = tuple(f"#0506070{i}" for i in range(9))
        apply_external_table(_table_bytes(
            names=names, rain_colors=colors, snow_colors=snow,
        ))
        np.testing.assert_array_equal(get_lut(99), get_lut(7))


class TestCaptureLuts:
    def test_capture_matches_get_lut(self):
        rain, snow, version = capture_luts(7)
        np.testing.assert_array_equal(rain, get_lut(7))
        np.testing.assert_array_equal(snow, get_lut(7, snow=True))
        assert version == current_version()

    def test_capture_version_changes_after_apply(self, packaged_palette):
        before_rain, before_snow, before = capture_luts(0)
        apply_external_table(_table_bytes())
        after_rain, after_snow, after = capture_luts(0)
        assert after != before
        assert after_rain is get_lut(0)
        assert after_snow is get_lut(0, snow=True)
        # The captured tables are the post-apply ones, not the old pair.
        assert after_rain[0].tolist() == [1, 2, 3, 4]
        assert before_rain is not after_rain


class TestColorizeWithLut:
    def test_pure_gather_matches_colorize(self):
        values = np.array([[0, 64, 255]], dtype=np.uint8)
        lut = get_lut(3)
        np.testing.assert_array_equal(
            colorize_with_lut(values, lut), colorize(values, 3),
        )
