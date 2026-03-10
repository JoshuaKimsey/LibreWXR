# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np

from librewrx.colors.schemes import colorize, get_lut, SCHEME_NAMES


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
        """Unknown scheme ID should fall back to Universal Blue."""
        lut = get_lut(99)
        expected = get_lut(2)
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
