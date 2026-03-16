# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import csv
from importlib.resources import files

import numpy as np

# Scheme ID -> column index in color_table.csv
SCHEME_NAMES = {
    0: "Black and White",
    1: "Original",
    2: "Universal Blue",
    3: "Titan",
    4: "The Weather Channel (TWC)",
    5: "Meteored",
    6: "NEXRAD Level III",
    7: "Rainbow @ Selex SI",
    8: "Dark Sky",
}

# Rain LUTs: scheme_id -> (256, 4) uint8 array mapping pixel value -> RGBA
_rain_luts: dict[int, np.ndarray] = {}
# Snow LUTs: same structure but for snow color variants
_snow_luts: dict[int, np.ndarray] = {}


def _parse_hex_rgba(hex_str: str) -> tuple[int, int, int, int]:
    """Parse '#RRGGBBAA' to (R, G, B, A)."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)


def _load_color_table() -> None:
    """Parse color_table.csv into LUTs."""
    global _rain_luts, _snow_luts

    csv_path = files("librewxr.colors").joinpath("color_table.csv")
    text = csv_path.read_text()
    reader = csv.reader(text.strip().splitlines())
    header = next(reader)

    # Column indices for each scheme (skip first column which is dBZ)
    scheme_cols = {sid: i + 1 for i, sid in enumerate(SCHEME_NAMES.keys())}

    # Rows 0-127: rain, rows 128-255: snow
    rain_rows: list[list[str]] = []
    snow_rows: list[list[str]] = []

    for i, row in enumerate(reader):
        if not row or not row[0].strip():
            continue
        if i < 128:
            rain_rows.append(row)
        else:
            snow_rows.append(row)

    for scheme_id, col_idx in scheme_cols.items():
        # Build 256-entry LUT for rain
        rain_lut = np.zeros((256, 4), dtype=np.uint8)
        for pixel_val in range(256):
            # Map pixel value to color table row: pixel_value // 2
            color_idx = pixel_val // 2
            if color_idx < len(rain_rows) and col_idx < len(rain_rows[color_idx]):
                r, g, b, a = _parse_hex_rgba(rain_rows[color_idx][col_idx])
                rain_lut[pixel_val] = [r, g, b, a]
        _rain_luts[scheme_id] = rain_lut

        # Build 256-entry LUT for snow
        snow_lut = np.zeros((256, 4), dtype=np.uint8)
        for pixel_val in range(256):
            color_idx = pixel_val // 2
            if color_idx < len(snow_rows) and col_idx < len(snow_rows[color_idx]):
                r, g, b, a = _parse_hex_rgba(snow_rows[color_idx][col_idx])
                snow_lut[pixel_val] = [r, g, b, a]
        _snow_luts[scheme_id] = snow_lut

    # Scheme 255 (Raw): identity mapping - pixel value becomes grayscale
    raw_lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        raw_lut[i] = [i, i, i, 255 if i > 0 else 0]
    _rain_luts[255] = raw_lut
    _snow_luts[255] = raw_lut


def get_lut(scheme: int, snow: bool = False) -> np.ndarray:
    """Get the 256-entry RGBA LUT for a color scheme.

    Returns array of shape (256, 4) dtype uint8.
    """
    if not _rain_luts:
        _load_color_table()

    luts = _snow_luts if snow else _rain_luts
    if scheme not in luts:
        scheme = 2  # default to Universal Blue
    return luts[scheme]


def colorize(values: np.ndarray, scheme: int, snow: bool = False) -> np.ndarray:
    """Apply color scheme to raw pixel values.

    Args:
        values: uint8 array of any shape
        scheme: color scheme ID
        snow: use snow color variant

    Returns:
        RGBA array of shape (*values.shape, 4)
    """
    lut = get_lut(scheme, snow)
    return lut[values]
