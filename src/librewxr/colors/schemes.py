# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import csv
import hashlib
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

# Scheme ID -> display name.  Populated at import time from the packaged
# ``color_table.csv`` header, so the registry is a complete, non-empty
# truth the moment the module imports (scripts import and read it without
# ever touching a LUT).  An external palette applies *in place* via
# ``apply_external_table``: ``api.routes`` imports this exact dict object
# by name, so it must be mutated, never rebound.
SCHEME_NAMES: dict[int, str] = {}

# Rain LUTs: scheme_id -> (256, 4) uint8 array mapping pixel value -> RGBA
_rain_luts: dict[int, np.ndarray] = {}
# Snow LUTs: same structure but for snow color variants
_snow_luts: dict[int, np.ndarray] = {}

# Content hash (``sha256(raw_bytes)[:12]``) of the table currently applied.
# Always a concrete value once loaded - there is no None/absent state.
_TABLE_VERSION: str = ""

# Guards the lazy-init in ``get_lut`` and the apply/rebind in
# ``apply_external_table``.  Without this, a tile renderer running many
# parallel ``colorize()`` calls on a fresh process can race: thread A is
# partway through ``_load_color_table`` (so the globals are non-empty but
# missing some scheme IDs), thread B sees the truthy dict, skips the
# reload, and KeyErrors on whatever scheme A hasn't populated yet.  Lock +
# atomic reassignment in ``_load_color_table`` eliminate that window.
_load_lock = threading.Lock()

_HEX_RGBA_RE = re.compile(r"^#[0-9A-Fa-f]{8}$")

# One header row plus one row per pixel value (0-255): rows 0-127 are the
# rain block, rows 128-255 the snow block.
_HEADER_ROWS = 1
_DATA_ROWS = 256
_RAIN_ROWS = 128
_MAX_SCHEMES = 255


@dataclass
class ParsedTable:
    """A validated color table, fully materialized.

    Produced by :func:`parse_color_table` and consumed either by
    :func:`apply_external_table` (global mutation) or by the pipeline's
    dry-run validation (no mutation at all).  Both LUT dicts include the
    synthesized Raw entry at key 255; ``names`` never includes it.
    """

    names: dict[int, str]
    rain_luts: dict[int, np.ndarray]
    snow_luts: dict[int, np.ndarray]
    version: str


@lru_cache(maxsize=1)
def packaged_table_bytes() -> bytes:
    """The packaged ``color_table.csv`` bytes (the built-in palette)."""
    return files("librewxr.colors").joinpath("color_table.csv").read_bytes()


def _parse_hex_rgba(hex_str: str) -> tuple[int, int, int, int]:
    """Parse '#RRGGBBAA' to (R, G, B, A)."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)


def _raw_lut() -> np.ndarray:
    """Scheme 255 (Raw): identity mapping - pixel value becomes grayscale."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        lut[i] = [i, i, i, 255 if i > 0 else 0]
    return lut


def parse_color_table(raw: bytes) -> ParsedTable:
    """Parse and strictly validate a color table, without mutating globals.

    Contract (any violation raises ``ValueError``):

    - UTF-8 decodable.
    - Exactly one header row plus 256 data rows.
    - Header cell 0 is decorative (any content); header cells 1..N are the
      scheme display names, stripped, non-empty and unique.
    - Scheme column count 1..255 (IDs ``0..N-1`` appear in column order).
    - Every data row has exactly ``N+1`` cells; every scheme cell matches
      ``#RRGGBBAA``.
    - Data rows 0-127 are the rain block, 128-255 the snow block; LUT
      value ``v`` reads block row ``v // 2``.
    - ``version`` is ``sha256(raw)[:12]``.

    Scheme 255 (Raw) is synthesized here and deliberately absent from
    ``names`` (it is not a dataset column).
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"color table is not valid UTF-8: {e}") from e

    try:
        # ``splitlines`` normalizes CRLF and drops a single trailing
        # newline, so a Windows-saved table validates the same as the
        # packaged (LF, no trailing newline) one.
        rows = list(csv.reader(text.splitlines()))
    except csv.Error as e:
        raise ValueError(f"color table is not valid CSV: {e}") from e

    if len(rows) != _HEADER_ROWS + _DATA_ROWS:
        raise ValueError(
            f"color table must have {_HEADER_ROWS} header row and "
            f"{_DATA_ROWS} data rows, got {len(rows)} row(s)"
        )

    header = rows[0]
    if len(header) < 2:
        raise ValueError("color table header has no scheme columns")
    scheme_count = len(header) - 1
    if scheme_count > _MAX_SCHEMES:
        raise ValueError(
            f"color table has {scheme_count} scheme columns, max {_MAX_SCHEMES}"
        )

    names: dict[int, str] = {}
    seen: set[str] = set()
    for idx in range(1, scheme_count + 1):
        name = header[idx].strip()
        if not name:
            raise ValueError(f"scheme column {idx} has an empty name")
        if name in seen:
            raise ValueError(f"duplicate scheme name {name!r}")
        seen.add(name)
        names[idx - 1] = name

    data_rows = rows[1:]
    for row_idx, row in enumerate(data_rows):
        if len(row) != scheme_count + 1:
            raise ValueError(
                f"data row {row_idx} has {len(row)} cell(s), "
                f"expected {scheme_count + 1}"
            )
        for cell in row[1:]:
            if not _HEX_RGBA_RE.match(cell):
                raise ValueError(
                    f"data row {row_idx} has invalid color cell {cell!r}"
                )

    rain_rows = data_rows[:_RAIN_ROWS]
    snow_rows = data_rows[_RAIN_ROWS:]

    raw_lut = _raw_lut()
    rain_luts: dict[int, np.ndarray] = {}
    snow_luts: dict[int, np.ndarray] = {}
    for scheme_id in range(scheme_count):
        col = scheme_id + 1

        rain_lut = np.zeros((256, 4), dtype=np.uint8)
        for pixel_val in range(256):
            rain_lut[pixel_val] = _parse_hex_rgba(
                rain_rows[pixel_val // 2][col]
            )
        rain_luts[scheme_id] = rain_lut

        snow_lut = np.zeros((256, 4), dtype=np.uint8)
        for pixel_val in range(256):
            snow_lut[pixel_val] = _parse_hex_rgba(
                snow_rows[pixel_val // 2][col]
            )
        snow_luts[scheme_id] = snow_lut

    rain_luts[255] = raw_lut
    snow_luts[255] = raw_lut

    version = hashlib.sha256(raw).hexdigest()[:12]
    return ParsedTable(
        names=names, rain_luts=rain_luts, snow_luts=snow_luts, version=version,
    )


def _fallback_scheme(luts: dict[int, np.ndarray], scheme: int) -> int:
    """Resolve an unknown scheme ID: 7 when present, else the lowest ID."""
    if scheme in luts:
        return scheme
    if 7 in luts:
        return 7
    return min(luts)


def _lookup(luts: dict[int, np.ndarray], scheme: int) -> np.ndarray:
    return luts[_fallback_scheme(luts, scheme)]


def _load_color_table() -> None:
    """Parse the packaged color_table.csv into the module globals.

    Builds the rain / snow LUT dicts in local variables and assigns them
    to the module globals in a single statement at the end, so no caller
    can ever observe a partially-populated dict.  Combined with the
    ``_load_lock`` in ``get_lut`` this makes lazy init safe under
    arbitrary concurrent access.
    """
    global _rain_luts, _snow_luts, _TABLE_VERSION

    parsed = parse_color_table(packaged_table_bytes())

    # Atomic publish: every concurrent reader sees either the empty
    # dict (load hasn't run) or the fully-populated dict (load done).
    _rain_luts = parsed.rain_luts
    _snow_luts = parsed.snow_luts
    _TABLE_VERSION = parsed.version


def _ensure_loaded() -> None:
    if not _rain_luts:
        with _load_lock:
            if not _rain_luts:  # double-check inside the lock
                _load_color_table()


def get_lut(scheme: int, snow: bool = False) -> np.ndarray:
    """Get the 256-entry RGBA LUT for a color scheme.

    Returns array of shape (256, 4) dtype uint8.  Unknown scheme IDs
    fall back to scheme 7 (Rainbow @ Selex SI) when present, otherwise to
    the lowest available ID.
    """
    _ensure_loaded()
    luts = _snow_luts if snow else _rain_luts
    return _lookup(luts, scheme)


def current_version() -> str:
    """Content hash of the palette currently applied (12 hex chars)."""
    _ensure_loaded()
    return _TABLE_VERSION


def capture_luts(scheme: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Capture (rain_lut, snow_lut, version) in one locked call.

    Taking all three under a single acquisition means a concurrent
    ``apply_external_table`` can never hand a caller a rain LUT from one
    palette and a snow LUT from another (or a LUT/version mismatch).
    """
    _ensure_loaded()
    with _load_lock:
        rain = _lookup(_rain_luts, scheme)
        snow = _lookup(_snow_luts, scheme)
        return rain, snow, _TABLE_VERSION


def apply_external_table(raw: bytes) -> str:
    """Validate and adopt an external color table, returning its version.

    All validation happens in :func:`parse_color_table` before any global
    is touched, so malformed bytes raise ``ValueError`` and leave the
    registry, LUTs and version exactly as they were.  On success the LUT
    dicts are rebound atomically (same pattern as the lazy load) and
    ``SCHEME_NAMES`` is mutated *in place* - ``api.routes`` holds this
    exact dict object by name, so it must never be rebound.
    """
    global _rain_luts, _snow_luts, _TABLE_VERSION

    parsed = parse_color_table(raw)

    with _load_lock:
        _rain_luts = parsed.rain_luts
        _snow_luts = parsed.snow_luts
        SCHEME_NAMES.clear()
        SCHEME_NAMES.update(parsed.names)
        _TABLE_VERSION = parsed.version

    return parsed.version


def colorize_with_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Gather ``lut`` at ``values``: the pure step of :func:`colorize`.

    ``lut`` is a (256, 4) uint8 RGBA table, so the result has shape
    ``(*values.shape, 4)``.
    """
    return lut[values]


def colorize(values: np.ndarray, scheme: int, snow: bool = False) -> np.ndarray:
    """Apply color scheme to raw pixel values.

    Args:
        values: uint8 array of any shape
        scheme: color scheme ID
        snow: use snow color variant

    Returns:
        RGBA array of shape (*values.shape, 4)
    """
    return colorize_with_lut(values, get_lut(scheme, snow))


# Populate the name registry + version at import time.  The LUT dicts stay
# empty on purpose: building them is still lazy (first get_lut/colorize),
# which keeps a script that only reads SCHEME_NAMES cheap.
_PARSED_PACKAGED = parse_color_table(packaged_table_bytes())
SCHEME_NAMES = _PARSED_PACKAGED.names
_TABLE_VERSION = _PARSED_PACKAGED.version
