#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generate the LibreWXR color-scheme preview maps.

Writes two PNGs to ``docs/``:

  * ``color-schemes-rain.png`` — rain palette for every scheme in
    ``SCHEME_NAMES``, stacked one per row, dBZ axis 0..85.
  * ``color-schemes-snow.png`` — snow palette equivalents, dBZ -10..50.

One-off authoring tool — not a runtime dependency.  Regenerate after
editing ``src/librewxr/colors/color_table.csv`` or adding a new scheme
to ``SCHEME_NAMES``.

# Regenerate with:
#   python3 -m venv /tmp/coverage-map-venv
#   /tmp/coverage-map-venv/bin/pip install matplotlib numpy
#   /tmp/coverage-map-venv/bin/python scripts/generate_color_scheme_previews.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RAIN_OUTPUT = REPO_ROOT / "docs" / "color-schemes-rain.png"
SNOW_OUTPUT = REPO_ROOT / "docs" / "color-schemes-snow.png"

# Pull the scheme list + LUTs straight from the project — adding a new
# scheme to ``SCHEME_NAMES`` propagates through automatically.
sys.path.insert(0, str(REPO_ROOT / "src"))
from librewxr.colors.schemes import SCHEME_NAMES, get_lut  # noqa: E402


# Pixel encoding (matches ``data.store`` / ``colorize``):
#   pixel_value = (dBZ + 32) * 2
def pixel_for_dbz(dbz: np.ndarray) -> np.ndarray:
    return np.clip((dbz + 32) * 2, 0, 255).astype(np.uint8)


def render_panel(
    output_path: Path,
    snow: bool,
    dbz_min: int,
    dbz_max: int,
    title: str,
) -> None:
    """One PNG containing one horizontal strip per scheme, stacked vertically."""
    schemes = list(SCHEME_NAMES.items())
    dbz_axis = np.arange(dbz_min, dbz_max + 1)
    pixel_values = pixel_for_dbz(dbz_axis)

    n = len(schemes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 0.55 * n + 1.2), dpi=140)
    if n == 1:
        axes = [axes]

    for ax, (scheme_id, name) in zip(axes, schemes):
        lut = get_lut(scheme_id, snow=snow)
        strip = lut[pixel_values].reshape(1, -1, 4)
        ax.imshow(
            strip, aspect="auto",
            extent=[dbz_min, dbz_max, 0, 1],
            interpolation="nearest",
        )
        ax.set_yticks([])
        # Tick every 5 dBZ.
        tick_start = (dbz_min // 5) * 5
        ax.set_xticks(np.arange(tick_start, dbz_max + 1, 5))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(
            f"Scheme {scheme_id}: {name}",
            fontsize=10, loc="left", pad=4,
        )
        # Only the bottom strip needs an x-axis label.
        if ax is axes[-1]:
            ax.set_xlabel("dBZ", fontsize=10)
        else:
            ax.set_xticklabels([])

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    RAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Rain palettes: 0..85 dBZ — covers the full reflectivity range used
    # by every shipping scheme (Valerio caps at 85; most others fade out
    # well below).
    render_panel(
        output_path=RAIN_OUTPUT,
        snow=False,
        dbz_min=0, dbz_max=85,
        title="LibreWXR — Rain Color Schemes",
    )

    # Snow palettes: -10..50 dBZ — frozen-precip reflectivity rarely
    # exceeds 40 dBZ in practice; this range shows where each scheme
    # places its visible band.
    render_panel(
        output_path=SNOW_OUTPUT,
        snow=True,
        dbz_min=-10, dbz_max=50,
        title="LibreWXR — Snow Color Schemes",
    )
