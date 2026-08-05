# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Adaptive lossless 8-bit palette (PNG8) encoding for tile PNG output.

Tiles whose final RGBA pixels contain ``_PALETTE_MIN_COLORS``..256 unique
colors are encoded as exact-palette P-mode PNGs: the palette is built from
the actual unique colors present in the image (no quantizer), so decoding
the PNG and converting back to RGBA reproduces the input pixels bit-for-bit,
including full 8-bit alpha via the tRNS chunk.  Everything else keeps the
plain 32-bit RGBA encoding.

The palette path must stay numpy-only on the hot path: the only Python
lists built here are the <=256-entry palette / transparency arrays, which
are negligible next to the pixel counting work.
"""
import io

import numpy as np
from PIL import Image

# Minimum unique colors before the palette path is worth taking.  Pillow
# does NOT pad a hand-supplied palette — measured with Pillow 12.3.0: the
# PLTE chunk scales as 3 * n_entries (6 B at 2 colors, 9 B at 3, 120 B at
# 40, and only reaches the 768 B cap at the 256-entry limit), so a 2-color
# tile is already smaller as PNG8 than as RGBA.  1-color tiles stay on the
# RGBA path so fully transparent tiles keep decoding as RGBA (pre-change
# behavior).
_PALETTE_MIN_COLORS = 2


def encode_png(img: Image.Image) -> bytes:
    """Encode an RGBA image as PNG, adaptively PNG8-palette or plain RGBA.

    Tiles with ``_PALETTE_MIN_COLORS``..256 unique RGBA colors use an
    exact-color palette (lossless, alpha preserved).  All others use the
    previous 32-bit RGBA encoding.  Output is byte-for-byte deterministic
    for identical input.
    """
    arr = np.asarray(img)
    packed = (
        (arr[..., 0].astype(np.uint32) << 24)
        | (arr[..., 1].astype(np.uint32) << 16)
        | (arr[..., 2].astype(np.uint32) << 8)
        | arr[..., 3].astype(np.uint32)
    )
    # ``np.unique`` sorts, which makes the palette order (and therefore the
    # encoded bytes) deterministic for a given input image.
    uniq, inverse = np.unique(packed.ravel(), return_inverse=True)
    n_colors = len(uniq)
    if _PALETTE_MIN_COLORS <= n_colors <= 256:
        idx = inverse.reshape(arr.shape[:2]).astype(np.uint8)
        out = Image.fromarray(idx, mode="P")
        palette = np.empty((n_colors, 3), dtype=np.uint8)
        palette[:, 0] = (uniq >> 24) & 0xFF
        palette[:, 1] = (uniq >> 16) & 0xFF
        palette[:, 2] = (uniq >> 8) & 0xFF
        alphas = (uniq & 0xFF).astype(np.uint8)
        out.putpalette(list(palette.ravel()))
        out.info["transparency"] = bytes(alphas.tolist())
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True, compress_level=6)
        return buf.getvalue()
    # 1 color, or more than 256 unique colors: plain 32-bit RGBA (the
    # previous PNG encoding).  Level 6, not 1: Pillow wheels link the host
    # libz (stock zlib on Debian, zlib-ng on Fedora) and the two diverge up
    # to ~4x in level-1 output size; level 6 converges and the palette path
    # already uses 6.
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()
