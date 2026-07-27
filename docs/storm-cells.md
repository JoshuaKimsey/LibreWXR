# Storm-Cell Detection

LibreWXR detects convective storm cells on every radar fetch cycle and
optionally overlays them on radar tiles via the `?cells=light|dark` URL
parameter -- the visual parallel to the existing `?arrows=` motion-arrow
overlay.

## How it works

Each fetch cycle (every `LIBREWXR_FETCH_INTERVAL` seconds), the latest
radar frame per region is thresholded at a configurable dBZ cutoff
(default 40 dBZ -- moderate-to-heavy convection). Contiguous pixels above
the threshold are grouped into cells via `cv2.connectedComponentsWithStats`.
Cells smaller than a minimum area (default 25 km^2) are filtered out as
noise. For each surviving cell, the optical flow field (the same one that
powers the `?arrows=` overlay) is sampled at the cell's centroid to derive
a motion vector (speed in km/h + compass heading).

Detected cells are stored in `StormCellStore` and read by the tile
renderer on demand -- the `?cells=` parameter is a present-time overlay
that doesn't affect the tile geometry cache.

## Enabling the overlay

Add `?cells=light` or `?cells=dark` to any radar tile URL:

```
# Light cells (for dark map themes)
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?cells=light

# Dark cells (for light map themes)
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?cells=dark

# Combined with motion arrows
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?arrows=light&cells=dark
```

The `?cells=` and `?arrows=` parameters are independent -- use either,
both, or neither.

## Visual encoding

Each detected cell is drawn as:

- **Filled circle** at the cell's centroid -- radius scales logarithmically
  with area (25 km^2 -> ~4px, 1000 km^2 -> ~12px). Color depends on the style
  (`light` = white, `dark` = dark grey).
- **Motion arrow** from the centroid (when motion data is available) --
  shows the storm's direction and relative speed, derived from the optical
  flow field. Cells without motion data (e.g. nowcast disabled) show only
  the circle, no arrow.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LIBREWXR_STORM_CELLS_ENABLED` | `true` | Master switch for storm-cell detection. When `false`, no detection runs and `?cells=` has no effect. |
| `LIBREWXR_STORM_CELLS_MIN_DBZ` | `40` | Minimum dBZ for a pixel to be part of a detected cell. Lower values detect more cells (including weak/shallow convection); higher values detect only strong cells. |
| `LIBREWXR_STORM_CELLS_MIN_AREA_KM2` | `25.0` | Minimum cell area in km^2. Filters out small/noise cells. Increase to show only large MCS-type systems; decrease to show individual cells. |

## Limitations

- **Latest-detection only.** The overlay uses the latest detection result
  regardless of which timestamp the tile URL requests. This is the same
  approximation the `?arrows=` overlay uses (arrows use the latest flow
  field regardless of timestamp). Historical cell positions are not
  reconstructed.
- **Radar-only.** Storm cells are detected on radar frames, not NWP or
  satellite. Areas without radar coverage show no cells.
- **Per-region detection.** Cells are detected independently per radar
  region. A storm spanning a region boundary may appear as two adjacent
  cells. Deduplication is deferred to the MCP `get_storm_cells` tool
  (Phase 2, not yet shipped).

## Health monitoring

The `/health` endpoint surfaces storm-cell status:

```json
{
  "storm_cells": {
    "enabled": true,
    "count": 42,
    "last_updated": 1785130265,
    "per_region": {"USCOMP": 15, "CACOMP": 3, "OPERA": 24}
  }
}
```

If `enabled: true` but `count: 0`, either no cells were detected in the
latest cycle (clear weather) or the detection failed silently -- check the
startup log for "Storm-cell detection failed" exceptions.
