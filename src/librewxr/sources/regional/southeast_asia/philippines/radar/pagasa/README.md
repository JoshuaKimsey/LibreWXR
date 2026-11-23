# PAGASA PANAHON radar composite (Philippines)

National radar mosaic from **PAGASA / DOST** (Philippine Atmospheric,
Geophysical and Astronomical Services Administration), served anonymously
from the PANAHON web app's same-origin API at `panahon.gov.ph` (the old
`cdn.panahon.gov.ph` host was retired in November 2026).

## Coverage

| Region   | Footprint                                                    |
| -------- | ------------------------------------------------------------ |
| `PHCOMP` | Philippines (Luzon + Visayas + Mindanao + W. Palawan, plus edges of Sabah and N. Sulawesi) |

Single 2048×2048 national mosaic — one timeline + image fetch pair per
cycle.

## Cadence & latency

- Native cadence: **10 min**, matching LibreWXR's store cadence 1:1, so
  each requested store slot maps onto a native frame.
- The timeline endpoint carries **9 frames** (~90 min of backfill), each
  with an explicit UTC `observed_at_unix` timestamp — no publish-lag math
  or timestamp inference needed.  Archive lookups older than the rolling
  buffer return no data.
- While the upstream radar renderer is unavailable, the image endpoint
  answers HTTP 200 with a universal 1×1 RGBA placeholder rather than an
  error.  The source's 2048×2048 size guard rejects it, so those frames
  are skipped with a warning instead of silently replacing real data.

## Endpoints

- `GET {base}/api/v1/radar/timeline?sublayer=mosaic-reflectivity` — JSON
  timeline: 9 frames at 10-min cadence with `observed_at_unix`, plus
  `tile_version`, `scale`, `colormap` and `bounds`.  The per-entry
  `image_url` is always empty now, so the source synthesizes the image URL
  from the timestamp.
- `GET {base}/api/v1/radar-data-image?sublayer=mosaic-reflectivity&t={observed_at_unix}&mode=dbz&size=2048&v={tile_version}` —
  2048×2048 RGBA PNG for that timestamp.

The host soft-blocks non-browser User-Agents with a 1280×720 decoy PNG at
HTTP 200 (verified 2026-11-23), so the source sends a browser UA on every
request.

## Format

The PNG uses the 14-stop 0-80 dBZ colormap carried in the timeline
response (`colormap` plus `scale.max`).  Those fields are authoritative —
upstream has retuned its palette before — and a verbatim fallback table
ships in `source.py` for a response that omits or mangles them.  Each stop
decodes to the midpoint of its dBZ range, with the upper bound clamped to
`scale.max` (so the open-ended top stop `[65, 999]` decodes to 72.5).
Zero anti-aliasing — every visible pixel is an exact palette match, decoded
by nearest-RGB lookup.  Alpha is discretised per stop: the three leading
weak-echo gray stops decode to no-data (they sit below LibreWXR's default
10 dBZ noise floor anyway); precip stops are α=255.

The mosaic is a union of ~240 km coverage circles around the national
station network (nine stations in the app's current list: Baguio, Baler,
Echague, Daet, Guiuan, Iloilo, Davao, Kabacan, Panabo).

## License & attribution

Public domain per Philippine IP code **RA 8293 §176** (government-works
exception).  Caveat: §176.1 requires PAGASA approval for for-profit
exploitation of the work — fine for a self-hosted AGPL instance;
worth flagging if you ever operate a paid hosting service.  Attribution
to **PAGASA / DOST**.

## Stations

Nine stations (Baguio, Baler, Echague, Daet, Guiuan, Iloilo, Davao,
Kabacan, Panabo).  Coordinates and range settings live in `stations.py`
and feed `data/coverage.py` via the `RadarSourceContribution`.