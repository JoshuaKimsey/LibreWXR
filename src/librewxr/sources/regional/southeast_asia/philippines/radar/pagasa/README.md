# PAGASA PANAHON radar composite (Philippines)

National radar mosaic from **PAGASA / DOST** (Philippine Atmospheric,
Geophysical and Astronomical Services Administration), served anonymously
from the PANAHON web app's API at `cdn.panahon.gov.ph/api/v1`.

## Coverage

| Region   | Footprint                                                    |
| -------- | ------------------------------------------------------------ |
| `PHCOMP` | Philippines (Luzon + Visayas + Mindanao + W. Palawan, plus edges of Sabah and N. Sulawesi) |

Single 2048×2048 national mosaic — one timeline + image fetch pair per
cycle.

## Cadence & latency

- Native cadence: **15 min** (LibreWXR's store cadence is 10 min, so
  each requested store slot is rounded to the nearest native frame,
  ≤7.5 min off — invisible in a RainViewer-style animation).
- The timeline endpoint carries **6 frames** (~75 min of backfill),
  each with an explicit UTC `observed_at_unix` timestamp — no
  publish-lag math or timestamp inference needed.  Archive lookups
  older than the rolling buffer return no data.

## Endpoints

- `GET {base}/api/v1/radar/timeline` — JSON timeline: 6 frames at
  15-min cadence with `observed_at_unix` and per-frame `image_url`.
- `GET {base}/api/v1/radar-image?sublayer=hybrid-reflectivity&index=N` —
  2048×2048 RGBA PNG for frame index `N`.

## Format

The PNG uses a 13-stop linear 0–75 dBZ palette taken verbatim from the
PANAHON JS bundle's `generateGradientColormap(palette, 13, [0, 75])`
call.  Zero anti-aliasing — every visible pixel is an exact palette
match (100% match rate observed in a 2026-08-19 probe), decoded by
nearest-RGB lookup.  Alpha is discretised per stop: the three weak-echo
gray stops render with α<255 and decode to no-data (they sit below
LibreWXR's default 10 dBZ noise floor anyway); precip stops are α=255.

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