# Lightning

LibreWXR ingests total-lightning flash points from two satellite
families -- NOAA GOES GLM (Geostationary Lightning Mapper) on the
GOES-East + GOES-West geostationary disks, and the EUMETSAT Meteosat-12
MTG Lightning Imager over the Eastern Hemisphere -- and optionally
overlays them on radar tiles via the `?lightning=` URL parameter -- the
storm-scale parallel to the existing `?arrows=` motion-arrow and
`?cells=` storm-cell overlays. The same strikes are served as GeoJSON at
`GET /v2/lightning` and through the MCP `get_recent_lightning` tool.
Strikes are retained for two hours by default -- matching the 12-frame x
10-minute radar history -- so every radar frame in the timeline can carry
its own lightning overlay.
GOES GLM flashes are US public domain, read from the anonymous NOAA NODD
S3 buckets (courtesy credit: "Lightning: NOAA GOES-R Geostationary
Lightning Mapper (GLM)"); Meteosat strikes come from the EUMETSAT Data
Store under a free account (required attribution: "Contains modified
EUMETSAT Meteosat data 2026").

## Table of Contents

- [How it works](#how-it-works)
- [Enabling the overlay](#enabling-the-overlay)
- [Frame semantics](#frame-semantics)
- [Visual encoding](#visual-encoding)
- [Configuration](#configuration)
- [REST endpoint](#rest-endpoint)
- [MCP tool](#mcp-tool)
- [Health monitoring](#health-monitoring)
- [Coverage and attribution](#coverage-and-attribution)
- [See also](#see-also)

## How it works

The NOAA family fetches GLM L2 LCFA total-lightning flash files from
the anonymous NODD buckets (`noaa-goes19` for GOES-East and `noaa-goes18`
for GOES-West) on the pipeline's clock-aligned loop, every
`LIBREWXR_LIGHTNING_FETCH_INTERVAL` seconds (default 300, i.e. every
5 minutes). Each satellite keeps a resume watermark, so pipeline restarts
pick up where they left off instead of re-decoding old files. The GLM
listing walks every S3 day-directory from the retention cutoff back to
now, so a cold start just after midnight UTC still reaches the previous
UTC day's files instead of missing them. The loop is deliberately its own
knob: disabling alerts never disables lightning, and vice versa.

Two families feed that one loop. The NOAA side is anonymous S3, gated
by `LIBREWXR_LIGHTNING_NOAA_ENABLED` (default true). The EUMETSAT family
(`data/lightning_eumetsat.py`, `LIBREWXR_LIGHTNING_EUMETSAT_ENABLED`,
default true) is Meteosat-12 (MTG-I1 at 0°E) L2 per-flash points from the
EUMETSAT Data Store collection `EO:EUM:DAT:0691`, and -- the project's
only API-key source -- it stays inert until
`LIBREWXR_LIGHTNING_EUMETSAT_API_KEY` and
`LIBREWXR_LIGHTNING_EUMETSAT_API_SECRET` are both configured. A single
multi-family coordinator drives both into the same `LightningStore` and
the same on-disk artifact, tagging each point with its satellite
identity (`goes18` / `goes19` / `meteosat12`); a family that is disabled,
credential-less, or stale simply contributes nothing.

Decoded flashes are held in `LightningStore` (`data/lightning_store.py`)
as structured points -- every flash inside the retention window
(`LIBREWXR_LIGHTNING_MAX_AGE`, default 7200 s = 2 hours, matching the
12-frame x 10-minute radar history so every radar frame's tile slot can
carry lightning). The store's RAM ceiling is a warn-once circuit breaker,
not a data policy, and it scales with retention: 500k points per 30
minutes of history, floored at 500k (2,000,000 at the 2-hour default).
Drop-oldest trimming is unchanged; under normal load even the 2-hour
window holds tens of thousands of points.

The pipeline writes the held points to a shared on-disk artifact,
`<cache_dir>/lightning/current.npz`, with per-satellite resume
watermarks in `<cache_dir>/lightning/watermark.json`. Lightning has
deliberately NO `state.json` section -- the artifact IS the cross-process
handoff, the same shared-tile-store / precip-mask pattern. Render workers
and the stdio MCP transport refresh from the artifact via a cheap `mtime`
stat on demand.

A satellite feed going stale is a degraded hemisphere, never a crash: its
last-known points persist until they age out of the retention window, and
the staleness surfaces in `/health`.

## Enabling the overlay

Add `?lightning=` to any radar tile URL:

```
# Energy-scaled dots (default style)
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?lightning=dots

# Hand-authored vector bolt glyphs
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?lightning=bolts

# Combined with motion arrows and storm cells
https://api.librewxr.net/v2/radar/{timestamp}/256/{z}/{x}/{y}/10/1_1.png?arrows=light&cells=dark&lightning=dots
```

| `?lightning=` value | Effect |
|---|---|
| *(omitted)* / `""` | Off (default) |
| `1`, `true`, `dots` | Small energy-scaled dots |
| `bolts` | Hand-authored vector bolt glyphs |

Unknown values fall back to off silently, matching the `?arrows=` and
`?cells=` precedent. `?lightning=`, `?arrows=`, and `?cells=` are
independent -- use any combination, or none.

## Frame semantics

Every requested frame draws the strikes that happened during its OWN
10-minute window `(T-600, T]`: the lower edge is exclusive (a strike
exactly on a frame boundary belongs to the earlier frame) and the upper
edge is inclusive. Timeline playback therefore replays the storm strike
by strike instead of showing a static latest snapshot. The lightning slots
now cover the full radar frame history: at the default two-hour retention
every one of the 12 past frames' 10-minute slots is populated. Strikes
older than `LIBREWXR_LIGHTNING_MAX_AGE` age out of the store, and tiles
whose window no longer covers them render plain.

`LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE` (default 1000) is a purely
presentational cap: on outbreak-dense tiles, glyphs are chosen
strongest-by-energy so the busiest tiles stay legible and cheap to draw.
It removes nothing from the store, `/v2/lightning`, or the MCP tool --
those always see every flash in the window. Overlay cache entries are
keyed by a deterministic content fingerprint of the tile's slot strikes,
so identical content always encodes to identical bytes.

## Visual encoding

Each strike is drawn as:

- **Dots** (default) -- a small filled circle, sized modestly by energy
  on a `log10` scale (flash energies are joule-scale, roughly 1e-15 to
  4e-12 J). Meteosat-12 reports flash radiance instead (see
  [Coverage and attribution](#coverage-and-attribution)); the dot-size
  heuristic saturates at its maximum radius for both families, so a tile
  carrying both renders with no visual discontinuity.
- **Bolts** -- a hand-authored vector bolt polygon, deliberately NOT a
  text or emoji glyph, so there is no font dependency and it renders
  identically on every host.

Both styles draw at constant brightness, with no age fade. Most tiles
hold zero flashes and skip the layer entirely; the layer combines freely
with the motion-arrow and storm-cell overlays.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LIBREWXR_LIGHTNING_ENABLED` | `true` | Master switch for the lightning layer and the MCP `get_recent_lightning` tool. When `false`, the fetcher loop does not run, `?lightning=` is a no-op on the tile endpoint, and `/v2/lightning` returns 503. |
| `LIBREWXR_LIGHTNING_NOAA_ENABLED` | `true` | Per-family switch for the NOAA GOES GLM flash family. Satellite identity is internal and never configured; per-family booleans are the knob. FengYun's boolean lands alongside its decoder. |
| `LIBREWXR_LIGHTNING_EUMETSAT_ENABLED` | `true` | Per-family switch for the EUMETSAT MTG LI flash family (Meteosat-12). Active only when both API credentials below are set -- without them the family stays off and the instance runs fully anonymous on GOES GLM alone. |
| `LIBREWXR_LIGHTNING_EUMETSAT_API_KEY` | *(empty)* | EUMETSAT Data Store consumer key. Free account at https://user.eumetsat.int, Data Store terms accepted at https://data.eumetsat.int, key pair generated at https://api.eumetsat.int/api-key. Empty = family off. |
| `LIBREWXR_LIGHTNING_EUMETSAT_API_SECRET` | *(empty)* | EUMETSAT Data Store consumer secret; both halves of the key pair must be set for the family to run. |
| `LIBREWXR_LIGHTNING_FETCH_INTERVAL` | `300` | Clock-aligned fetch cadence, in seconds. |
| `LIBREWXR_LIGHTNING_MAX_AGE` | `7200` | Seconds of flash history retained; also clamps the `minutes=` parameter on the REST endpoint and the MCP tool. Default 7200 s = 2 hours, matching the radar frame history so older frames keep their lightning overlay. The store's RAM ceiling scales with this (500k points per 30 minutes, 2,000,000 at the default); API and WebSocket snapshot defaults stay at 30 minutes. |
| `LIBREWXR_LIGHTNING_MAX_DRAW_PER_TILE` | `1000` | Presentational strongest-by-energy draw cap per tile; removes nothing from the store or the API. |

See [docs/configuration-reference.md](configuration-reference.md) for the
full per-variable descriptions.

## REST endpoint

The held strikes are exposed as a JSON API -- the programmatic
counterpart to the `?lightning=` tile overlay:

```
GET /v2/lightning
GET /v2/lightning?lat={lat}&lon={lon}&radius_km={radius}
GET /v2/lightning?bbox=west,south,east,north
```

Returns a GeoJSON `FeatureCollection` of `Point` features (coordinates
`[lon, lat]`), each carrying `utc`, `energy`, and `satellite`
(`goes18` / `goes19` / `meteosat12`) properties. `energy` is flash energy
in joules for the GOES GLM family and flash radiance in `mW.m-2.sr-1` for
`meteosat12` -- the `satellite` value is the unit key, and `/health`
reports the per-family units.

| Query parameter | Description |
|---|---|
| *(none)* | Strikes from the last 30 minutes (the `minutes` default), not the whole held window |
| `lat` + `lon` | Strikes within `radius_km` of the point (both required together, else 400) |
| `radius_km` | Search radius in km (default 25, ignored without lat/lon) |
| `bbox` | `west,south,east,north` rectangle (alerts-style validation) |
| `minutes` | Lookback window in minutes, clamped to `LIBREWXR_LIGHTNING_MAX_AGE` (default 30; `minutes=120` returns the full two-hour history) |
| `limit` | Maximum number of strikes returned, newest first (default: the full requested window) |

When both a point and a `bbox` are supplied, the point wins (mirroring
`/v2/alerts`). Returns `503 Service Unavailable` when lightning is
disabled (`LIBREWXR_LIGHTNING_ENABLED=false`).

The default 30-minute window is a bounded response even during a severe
outbreak; the full history is an explicit `minutes` request
(`?minutes=120` with the default retention), which can be large. For
routine polling prefer `bbox` or point + `radius_km` (with `minutes`),
and use `limit` to cap a large window. Large JSON responses (>= 1 KiB)
are gzipped by the server whenever the client advertises gzip.

The same payload is also pushed over the WebSocket channel
(`/v2/ws`, topic `lightning`), where it arrives as a snapshot pinned to
the last 30 minutes -- a bounded bootstrap regardless of retention --
followed by a live delta stream in ascending time order
(the REST endpoint returns newest first) -- see
[docs/web-integration-guide.md](web-integration-guide.md#websocket-endpoint).

## MCP tool

The same strikes are available to AI agents through the MCP
`get_recent_lightning` tool on BOTH transports (HTTP at `/mcp` and the
stdio `python -m librewxr.mcp`):

```
get_recent_lightning(lat, lon, radius_km=100.0, bbox=None, minutes=30.0, limit=2000)
```

Returns a newest-first list of strikes, each a dict of
`{lat, lon, utc (ISO 8601), energy, satellite}`, where `satellite` is
`goes18`, `goes19`, or `meteosat12` and `energy` follows that family's
unit (joules for GOES GLM, `mW.m-2.sr-1` radiance for `meteosat12`).
`lat` + `lon` filter by radius, `bbox=[west, south, east, north]` filters
to a rectangle, and `minutes` (default 30, clamped to the two-hour
`LIBREWXR_LIGHTNING_MAX_AGE` retention -- pass `minutes=120` for the
full history) and `limit` apply on every mode. It returns
an empty list `[]` when lightning is disabled and never raises.

See [docs/mcp-server.md](mcp-server.md) for the full tool reference and
transport setup.

## Health monitoring

The `/health` endpoint surfaces an additive `lightning` section:

```json
{
  "lightning": {
    "enabled": true,
    "points": 12345,
    "version": 1785130265,
    "last_updated": 1785130265,
    "window_start_s": 1785123065,
    "window_end_s": 1785130265,
    "ceiling_trimmed_total": 0,
    "satellites": {
      "goes18": {
        "bucket": "noaa-goes18",
        "last_key": "OR_GLM-L2-LCFA_G18_s2026282175940_e2026282180000_c20262821800110.nc",
        "last_e": 1785130200,
        "units": "J",
        "stale": false,
        "last_fetch_ok": true
      },
      "goes19": {
        "bucket": "noaa-goes19",
        "last_key": "OR_GLM-L2-LCFA_G19_s2026282175940_e2026282180000_c20262821800110.nc",
        "last_e": 1785130200,
        "units": "J",
        "stale": false,
        "last_fetch_ok": true
      },
      "meteosat12": {
        "source": "EO:EUM:DAT:0691",
        "last_key": "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-LFL--FD--x-x--ARC-x_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005_N__O_0056_0000",
        "last_e": 1785130200,
        "units": "mW.m-2.sr-1",
        "stale": false,
        "last_fetch_ok": true
      }
    }
  }
}
```

`enabled` reports the master switch; `points` is how many flashes the
store currently holds; `version` and `last_updated` identify the artifact
version (equal -- the version is the write timestamp); `window_start_s` /
`window_end_s` bound the retained window (epoch seconds; the example spans
the two-hour default); `ceiling_trimmed_total` is the running count of
points dropped by the retention-scaled RAM ceiling; and `satellites`
carries each satellite's fetch status, including the `stale` flag and the
source handle -- the S3 `bucket` for the NOAA family, the Data Store
collection `source` for EUMETSAT.

`units` is carried per family: `"J"` for the GOES GLM flashes,
`"mW.m-2.sr-1"` (flash radiance) for Meteosat-12. Each `satellites` entry
reports its own `units`, so a client can always disambiguate a per-flash
`energy` number. The store's own `units` meta (alongside `satellites` in
the artifact) becomes `"mixed"` once both families carry points -- `"J"`
while only GOES GLM does. When lightning is disabled the section degrades
to `{"enabled": false}`.

## Coverage and attribution

Lightning coverage follows the two observation footprints. The NOAA
family comes from the GOES-East + GOES-West geostationary disks, roughly
the Americas, the Atlantic, and the eastern Pacific, out to about +/-57
degrees latitude per satellite. The EUMETSAT family is Meteosat-12
(MTG-I1) at 0°E, whose Lightning Imager has been operational since
2024-10-31 out to about 79°N–79°S -- Europe, Africa, the Atlantic, and
the western Indian Ocean. Together the two families cover the Americas,
the Atlantic, Europe, and Africa. South and East Asia and the Maritime
Continent (roughly 80°E–170°E) still carry no observed lightning; the
model-derived lightning diagnostics surveyed in
[docs/source-survey.md](source-survey.md) remain the threat proxy there.

The two families are fetched differently. GOES GLM is US public domain,
read from the anonymous NOAA NODD buckets. Meteosat-12 is EUMETSAT Core
Data under CC-BY-4.0 -- free and unrestricted, including redistribution
-- but the Data Store in front of it requires a free account and an API
key. Register at https://user.eumetsat.int, accept the Data Store terms
on your first visit to https://data.eumetsat.int, then generate the
consumer-key / consumer-secret pair at https://api.eumetsat.int/api-key
and set `LIBREWXR_LIGHTNING_EUMETSAT_API_KEY` /
`LIBREWXR_LIGHTNING_EUMETSAT_API_SECRET`. The pipeline exchanges the
pair for 8-hour bearer tokens automatically; searches are anonymous, only
downloads are authenticated. Both halves must be set -- without
credentials the EUMETSAT family stays off, makes no request, and the
instance remains fully anonymous on GOES GLM alone.

The Atlantic overlap between GOES-East and Meteosat-12 is accepted as a
double-draw: the same storm can be drawn twice near the seam, the same
standing ruling that covers the GOES-East / GOES-West overlap over the
Americas. Deduplicating flashes across two different instruments would
mean cross-instrument matching, which is not worth it for a visual
overlay.

`energy` is not the same physical quantity across the two families.
GOES GLM publishes flash radiant energy in joules (roughly 1e-15 to
4e-12 J); Meteosat-12 publishes flash radiance in `mW.m-2.sr-1`
(milliwatts per square metre per steradian). LibreWXR stores each
family's value exactly as published -- no cross-family rescale, no
synthetic conversion -- and carries the unit alongside it: the
`satellite` property keys the unit on `/v2/lightning` and the MCP tool,
and `/health` reports it per satellite (top-level `"mixed"` once both
families hold points). Rendering is unaffected because the overlay's
dot-size heuristic saturates at its maximum radius for both families.

LibreWXR reprojects and styles the flashes, so a courtesy credit is
requested for the NOAA data (not required):

> Lightning: NOAA GOES-R Geostationary Lightning Mapper (GLM)

EUMETSAT requires attribution when its data is redistributed:

> Contains modified EUMETSAT Meteosat data 2026

## See also

- [docs/lightning-implementation-plan.md](lightning-implementation-plan.md)
  -- the living decision log for the feature (phases, maintainer rulings,
  and as-landed notes).
- [docs/mcp-server.md](mcp-server.md) -- the `get_recent_lightning` tool
  and both MCP transports.
- [docs/configuration-reference.md](configuration-reference.md) -- full
  descriptions of every `LIBREWXR_LIGHTNING_*` variable.
