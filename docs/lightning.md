# Lightning

LibreWXR ingests NOAA GOES GLM (Geostationary Lightning Mapper)
total-lightning flash points from the GOES-East + GOES-West
geostationary disks and optionally overlays them on radar tiles via the
`?lightning=` URL parameter -- the storm-scale parallel to the existing
`?arrows=` motion-arrow and `?cells=` storm-cell overlays. The same
strikes are served as GeoJSON at `GET /v2/lightning` and through the MCP
`get_recent_lightning` tool. Flashes are US public domain, read from the
anonymous NOAA NODD S3 buckets (courtesy credit: "Lightning: NOAA GOES-R
Geostationary Lightning Mapper (GLM)").

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

The pipeline (`data/lightning_fetcher.py`) fetches GLM L2 LCFA
total-lightning flash files from the anonymous NODD buckets
(`noaa-goes19` for GOES-East and `noaa-goes18` for GOES-West) on its own
clock-aligned loop, every `LIBREWXR_LIGHTNING_FETCH_INTERVAL` seconds
(default 300, i.e. every 5 minutes). Each satellite keeps a resume
watermark, so pipeline restarts pick up where they left off instead of
re-decoding old files. The loop is deliberately its own knob: disabling
alerts never disables lightning, and vice versa.

Decoded flashes are held in `LightningStore` (`data/lightning_store.py`)
as structured points -- every flash inside the retention window
(`LIBREWXR_LIGHTNING_MAX_AGE`, default 1800 s = 30 minutes, matching the
standard "wait 30 minutes after the last thunder" guidance). The store's
500k-point RAM ceiling is a warn-once circuit breaker, not a data policy;
under normal load the 30-minute window holds tens of thousands of points.

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
by strike instead of showing a static latest snapshot. Strikes older than
`LIBREWXR_LIGHTNING_MAX_AGE` age out of the store and render plain on
tiles whose window no longer covers them.

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
  4e-12 J).
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
| `LIBREWXR_LIGHTNING_NOAA_ENABLED` | `true` | Per-family switch for the NOAA GOES GLM flash family. Satellite identity is internal and never configured; per-family booleans are the knob. EUMETSAT / FengYun booleans land alongside their decoders. |
| `LIBREWXR_LIGHTNING_FETCH_INTERVAL` | `300` | Clock-aligned fetch cadence, in seconds. |
| `LIBREWXR_LIGHTNING_MAX_AGE` | `1800` | Seconds of flash history retained; also clamps the `minutes=` parameter on the REST endpoint and the MCP tool. |
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
`[lon, lat]`), each carrying `utc`, `energy` (joules), and `satellite`
(`goes18` / `goes19`) properties.

| Query parameter | Description |
|---|---|
| *(none)* | Every strike in the held window |
| `lat` + `lon` | Strikes within `radius_km` of the point (both required together, else 400) |
| `radius_km` | Search radius in km (default 25, ignored without lat/lon) |
| `bbox` | `west,south,east,north` rectangle (alerts-style validation) |
| `minutes` | Lookback window in minutes, clamped to `LIBREWXR_LIGHTNING_MAX_AGE` |
| `limit` | Maximum number of strikes returned, newest first (default: the full window) |

When both a point and a `bbox` are supplied, the point wins (mirroring
`/v2/alerts`). Returns `503 Service Unavailable` when lightning is
disabled (`LIBREWXR_LIGHTNING_ENABLED=false`).

The no-args mode returns the entire held window, which during a severe
outbreak can be a large response -- prefer `bbox` or point + `radius_km`
(with `minutes`) for routine use, and use `limit` to cap a large window.
Large JSON responses are gzipped by the server whenever the client
advertises gzip.

## MCP tool

The same strikes are available to AI agents through the MCP
`get_recent_lightning` tool on BOTH transports (HTTP at `/mcp` and the
stdio `python -m librewxr.mcp`):

```
get_recent_lightning(lat, lon, radius_km=100.0, bbox=None, minutes=30.0, limit=2000)
```

Returns a newest-first list of strikes, each a dict of
`{lat, lon, utc (ISO 8601), energy (joules), satellite}`. `lat` + `lon`
filter by radius, `bbox=[west, south, east, north]` filters to a
rectangle, and `minutes` (clamped to `LIBREWXR_LIGHTNING_MAX_AGE`) and
`limit` apply on every mode. It returns an empty list `[]` when lightning
is disabled and never raises.

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
    "window_start_s": 1785128465,
    "window_end_s": 1785130265,
    "ceiling_trimmed_total": 0,
    "satellites": {
      "goes18": {
        "bucket": "noaa-goes18",
        "last_key": "OR_GLM-L2-LCFA_G18_s2026282175940_e2026282180000_c20262821800110.nc",
        "last_e": 1785130200,
        "stale": false,
        "last_fetch_ok": true
      },
      "goes19": {
        "bucket": "noaa-goes19",
        "last_key": "OR_GLM-L2-LCFA_G19_s2026282175940_e2026282180000_c20262821800110.nc",
        "last_e": 1785130200,
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
`window_end_s` bound the retained window (epoch seconds);
`ceiling_trimmed_total` is the running count of points dropped by the
500k RAM ceiling; and `satellites` carries each satellite's fetch status,
including the `stale` flag. When lightning is disabled the section
degrades to `{"enabled": false}`.

## Coverage and attribution

Lightning coverage follows the GOES GLM observation footprint: strikes
come from the GOES-East + GOES-West geostationary disks, roughly the
Americas, the Atlantic, and the eastern Pacific, out to about +/-57
degrees latitude per satellite. Africa, Asia, and the Indian Ocean sit
outside GLM's view, so no lightning overlay is available there.

The data is US public domain, ingested from the anonymous NOAA NODD
buckets. LibreWXR reprojects and styles the flashes, so a courtesy credit
is requested (not required):

> Lightning: NOAA GOES-R Geostationary Lightning Mapper (GLM)

## See also

- [docs/lightning-implementation-plan.md](lightning-implementation-plan.md)
  -- the living decision log for the feature (phases, maintainer rulings,
  and as-landed notes).
- [docs/mcp-server.md](mcp-server.md) -- the `get_recent_lightning` tool
  and both MCP transports.
- [docs/configuration-reference.md](configuration-reference.md) -- full
  descriptions of every `LIBREWXR_LIGHTNING_*` variable.
