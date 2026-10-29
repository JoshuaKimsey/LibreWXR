# Weather Alerts

LibreWXR ingests global weather alerts (WMO CAP) from the WMO alert hub
at [severeweather.wmo.int](https://severeweather.wmo.int/) and US alerts
directly from the NWS API, resolves national area codes to polygons
through pluggable geocode tables, and serves the result as GeoJSON at
`GET /v2/alerts` and through the MCP `get_active_alerts` tool.

## Table of Contents

- [How it works](#how-it-works)
- [Geocode tables](#geocode-tables)
- [CAP entry semantics](#cap-entry-semantics)
- [Configuration](#configuration)
- [REST endpoint](#rest-endpoint)
- [MCP tool](#mcp-tool)
- [Health monitoring](#health-monitoring)
- [Coverage and attribution](#coverage-and-attribution)
- [See also](#see-also)

## How it works

The ingest is pipeline-owned (`data/alerts_fetcher.py`) and runs on its
own clock-aligned loop, every `LIBREWXR_ALERTS_FETCH_INTERVAL` seconds
(default 300, i.e. every 5 minutes). Each cycle:

1. Fetches the hub's agency registry, `json/sources.json` -- per-agency
   `sourceId`, `capAlertFeed` (the agency's own feed URL, sometimes a
   comma-separated list of several feeds) and `capAlertFeedStatus` -- and
   `v2/json/wmo_all.json`, the index of active alerts with their CAP
   `id`s, in parallel with the NWS `/alerts/active` call for the US
   slice.
2. Selects agencies whose `capAlertFeedStatus` is `operating`, that are
   not in `_EXCLUDED_SOURCE_IDS`, and that either match an agency path
   segment in `wmo_all.json` exactly, or match it on the
   `(authorityCountry, authorityAbbrev)` family. The family fallback
   recovers language-suffix drift between the registry and the hub (the
   registry says `de-dwd-en` while the hub publishes `de-dwd-de`, and
   likewise for `ir-irimo`, `kz-kazhydromet`, `ly-nmc` and `th-tmd`).
3. Fetches each selected agency's declared `capAlertFeed` URL(s). The
   legacy per-agency proxy `v2/cap-alerts/{sid}/rss.xml` is only used as
   a fallback when an agency declares no feed -- those proxy URLs are
   dead for 48 of the 49 selected agencies, which is why the declared
   feeds are the primary path.
4. Parses the feeds as RSS 2.0 or Atom. Atom entries are read through
   their `application/cap+xml` link (many agencies, MeteoAlarm among
   them, serve Atom), the `cap:identifier` element is used as the id
   candidate, and the decoy geocode links MeteoAlarm ships are ignored.
5. Matches hub ids with the `urn:oid:` prefix stripped, first exactly
   and then by bidirectional substring -- feeds append query parameters
   to the hub id or embed it inside the CAP href, so either direction
   alone would miss real alerts. A feed whose id scheme matches nothing
   at all (INMET publishes URL guids) still gets its newest 100 items
   fetched; stale entries are filtered at serve time.
6. Downloads each candidate CAP document and turns it into one alert
   entry (see [CAP entry semantics](#cap-entry-semantics)).

The cycle ends by replacing the store and logging a summary line,
`Alerts updated: N total (N WMO + N NWS) from K/J WMO feed URLs`, so the
operator-visible health of the ingest is one grep away. Feeds that fail
are counted rather than silently dropped; a cycle in which every
attempted feed URL failed logs
`WMO CAP feeds failing: N of M feed URLs unreachable (last failures: ...)`
and marks the store degraded (see [Health monitoring](#health-monitoring)).

Two sources are deliberately excluded from the WMO feed path:

- **US NWS** -- handled directly by the NWS API path, so the hub's US
  agency entries never double-ingest.
- **`cn-cma-xx` (China)** -- publishes `CPEAS` 12-digit admin codes and
  no usable polygons, and no resolver covers CPEAS yet. It is the
  largest single hub source (roughly 21% of active items), so it is
  excluded rather than downloaded and discarded; the rationale is
  documented inline at the exclusion in `_EXCLUDED_SOURCE_IDS`. It stays
  parked: a resolver would now be a trivial registry data drop, but no
  openly licensed geometry source exists (township polygons are not
  public, and the county-level DataV/AMAP source prohibits caching and
  redistribution).

A sources.json or wmo_all.json outage does not freeze the US slice: the
last-known non-NWS alerts are kept and refreshed NWS alerts are merged
in, with the store flagged degraded.

## Geocode tables

Many CAP areas carry no `<polygon>` at all -- they reference a national
or regional area code instead. The tables that turn those codes into
polygons live in `src/librewxr/data/alert_geocodes.py` and are a
registry of `GeocodeProvider`s: each provider claims one or more CAP
`valueName` strings and loads a `{code: Polygon}` table. Resolution is
routed by `valueName` in registration order, and the first claiming
provider that actually holds the code wins -- a claiming provider that
misses does not stop the search.

| Provider | Claimed `valueName`s | Source | Size | Notes |
|---|---|---|---|---|
| `meteoalarm` | `EMMA_ID`, `NUTS2`, `NUTS3` | MeteoAlarm region tables (GeoJSON + alias CSV), as published by the Pirate Weather project | ~33 MB | Europe. The NUTS2/NUTS3 aliases resolve the NUTS codes some agencies publish (Belgium's `BE23`, for example) |
| `dwd_warncell` | `WARNCELLID` | DWD GeoServer WFS GeoJSON (EPSG:4326) | ~12 MB | Germany. Four layers -- Gemeinden, Kreise, Kueste, Binnenseen -- 11,647 cells; ids are zero-padded to the 9-digit form CAP publishes |
| `bom_amoc` | `AMOC-AREACODE` | BoM AAC shapefile zips over anonymous FTP | ~22 MB | Australia. Products IDM00001 (PW), IDM00003 (MW), IDM00007 (FW), IDM00014 (ME) and IDM00017 (RC); 958 areas; GDA94 geometry is treated as WGS84 |

Operational notes:

- The tables download on first startup (~67 MB in total), cache under
  `<LIBREWXR_ALERTS_CACHE_DIR or LIBREWXR_CACHE_DIR>/alerts/`, and load
  once per process lifetime. Only the pipeline process loads them --
  render workers read the resulting alerts from `state.json` and never
  touch a table.
- They do not refresh on their own; upstream changes arrive with a cache
  clear. A corrupt cached table file is self-healing: the bad file is
  deleted and re-downloaded on the next cycle.
- BoM publishes its AAC geometry on anonymous FTP only (the HTTPS paths
  answer 403), so that loader downloads over FTP with bounded timeouts
  and needs the `pyshp` dependency to read the shapefile zips.
- Invalid table polygons are repaired with `buffer(0)`; a repair that
  still fails drops the polygon.
- A code that no provider resolves drops its area -- visible at `DEBUG`
  as a `Dropped CAP alert` line, so the gap is discoverable without
  inflating the store with geometry-less alerts.

> **Table credits.** MeteoAlarm warning data, CC BY 4.0-equivalent (via
> the region tables published by the Pirate Weather project). DWD
> Warngebiete © Deutscher Wetterdienst, CC BY 4.0; underlying
> administrative geometry © GeoBasis-DE / BKG 2021 (Daten modifiziert),
> used under GeoNutzV. BoM AAC areas © Commonwealth of Australia,
> Bureau of Meteorology, CC BY 4.0.

## CAP entry semantics

Each CAP document yields one alert entry per `<info>` block. Only the
first language encountered is kept -- infos in any other language are
skipped -- and `urgency=past` infos are dropped. Within an `<info>`:

- Every area contributes geometry: `<polygon>` areas directly, geocode
  areas through the tables above. Geocode resolution runs even when the
  same info also has explicit polygons, so a mixed CAP keeps all of its
  coverage.
- The areas merge into ONE geometry: a single `Polygon` when only one
  area polygon survives, otherwise an assembled `MultiPolygon` of the
  exact parts (no union at ingest).
- A geocode-resolved polygon that lies fully inside an explicit polygon
  of the same info is pruned. This is what keeps DWD's
  aggregate-polygon-plus-warncells documents from carrying the same
  coverage twice.
- `areaDesc` becomes the info's unique area descriptions joined with
  `"; "` -- one string per entry, the same multi-area string shape WMO
  feeds already publish natively.

The practical effect is one entry per info rather than one per area: a
DWD warning that lists a hundred warn cells is a single stored alert
with a multi-part geometry, not a hundred.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LIBREWXR_ALERTS_ENABLED` | `true` | Master switch for the alert ingest, `/v2/alerts` and the MCP `get_active_alerts` tool. |
| `LIBREWXR_ALERTS_FETCH_INTERVAL` | `300` | Clock-aligned ingest cadence, in seconds. |
| `LIBREWXR_ALERTS_CONCURRENCY` | `5` | Max concurrent HTTP connections for the feed and CAP fetches. |
| `LIBREWXR_ALERTS_CACHE_DIR` | *(empty)* | Cache directory for the geocode tables and the NWS zone cache. Empty = system temp. |

The BoM table additionally needs `pyshp>=2.3`, which ships with the
project's dependencies and is imported lazily by that one loader.

See [docs/configuration-reference.md](configuration-reference.md) for
the full per-variable descriptions.

## REST endpoint

```
GET /v2/alerts
GET /v2/alerts?lat={lat}&lon={lon}
GET /v2/alerts?bbox=west,south,east,north
```

Returns a GeoJSON `FeatureCollection` of `Polygon` or `MultiPolygon`
features (merged alerts carry a `MultiPolygon`; the coordinate nesting
differs accordingly). Each feature's properties are `title`, `severity`,
`time`, `expires`, `description`, `regions` and `uri`.

| Query parameter | Description |
|---|---|
| *(none)* | Every active alert worldwide |
| `lat` + `lon` | Alerts whose geometry contains the point (both required together) |
| `bbox` | `west,south,east,north` rectangle; returns alerts whose geometry intersects it |
| `simplify` | Geometry simplification tolerance in metres (default 1000, `0` disables) |

Filtering runs against the full-resolution geometry, features are grouped
per alert `uri` and unioned before encoding, and `regions` is a list
carrying the entry's `areaDesc` string -- for merged alerts that is the
semicolon-joined list of area descriptions. Expiry is evaluated at
request time: features are filtered to `expires` in the future, and
alerts with no expiry always pass. Returns `503 Service Unavailable`
when alerts are disabled (`LIBREWXR_ALERTS_ENABLED=false`).

The same payload is also pushed over the WebSocket channel
(`/v2/ws`, topic `alerts`) -- see
[docs/web-integration-guide.md](web-integration-guide.md#websocket-endpoint).

## MCP tool

The same alerts are available to AI agents through the MCP
`get_active_alerts` tool on both transports (HTTP at `/mcp` and the
stdio `python -m librewxr.mcp`):

```
get_active_alerts(lat, lon, radius_km=25, severity=None)
```

It filters by radius around a point with an optional CAP severity, and
returns an empty `FeatureCollection` (never raises) when alerts are
disabled or nothing matches.

See [docs/mcp-server.md](mcp-server.md) for the full tool reference and
transport setup.

## Health monitoring

The `/health` endpoint carries an additive `alerts` section:

```json
{
  "alerts": {
    "enabled": true,
    "count": 1999,
    "last_updated": 1785130265,
    "ingest_ok": true
  }
}
```

`enabled` reports the master switch, `count` is how many alert entries
the store currently holds, `last_updated` is the epoch second of the last
successful store replace, and `ingest_ok` mirrors the store's
`fetch_success` flag. `ingest_ok` degrades to `false` only when every
attempted WMO feed URL failed in the last cycle -- an agency that simply
has no active alerts is healthy and leaves the flag alone. When alerts
are disabled the section degrades to `{"enabled": false}`.

## Coverage and attribution

Coverage follows the contributing agencies: WMO-hub agencies worldwide,
with Europe resolving through MeteoAlarm's EMMA_ID/NUTS2/NUTS3 regions,
Germany through DWD's warn cells, and Australia through BoM's AMOC/AAC
areas. Areas whose code no table resolves are dropped, so a region can
simply have no polygon coverage rather than a guessed one. US alerts
come directly from the NWS API, with zone-based alerts (e.g. Tornado
Watches) resolved to zone polygons at ingest and the zone geometry
disk-cached for 30 days.

Credits:

> MeteoAlarm warning data, CC BY 4.0-equivalent (region tables published
> by the Pirate Weather project).

> DWD Warngebiete © Deutscher Wetterdienst, CC BY 4.0. Underlying
> administrative geometry © GeoBasis-DE / BKG 2021 (Daten modifiziert),
> used under GeoNutzV.

> BoM AMOC/AAC areas © Commonwealth of Australia, Bureau of Meteorology,
> CC BY 4.0.

## See also

- [docs/web-integration-guide.md](web-integration-guide.md) -- drawing
  the `/v2/alerts` response on a Leaflet or MapLibre map.
- [docs/mcp-server.md](mcp-server.md) -- the `get_active_alerts` tool and
  both MCP transports.
- [docs/configuration-reference.md](configuration-reference.md) -- full
  descriptions of every `LIBREWXR_ALERTS_*` variable.
