# MCP Server — Model Context Protocol

LibreWXR ships a built-in [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that exposes live weather data to LLM agents and MCP-capable tools. Phase 1 provides two tools: **`get_precip_nowcast`** (point-based precipitation forecast) and **`get_active_alerts`** (WMO CAP weather alerts near a point). Two transports are available: an **HTTP transport** mounted inside the main FastAPI app, and a **stdio transport** for local desktop agents like Claude Desktop.

---

## Install

The MCP server requires the optional `[mcp]` extra, which pulls in `fastmcp`:

```bash
pip install -e ".[mcp]"
```

Without this extra, the HTTP transport is silently disabled at startup (a warning is logged), and the stdio console entry point (`python -m librewxr.mcp` / `librewxr-mcp`) will not import. The base LibreWXR installation works fine without it — MCP is entirely optional.

---

## HTTP Transport (primary)

The HTTP transport is mounted inside the main LibreWXR FastAPI app as a sub-application using FastMCP's SSE (Server-Sent Events) streaming transport. It is enabled by default and controlled by the following environment variables:

- **`LIBREWXR_MCP_ENABLED`** (default `true`) — master switch. When `false`, no MCP route is mounted and the app boots without any MCP lifespan overhead.
- **`LIBREWXR_MCP_PATH`** (default `/mcp`) — the URL path where the MCP transport is mounted.

The HTTP transport works in **both single and multi deployment modes** — it reads live data from the server's in-memory stores (single mode) or from the memmap snapshot (multi mode), whichever is active.

### Stateless transport

The HTTP transport runs in **stateless mode**: every request is self-contained, there is **no `Mcp-Session-Id`** to obtain or carry, and clients do not need to track any per-session state. Each request is served by a fresh transport with no session store — this is what allows multi-worker deployments (e.g. 16 render workers behind one load balancer or Cloudflare tunnel) to serve a client's requests from any worker. Clients can simply POST `initialize`, then `tools/list` / `tools/call` requests, with no session-id header at all.

### Connecting from an MCP client

Point an MCP-capable client (n8n, Claude Code in proxy mode, custom scripts using the MCP SDK) at:

```
http://<host>:<port>/mcp/
```

A POST to `/mcp` (without trailing slash) returns a **307 redirect** to `/mcp/` — this is Starlette's standard mount-prefix behaviour, not a bug. Most HTTP clients (including most MCP SDK implementations) follow POST redirects automatically, but some (notably `httpx` by default) do not. If your client fails to connect, ensure you are targeting `/mcp/` directly, or set `follow_redirects=True` on the client's HTTP transport.

For deployments behind a reverse proxy, the endpoint is at `<public_url>/mcp/` — ensure your proxy passes `/mcp/` (with trailing slash) through to the upstream unchanged. For example, if your instance is at `https://radar.example.com`, a client connects to `https://radar.example.com/mcp/`.

### FastMCP Host/Origin guard

FastMCP 3.4.4 ships an opt-in Host/Origin guard that rejects requests with unexpected `Host` or `Origin` headers. LibreWXR **does not enable this guard** by default, so reverse-proxy deployments work out of the box. If you enable it explicitly (e.g. via middleware), set `allowed_hosts` and `allowed_origins` to match the proxy's Host header and the client's Origin header — otherwise SSE connections will be rejected.

---

## stdio Transport (for local agents)

The stdio transport runs as a standalone process and communicates over standard input/output. It is intended for desktop-based MCP clients such as Claude Desktop, Cursor, or any tool that launches subprocesses speaking MCP over stdio.

Run it with:

```bash
python -m librewxr.mcp
```

Or use the console entry point:

```bash
librewxr-mcp
```

### Requirements

- **`LIBREWXR_CACHE_DIR`** must point to a directory where a running LibreWXR server (single OR multi mode) is writing `state.json`. The stdio process reads this snapshot and polls its mtime to stay in sync, so it always serves the same data the server just rendered.
- In **single mode**, the server dumps `state.json` at the end of every fetch cycle via the `on_cycle_complete` hook wired in `main.py`. No extra configuration is needed.
- In **multi mode**, the data pipeline sidecar owns the `state.json` snapshot — the render workers and the MCP stdio process all read the same file.

### Example Claude Desktop configuration

```json
{
  "mcpServers": {
    "librewxr": {
      "command": "/path/to/librewxr-mcp",
      "env": {
        "LIBREWXR_CACHE_DIR": "/var/lib/librewxr"
      }
    }
  }
}
```

Replace `/path/to/librewxr-mcp` with the full path to the `librewxr-mcp` entry point (run `which librewxr-mcp` to find it if installed in a virtualenv or system-wide), and `/var/lib/librewxr` with the directory configured on your LibreWXR server.

The stdio process polls `state.json`'s mtime and refreshes its internal stores in-place, so it tracks new fetch cycles automatically without restarting.

---

## Tool Reference

### `get_precip_nowcast(lat, lon, minutes=60)`

Returns a list of future nowcast frames (precipitation forecast) for a single geographic point.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `lat` | float | (required) | Latitude of the point (degrees, -90 to 90) |
| `lon` | float | (required) | Longitude of the point (degrees, -180 to 180) |
| `minutes` | int | `60` | Forecast horizon in minutes. Clipped to the active nowcast window. |

**Returns:** A list of dicts, one per forecast frame, each containing:

| Field | Type | Description |
|---|---|---|
| `time` | string | ISO 8601 timestamp of the forecast frame |
| `minutes_offset` | int | Minutes from now this frame represents |
| `dbz` | float | Reflectivity in dBZ at the point for this frame |
| `rate_mmh` | float | Precipitation rate in mm/h (derived from dBZ via Marshall-Palmer Z-R) |
| `source` | string | `"radar"` when the value comes from radar extrapolation, `"nwp"` when it comes from the model chain, `"none"` when there is no data |
| `blend_weight` | float | Weight of the radar source in the blended nowcast (0.0 = pure NWP, 1.0 = pure radar). Only meaningful when source is not `"none"`. |
| `coverage` | string | `"in_range"` when the point is inside radar coverage, `"out_of_range"` when it is not |

**Behaviour:**

- Returns FUTURE frames only — the latest observed radar frame at t=0 is **not** included.
- When the point is outside radar coverage, the nowcast falls back to the active NWP chain (the same chain that drives the tile renderer's model layer).
- Returns an empty list `[]` when nowcasting is disabled by configuration (`LIBREWXR_NOWCAST_ENABLED=false`).
- Never raises on invalid input — out-of-bounds coordinates return empty results.

### `get_active_alerts(lat, lon, radius_km=25, severity=None)`

Returns weather alerts (WMO CAP format) active within a given radius of a point.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `lat` | float | (required) | Latitude of the centre point (degrees) |
| `lon` | float | (required) | Longitude of the centre point (degrees) |
| `radius_km` | float | `25` | Search radius in kilometres |
| `severity` | string | `None` | Optional filter. One of: `"Extreme"`, `"Severe"`, `"Moderate"`, `"Minor"`. When unset, all severities are returned. |

**Returns:** A GeoJSON FeatureCollection (`dict` with `"type": "FeatureCollection"` and a `"features"` list). Each feature is a GeoJSON Feature with:

| Property | Type | Description |
|---|---|---|
| `id` | string | CAP alert identifier (unique per source) |
| `event` | string | Event description (e.g. "Tornado Warning") |
| `headline` | string | Short headline |
| `description` | string | Full alert text |
| `severity` | string | One of `Extreme`, `Severe`, `Moderate`, `Minor`, `Unknown` |
| `urgency` | string | One of `Immediate`, `Expected`, `Future`, `Past`, `Unknown` |
| `certainty` | string | One of `Observed`, `Likely`, `Possible`, `Unlikely`, `Unknown` |
| `onset` | string | ISO 8601 start time |
| `expires` | string | ISO 8601 expiry time |
| `polygon` | list[list[float]] | `None` or a list of `[lon, lat]` coordinate pairs forming the alert polygon |
| `sender_name` | string | Human-readable name of the issuing agency |

**Behaviour:**

- US zone-based alerts (e.g. Tornado Watches) are resolved to zone polygons at ingest, so the lookup never queries NWS at request time. These appear in the FeatureCollection with their NWS-sourced properties.
- Returns an **empty FeatureCollection** (`{"type": "FeatureCollection", "features": []}`) when alerts are disabled by configuration (`LIBREWXR_ALERTS_ENABLED=false`) or when no alerts match the query.
- Never raises — invalid coordinates, network errors on the upstream API, or missing alert data all result in an empty FeatureCollection.

---

## Discovery

LibreWXR self-describes its MCP endpoint with two draft metadata documents (neither is a ratified standard yet):

### MCP Server Card (SEP-2127 draft)

`GET <mcp path>/server-card` (e.g. `GET /mcp/server-card`) serves a [SEP-2127](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2127) (draft) MCP Server Card with media type `application/mcp-server-card+json`. The card tells MCP clients where the streamable-HTTP remote lives and which protocol versions it supports, without an `initialize` round-trip.

Server cards intentionally do **not** enumerate tools — clients list tools at runtime via the MCP protocol's `tools/list` method.

The endpoint supports conditional GET: responses carry an `ETag` (the first 16 hex chars of the SHA-256 of the body) and an `If-None-Match` request header equal to the current ETag returns a `304 Not Modified` with an empty body. All responses also carry `Cache-Control: public, max-age=3600` and CORS headers (`Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET`, `Access-Control-Allow-Headers: Content-Type, If-None-Match`, `Access-Control-Expose-Headers: ETag`).

Example body:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json",
  "name": "io.github.joshuakimsey/librewxr-mcp",
  "title": "LibreWXR MCP",
  "description": "Precipitation nowcasts, active weather alerts, and storm-cell data for any point on Earth.",
  "version": "0.1.0",
  "websiteUrl": "http://localhost:8080",
  "repository": { "source": "github", "url": "https://github.com/JoshuaKimsey/LibreWRX" },
  "remotes": [
    {
      "type": "streamable-http",
      "url": "http://localhost:8080/mcp/",
      "supportedProtocolVersions": ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
    }
  ]
}
```

The card URL follows `LIBREWXR_MCP_PATH` (default `/mcp`), and every advertised URL derives from `LIBREWXR_PUBLIC_URL` (default `http://localhost:8080`) — behind a reverse proxy, set `LIBREWXR_PUBLIC_URL` to the public base so the card advertises reachable URLs.

### AI Catalog (proposal)

`GET /.well-known/ai-catalog.json` serves the AI Catalog (proposal) entry with media type `application/ai-catalog+json`: a directory-of-directories pointer that resolves to the server card above.

Example body:

```json
{
  "specVersion": "1.0",
  "entries": [
    {
      "identifier": "urn:air:localhost:mcp:librewxr-mcp",
      "type": "application/mcp-server-card+json",
      "url": "http://localhost:8080/mcp/server-card"
    }
  ]
}
```

The catalog returns `404 Not Found` when MCP is disabled (`LIBREWXR_MCP_ENABLED=false`) or the HTTP transport failed to mount.

> **Note:** both documents are draft proposals. There is **no** standardized `/.well-known/mcp.json` — LibreWXR does not serve one, so do not point clients at it.

---

## Deployment Notes

### Single-mode state.json enabler

In single mode, the server writes `state.json` to `LIBREWXR_CACHE_DIR` at the end of each fetch cycle. This is handled by the `on_cycle_complete` hook wired in `main.py`. As long as `LIBREWXR_CACHE_DIR` is set (it is required for the stdio transport), the snapshot is produced automatically — no extra configuration.

### stdio transport must see the same cache directory

The stdio process (`python -m librewxr.mcp` / `librewxr-mcp`) reads `state.json` from `LIBREWXR_CACHE_DIR`. **This must be the same directory** the running server writes to. In Docker, this means both the server container and the MCP stdio process must use the same volume mount. In a local dev setup, point both at the same path on disk.

### What the tools can see

The server-side configuration determines what data the MCP tools can access:

- `LIBREWXR_ENABLED_REGIONS` — radar regions the nowcast can extrapolate from.
- `LIBREWXR_RADAR_ENABLED` — when `false`, radar data is unavailable (nowcast may still return NWP-only frames).
- `LIBREWXR_REGIONAL_NWP_ENABLED` — when `false`, the nowcast blends against ECMWF IFS only (no regional model for the point).
- `LIBREWXR_ALERTS_ENABLED` — when `false`, `get_active_alerts` returns an empty FeatureCollection.
- `LIBREWXR_NOWCAST_ENABLED` — when `false`, `get_precip_nowcast` returns `[]`.
- `LIBREWXR_NOWCAST_BLEND_MODE` — controls how radar and NWP are blended in the nowcast (see [configuration reference](configuration-reference.md#nowcasting)).

The stdio MCP process reads the same `config.py` settings (it re-uses the `Settings` model), so keep the server's `.env` and the MCP process's environment in sync if you run them separately.

---

### `get_storm_cells(lat, lon, radius_km=100)`

Returns a list of detected storm cells within `radius_km` of the point, sourced from the latest radar frame's connected-component detection.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `lat` | float | (required) | Latitude of the centre point (degrees) |
| `lon` | float | (required) | Longitude of the centre point (degrees) |
| `radius_km` | float | `100` | Search radius in kilometres |

**Returns:** A list of dicts, one per detected cell within range. Each cell contains:

| Field | Type | Description |
|---|---|---|
| `lat` | float | Cell centroid latitude |
| `lon` | float | Cell centroid longitude |
| `area_km2` | float | Cell area in square kilometres (approximate) |
| `max_dbz` | float | Maximum reflectivity within the cell (dBZ) |
| `motion_speed_kmh` | float or null | Storm motion speed in km/h. `null` when no optical-flow data is available (e.g. nowcast disabled). |
| `motion_heading_deg` | float or null | Storm motion compass heading in degrees (0=N, 90=E). `null` when no flow data is available. |
| `region` | string | Name of the radar region the cell was detected in |

**Behaviour:**

- Returns an empty list `[]` when storm-cell detection is disabled by configuration (`LIBREWXR_STORM_CELLS_ENABLED=false`) or when no cells are within the search radius.
- `motion_speed_kmh` and `motion_heading_deg` are `null` (not missing) when no optical-flow data was available — this is the JSON-safe representation (NaN is not valid JSON).
- Never raises — invalid coordinates, disabled detection, or empty data all result in `[]`.

---

## Deployment Notes

## See Also

- [Configuration Reference](configuration-reference.md#mcp-server) — full env var reference for `LIBREWXR_MCP_ENABLED` and `LIBREWXR_MCP_PATH`.
- [Getting Started](../README.md) — general setup instructions for LibreWXR.
- [Model Context Protocol Specification](https://modelcontextprotocol.io/) — official MCP docs.
