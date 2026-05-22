# Source Survey

A snapshot of the data sources LibreWXR has evaluated for inclusion in the core project: what shipped, what's queued, what was investigated and ruled out, and the reasoning behind each call. Covers both radar composites and regional NWP grids.

The open-data criteria these decisions apply are documented in [`adding-a-source.md`](adding-a-source.md#upstream-contribution-criteria). Self-hosters running their own LibreWXR instance are not bound by them — this document is the upstream selection record, not a prescription for every deployment.

## Conventions

Sources are tracked across four states:

- **Implemented** — shipping in the current main branch.
- **Tier 1** — validated against the upstream endpoints, ready to implement, no known blockers remaining.
- **Tier 2** — viable but deferred. Some have soft frictions (WAFs, undocumented bounds, missing palette legends); others are blocked on external action (operator outreach, license clarification).
- **Tier 3** — not currently viable. Usually a structural issue: API-key gate, commercial state-owned-enterprise model, app-only distribution, ToU forbids redistribution, or coverage entirely overlaps an existing source.

Sources that shipped and were later removed are recorded in [Reverted and removed](#reverted-and-removed) with the lessons learned. Those reverts informed criteria refinements in `adding-a-source.md` and are worth keeping visible — half of the value of a survey like this is the negative results.

## Table of Contents

- [Radar — Implemented](#radar--implemented)
- [Radar — Reverted and removed](#radar--reverted-and-removed)
- [Radar — Tier 1](#radar--tier-1)
- [Radar — Tier 2](#radar--tier-2)
- [Radar — Tier 3](#radar--tier-3)
- [NWP — Implemented](#nwp--implemented)
- [NWP — Tier 2](#nwp--tier-2)
- [NWP — Tier 3](#nwp--tier-3)
- [Context: WMO and open-data policy](#context-wmo-and-open-data-policy)

## Radar — Implemented

### United States — IEM NEXRAD and MRMS

Two complementary US sources, selectable via `LIBREWXR_NA_SOURCE`:

- **MRMS** (default) — NCEP Multi-Radar Multi-Sensor, GRIB2 MergedBaseReflectivityQC at 0.01° (\~1 km) on CONUS plus region-aware products for Alaska, Hawaii, Caribbean, Guam. Decoded with eccodes; no GDAL dependency. Anonymous HTTPS at `mrms.ncep.noaa.gov`. US Government public domain.
- **IEM NEXRAD** — Iowa Environmental Mesonet N0Q composites, palette-indexed PNG, uint8 dBZ encoding. Anonymous. Legacy / fallback source.

Regions covered by both: USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP. Per-region MRMS products are routed through separate per-product source instances inside the same package.

### Canada — ECCC MSC GeoMet WMS

Region: CACOMP. Source: MSC GeoMet WMS, `RADAR_1KM_RRAI` layer with `Radar-Rain_Dis-14colors` discrete style. Pre-coloured PNG only (no raw data access) — decoded via palette reverse-engineering back to mm/h, then converted to dBZ via Marshall-Palmer Z-R. Resolution: 0.025° (\~2.5 km), \~3560×1720 grid. Anonymous. Selectable independently from the US source via `LIBREWXR_CA_SOURCE`.

### Europe — OPERA Pan-European CIRRUS Composite

Region: OPERA (\~155 radars across 24 countries). Source: EUMETNET OPERA CIRRUS MAX reflectivity via MeteoGate/Cloudferro S3 — anonymous, no key required. ODIM HDF5, float64 dBZ. Resolution 3800×4400 at 1 km, LAEA projection. 5-minute cadence, rolling 24-hour archive. Sentinels: nodata `-9999000.0`, undetect `-8888000.0` (both map to 0 in uint8, so clear sky falls through to the NWP chain). URL pattern: `https://s3.waw3-1.cloudferro.com/openradar-24h/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@DBZH.h5`.

The legal lever that opened European radar data is the **EU High Value Datasets regulation (EU 2023/138)**, not WMO policy — it legally requires European meteorological data to be free, open-licensed, and API-accessible.

### El Salvador — MARN/SNET San Andrés

Region: SVCOMP (group `CENTRAL_AMERICA`). Single S-band radar at San Andrés, 120 km product (`esar82/Images/`) from the anonymous GCS bucket `radar-images-sv`. Format: PNG with continuous HSV-style hue gradient (green → cyan → blue → magenta on the saturated outer ring); decoded by arc-detect + linear hue → dBZ map. Resolution 409×342 at \~1 km, regular lat/lon. 5-minute cadence, \~24-hour bucket retention. Filenames embed local time (UTC-6, no DST). License: MARN explicitly permits reproduction with citation.

**Implementation surprise worth flagging for any contributor working a similar product**: the published legend image (`escalaPropuesta2013SNEThW_.png`) is for the 60 km multi-radar composite, NOT for `esar82`. The actual 120 km product uses a smooth HSV gradient with no discrete bins. Decoding against the legend's discrete palette yields garbage; decoding against the hue arc yields the right answer.

### Taiwan — CWA QPESUMS Composite

Region: TWCOMP (group `TAIWAN`). Source: CWA O-A0059-001 / 雷達合成回波 on the anonymous AWS S3 bucket `cwaopendata` in `ap-northeast-1`. UTF-8 XML with raw dBZ as comma-separated scientific-notation floats inside a single `<content>` element. Resolution 921×881 at 0.0125° (\~1.4 km), regular lat/lon. 10-minute cadence. Coverage: Taiwan plus a substantial western Pacific buffer for typhoon tracking. License: data.gov.tw Open Government Data License v1.0, attribution required.

URL pattern: `https://cwaopendata.s3.ap-northeast-1.amazonaws.com/history/Observation/{YYYYMMDDHHMM}compref_mosaic.xml` — **no separator dot** between timestamp and product name. The interleaved QPESUMS gauge keys *do* use a dot; easy to mix up. Filename timestamps are Taipei local (UTC+8, no DST). Row order is south-to-north (first value is SW corner) — vertical flip on decode. Datum is TWD67 (sub-pixel offset vs WGS84 at 1.4 km; treated as lat/lon).

### Malaysia — MET Malaysia 12-radar composite

Regions: MYPENINSULAR (Peninsular Malaysia + N. Sumatra + Singapore via KLIA's 240 km Doppler reach) and MYEAST (Borneo + Brunei), both in group `SOUTHEAST_ASIA`. Source: Jabatan Meteorologi Malaysia national radar composite via anonymous HTTPS at `api.met.gov.my`.

Format: animated GIF89a, 1352×570, 6 frames at 10-min cadence (\~60 min of backfill per fetch). Decoded via 18-stop palette → dBZ table (Marshall-Palmer Z = 200·R^1.6 applied to the legend's mm/h stops). Frames cropped to two sub-rectangles. Combined bounding box 96.92–121.19°E × -1.48–9.18°N, split across the South China Sea gap. Single fixed URL `https://api.met.gov.my/static/images/radar-latest.gif` — no per-frame paths. License: CC-BY-4.0 (METMalaysia / api.met.gov.my), attribution required.

**Two implementation gotchas worth documenting for contributors hitting similar products:**

1. **Timestamp anchoring**: the GIF carries no per-frame timestamps (only burned-in chrome text). MET publishes each 10-min slot \~11 min after its real data time, so anchoring the "newest" frame on the real publication time leaves the current slot perpetually empty in the served frames. The fix is to relabel frames against the current wall-clock 10-min slot rather than the upstream timestamp. Configurable lag defaults to 600 s.
2. **State-boundary gaps**: a burned-in state-boundary line leaves hairline gaps in the decoded reflectivity. Post-decode morphological close fills them.

Vendor credit visible in the burned-in chrome ("Rainbow 5 / LEONARDO Germany GmbH") is a tool credit, not a license obligation.

## Radar — Reverted and removed

Two sources shipped and were removed within the same day. Both reverts taught something concrete that informed the upstream-contribution criteria.

### Singapore — MSS Changi (shipped and removed 2026-05-15 / 2026-05-16)

Shipped 2026-05-15 as `SEACOMP` (480 km / 30-min, with optical-flow interpolation), migrated to the 50 km / 5-min `SGCOMP` product on 2026-05-16, then removed later the same day. The technical implementation worked fine; the problem was compositing.

When MET Malaysia shipped on 2026-05-16, its KLIA radar (in the MYPENINSULAR region) covered Singapore via the standard 240 km Doppler reach. SGCOMP and MYPENINSULAR disagreed in detail along the SGCOMP rectangle's edges, producing a visible seam in the rendered tiles. Two sources fighting over the same area, neither obviously correct.

The call was to drop the smaller-domain source rather than special-case the seam. Singapore is now covered by MET Malaysia at coarser resolution but visually consistent with the surrounding region.

**Lesson**: when a new source overlaps an existing one, the question isn't just "is the new data better?" — it's "do they agree at the seam?" If they don't, picking one and removing the other is cleaner than trying to feather across them. Same posture as the NWP chain's feathered hand-off, just applied to radar.

If a sharper Singapore-specific layer is wanted again, the path is documented: anonymous HTTPS at `https://www.weather.gov.sg/files/rainarea/50km/v2/dpsri_70km_{YYYYMMDDHHMM}0000dBR.dpsri.png`, RGBA PNG 217×120 at \~0.5 km/px, Singapore Open Data Licence v1.0. The composite-with-MMD seam is the open design problem that blocks re-adoption.

### Philippines — PAGASA PANAHON (shipped and reverted 2026-05-16)

Shipped as `PHCOMP` in group `SOUTHEAST_ASIA` on 2026-05-16 and reverted later the same day. The technical implementation worked; the upstream product was the problem.

PAGASA publishes a single 2048×2048 PNG covering the whole archipelago, but the image carries data only inside each station's range circle. Three of the eight contributing radars (Echague, Kabacan, Panabo) publish at \~80 km rather than the assumed 240 km, leaving the coverage mask claiming radar coverage in areas with no data. This rendered as a sharp black hole over northern Luzon where the mask blocked the IFS fallback. Across a typical frame only \~2065 pixels (\~0.05% of the grid) carried real returns.

**Lesson**: "national mosaic" framing in upstream documentation can be misleading when stations publish at heterogeneous ranges. The negative-space behaviour (where the source claims coverage but has no data) matters as much as the positive-space behaviour, because LibreWXR's coverage mask suppresses the fallback NWP chain inside the claimed area. PAGASA's product is technically a national composite; visually it's eight disjoint circles.

What it would take to revisit: per-station ranges (the current radar infrastructure assumes one range per region), max-pool downsampling for sparse high-res frames at low zoom, and optional disabling of the NWP fallback over PHCOMP to make the rendered output match PANAHON's radar-only appearance. Defensible to leave it removed indefinitely.

Worth noting the license itself was fine — PAGASA's site-wide public-domain statement covered the radar imagery. The revert was a data-quality call, not a licensing one. An open license is a prerequisite for upstream inclusion, but the data also has to be usable in practice.

## Radar — Tier 1

Validated against the upstream endpoints. License and access path are clear. Implementation is queued behind whatever else is in flight.

### Cayman Islands — CINWS

Source: weather.gov.ky CDN, anonymous, no WAF. URL pattern `https://www.weather.gov.ky/cdn/assets/images/radar/{product}/{TIMESTAMP}{type}.jpg`. Timestamp format `YYYYMMDDHHMMSSSS` (UTC, clock-aligned, \~5-minute cadence). JPEG 1276×1000 RGB, \~65 KB per frame.

Products available include PPI 400 km (largest coverage — recommended primary), CAPPI 250 km, 40 nautical-mile zoom, DPSRI rainfall rate, and a Sister Islands product. The DPSRI naming and file-name suffixes (`dBZ.cappi.jpg`, `dBZ.ppi.jpg`, `dBR.dpsri.jpg`) suggest the same vendor product line as the old Singapore MSS feed — Selex/Leonardo. Bermuda likely runs the same family.

**Open questions before implementation:**
- **Bounds**: not found in the radar HTML or in the JS bundles checked so far. The fallback is centre-on-radar + per-product range arithmetic, verified empirically by aligning the Cuba/Jamaica coastline against a basemap. Final bounds should come from a deeper JS dig or from CINWS directly.
- **JPEG decode**: lossy compression means RGB triplets drift around discrete intensity bands. The mitigation is fuzzy palette matching (snap each pixel to the nearest of N known colours within a tolerance), but the band-width needs measurement against a known precipitation frame.
- **License**: not explicitly published on the radar page. CINWS recently upgraded their site (May 2026) with explicit language about users being able to "download weather data directly for research, planning and operational use" — implying open-access intent. Direct confirmation from CINWS via their website contact form is still warranted before shipping.

### Bermuda — BWS

Source: weather.bm, anonymous. URL pattern `https://www.weather.bm/images/Radar/CurrentRadarAnimation_500km_dBZ_0deg/{YYYY-MM-DD-HHMM}_500km_dBz_0deg.png` (note mixed casing: `dBZ` in the directory path, `dBz` in the filename). Frame manifest exposed at `https://www.weather.bm/tools/graphics.asp?name=500KM%20PPI` — 5 frames, \~30 min trail, page-scrape pattern. Format: PNG 1032×700, 8-bit RGB, **6-min cadence**, UTC timestamps.

Centre at BWS site \~32.36 °N, -64.69 °W; 500 km range ≈ 28–37 °N × -70 to -59 °W. Projection is almost certainly azimuthal-equidistant centred on the radar (the Selex/Leonardo default). The image has burned-in Bermuda outline + coordinate grid, which should make ground-truthing trivial. Product is dBZ — no Z-R inversion needed.

License: permissive with attribution. BWS disclaimer explicitly permits redistribution of "imagery" provided users acknowledge BWS as the source and quote issue time and validity. The same disclaimer prohibits "deep links, image links, or trademarked 'hot' links without written agreement" — LibreWXR's pattern is server-side fetch + re-render rather than hotlinking, but a courtesy confirmation to BWS is appropriate.

**Coverage value**: this is the largest single gap-fill in the entire radar backlog. USCOMP drops off \~80 km offshore, OPERA is \~3500 km east, MRMS Caribbean (PRCOMP) is \~1500 km south. Bermuda fills a unique slice of the western Atlantic on the hurricane track.

Open questions before implementation: exact palette / dBZ scale (legend needed — sample a precip frame or correspond with BWS); backfill depth (page exposes only 5 frames; older timestamps return 404); projection confirmation. Adding a new `proj="aeqd"` branch to the `RegionDef` machinery is the largest scoped change relative to the existing source pattern.

## Radar — Tier 2

Sources with viable open access but at least one outstanding blocker — either a license question pending an operator response, a technical friction that adds material engineering cost, or a deferral while higher-value Tier 1s ship first.

### Météo-France Antilles-Guyane (MQCOMP + GFCOMP)

Source: `rwg.meteofrance.com` WMS GetMap for `BASE_REFLECTIVITY_ANTILLES` (Guadeloupe + Martinique mosaic) and `BASE_REFLECTIVITY_GUYANE_400` (French Guiana). Anonymous after a one-step JWT bootstrap: `GET https://meteofrance.gp/fr/images-radar/mosaique-antilles`, extract the `mfsession` cookie, `urldecode` + `rot13` to a JWT, then include the JWT in the WMS query. No account, no registration. 5-minute cadence, TIME at 5-min boundaries.

Format: PNG RGBA via the `synopsis_reflectivity_oppidum_transparence` style — colour-keyed reflectivity, palette decode required. Native dimensions: ANTILLES mosaic 700×600, GUYANE_400 700×700. EPSG:3857 projection.

License: **Etalab Licence Ouverte v2.0** with attribution "Météo-France" — the same licence as the already-shipped AROME Antilles NWP source. The official Données Radar API on `portail-api.meteofrance.fr` requires account registration (auto-Tier-3 under the no-API-keys rule in `adding-a-source.md`), but the public viewer's WMS at `rwg.meteofrance.com` exposes the same data with the session-cookie bootstrap and is covered by the same Etalab licence.

**Why this is uniquely valuable**: pairs radar + high-res NWP across the same domain. First source to do so outside CONUS and Europe, since AROME Antilles is already running for the model side. Three regions (Guadeloupe, Martinique, French Guiana) are otherwise covered only by the IFS global base.

Open questions: JWT lifetime under sustained use (needs probing across hours — short-lived sessions need a refresh cycle, long-lived sessions can bootstrap once at startup); time-archive depth (capabilities suppresses the list; the viewer animation length suggests \~1–2 h); courtesy email to `contact.api@meteo.fr` for a paper trail on the `rwg.meteofrance.com` endpoint's specific terms.

### Argentina — SMN / SINARAME

17-radar SINARAME network operated by SMN — modern, recently expanded. LibreWXR already consumes SMN's WRF Argentina model from their AWS Open Data bucket (`smn-ar-wrf-dataset`), so same agency, same email contact.

**Blocker**: no SINARAME bucket exists on AWS today. The radar page at `smn.gob.ar/radar` is fronted by Cloudflare's anti-bot interstitial — sustained automated access doesn't work, and bypassing Cloudflare programmatically doesn't fit a self-hostable design.

**Recommended action**: ask SMN's open-data contact (`odp-aws@smn.gov.ar`) whether SINARAME composite reflectivity could be added to the existing AWS Open Data Sponsorship arrangement, similar to how Colombia's IDEAM publishes `s3-radaresideam`. SMN has publicly stated openness in principle ("the beneficial potential of the radars justifies making it public") and the infrastructure pattern is already in place.

If SMN says yes, the format is likely Sigmet IRIS RAW (same as Colombia), which would need `pyart`/`xradar` for decode. Tier 1 candidate at that point. If they say no, defer indefinitely — don't try to scrape past Cloudflare.

Coverage value: significant. WRF-SMN already covers Argentina + Chile + Uruguay + Paraguay + parts of Bolivia and Brazil at 4 km / hourly. Observed radar over the same domain would be a meaningful upgrade from "model only."

### Mexico — CONAGUA SMN

16-station network with per-station rendered GIF images, no national mosaic. Comparable in shape to the Canada source (palette reverse + Marshall-Palmer Z-R) but with multi-station compositing layered on top — effort estimated at \~3–5× a single-region Tier 1.

Source plumbing (no auth, but SSL cert chain is broken on the host — needs `-k` / cert pinning):
- File listing: `https://smn.conagua.gob.mx/tools/PHP/RDA/static/php/RDA_repository.php?dir=ecos&type=json`
- Image base: `https://smn.conagua.gob.mx/tools/GUI/visor_radares_v3/ecos/{filename}`
- Per-station config: `radarsDB.json` (\~13 MB) carries `radarsInfo` (16 radars), per-product `map.bounds`, `range` km, etc.

Filename pattern: `{STATION}_{PRODUCT}_{MOMENT}_XXX_PXXX_YYYYMMDD_HHMMSS.gif`. Per-station cadence \~3–5 minutes, not clock-aligned; round to nearest 10-min boundary. SMN does **not** publish a single national composite — the `config.mosaico` entry in `radarsDB.json` is a Leaflet viewer view, not a product. A national MXCOMP would have to be built by max-pooling the 16 per-station rectangles onto a unified grid.

**Blocker**: no open-data licence statement on the smn.conagua endpoints. The NCAR EOL archive of past SMN radar (`https://data.eol.ucar.edu/dataset/82.093`) is research-use-only and explicitly not redistributable. Direct confirmation from `ventanillaunica.smn@conagua.gob.mx` is needed before shipping. Rain Viewer lists SMN as a source, but Rain Viewer's terms aren't AGPL-compatible and they may have a private agreement.

Format gotcha: palette table is not documented. Capture it from a representative reflectivity GIF; the React viewer renders the legend client-side, so the palette table may be embedded in `radarsDB.json` or the JS bundle — worth a deeper search before manual colour-picking.

### Thailand — TMD

National composite at `weather.tmd.go.th/composite/index_composite.html`, driven by the HAniS animation framework. The radar layer is a transparent overlay (Singapore-style), not baked into a basemap — the architectural prerequisite for being usable. Frames composed from four layered PNGs at 1173×1668 RGBA, of which only `images/zr{NNNN}.png` (\~108 KB, \~525 distinct RGBA colours) is the radar data.

Frame manifest at `images_composite.list` — 24 frames at **15-minute cadence**, 6 hours of rolling archive. Indexed filenames rotate; the manifest is the authoritative index → timestamp map. Always parse it first.

**Three frictions, all real:**

1. **Imperva/Incapsula WAF on the host**, visible from the `<script src="/_Incapsula_Resource?...">` tag. Basic clients work today, but sustained server-IP access is at real risk over time. Mitigations (residential proxy, browser-fingerprint emulation) don't fit a self-hostable design.
2. **No published bounds**. Thailand's geographic bbox doesn't match the image aspect ratio, so either the image extends substantially over neighbouring countries or it uses a non-equirectangular projection. Empirical georeferencing against known features (Bangkok, Phuket, Mae Sai) is required.
3. **License unclear**. Radar composite page says only "Copyright © 2022 Thai Meteorological Department" with no open-data license referenced. Email confirmation from TMD is needed before shipping.

If revisited: the official TMD data API at `data.tmd.go.th/api/index1.php` may have radar imagery on a documented endpoint that bypasses the composite-page WAF — worth checking. WMO WIS2 publication is the other long-horizon trigger.

### Bahrain / Gulf Cooperation Council (gccmet.net)

URL extraction resolved 2026-05-15: `https://www.bahrainweather.gov.bh/o/ibl-image-player/images/306/cappi_Horizontal-Reflectivity_GCC_YYYYMMDDHHMM`. Anonymous, no session needed. 5-min cadence, clock-aligned, 24-hour archive. The portal is built on Liferay with IBL Software's `ibl-image-player` portlet; the 24-hour frame manifest is baked directly into the rendered HTML.

**Why this could be huge**: a single source covering all six GCC member states (Saudi Arabia, UAE, Kuwait, Qatar, Oman, Bahrain). Particularly valuable since Saudi Arabia is independently Tier 3.

**Format problems making this harder than estimated**, all newly-discovered during pre-flight:

1. **JPEG-RGB, not PNG-palette**. DCT compression smears RGB triplets across block boundaries, so the standard nearest-colour palette decode is lossy. Fuzzy colour matching with tolerance is needed.
2. **Burned-in coastlines and country borders**. Vector overlay (thin white-grey lines) is rendered into each JPEG, not delivered as a separate layer. Decoding masks the overlay pixels as phantom dBZ unless explicitly masked — and the mask carves no-data holes along every coast.
3. **No bounds published**. Image is approximately Mercator centred \~25°N, 53°E with rough bbox \~14–32°N × 42–62°E, but the bbox must be fitted empirically.
4. **No colormap legend visible in dry-region frames**. Most of the GCC is desert with very few radar returns in normal conditions; legend sampling has to wait for a rain event or guess from IBL Visual Weather's stock palette.
5. **License unclear**. Footer reads "© 2025 Ministry of Transportation and Telecommunications Kingdom of Bahrain"; portal vendor is IBL Software Engineering (Slovakia). No open-data statement.

**Recommended action before writing code**: email gccmet.net or Bahrain MoT asking for a GRIB2/NetCDF/GeoTIFF feed. They publish the visualisation openly, so the marginal ask is small. Most of the engineering cost (JPEG fuzzy decode + coastline masking + bbox fitting) evaporates if they share the underlying raster.

If the email goes nowhere, this sits at the back of Tier 2 behind cleaner-format options. The image-based path is tractable but the cost / value is worse than every Tier 1.

### Brazil — CPTEC/INPE sigma

**Path A (DECEA REDEMET) is ruled out** — `api-redemet.decea.mil.br` requires API-key registration. Auto-Tier-3 under the no-API-keys rule.

**Path B (CPTEC/INPE sigma reverse-engineering)** is the only acceptable path. The viewer at `sigma.cptec.inpe.br/radar/` aggregates data from all five Brazilian radar networks (INPE + DECEA + IPMet + CEMADEN + CENSIPAM) without an auth gate. Internal plumbing:

- Product catalog: `https://s0.cptec.inpe.br/webdsa/json_dsa/dados_radar.json` (\~756 KB). Keyed by 4-digit `idSubprod` codes; each entry has `filePath`, `fileDate`, `fileTime`, and a fully-qualified `url` to the PNG.
- Image CDN: `https://satelite.cptec.inpe.br/repositorio7/{radar}/cappi/maxcappi/YYYY/MM/R{station_id}_{YYYYMMDDHHMM}.png` — anonymous, no auth.

**Why this is back of Tier 2 and not Tier 1:**

- Per-radar PNGs, not a national mosaic — despite the viewer's appearance, the underlying data is per-station. A national composite would have to be built from \~24 stations, comparable to Mexico's complexity.
- No documented URL contract — `dados_radar.json` and `repositorio7/...` paths are the viewer's internal plumbing, not a published API. More fragile than Mexico (which at least has documented per-station bounds in `radarsDB.json`).
- Sample inspection has shown 2018-era timestamps in the top entries of the catalog. Either deep-archive priority or stale catalog — must verify current data freshness as the very first implementation step.
- Whether each per-radar PNG carries a transparent radar overlay or a basemap-baked layer hasn't been confirmed.
- Aggregator dependency: if CPTEC drops one of the five contributing networks, Brazil coverage degrades silently.
- License not explicitly published on the radar viewer page. Email confirmation needed.

Defensible to leave Brazil at the back of Tier 2 indefinitely — Argentina (single-email blocker) and Mexico (cleaner architecture) are higher-value Tier 2 targets and IFS already provides global fallback over Brazil.

### Colombia — IDEAM (archive-only)

Anonymous AWS S3 bucket `s3-radaresideam` in `us-east-1`. AWS Open Data Sponsorship Program — clean license. Four radars currently active (Barrancabermeja, Bogota, Guaviare, santa_elena). Format: Sigmet IRIS RAW (.RAWXXXX) volumetric scans — best raw data quality of any source surveyed; this is what `pyart` / `xradar` consume natively for research-grade work. Archive depth: 2018 → present.

**Blocker for the real-time pipeline**: \~24-hour publication delay. Querying today's directory returns zero keys; querying two days back returns data with all `LastModified` timestamps at \~04:04 UTC the following day. Useless for live precipitation overlay where users see "current" weather, but **Tier 1 for an archive/research mode** if LibreWXR ever adds one.

If revisited: trigger is IDEAM shortening the publication delay (would be a meaningful operational change — worth checking annually). Decoding would need `pyart` or `xradar`, which is a meaningful new dependency LibreWXR currently avoids.

## Radar — Tier 3

Sources that aren't currently viable. Most blockers are structural rather than technical — API-key gates, commercial state-owned-enterprise models, app-only distribution policies, or terms of service that explicitly forbid commercial redistribution. Documented so future surveys don't re-discover the same blocker.

### Australia — Bureau of Meteorology

Partially viable but deferred. The FTP composite (IDR00004) at `ftp.bom.gov.au/anon/gen/radar/` is reachable anonymously and the data exists in a documented form (512×512 palette-indexed PNG, 10-min cadence). But:

- **Data licensing unclear**. BOM's copyright notice and automated-access warnings are aggressive; redistribution terms for the anonymous FTP aren't explicitly stated.
- **Projection complexity**. The composite is 512×512 but exact geographic bounds are undocumented. Equirectangular estimates by marker-matching land 25 of 62 markers within 5 px (avg 3.4 px error) — western Australian markers show systematic displacement suggesting meridian convergence, possibly Lambert conformal.
- **WMTS inaccessible server-side**. The higher-quality `api.bom.gov.au` WMTS layer with actual dBZ tiles is behind Akamai CDN that blocks non-browser requests.

If revisited: contact BOM about data access terms; check whether `api.bom.gov.au` has a public API key programme (would still need self-hosting compatibility); study `Makin-Things/bom-radar-card`'s projection handling for the WMTS approach.

### Japan — JMA

Not openly available for programmatic access. Worth re-checking — JMA has been expanding open-data initiatives over time, but the current radar imagery is not on an anonymous endpoint.

### South Korea — KMA

Korean-only documentation; unclear format and access. Worth re-checking if English documentation becomes available.

### India — IMD

Not investigated in depth. Large coverage area would be high-impact if open access exists.

### Netherlands — KNMI

ODIM HDF5 data is available but requires free API-key registration. Automatic Tier 3 under the no-API-keys rule. Coverage also overlaps entirely with OPERA, so even if the API gate were lifted the marginal value would be near-zero.

### Finland — FMI standalone

`fmi::radar::composite::dbz` via WMS — raw uint8 GeoTIFF, anonymous, free. Skipped because coverage overlaps entirely with OPERA. Only useful in deployments where OPERA is disabled.

### New Zealand — MetService / NIWA

Structural blocker. Both NZ meteorological agencies operate under commercial mandates that prevent unauthenticated public radar publication.

- **MetService open-access tier** is SFTP-by-email (`data.manager@metservice.com`) — credential-gated, equivalent to an API key under the upstream criteria.
- **MetService modern API** at `data.metservice.com` is explicitly API-key gated.
- **NIWA** operates a commercial delivery model (web interfaces, contracted APIs, GIS).

MetService is a State-Owned Enterprise, required by mandate to operate commercially. MBIE's *Weather Permitting* review (2024-ish) explicitly flagged this: *"access to observational weather data in New Zealand is more restricted compared with some other countries, largely because of the state-owned enterprise (SOE) and Crown research institute (CRI) models."* Neither agency is going to mirror radar to anonymous AWS/CDN, because that would undercut their own commercial offerings.

Trigger to revisit: NZ government policy change repurposing MetService away from the SOE model, or MetService voluntarily publishing radar to a CC-BY-licensed anonymous endpoint. Neither is on the horizon.

### Costa Rica — IMN

Paid / service-request model. IMN's data offerings explicitly note "information has associated costs depending on what is requested" — a pay-per-request model administered through service-request forms. AGPL self-hosted redistribution is incompatible with per-request paid licensing.

### Hong Kong — HKO

Technically reachable anonymously (manifest at `nradar_img.json`, 6-min cadence, JPEG 577×400 at 64/128/256 km ranges), but the licence explicitly prohibits commercial redistribution: *"the use of the Materials for commercial purposes is strictly prohibited unless... prior written authorisation is obtained."* Radar imagery is **not** on the `data.gov.hk` Open Data API — the permissive open-data terms don't cover it.

If the licence ever opens, implementation effort would be \~1.5–2 days. The product is rainfall rate (mm/h, 13 discrete bins), not raw dBZ — would need Marshall-Palmer inverse. Burned-in logo / legend / range rings / Chinese text / terrain basemap, plus circular crop with corner-fill, plus JPEG lossy compression all make palette matching harder than a clean PNG-palette source.

Trigger to revisit: HKO adds radar to `data.gov.hk` (where they already publish warnings, climate, station data under permissive terms).

### Saudi Arabia — NCM

No anonymous public radar URL exists. NCM's ArcGIS Hub publishes climate, surface station, and alert data but conspicuously no radar. `meteo.ncm.gov.sa` is a registered-user portal (MeteoKSA). `beta.ncm.gov.sa` doesn't resolve from non-Saudi hosts.

Evidence the data is gated: the third-party viewer at `radar-flask.xyz` displays NCM radar with KML overlays, but its JS calls `/radar_image_proxy/${colormap}/${ts}` — a backend proxy on its own server, not direct NCM URLs. The proxy exists precisely because direct anonymous access doesn't work; the operator likely registered for MeteoKSA and runs the proxy from their own infrastructure. LibreWXR can't follow that path — would require either every self-hoster getting MeteoKSA credentials (API-key rule violation) or routing through a central proxy (not self-hostable).

Trigger to revisit: NCM publishes radar to their ArcGIS Hub, to anonymous AWS/CDN, or via a documented free-tier API on `data.ncm.gov.sa` or similar.

### Turkey — MGM

\~25-radar national network, but no open raw data. `www.mgm.gov.tr/sondurum/radar.aspx` serves rendered PNG snapshots only — per-station products and a national "combined" composite. Latest snapshot only, no time-series archive, no documented API, no published terms of use. The raw data lives behind MEVBİS (`Meteorolojik Veri-Bilgi Sunum ve Satış Sistemi` — literally "Sales System"), MGM's commercial portal with per-request licensing.

Turkey is fully inside the ICON-EU domain (29.5–70.5°N, 23.5°W–62.5°E), so the loss is \~155 km radar resolution → 7 km model resolution, not radar → 9 km IFS. Material but bounded.

Trigger to revisit: MGM joins OPERA (talked about for years, no movement) or publishes an open-data radar feed.

### Vietnam — NCHMF

Policy-blocked. The official site at `nchmf.gov.vn` is unreachable or intermittently slow from outside the country, but the deeper blocker is that the World Bank's SE Asia hydromet policy note explicitly identifies Vietnam as one of the countries where regional hydromet data sharing is not yet achieved — data access is restricted by national policy, not a documentation gap. Matches the regional pattern across Cambodia, Laos, Myanmar, Vietnam.

Trigger to revisit: Vietnam publishes radar to WMO WIS2 (currently sparse — only a few countries publish radar metadata there, and Vietnam is not among them).

### Indonesia — BMKG

App-only distribution policy. The national composite URL `https://inderaja.bmkg.go.id/Radar/Indonesia_ReflectivityQCComposite.png` is documented, but all `/Radar/*` requests return HTTP 301 → a generic splash image. BMKG's own viewer page states: *"Currently radar weather imagery can only be accessed through the Info BMKG application."* The open-data portal `data.bmkg.go.id` publishes forecasts, earthquakes, and nowcast alerts but radar is conspicuously absent.

Trigger to revisit: BMKG adds radar to `data.bmkg.go.id` (where they already publish forecasts/earthquakes/nowcasts under a citation-required open licence). If that happens, BMKG jumps straight to Tier 1 — the composite product clearly exists and the agency is comfortable with the open-data model for other products. This is the most promising future SE Asia candidate if the policy ever flips.

### Iceland — Vedur.is

Only pre-coloured images with map backgrounds baked in (540×383 PNG with grid lines, coastlines, legends). Not usable as a tile overlay. Same shape of blocker as Turkey but smaller.

## NWP — Implemented

LibreWXR's NWP chain blends multiple regional models on top of a global base, dispatched per-pixel via feathered hand-off. Lower priority numbers run first.

### HRRR-CONUS (priority 10)

3 km native LCC, NOAA S3 anonymous bucket `noaa-hrrr-bdp-pds`. Uses `wrfsubhf` for native 15-min steps plus linear interpolation (no optical flow needed at native cadence). Native `composite_reflectivity` (REFC) — no Z-R conversion. Snow mask supported via the shared `compute_snow_mask` helper.

### HRRR-Alaska (priority 11)

Same model as HRRR-CONUS, disjoint domain. 3 km native polar stereographic, hourly `wrfsfcf` (no subh available). Same `noaa-hrrr-bdp-pds` bucket, `/alaska/` prefix, `.ak.grib2` filename infix. Publishes \~80 minutes after run init (later than CONUS subh). Linear interpolation only — optical flow deferred per the original integration plan.

### HRDPS (priority 20)

ECCC continental at 2.5 km native rotated lat/lon, 6-hourly cycles, 1-hour APCP accumulation. Anonymous HTTPS via dd.weather.gc.ca date-prefixed archive path. Implementation gotcha worth flagging: ECCC GRIB stores rotation as **south pole** at (-36.08852°, 245.305°); the LibreWXR code uses **north pole** at the antipode, which is the same physical rotation but the rotated-lon axis runs backwards through the grid (col 0 = highest rlon). Sign flip in the column equation; verified against all four GRIB corners. Variable comes back as `unknown` (paramId table miss in cfgrib), not `tp` — fall back to first-2D-var-matching-shape.

### AROME Antilles (priority 25)

Météo-France 1.3 km native / 2.5 km public distribution (0.025° regular lat/lon, with NaN mask outside the trapezoidal native grid). Anonymous HTTPS via the OVH Swift bucket at `meteofrance-pnt.s3.rbx.io.cloud.ovh.net` (Météo-France migrated PNT distribution off `object.data.gouv.fr/meteofrance-pnt/` around 2026-01; the data.gouv.fr API still works but its file links now 302 to OVH). 4 cycles/day (00/06/12/18Z), 0–48 h horizon. License: Etalab Open Licence v2.0. Scan mode 0 (row 0 = north, no flip needed) — the first source where this is the case. `tp` shortName recognised directly (no paramId fallback). Cumulative-since-init precip → diff against prior step → Marshall-Palmer Z-R.

Snow mask not enabled — the domain is tropical.

### AROME Guyane (priority 26)

Same upstream + family base as AROME Antilles. URL token `GUYANE`. Domain ~1156 × 877 km covering French Guiana plus the eastern strip of Suriname and a sliver of Amapá, Brazil. Pairs with Antilles to give continuous Caribbean → Guianas coverage. Domain back-decoded from GRIB Section 3 on 2026-05-21.

### AROME Indien (priority 27)

URL token `INDIEN`. The largest AROME-OM grid at ~3742 × 2492 km — covers Réunion, Mayotte, the Comoros, almost all of Madagascar, and the Tanzanian coast. ~70 MB RAM is the highest memory cost of any AROME-OM variant, so disable it if low on memory and not serving Indian Ocean clients.

### AROME Nouvelle-Calédonie (priority 28)

URL token `NCALED`. Domain ~1357 × 1360 km covering New Caledonia + Loyalty Islands + the southern half of Vanuatu.

### AROME Polynésie (priority 29)

URL token `POLYN`. Domain ~1365 × 1404 km covering the Society Islands (Tahiti, Bora Bora) and the Tuamotu archipelago. The Marquesas are outside the public 0.025° grid even though they're part of the operational model's footprint. An earlier survey claimed Polynésie ran 12-hourly, but a bucket probe on 2026-05-21 confirmed 6-hourly cycles like every other AROME-OM variant.

### DMI HARMONIE-AROME DINI (priority 30)

2 km native LCC, 3-hourly cycles. Anonymous AWS S3 bucket `dmi-opendata` in `eu-north-1` (`--no-sign-request`). No `.idx` files — uses byte-range header walk to find the precip message offset. Domain covers UK + Ireland, France, Benelux, Germany, Switzerland, Austria, northern Italy, Czechia, Poland, **all of Scandinavia including most of Finland**, Denmark, and Iceland (DMI's published computational grid extends past the marketing maps).

Outside the DINI domain: Iberia, southern Italy, Greece, Balkans, Belarus / Ukraine, the very-northern tip of Norway, and Murmansk. ICON-EU fills those at priority 35.

### ICON-EU (priority 35)

7 km lat/lon, 3-hourly cycles. DWD opendata at `opendata.dwd.de/weather/nwp/icon-eu/`. `tot_prec` accumulated; difference consecutive steps for hourly rate; Marshall-Palmer Z-R with +6 dBZ calibration offset against OPERA radar.

### WRF-SMN Argentina (priority 40)

4 km LCC, first regional NWP for the South American Cone. Anonymous AWS Open Data bucket `smn-ar-wrf` in **us-west-2** (the registry page suggested us-east-1; bucket-region header revealed otherwise). 4 cycles/day (00/06/12/18Z), 72 h horizon. License: CC BY 2.5 Argentina, AGPL-redistributable.

**First non-GRIB source in the chain** — uses NetCDF4/HDF5 via h5py. Spherical LCC, single tangent at -35°S, sphere R = 6,370,000 m (1,229 m smaller than the usual WMO sphere; calibrated against the file's `Lambert_Conformal` grid_mapping attrs and 2-D lat/lon coord arrays). Southern-hemisphere LCC subtlety: `n = sin(-35°) < 0`, but the projected y still increases going north because the `n < 0` sign already lives in F and ρ_0. `PP` field cumulative-since-init mm → diff → Marshall-Palmer.

Per-file size (\~34 MB NetCDF4) makes serial download painfully slow, so this source uses a **parallel fetch pipeline** (concurrency 6) — Phase 1 downloads accumulations in parallel, Phase 2 walks them sequentially per run to compute diffs. Other sources don't need this since their files are 1–9 MB.

### ECMWF IFS (priority 1000, global base)

9 km native (O1280 reduced Gaussian → regridded to 0.1° lat/lon). Open-Meteo S3 mirror, CC-BY-4.0. Provides pseudo-reflectivity (precipitation rate → dBZ via Marshall-Palmer) plus snow/rain classification globally. Optical flow interpolation of hourly IFS → 10-minute frames. This is the global base layer, not a "fallback" — it provides coverage everywhere a regional model isn't running.

## NWP — Tier 2

Open access and additive value, but lower priority than what's currently shipping.

### UK Met Office UKV

1.5 km native / 0.018° (\~2 km) public distribution on AWS Open Data Registry (`registry.opendata.aws/met-office-uk-deterministic/`), anonymous. 8 cycles per day, **2-year rolling archive** (unusually long). License: **CC BY-SA 4.0** — ShareAlike clause is the only complication. Code (AGPL) and data (CC BY-SA) are separate works so combining is generally fine, but downstream tile redistribution carries the SA obligation.

Domain: UK + Ireland — high overlap with DMI DINI (already 2 km native over the same area), so the upgrade is marginal at typical zooms.

### Italy — ICON-2I

2.2 km native, 72 h horizon. ItalyMeteo. Would fill the southern Italy gap below DMI DINI's southern edge. Access details (license, fetch path) need verification — the catalog metadata was thin.

The cadence is a real downside: **12-hourly cycles** mean the forecast at "now" can be 10+ hours stale by next-cycle time. ICON-EU's 3-hourly cycles are fresher more of the time despite the lower resolution.

### AROME Réunion-Mayotte / French Guiana / New Caledonia / French Polynesia

Etalab-licensed, keyless on data.gouv.fr (same channel as AROME Antilles). 1.3 km native, hourly, 4× daily. Each domain is small and remote: Indian Ocean / Madagascar (Réunion-Mayotte), N. South America corner (French Guiana), SW Pacific (New Caledonia), S. Pacific (French Polynesia). Cheap to add once AROME Antilles is in — same source class with different domain constants. \~50 LOC + tests per added overseas variant.

### Canada CAPS (Canadian Arctic Prediction System)

ECCC End-Use Licence, no auth on Datamart. 2× daily, 48 h horizon. Pan-Arctic + mid-latitude extension. Niche but covers Greenland / Arctic Ocean / high Arctic where HRDPS continental and HRRR-Alaska don't reach.

## NWP — Tier 3

### MET Nordic / MEPS (Norway)

Deferred 2026-05-08 after DMI DINI's true grid extent was confirmed via Open-Meteo's published DINI map. DINI's computational grid actually reaches northern Scandinavia and most of Finland (the DMI marketing map shows only the inner "useful" area). MEPS is post-processed to 1 km vs DINI's 2 km native, but 1 km vs 2 km is only visible at zoom 9+, and the MEPS-unique territory is just the very-northern Norway tip and Murmansk strip — tens of thousands of people. `data.met.no` is THREDDS/NetCDF, a novel fetch path requiring new infrastructure.

Revisit if a Nordic-fidelity feature ever becomes a priority.

### KNMI HARMONIE-AROME Europe (P3)

Same physics as DMI DINI, but regridded down to 5.5 km (0.05° rotated lat-lon) — DMI gives \~7× more pixels at the same domain. KNMI also requires per-user API-key registration. Picking DMI DINI is strictly better.

### KNMI HARMONIE-AROME Netherlands (P1)

2 km native over NL + BE, hourly. DMI DINI already covers the Netherlands at the same 2 km native, so this is redundant. Same API-key blocker as the Europe variant.

### Météo-France ARPEGE Europe

Confusion-prone name — this is the **global ARPEGE model** at 11 km (0.1°), NOT the high-resolution AROME family. Coarser than ICON-EU at 7 km and barely different from IFS at 9 km. Skip entirely. The Météo-France model worth wanting is **AROME-France** at 1.3 km, which needs an API key (auto-Tier-3).

### ICON-D2 (DWD Germany)

2.2 km central-Europe coverage, hourly cycles. Subsumed by DMI DINI's 2 km coverage of the same area at 3-hourly cycles — DINI's higher native resolution and broader domain wins on every axis except cycle frequency, and the cycle-frequency advantage of ICON-D2 doesn't materially affect tile rendering since the blend at T+10min is dominated by the radar side anyway.

### DMI DINI overlap models (skip all)

These are all in the DMI DINI domain at coarser native resolution, often with API-key gates: KNMI harmonie-arome-netherlands (Cy43 P1, requires KNMI API key), Belgium alaro-belgium, Czechia chmi-lam, Slovakia aladin-slovakia (4.5 km, coarser than DINI), Slovenia aladin-slovenia (4 km), Croatia aladin-hr (password-gated open data), Ireland harmonie-arome-ireland (Met Éireann; DINI fully covers Ireland at similar res), Austria arome-austria, Hungary arome-hungary.

### Switzerland — ICON-CH-EPS

Ensemble-only, not deterministic. LibreWXR's tile rendering needs deterministic output.

### Israel — COSMO-IL / ICON-IL

Israeli IMS requires signing a Terms of Use PDF and emailing it. Open licence in principle but operational friction (each self-host operator must register individually) — auto-Tier-3 under the no-API-keys spirit even if not the letter.

### Norway — AROME-Arctic

Svalbard + Barents Sea. Deferred along with MET Nordic.

### Japan — JMA LFM / MSM

Structurally blocked (verified 2026-05-08). JMBSC is the sole distributor for all of JMA's raw GRIB output (GSM, MSM, MSM upper-level, LFM) — only via paid contract, no anonymous public endpoint. Open-Meteo ingests GSM + MSM via a private credentialed URL their docs explicitly state is "not publicly disclosed." Even with a JMBSC contract, AGPL self-host redistribution would likely violate terms. The only public JMA data is GSM via NOMADS / WMO GTS mirrors at 0.5° / \~55 km — worse than IFS, not worth integrating.

### China — CMA models

Structurally blocked (verified 2026-05-08). Open-Meteo's CMA downloader requires a `--server` private credentialed URL just like JMA. `data.cma.cn` and `nmc.gov.cn` are browse-only; the WIS2 node publishes discovery metadata but the actual GRIB feeds remain on the WMO-restricted GTS pipe. No CMA bucket on AWS Open Data Registry. GRAPES-GFS is also only 0.125° (\~14 km), coarser than IFS — no upgrade even if it opened up.

### Korea — KMA LDPS / KIM

Structurally blocked (verified 2026-05-08). Anonymous FTP at `ncms.kma.go.kr` that Open-Meteo was using was discontinued by KMA on 2026-03-31 when they switched from UM-based LDPS to KIM-based l010. The replacement `apihub.kma.go.kr` requires a per-user `authKey` query parameter with a 5 GB/day quota (20 GB for approved researchers). One LDPS model cycle (49 hourly steps × \~14 variables in GRIB2) blows through that ceiling — incompatible with serving tiles to multiple users from a single self-hosted instance.

### CWA Taiwan WRF

Dropped 2026-05-08. Projection math, byte-range fetch path, and bucket access all worked cleanly. **Killed by 6-hour temporal resolution** of the open-data publication: every product in `s3://cwaopendata/Model/` publishes only at 6-hour stride from analysis through forecast horizon. For a Rain Viewer-style animated nowcast API, each 10-min frame inside CWA's domain would render as the same 6-hour-mean rate — IFS at 1h / 9 km gives smoother animation than CWA at 6h / 3 km would.

### Other SE Asia models

BMKG Indonesia, PAGASA Philippines, Vietnam NCHMF, Thailand TMD, Mongolia — all blocked, mostly via registered-account requirements or app-only distribution policies. PAGASA's open directory at `pubfiles.pagasa.dost.gov.ph/tamss/nwp/wrf/` carries PNG/GIF charts only, not GRIB — worth a deeper dig if Philippines NWP ever becomes a priority.

### Brazil — CPTEC ETA / CPTEC-Regional

Open but native resolutions are \~7 km at 1–2× daily cycles. Marginal upgrade over IFS 9 km. WRF-SMN Argentina is the better S. America pick.

### US — NAM / RAP

Both 12–13 km, coarser than IFS at 9 km. Adding them buys little. RAP's hourly cadence is the only edge but not transformative. NBM (National Blend of Models) doesn't add resolution over HRRR. HiResW Guam is 3 km but GUCOMP radar exists for Guam and IFS 9 km is fine at typical zooms.

### Canada — HRDPS-West, RDPS

HRDPS-West is alpha datamart status with 2× daily cycles; HRDPS continental already covers BC at higher cadence. RDPS is 10 km, coarser than IFS 9 km.

## Context: WMO and open-data policy

The state of the world for radar redistribution, briefly:

- **WMO Unified Data Policy does NOT mandate radar data exchange** — radar is not classified as "core" data, so member states aren't required to share it openly.
- **WMO WIS2** (operational January 2025) has very sparse radar content. Only a few countries publish radar *metadata* there, and almost none publish the actual raster data through the WIS2 channel.
- **WMO does not produce any radar composites** — the organisation is infrastructure for sharing, not a producer.
- **The EU High Value Datasets regulation (EU 2023/138) is the legal lever** that forced European radar data open. It legally requires European meteorological data to be free, open-licensed, and API-accessible. The OPERA composite as LibreWXR consumes it is a direct consequence of that regulation.
- **AWS Open Data Sponsorship Program** is the other major channel through which countries publish radar imagery — Colombia (`s3-radaresideam`), Argentina's WRF model (`smn-ar-wrf-dataset`), and US NOAA HRRR (`noaa-hrrr-bdp-pds`) all use it. SMN Argentina's existing presence on the registry makes the radar-publication ask cheaper than starting from scratch with a country that has no AWS Open Data footprint.
- The **regional pattern** in Southeast Asia is policy-driven: Cambodia, Laos, Myanmar, Vietnam, Indonesia, and (until recently) Malaysia have historically blocked radar redistribution under national policy rather than technical limitation. Malaysia's MMD source shipping on CC-BY-4.0 is a regional shift worth watching — Indonesia's BMKG and Vietnam's NCHMF are the next plausible regional opens.

## Documentation references

- **OPERA / MeteoGate**: `https://github.com/EUMETNET/openradardata-documentation/` and `https://eumetnet.github.io/meteogate-documentation/`
- **Rain Viewer source list** (used as a discovery aid, not a licensing signal — Rain Viewer's agreements are private): `https://www.rainviewer.com/sources.html`
- **Open-Meteo source catalog** (similarly useful for NWP discovery, not for licensing): the model-by-country pages on `open-meteo.com`
- **AWS Open Data Registry** (search "meteorological", "radar", "weather"): `https://registry.opendata.aws/`
- **WMO WIS2 country registry**: the single signal that meaningfully changes Tier 3 status for several countries — worth checking annually
