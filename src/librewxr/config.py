# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# Per-mode defaults for settings whose sensible value depends on whether
# LibreWXR is running as one container (single) or split into a pipeline
# + N renderer workers (multi).  Anything the user leaves at the sentinel
# value 0 is filled in from here by ``_apply_mode_defaults`` below.
# Multi-mode values are PER WORKER — total RAM scales with workers count.
_MODE_DEFAULTS: dict[str, dict[str, int]] = {
    "single": {
        "workers": 1,
        "tile_cache_mb": 200,
        "coord_cache_size": 2048,
        "warmer_threads": 0,  # 0 keeps the "auto = CPU-1" behaviour in single mode
    },
    "multi": {
        "workers": 16,
        "tile_cache_mb": 128,
        "coord_cache_size": 512,
        "warmer_threads": 4,
    },
}


class Settings(BaseSettings):
    model_config = {"env_prefix": "LIBREWXR_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 8080
    public_url: str = "http://localhost:8080"
    # Optional direct TLS termination.  Leave both unset (the default) to
    # serve plain HTTP and let a reverse proxy / tunnel handle TLS.  Set
    # both to absolute paths to have uvicorn serve HTTPS itself.
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    fetch_interval: int = 600  # seconds between fetches (10 min = radar frame cadence)
    max_frames: int = 12
    max_zoom: int = 12
    # Deployment shape.  Drives sensible defaults for workers, tile cache,
    # coord cache, and warmer threads via ``_apply_mode_defaults``.
    #   single  - one container, fetcher + renderer in the same process
    #   multi   - pipeline sidecar + N renderer workers sharing memmap state
    # Reads LIBREWXR_MODE first, then falls back to Docker Compose's
    # COMPOSE_PROFILES so docker users only need to set one env var.  Any
    # token other than "multi" resolves to "single".
    mode: Literal["single", "multi"] = Field(
        "single",
        validation_alias=AliasChoices("LIBREWXR_MODE", "COMPOSE_PROFILES"),
    )
    # All four below use 0 as a "use mode default" sentinel.  Set an
    # explicit value to override the per-mode default in _MODE_DEFAULTS.
    tile_cache_mb: int = 0  # Max tile cache size in MB (byte-capped); 0 = mode default
    coord_cache_size: int = 0  # LRU entries per coordinate cache; 0 = mode default
    memory_limit_mb: int = 0  # Container memory limit in MB (0 = auto-detect)
    memory_pressure_check_interval: int = 30  # Seconds between memory pressure checks
    smooth_radius: float = 1.0  # Baseline Gaussian blur radius; renderer auto-scales it up at high zoom on coarse sources
    noise_floor_dbz: float = 10.0  # Minimum dBZ to display; lower values are zeroed out
    despeckle_min_neighbors: int = 3  # Min non-zero neighbors (of 8) to keep a pixel; 0 to disable
    webp_quality: int = 65  # WebP quality: 100 = lossless, 1-99 = lossy at that quality
    workers: int = 0  # Number of uvicorn worker processes; 0 = mode default
    warmer_threads: int = 0  # Render thread pool size; 0 = mode default (auto in single, 4 in multi)
    warm_coord_zoom: int = 6  # Pre-warm coordinate caches up to this zoom (0 = disable)
    warm_overview_zoom: int = 4  # Pre-render ALL tiles up to this zoom on each fetch (-1 = disable)
    warm_overview_zoom_regional: int = 6  # Pre-render tiles overlapping enabled regions up to this zoom (-1 = disable)
    enabled_regions: str = "ALL"  # Region spec: CONUS, US, ALL, or comma-separated region names
    # Global radar-layer toggle.  When False, no radar provider gets
    # instantiated — every MRMS / IEM / MSC / OPERA / MARN / CWA / MMD
    # fetcher is skipped, region-coverage masks are empty, and the radar
    # tile path returns "no data" so the IFS-derived cloud cover or the
    # real-satellite layer takes over the entire map.  Useful for
    # satellite-only deployments and for faster startup during
    # development of non-radar features.  Per-source toggles below
    # still apply when this is True.
    radar_enabled: bool = True
    # Master switch for every regional NWP source — HRRR, HRRR-Alaska,
    # HRDPS, AROME-OM family, DMI DINI, ICON-EU, WRF-SMN, JMA MSM.  When False,
    # the NWP chain collapses to ECMWF IFS alone, keeping the global
    # precipitation layer intact.  Useful for fast startup during
    # satellite-only or nowcast-only development.  Per-source
    # ``*_enabled`` toggles still apply when this is True.
    regional_nwp_enabled: bool = True
    # Master switch for the satellite layer.  Backs the /v2/satellite/...
    # endpoint with NOAA's hourly global GMGSI mosaic (LW + VIS via
    # composite — see docs/satellite-implementation-plan.md).  When
    # False, the satellite endpoint returns 503 and the catalog's
    # ``satellite.infrared`` array is empty (same pattern as
    # ``radar_enabled``).  Per-channel toggles below still apply when
    # this is True.
    satellite_enabled: bool = True
    # Per-channel GMGSI toggles.  LW backs the IR night side of the
    # composite; VIS adds the daytime reflected-sunlight overlay with
    # natural terminator crossfade.  Disabling VIS while LW stays on
    # degrades the composite to LW-only without breaking the endpoint.
    gmgsi_lw_enabled: bool = True
    gmgsi_vis_enabled: bool = True
    # Number of hourly satellite frames retained per channel.  GMGSI
    # publishes one frame per hour, so 12 ≈ 12 hours of animation.
    # At ~15 MB per channel per frame, 12 × 2 channels ≈ 360 MB resident.
    satellite_max_frames: int = 12
    # US-side radar data source (USCOMP / AKCOMP / HICOMP / PRCOMP / GUCOMP).
    # Three modes:
    #   mrms_fallback  - (default) MRMS primary + IEM fallback when MRMS fails.
    #                    Best coverage, slightly more bandwidth.
    #   mrms           - MRMS only, no fallback.  Pure MRMS where available,
    #                    gaps show as empty (IFS fills in).  Least bandwidth.
    #   iem            - Legacy IEM N0Q.  NEXRAD-only.
    # Canada-side (CACOMP) is controlled independently by ``ca_source``.
    na_source: Literal["mrms", "mrms_fallback", "iem"] = "mrms_fallback"
    # Canada-side radar data source (CACOMP).  Three modes, fully independent
    # of ``na_source``:
    #   mrms_with_msc_blend - (default) MRMS primary covering southern Canada,
    #                         MSC Canada blended in to fill gaps north of MRMS's
    #                         bbox, MSC fallback if MRMS fails.  Best coverage.
    #   mrms                - MRMS only.  Southern Canada covered by MRMS's CONUS
    #                         product; northern Canada (outside MRMS bbox) falls
    #                         through to IFS.  No MSC fetched at all.
    #   msc                 - MSC Canada standalone.  Native ECCC composite for
    #                         all of Canada, no MRMS contribution to CACOMP.
    ca_source: Literal["msc", "mrms", "mrms_with_msc_blend"] = "mrms_with_msc_blend"
    iem_base_url: str = "https://mesonet.agron.iastate.edu"
    msc_canada_base_url: str = "https://geo.weather.gc.ca"
    mrms_base_url: str = "https://mrms.ncep.noaa.gov/2D"
    opera_base_url: str = "https://s3.waw3-1.cloudferro.com"
    # El Salvador MARN/SNET 120 km radar via anonymous Google Cloud Storage
    # bucket ``radar-images-sv``.  5-min cadence, PNG with HSV-style hue
    # gradient.  Region group: CENTRAL_AMERICA.
    marn_base_url: str = "https://storage.googleapis.com"
    # Taiwan CWA QPESUMS composite (O-A0059-001) via anonymous AWS S3
    # (``cwaopendata`` in ``ap-northeast-1``).  10-min cadence, XML format,
    # raw dBZ in scientific-notation floats.  Region group: TAIWAN.
    cwa_base_url: str = (
        "https://cwaopendata.s3.ap-northeast-1.amazonaws.com"
    )
    # MET Malaysia (Jabatan Meteorologi Malaysia) radar composite via
    # anonymous HTTPS at ``api.met.gov.my``.  10-min native cadence — one
    # animated GIF per fetch carries 6 frames (~60 min of backfill).
    # Decoded via 18-stop palette → dBZ table, sub-rectangle split into
    # MYPENINSULAR + MYEAST regions.  CC-BY-4.0 licensed.  Sole source in
    # the SOUTHEAST_ASIA region group.
    mmd_base_url: str = "https://api.met.gov.my"
    mmd_enabled: bool = True
    # Publication lag (seconds) used as a ceiling when labelling frame
    # timestamps.  MET publishes each 10-min slot ~11 minutes after its
    # real data time, so the newest frame on the server is up to ~10 min
    # stale; the decoder labels the newest GIF frame at the current
    # wall-clock 10-min slot so the renderer's "current" slot is always
    # populated.  This setting still acts as a stale-content ceiling — a
    # response whose ``Last-Modified`` is more than this far behind wall
    # clock is treated as legitimately old data, not relabelled forward.
    mmd_publish_lag_sec: int = 600
    # DPC Italy national radar composite via the open Radar-DPC v2 REST API
    # (``radar-api.protezionecivile.it``).  Anonymous, 5-min native cadence,
    # Float32 GeoTIFF in spherical Transverse Mercator, CC-BY-SA 4.0.  Sole
    # Italian native source — pairs with OPERA in the EUROPE region group;
    # ITCOMP wins precedence over OPERA where it covers (Italy is not in
    # the EUMETNET OPERA station list, so OPERA-over-Italy is edge-of-range
    # data from neighbours).
    dpc_base_url: str = "https://radar-api.protezionecivile.it"
    dpc_enabled: bool = True
    # JMA HRPN (High-Resolution Precipitation Nowcast) composite for Japan.
    # Anonymous S3-backed CDN at ``www.jma.go.jp/bosai/jmatile/data/nowc/``,
    # 5-min cadence, XYZ tile pyramid in JMA's standard 10-stop palette PNG
    # encoding (4-bit palette for populated tiles, 8-bit RGBA for empty).
    # Two contributions from one package: the analysis leg (basetime ==
    # validtime, ingested as standard radar) and the forecast leg
    # (validtime > basetime, ingested via NowcastContribution to replace
    # internal optical-flow extrapolation for JPCOMP).  JMA Public Data
    # License v1.0 — CC-BY equivalent with commercial reuse explicitly
    # permitted, attribution required.
    jma_base_url: str = "https://www.jma.go.jp/bosai/jmatile/data/nowc"
    jma_enabled: bool = True
    # Tile zoom level used for the fetch+stitch pipeline.  JMA's HRPN
    # pyramid only serves real PNG palette data at EVEN zoom levels
    # (z=4, 6, 8, 10); odd zooms return empty RGBA placeholder PNGs.
    # z=8 (~1.25 km/px, ~420 tiles per frame) matches the JPCOMP grid
    # resolution (0.0125°/px ≈ 1.4 km/px) exactly.  Affordable now
    # that the JMA N2 forecast leg is gone — analysis-only at z=8
    # is ~420 tiles per 10-min cycle.  z=6 (~30 tiles, 5 km/px) is
    # the bandwidth-saver fallback if cold-start is a problem.
    # z=7 / z=9 must NOT be used — they look syntactically valid but
    # produce all-empty frames.
    jma_zoom: int = 8
    ecmwf_s3_bucket: str = "openmeteo"
    ecmwf_s3_region: str = "us-west-2"
    ecmwf_s3_prefix: str = "data_spatial/ecmwf_ifs"
    ecmwf_snow_ratio_threshold: float = 0.5
    # Temperature threshold (°C) for snow classification in regional NWP
    # sources that don't carry a native snow-ratio field.  Each source
    # computes snow = (T_2m < threshold).  1.5 °C matches Open-Meteo's
    # softer threshold and captures wet-snow at ground level without
    # painting cold rain as snow.  Used by HRRR, HRRR-Alaska, HRDPS,
    # DMI DINI, ICON-EU, and WRF-SMN.  IFS uses its native snowfall
    # ratio (``ecmwf_snow_ratio_threshold`` above) instead.
    regional_snow_temp_threshold: float = 1.5
    # Apply the same Farneback optical-flow temporal interpolation we use
    # on hourly IFS frames to regional NWP sources whose native cadence
    # is also hourly (WRF-SMN, DMI DINI — others coming).  Without this,
    # a moving precip cell appears to cross-fade between hourly bracket
    # frames at intermediate query times, producing a visible "two faint
    # copies" ghost.  With this on, the cell translates smoothly along
    # the motion vectors.  Adds ~5-10 s of CPU per source per fetch
    # cycle.  Set to False to fall back to linear time blending.
    regional_interpolation: bool = True
    ecmwf_max_timesteps: int = 0  # 0 = auto (derived from max_frames)
    ecmwf_interpolation: bool = True  # Optical flow interpolation of IFS hourly data to 10-min frames
    # Disable IFS entirely (skip the global precipitation fallback).  Useful
    # for isolating regional NWP layers during debugging — anywhere outside
    # the regional models will simply show zero precipitation.  Default
    # leaves IFS on; turn off only when you specifically want to see what
    # a regional model contributes on its own.
    ecmwf_enabled: bool = True
    # North American NWP source for the chain. "ifs" uses ECMWF IFS as the
    # only source (current behavior). "hrrr" prepends NOAA HRRR-subh as the
    # CONUS-priority source, falling back to IFS outside HRRR's domain.
    na_nwp_source: Literal["ifs", "hrrr"] = "ifs"
    hrrr_s3_bucket: str = "noaa-hrrr-bdp-pds"
    hrrr_s3_region: str = "us-east-1"
    hrrr_publish_delay_minutes: int = 55  # subh files typically publish ~55 min after run init
    # NOAA HRRR-Alaska is bundled with HRRR-CONUS — both are NCEP's
    # HRRR run on disjoint domains and share the same anonymous S3
    # bucket, so ``LIBREWXR_NA_NWP_SOURCE=hrrr`` enables both.  Alaska
    # runs separately at native 3 km polar stereographic, 3-hourly
    # cycles (00/03/06/09/12/15/18/21Z), 0-48 h horizon, hourly wrfsfcf
    # surface files (no subh — Alaska runs only have hourly steps).
    # The publish delay is set independently because the AK run takes
    # longer to finish than CONUS subh.
    hrrr_alaska_publish_delay_minutes: int = 80  # full run (0-48 h) typically published within ~80 min
    # European NWP profile for the chain.  Each profile names the full
    # set of European sources that get instantiated (the chain order
    # itself is fixed: narrowest-domain first).
    #   "ifs"                 - no regional NWP; IFS is the only source.
    #   "icon_eu_only"        - DWD ICON-EU (~7 km, 3-hourly) ahead of IFS.
    #   "dini_with_icon_eu"   - DMI HARMONIE-AROME DINI (2 km native LCC,
    #                           3-hourly) ahead of ICON-EU ahead of IFS.
    #                           DINI covers most of populated Europe;
    #                           ICON-EU stays in the chain to fill
    #                           Iberia, southern Italy, the Balkans, and
    #                           eastern Europe past Poland that DINI
    #                           doesn't reach.
    # See project memory entry "EU NWP profile naming refactor"
    # (project_eu_nwp_profile_refactor.md) for the planned future move
    # to a list-valued LIBREWXR_EU_NWP_CHAIN setting.
    eu_nwp_profile: Literal["ifs", "icon_eu_only", "dini_with_icon_eu"] = "ifs"
    icon_eu_base_url: str = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
    icon_eu_publish_delay_minutes: int = 240  # main runs typically publish ~3-4h after init; 4h is conservative
    # dBZ calibration shift applied after Z-R conversion of ICON-EU
    # precipitation rates.  Marshall-Palmer (Z = 200 * R^1.6) is for
    # stratiform rain at the surface; radar reflectivity is sampled at
    # the brightest part of the storm column and tends to read 5-10
    # dBZ higher than the surface rate would predict.  Tune up to make
    # convective cells closer in colour to OPERA radar.
    icon_eu_dbz_offset: float = 6.0
    # DMI HARMONIE-AROME DINI is published anonymously on AWS Open Data
    # (s3://dmi-opendata in eu-north-1).  Each (run, lead) is a single
    # ~600 MB GRIB2 file; we fetch only the tp message (~9 MB) per leadtime
    # via byte-range, locating it via a one-per-run header walk.
    dmi_dini_s3_bucket: str = "dmi-opendata"
    dmi_dini_s3_region: str = "eu-north-1"
    dmi_dini_publish_delay_minutes: int = 180  # files publish ~3 h after run init
    # Same Marshall-Palmer caveat as ICON-EU.  HARMONIE has no native
    # composite reflectivity output so we derive dBZ from accumulated tp.
    dmi_dini_dbz_offset: float = 6.0
    # ECCC HRDPS continental: 2.5 km native rotated lat/lon, 4 cycles/day
    # (00/06/12/18 UTC), 48 h horizon, 1-hour APCP accumulation.  Anonymous
    # HTTPS via dd.weather.gc.ca — no auth, no API key.  Independent toggle
    # from na_nwp_source: HRDPS covers Canada + the northern fringe of
    # CONUS, disjoint enough from HRRR's CONUS focus to layer cleanly
    # (HRRR first inside CONUS where it's denser; HRDPS second to fill
    # Canada).  The base URL is the dd.weather.gc.ca root — the URL
    # builder appends the date-prefixed archive path so we can fetch
    # runs from yesterday across midnight UTC without the ``/today/``
    # tree rolling out from under us.
    hrdps_enabled: bool = True
    hrdps_base_url: str = "https://dd.weather.gc.ca"
    hrdps_publish_delay_minutes: int = 240   # ~3.5-4 h after init; 4 h conservative
    hrdps_dbz_offset: float = 6.0            # same Marshall-Palmer caveat as DINI/ICON-EU
    # Météo-France AROME Antilles: 1.3 km native, 0.025° (~2.5 km) regular
    # lat/lon public dist, 4 cycles/day (00/06/12/18 UTC), 48 h horizon,
    # cumulative-since-init total precipitation.  Anonymous via the
    # data.gouv.fr open-data portal — no API key.  Independent toggle
    # since the Antilles domain is disjoint from every other regional
    # source in the chain (Caribbean, abuts the existing PRCOMP radar).
    arome_antilles_enabled: bool = True
    # Météo-France migrated PNT distribution off object.data.gouv.fr to
    # direct OVH Swift hosting around 2026-01.  Old host returns 404 for
    # all runs; the data.gouv.fr API now redirects every download to the
    # OVH bucket below.  Same path layout (/pnt/{run}/arome-om/...).
    arome_antilles_base_url: str = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
    arome_antilles_publish_delay_minutes: int = 420   # ~7 h after init for full 0..48h
    arome_antilles_dbz_offset: float = 6.0            # same Marshall-Palmer caveat as DINI/ICON-EU
    # Météo-France AROME Outre-Mer variants — same upstream + format +
    # decoder as Antilles via the shared AROMEOverseasGrid base.  Each
    # covers a separate French overseas territory (and surrounding
    # waters / nearby islands).  4 cycles/day, 48 h horizon, 0.025°
    # regular lat/lon, anonymous, Etalab v2.0.  Defaults enabled.
    #
    # GUYANE — French Guiana (+ Suriname, Amapá Brazil borders).
    arome_guyane_enabled: bool = True
    arome_guyane_base_url: str = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
    arome_guyane_publish_delay_minutes: int = 420
    arome_guyane_dbz_offset: float = 6.0
    # INDIEN — Réunion, Mayotte, Comoros, much of Madagascar + adjacent
    # SW Indian Ocean (~3700×2500 km, the largest of the AROME-OM grids).
    arome_indien_enabled: bool = True
    arome_indien_base_url: str = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
    arome_indien_publish_delay_minutes: int = 420
    arome_indien_dbz_offset: float = 6.0
    # NCALED — Nouvelle-Calédonie + adjacent SW Pacific (Vanuatu side).
    arome_ncaled_enabled: bool = True
    arome_ncaled_base_url: str = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
    arome_ncaled_publish_delay_minutes: int = 420
    arome_ncaled_dbz_offset: float = 6.0
    # POLYN — French Polynesia (Society, Tuamotu, Marquesas archipelagoes)
    # spread across the S Pacific.
    arome_polyn_enabled: bool = True
    arome_polyn_base_url: str = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
    arome_polyn_publish_delay_minutes: int = 420
    arome_polyn_dbz_offset: float = 6.0
    # SMN Argentina WRF-DET: 4 km LCC over Argentina + Chile + Uruguay
    # + Bolivia + Paraguay + S. Brazil — first regional NWP for the
    # South American Cone.  Anonymous AWS Open Data S3 (smn-ar-wrf in
    # us-east-1), 4 cycles/day (00/06/12/18 UTC), 72 h horizon, NetCDF4
    # files (~34 MB each).  Independent toggle since this is the only
    # source covering the South American chain slot.
    wrf_smn_enabled: bool = True
    wrf_smn_s3_bucket: str = "smn-ar-wrf"
    wrf_smn_s3_region: str = "us-west-2"   # bucket region per x-amz-bucket-region
    wrf_smn_publish_delay_minutes: int = 240          # ~3-4 h after init for full 0..72h
    wrf_smn_dbz_offset: float = 6.0                   # same Marshall-Palmer caveat as DINI/ICON-EU
    # JMA MSM (Mesoscale Model) for Japan + adjacent East Asia: native 5 km
    # (0.0625° lon × 0.05° lat), 8 cycles/day (00/03/06/09/12/15/18/21 UTC),
    # 78 h horizon from the main 00Z/12Z runs (39 h others), hourly steps.
    # Distributed via Open-Meteo's anonymous AWS Open Data mirror (same
    # ``openmeteo`` bucket we use for IFS).  Pairs with the JMA HRPN radar
    # (analysis leg → JPCOMP) to give the East Asia chain slot a high-
    # resolution Japanese mesoscale NWP overlay instead of falling through
    # to global IFS.  ~3 MB per fetch (~4 .om files × ~800 KB each in the
    # 3-hour window), ~50 MB resident.  Independent toggle since this is
    # the only source covering the East Asia NWP chain slot.
    jma_msm_enabled: bool = True
    jma_msm_s3_bucket: str = "openmeteo"
    jma_msm_s3_region: str = "us-west-2"
    jma_msm_s3_prefix: str = "data_spatial/jma_msm"
    jma_msm_publish_delay_minutes: int = 300          # ~5 h after init for full run write-out
    jma_msm_dbz_offset: float = 6.0                   # same Marshall-Palmer caveat as DINI/ICON-EU
    nowcast_enabled: bool = True  # Generate precipitation nowcast via radar extrapolation + IFS
    nowcast_frames: int = 6  # Number of 10-min forecast frames (6 = 60 min)
    nowcast_blend_mode: str = "blended"  # "radar", "blended", or "model"
    cache_dir: str = ""  # Persistent cache directory for fetched grids; empty = in-memory only

    # Multi-worker tile-server split.  When render_only is True, this
    # process skips fetcher / NWP grid / satellite / nowcast initialisation
    # and instead memory-maps an existing snapshot under cache_dir
    # written by ``python -m librewxr.data_pipeline``.  cache_dir is
    # required in render-only mode.
    render_only: bool = False
    # Seconds between state.json mtime polls in render-only mode.  The
    # file is rewritten once per fetch_interval (default 600 s) so a 1 s
    # poll is responsive without burning CPU.
    state_poll_interval: float = 1.0
    # Seconds to wait for the data pipeline to write its first state.json
    # before failing loudly.  0 = wait forever.
    state_wait_timeout: float = 300.0

    # WMO CAP Weather Alerts
    alerts_enabled: bool = True
    alerts_fetch_interval: int = 300  # 5 minutes, aligned to clock boundaries
    alerts_cache_dir: str = ""  # Cache dir for meteoalarm data; empty = system temp
    alerts_concurrency: int = 5  # Max concurrent WMO HTTP connections

    # Tile request tracking — observational only; surfaces hot tiles in /health
    # for diagnostic use.  Originally instrumentation for a planned adaptive
    # warming pass, but multi-worker mode made cold-render stalls imperceptible
    # in practice, so the warming policy itself isn't shipping.
    tile_tracking_enabled: bool = True
    tile_tracking_min_zoom: int = 7  # Track only z >= this (overview zooms are pre-warmed anyway)
    tile_tracking_max_entries: int = 10_000  # Cap per-tile counters; halves when full
    download_retries: int = 1  # Retries on transient download errors (0 = no retry, 1 = one retry)
    # Maximum number of NWP auxiliary-grid fetches running in parallel
    # inside one fetch cycle.  Each grid loads tens-to-hundreds of MB
    # during decode, so the cap bounds peak RAM at ~N × per-grid working
    # set.  4 fits comfortably in 8 GB; bump higher for fatter rigs to
    # bring fetch-cycle wall time closer to the slowest single source.
    nwp_fetch_concurrency: int = 4
    cors_origins: list[str] = ["*"]

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v):
        """Parse COMPOSE_PROFILES-style comma lists down to a single mode token.

        COMPOSE_PROFILES is comma-separated (e.g. ``"multi,manual"``); the
        ``manual`` profile is used for one-off services like clear-cache,
        so we only care whether ``multi`` is in the list.  Anything else
        falls back to ``single`` rather than failing Literal validation —
        unrelated values in COMPOSE_PROFILES shouldn't crash startup.
        """
        if not isinstance(v, str):
            return v
        tokens = {t.strip() for t in v.split(",") if t.strip()}
        if "multi" in tokens:
            return "multi"
        if "single" in tokens:
            return "single"
        return "single"

    @model_validator(mode="after")
    def _apply_mode_defaults(self):
        """Fill in mode-appropriate defaults for any setting left at 0."""
        defaults = _MODE_DEFAULTS[self.mode]
        for name, value in defaults.items():
            if getattr(self, name) == 0:
                setattr(self, name, value)
        return self

    def get_ecmwf_max_timesteps(self) -> int:
        """Return effective ECMWF timestep count.

        If ecmwf_max_timesteps > 0, use it as-is (user override).
        Otherwise auto-derive from max_frames + nowcast_frames, with two
        extra hourly buckets so the window still covers both ends of the
        radar/nowcast span when ``now`` falls between hourly IFS marks
        (worst case can drift the window up to ~1h on each side).
        """
        if self.ecmwf_max_timesteps > 0:
            return self.ecmwf_max_timesteps
        past_hours = math.ceil(self.max_frames * self.fetch_interval / 3600)
        future_hours = (
            math.ceil(self.nowcast_frames * self.fetch_interval / 3600)
            if self.nowcast_enabled else 0
        )
        return past_hours + future_hours + 2

    def get_enabled_regions(self) -> list[str]:
        """Resolve the region spec into individual region names."""
        from librewxr.data.regions import resolve_regions
        return resolve_regions(self.enabled_regions)


settings = Settings()
