# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Antilles regional precipitation source.

Implements the NWPSource Protocol for the Caribbean overseas variant of
the AROME convection-permitting model — 1.3 km native resolution, 0.025°
(~2.5 km) regular lat/lon public distribution grid, covering Guadeloupe,
Martinique, Saint Martin, Saint-Barthélemy and the surrounding waters of
the eastern Caribbean.

Four daily cycles (00/06/12/18 UTC), 48 h forecast horizon, hourly
forecast steps.  AROME's surface output has no native composite
reflectivity field, so we derive dBZ from accumulated ``tp`` (Total
Precipitation, kg/m² ≡ mm) by differencing consecutive forecast steps
and applying the same Marshall-Palmer Z-R conversion ECMWFGrid /
ICONEUGrid / DMIDiniGrid use.

Distribution: anonymous data.gouv.fr open-data S3-style object storage
(``object.data.gouv.fr/meteofrance-pnt/``), no auth, plain HTTPS.  Each
(run, leadtime) is a single-message GRIB2 file ≈ 2-4 MB; we fetch the
whole file per leadtime — small enough that a byte-range header walk
isn't worth the round-trip overhead.

Projection: regular lat/lon (gridType=regular_ll), so there's no
projection math — fractional row/col come straight from the lat/lon
deltas.  Scan mode 0 (i+, j-) means cfgrib returns the file with row 0
at the NORTHERN edge already, so no flip is needed (different from
HRRR / DMI DINI / HRDPS, which all use scan mode 64).

Data attribution: Météo-France, Etalab Open Licence v2.0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)


# ── AROME Antilles regular lat/lon grid parameters ─────────────────────
#
# Source: GRIB Section 3 of a representative
# arome-om-ANTIL__0025__SP1__006H file decoded on 2026-05-08.
# All four corners back-decode to within float precision of cfgrib's
# reported lat/lon arrays.

AROME_ANT_LAT_NORTH = 22.9               # row 0 (top, scan mode 0)
AROME_ANT_LAT_SOUTH = 9.7                # row Nj-1 (bottom)
AROME_ANT_LON_WEST_DEG_E = 284.7         # col 0 (longitude in [0, 360) east-positive)
AROME_ANT_LON_EAST_DEG_E = 308.3         # col Ni-1
AROME_ANT_GRID_DLAT = 0.025              # row spacing (deg)
AROME_ANT_GRID_DLON = 0.025              # col spacing (deg)
AROME_ANT_GRID_WIDTH = 945               # Ni — points along parallel
AROME_ANT_GRID_HEIGHT = 529              # Nj — points along meridian


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the AROME-Antilles grid.

    Row 0 is the NORTHERN edge (lat = 22.9°N), row HEIGHT-1 the southern
    edge.  Column 0 is the western edge (lon = -75.3°E ≡ 284.7°E),
    column WIDTH-1 the eastern edge (lon = -51.7°E ≡ 308.3°E).  Inputs
    in standard geographic conventions: lat in [-90, 90], lon in
    [-180, 180]; we wrap lon onto the bucket's [0, 360) convention via a
    simple modulo so query points work in either form.

    Out-of-domain points still return values; callers should test
    ``domain_mask`` first.
    """
    # Fold lon onto [0, 360) and unwrap so column math is monotonic
    # across the western/Caribbean longitudes regardless of input form.
    lon_e = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    row = (AROME_ANT_LAT_NORTH - np.asarray(lat, dtype=np.float64)) / AROME_ANT_GRID_DLAT
    col = (lon_e - AROME_ANT_LON_WEST_DEG_E) / AROME_ANT_GRID_DLON
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the AROME-Antilles grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < AROME_ANT_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < AROME_ANT_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# Width of the soft transition zone at the AROME Antilles domain edge in
# grid cells.  Inside the inner region (≥ FEATHER_DISTANCE_CELLS from
# any edge) AROME is trusted at full weight; over the feather zone the
# weight tapers linearly to 0 at the edge so chain blending hands
# control to IFS smoothly.  The Antilles domain is small (~2600 km
# east-west, ~1500 km north-south), so we feather a tighter ~50 km
# (20 cells × 0.025° × ~110 km/° ≈ 55 km) instead of HRRR/DMI's 75 km.

AROME_ANT_FEATHER_DISTANCE_CELLS = 20


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Float32 weights in [0, 1]: 1 deep inside AROME Antilles, 0 outside."""
    row, col = grid_indices(lat, lon)
    dist_cells = np.minimum(
        np.minimum(row, (AROME_ANT_GRID_HEIGHT - 1) - row),
        np.minimum(col, (AROME_ANT_GRID_WIDTH - 1) - col),
    )
    weight = np.clip(
        dist_cells / float(AROME_ANT_FEATHER_DISTANCE_CELLS), 0.0, 1.0,
    )
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches ECMWFGrid / ICONEUGrid / DMIDiniGrid / HRDPS) ─

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    Same Marshall-Palmer rain Z-R as DMI DINI: Z = 200 * R^1.6.  The
    optional ``dbz_offset`` shifts the result uniformly to compensate
    for the model-vs-radar intensity bias (radar samples the brightest
    part of the storm column while the model gives surface rate).
    """
    rate = np.where(np.isfinite(precip_mm_per_hour), precip_mm_per_hour, 0.0)
    rate = np.maximum(rate, 0.0)
    eps = 1e-6
    z = ZR_A_RAIN * np.power(rate + eps, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, eps)) + dbz_offset
    encoded = np.clip((dbz + 32.0) * 2.0 + 0.5, 0, 255)
    encoded[rate <= 0.0] = 0
    return encoded.astype(np.uint8)


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 6 * 3600        # AROME-OM deterministic runs every 6 h
BRACKET_INTERVAL_SECONDS = 3600          # forecast steps are 1 hour apart
MAX_FORECAST_HOURS = 48                  # all runs reach +48 h

# Two cycles of lookback (12 h) is plenty: each run covers +48 h, far
# more than any reasonable active history+horizon window.  Same shape
# as HRDPS — every AROME-OM cycle reaches full horizon, no ICON-EU-style
# intermediate-run truncation.
RUN_LOOKBACK_CYCLES = 2


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 6-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(lead_seconds: int) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` such that L0 ≤ L < L1.

    Both leads are exact hour multiples (multiples of 3600 s, ≥ 0).
    Alpha is the lerp weight: 0 at L0, 1 at L1.
    """
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // BRACKET_INTERVAL_SECONDS) * BRACKET_INTERVAL_SECONDS
    l1 = l0 + BRACKET_INTERVAL_SECONDS
    alpha = (lead_seconds - l0) / BRACKET_INTERVAL_SECONDS
    return l0, l1, alpha


# ── data.gouv.fr file URLs ────────────────────────────────────────────
#
# Filename scheme on object.data.gouv.fr/meteofrance-pnt/:
#   pnt/{run-iso}/arome-om/ANTIL/0025/SP1/
#   arome-om-ANTIL__0025__SP1__{LLL}H__{run-iso}.grib2
# where {run-iso} is e.g. "2026-05-08T00:00:00Z" (ISO 8601 with literal
# colons — the bucket URL-encodes them on listing but accepts them
# literally on GET) and {LLL} is the zero-padded 3-digit lead hour
# (000..048).
#
# We fetch SP1 only — that's the "surface package 1" that contains
# accumulated total precipitation and 2 m temperature.  SP2/SP3 carry
# radiation/clouds/winds we don't need; HP1-3 and IP1-5 are upper-air.

_AROME_ANT_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _format_run_ts(dt: datetime) -> str:
    """Format a datetime as Météo-France's ISO 8601 with colons."""
    return dt.strftime(_AROME_ANT_TS_FMT)


def file_url(run: datetime, step_hour: int) -> str:
    """Construct the data.gouv.fr URL for one AROME Antilles SP1 file.

    ``step_hour`` is the forecast hour 0..48; step 0 is the analysis
    (no precipitation accumulated since init = 0).
    """
    base = settings.arome_antilles_base_url.rstrip("/")
    run_str = _format_run_ts(run)
    lll = f"{step_hour:03d}"
    return (
        f"{base}/pnt/{run_str}/arome-om/ANTIL/0025/SP1/"
        f"arome-om-ANTIL__0025__SP1__{lll}H__{run_str}.grib2"
    )


# ── GRIB2 message decode ──────────────────────────────────────────────


def _suppress_eccodes_stderr():
    from librewxr.data.sources import _suppress_eccodes_stderr as _s
    return _s()


def decode_tp_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode the ``tp`` GRIB2 message into a 2D float32 array.

    Returns ``None`` on parse failure.  Output shape is
    ``(AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH)`` with row 0 at the
    NORTHERN edge.  Scan mode 0 in the source GRIB already gives row 0
    at the north, so no flip is normally needed; we still verify against
    the ``latitude`` coordinate when present so the code self-corrects
    if a future cfgrib release ever permutes the orientation.
    """
    import xarray as xr

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(grib_bytes)
            tmp_path = tmp.name
        with _suppress_eccodes_stderr():
            ds = xr.open_dataset(
                tmp_path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"shortName": "tp"},
                },
            )
        ds = ds.compute()
    except Exception:
        logger.exception("Failed to decode AROME Antilles tp GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    if "tp" in ds.data_vars:
        arr = ds["tp"].values
    else:
        for name, da in ds.data_vars.items():
            if (
                da.ndim == 2
                and da.shape == (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH)
            ):
                logger.warning(
                    "AROME Antilles tp variable not named 'tp' (got %r); "
                    "using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning("AROME Antilles GRIB had no recognised tp field")
            return None

    if arr.shape != (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH):
        logger.warning(
            "AROME Antilles tp has unexpected shape %s (expected %s); skipping",
            arr.shape, (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH),
        )
        return None

    # Self-correcting orientation: row 0 should be the NORTHERN edge.
    # cfgrib historically returns AROME with row 0 at the north already
    # (scan mode 0); flip only if the latitude coord is increasing with
    # row index.
    if "latitude" in ds.coords:
        lat_arr = np.asarray(ds["latitude"].values)
        if lat_arr.ndim == 1 and lat_arr.size > 1:
            needs_flip = lat_arr[0] < lat_arr[-1]
        elif lat_arr.ndim == 2 and lat_arr.shape[0] > 1:
            needs_flip = lat_arr[0, 0] < lat_arr[-1, 0]
        else:
            needs_flip = False
    else:
        needs_flip = False
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


# ── AROMEAntillesGrid: the public NWPSource implementation ────────────


class AROMEAntillesGrid:
    """Météo-France AROME Antilles as an NWPSource for the Caribbean slot.

    Implements the NWPSource Protocol.  Frames are stored at native
    0.025° (~2.5 km) regular lat/lon resolution as uint8 dBZ-encoded
    arrays keyed by ``(run_unix_ts, lead_seconds)``.  Sampling at a query
    (lat, lon, ts) does:

    1. Pick the freshest run whose forecast covers ``ts`` and has both
       bracket frames loaded.
    2. Lerp between the two bracketing 1-hour frames in time.
    3. Index the regular lat/lon grid at the query (lat, lon) and sample.
    """

    name = "arome_antilles"

    def __init__(self, cache_dir: Path | None = None):
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        # Raw accumulated tp values keyed by (run_ts, step_hour); kept
        # only long enough to compute the rate at the next step.
        self._accum: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "arome_antilles"
            self._persistent = True
        else:
            self._memmap_dir = Path(
                tempfile.mkdtemp(prefix="librewxr_arome_antilles_")
            )
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "AROME Antilles memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )
        if self._persistent:
            self._load_cached_frames()

    # ── Cache management ──────────────────────────────────────────────

    def _frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Atomic-write ``data`` and return a read-only memmap view."""
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    def _load_cached_frames(self) -> None:
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        loaded = 0
        pat = re.compile(r"^r(\d+)_l(\d+)$")
        for path in self._memmap_dir.glob("r*_l*.dat"):
            m = pat.match(path.stem)
            if m is None:
                continue
            run_ts = int(m.group(1))
            lead_s = int(m.group(2))
            try:
                mm = np.memmap(
                    path, dtype=np.uint8, mode="r",
                    shape=(AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH),
                )
            except Exception:
                logger.warning("Failed to memmap cached %s, removing", path)
                path.unlink(missing_ok=True)
                continue
            self._frames[(run_ts, lead_s)] = mm
            if self._latest_run_ts is None or run_ts > self._latest_run_ts:
                self._latest_run_ts = run_ts
            loaded += 1
        if loaded:
            logger.info(
                "AROME Antilles: loaded %d cached frame(s) from disk", loaded,
            )

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode).

        The on-disk layout in ``_memmap_dir`` is the canonical state;
        ``__setstate__`` rebuilds the in-memory frame dict by rescanning
        disk via ``_load_cached_frames``.
        """
        return {
            "memmap_dir": str(self._memmap_dir),
            "latest_run_ts": self._latest_run_ts,
            "frame_keys": [[run, lead] for (run, lead) in self._frames.keys()],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state by rescanning ``memmap_dir`` from disk."""
        self._memmap_dir = Path(state["memmap_dir"])
        self._persistent = True
        self._client = None
        self._fetch_lock = asyncio.Lock()
        self._frames = {}
        self._latest_run_ts = None
        self._load_cached_frames()

    @property
    def data_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._frames.values())

    @property
    def latest_run_iso(self) -> str | None:
        if self._latest_run_ts is None:
            return None
        return datetime.fromtimestamp(
            self._latest_run_ts, tz=timezone.utc,
        ).isoformat()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    # ── NWPSource Protocol ────────────────────────────────────────────

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return domain_mask(lat, lon)

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return feather_mask(lat, lon)

    def has_data(self) -> bool:
        return bool(self._frames)

    def has_data_at(self, timestamp: int) -> bool:
        run = self._pick_run(timestamp)
        if run is None:
            return False
        lead = timestamp - run
        l0, l1, _ = bracket_lead_seconds(lead)
        return ((run, l0) in self._frames) and ((run, l1) in self._frames)

    @property
    def supports_snow(self) -> bool:
        return False

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        return np.zeros(lat.shape, dtype=bool)

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        if timestamp is None or not self._frames:
            return np.zeros(lat.shape, dtype=np.uint8)
        run = self._pick_run(timestamp)
        if run is None:
            return np.zeros(lat.shape, dtype=np.uint8)
        lead = timestamp - run
        l0, l1, alpha = bracket_lead_seconds(lead)
        f0 = self._frames.get((run, l0))
        f1 = self._frames.get((run, l1))
        if f0 is None or f1 is None:
            return np.zeros(lat.shape, dtype=np.uint8)
        if alpha == 0.0:
            grid = f0
        elif alpha == 1.0:
            grid = f1
        else:
            grid = (
                (1.0 - alpha) * f0.astype(np.float32)
                + alpha * f1.astype(np.float32)
                + 0.5
            ).astype(np.uint8)
        return _sample_grid(grid, lat, lon, bilinear=bilinear)

    # ── Run selection ─────────────────────────────────────────────────

    def _pick_run(self, timestamp: int) -> int | None:
        """Pick the freshest run whose bracket is loaded for ``timestamp``."""
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (0 <= lead <= MAX_FORECAST_HOURS * 3600):
                continue
            l0, l1, _ = bracket_lead_seconds(lead)
            if (run, l0) in self._frames and (run, l1) in self._frames:
                return run
        return None

    # ── Fetch loop ────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=4, max_connections=8,
                ),
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        """Refresh the in-memory window — same shape as DMIDiniGrid.fetch.

        Walks back through 6-hourly Météo-France cycles to cover the
        active history window; each run's forecast hours that overlap
        the window are downloaded individually (one HTTP GET per
        (run, hour); files are tiny single-message GRIBs ≤ 5 MB).
        """
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.arome_antilles_publish_delay_minutes * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if (
                self._latest_run_ts is None
                or latest_run_ts > self._latest_run_ts
            ):
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            earliest_run = max(
                floor_cycle(window_start - CYCLE_INTERVAL_SECONDS),
                latest_run_ts - RUN_LOOKBACK_CYCLES * CYCLE_INTERVAL_SECONDS,
            )
            runs_to_consider = list(range(
                earliest_run, latest_run_ts + 1, CYCLE_INTERVAL_SECONDS,
            ))
            if not runs_to_consider:
                logger.debug("AROME Antilles fetch: no runs available for window")
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                min_lead = max(
                    0, window_start - run_ts - BRACKET_INTERVAL_SECONDS,
                )
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                # Need step F-1 to compute the rate at step F via diff,
                # so always start one step earlier than strictly needed.
                # Step 0 has no precip data of value (cumulative-since-init
                # at init = 0); the diff path treats it as a zero baseline.
                min_step = max(0, (min_lead // BRACKET_INTERVAL_SECONDS) - 1)
                max_step = min(
                    MAX_FORECAST_HOURS,
                    -(-max_lead // BRACKET_INTERVAL_SECONDS),
                )
                for step in range(int(min_step), int(max_step) + 1):
                    added = await self._fetch_one_step(run_dt, step, client)
                    if added > 0:
                        total_fetched += added
                    elif added < 0:
                        total_failed += 1

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "AROME Antilles: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched, len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "AROME Antilles: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_step(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        """Fetch one step's tp file, diff against prior step, encode, and store.

        Returns 1 on success, 0 if already loaded, -1 on fetch error.
        Step 0 caches a zero baseline so step 1 can diff against it
        cleanly; no actual file fetch occurs for step 0.
        """
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS

        # Step 0: cumulative precip is zero everywhere at model init.
        if step_hour == 0:
            if (run_ts, 0) not in self._accum:
                self._accum[(run_ts, 0)] = np.zeros(
                    (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH),
                    dtype=np.float32,
                )
            return 0

        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = file_url(run, step_hour)
        from librewxr.data.retry import retry_get
        resp = await retry_get(client, url, log_name="AROME Antilles data")
        if resp is None:
            return -1
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 404 is expected when polling at the leading edge of a run
            # that hasn't fully published yet — log at debug, not warning.
            if getattr(e.response, "status_code", None) == 404:
                logger.debug("AROME Antilles not yet published for %s", url)
            else:
                logger.warning("AROME Antilles fetch failed for %s: %s", url, e)
            return -1
        grib_bytes = resp.content

        accum = decode_tp_message(grib_bytes)
        if accum is None:
            return -1

        # Ensure we have step F-1 cached for the diff.
        prev_key = (run_ts, step_hour - 1)
        prev = self._accum.get(prev_key)
        if prev is None and step_hour - 1 >= 0:
            await self._fetch_one_step(run, step_hour - 1, client)
            prev = self._accum.get(prev_key)
        if prev is None:
            return -1

        rate_mm_per_hour = accum - prev
        encoded = precip_rate_to_dbz_encoded(
            rate_mm_per_hour,
            dbz_offset=settings.arome_antilles_dbz_offset,
        )
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm
        self._accum[(run_ts, step_hour)] = accum

        return 1

    # ── Eviction ──────────────────────────────────────────────────────

    def _evict_outside_window(
        self, window_start: int, window_end: int,
    ) -> None:
        slack = BRACKET_INTERVAL_SECONDS
        ws = window_start - slack
        we = window_end + slack
        stale_frames = []
        for key in self._frames:
            run_ts, lead = key
            valid_time = run_ts + lead
            if valid_time < ws or valid_time > we:
                stale_frames.append(key)
        for key in stale_frames:
            self._frames.pop(key, None)
            try:
                self._frame_path(*key).unlink(missing_ok=True)
            except OSError:
                pass
        stale_accums = []
        for (run_ts, step_h) in self._accum:
            valid_time = run_ts + step_h * BRACKET_INTERVAL_SECONDS
            if valid_time < ws - BRACKET_INTERVAL_SECONDS or valid_time > we:
                stale_accums.append((run_ts, step_h))
        for k in stale_accums:
            self._accum.pop(k, None)
        if stale_frames:
            logger.info(
                "AROME Antilles: evicted %d out-of-window frame(s)",
                len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._accum.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("AROME Antilles memmap directory cleaned up")
        else:
            logger.info(
                "AROME Antilles cache retained at %s for warm restart",
                self._memmap_dir,
            )


# ── Grid sampling ────────────────────────────────────────────────────


def _sample_grid(
    grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bilinear: bool = False,
) -> np.ndarray:
    """Sample a uint8 regular lat/lon grid at (lat, lon) points."""
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < AROME_ANT_GRID_HEIGHT)
            & (col >= 0)
            & (col < AROME_ANT_GRID_WIDTH)
        )
        out = np.zeros(lat.shape, dtype=np.uint8)
        if in_domain.any():
            out[in_domain] = grid[row[in_domain], col[in_domain]]
        return out

    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1
    in_domain = (
        (r0 >= 0)
        & (r1 < AROME_ANT_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < AROME_ANT_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, AROME_ANT_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, AROME_ANT_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, AROME_ANT_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, AROME_ANT_GRID_WIDTH - 1)
    dr = np.clip(row_f - r0, 0.0, 1.0).astype(np.float32)
    dc = np.clip(col_f - c0, 0.0, 1.0).astype(np.float32)
    v00 = grid[r0c, c0c].astype(np.float32)
    v01 = grid[r0c, c1c].astype(np.float32)
    v10 = grid[r1c, c0c].astype(np.float32)
    v11 = grid[r1c, c1c].astype(np.float32)
    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
    interp = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    sampled = np.where(any_zero, v00, interp)
    out = np.clip(sampled + 0.5, 0, 255).astype(np.uint8)
    out[~in_domain] = 0
    return out
