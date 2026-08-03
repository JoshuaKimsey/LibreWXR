# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NWPSource Protocol and NWPChain dispatcher for multi-model NWP fallback.

Phase 1 of the multi-model NWP integration: defines the contract that any
numerical-weather-prediction source (ECMWF IFS, NOAA HRRR, DWD ICON-D2, ...)
must satisfy, plus a chain dispatcher that walks sources in priority order
and fills pixels from the first source with both coverage and data.

Each source handles its own quirks internally — Z-R conversion, projection
sampling, fetch cadence — so the renderer talks to a single uniform interface.
"""

from typing import Protocol, runtime_checkable

import numpy as np

# NOTE: ``precip_bbox`` is deliberately NOT a member of the runtime-checkable
# ``NWPSource`` Protocol: ``@runtime_checkable`` isinstance() requires every
# protocol member to exist on the concrete class, which would fail for the
# regional sources that intentionally don't implement it (spec: no regional
# source edits).  ``NWPChain.has_precip_in_bbox`` probes it via ``getattr``
# instead, so an absent method is equivalent to "conservative always has
# precip".  Optional methods are documented in the chain instead.


@runtime_checkable
class NWPSource(Protocol):
    """A numerical weather prediction data source."""

    name: str

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Return uint8 dBZ-encoded precipitation at each (lat, lon) point.

        Encoding matches the radar pipeline: pixel = (dBZ + 32) * 2.
        Output shape == lat.shape.
        """
        ...

    @property
    def supports_snow(self) -> bool:
        """Whether this source can classify precipitation as rain vs. snow.

        Sources that lack a snow-ratio field (e.g. HRRR, DMI DINI, ICON-EU)
        return ``False`` so the chain dispatcher skips their expensive
        ``domain_mask`` and falls through to a source that can (IFS).
        """
        ...

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Return bool mask: True where precipitation is snow. Shape == lat.shape."""
        ...

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return bool mask: True where this source has coverage. Shape == lat.shape."""
        ...

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return float32 mask in [0, 1] for soft chain blending.

        Values close to 1.0 mean "trust this source fully here"; values
        close to 0.0 hand control to the next source in the chain.
        Sources with a hard boundary (e.g., the global IFS fallback)
        return ``domain_mask(lat, lon).astype(float32)``.  Sources with
        a finite domain (HRRR, future regional NWP) return a smooth
        taper to 0 at the boundary so chain blending produces a
        continuous transition instead of a visible seam.
        """
        ...

    def has_data_at(self, timestamp: int) -> bool:
        """Whether this source can answer for the given valid time right now."""
        ...

    def has_data(self) -> bool:
        """Whether this source has any data loaded at all."""
        ...


class NWPChain:
    """Dispatches sample / snow_mask queries across NWP sources in priority order.

    ``sample`` does a soft, weight-accumulating blend across sources.
    Each source contributes ``remaining_weight × its_feather`` of its
    sampled values, with ``remaining`` decreasing as preceding sources
    fill up.  When a source's feather is binary (1 inside / 0 outside,
    e.g. the global IFS fallback) the blend collapses to a hard fill —
    so a chain of binary-feather sources behaves identically to a
    first-fill dispatcher.  When a source's feather tapers smoothly
    near its boundary (e.g. HRRR's LCC edge), the chain produces a
    continuous transition into the next source instead of a visible
    seam.

    ``get_snow_mask`` stays a hard first-fill: blending booleans is
    meaningless and the snow flag is per-pixel categorical.
    """

    def __init__(self, sources: list[NWPSource]):
        self._sources = list(sources)
        self._domain_bboxes: list[tuple[float, float, float, float] | None] = []
        for src in self._sources:
            self._domain_bboxes.append(self._probe_domain_bbox(src))

    @staticmethod
    def _probe_domain_bbox(src) -> tuple[float, float, float, float] | None:
        """Probe src.domain_mask on a 1-degree global grid once at init.

        domain_mask is projection-only for every concrete NWP source (verified: reads
        only grid constants, never _timesteps), so this is safe on a freshly-constructed
        grid without timestep data loaded.
        """
        lat = np.linspace(90.0, -90.0, 181, dtype=np.float32)
        lon = np.linspace(-180.0, 180.0, 361, dtype=np.float32)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        try:
            mask = src.domain_mask(lat_grid, lon_grid)
        except Exception:
            return None
        if not mask.any():
            return None
        lat_any = mask.any(axis=1)
        lon_any = mask.any(axis=0)
        min_lat = float(lat[lat_any].min()) - 1.0
        max_lat = float(lat[lat_any].max()) + 1.0
        min_lon = float(lon[lon_any].min()) - 1.0
        max_lon = float(lon[lon_any].max()) + 1.0
        min_lat = max(min_lat, -90.0); max_lat = min(max_lat, 90.0)
        min_lon = max(min_lon, -180.0); max_lon = min(max_lon, 180.0)
        return (min_lon, min_lat, max_lon, max_lat)

    @staticmethod
    def _bbox_intersects(a: tuple[float, float, float, float],
                         b: tuple[float, float, float, float]) -> bool:
        """AABB intersection. Both bboxes are (west, south, east, north), non-wrapping."""
        return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

    def has_precip_in_bbox(self, timestamp: int,
                           tile_bbox: tuple[float, float, float, float]) -> bool:
        """Return whether any NWP source may have precip in ``tile_bbox``.

        Conservative: returns True (assume precip) when a source is unsupported,
        when its precip_bbox is missing/antimeridian-spanning, or on any error.
        """
        for src, dom_bbox in zip(self._sources, self._domain_bboxes):
            if dom_bbox is None or not self._bbox_intersects(dom_bbox, tile_bbox):
                continue  # source doesn't cover this tile
            bbox_fn = getattr(src, "precip_bbox", None)
            if bbox_fn is None:
                return True  # unsupported source — assume has precip
            try:
                bbox = bbox_fn(timestamp)
            except Exception:
                return True  # error — conservative
            if bbox is None:
                return True  # missing/antimeridian-spanning — assume has precip
            if self._bbox_intersects(bbox, tile_bbox):
                return True
        return False

    @property
    def sources(self) -> list[NWPSource]:
        return list(self._sources)

    def has_data(self) -> bool:
        """True if any registered source has data loaded."""
        return any(src.has_data() for src in self._sources)

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=np.float32)
        remaining = np.ones(lat.shape, dtype=np.float32)
        for src in self._sources:
            if not (remaining > 0.0).any():
                break
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            feather = src.feather_mask(lat, lon).astype(np.float32, copy=False)
            weight = remaining * feather
            relevant = weight > 0.0
            if not relevant.any():
                continue
            sub_lat = lat[relevant]
            sub_lon = lon[relevant]
            sample_vals = src.sample(sub_lat, sub_lon, timestamp, bilinear)
            contribution = np.zeros(lat.shape, dtype=np.float32)
            contribution[relevant] = sample_vals.astype(np.float32, copy=False)
            out += weight * contribution
            remaining *= 1.0 - feather
        # NaN values from out-of-domain LCC projections (HRRR, DINI,
        # WRF-SMN) can flow through the feather-weighted blend into
        # ``out``.  clip + astype on NaN produces a RuntimeWarning;
        # the resulting 0 values are filtered by domain_mask downstream.
        with np.errstate(invalid="ignore"):
            return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=bool)
        unfilled = np.ones(lat.shape, dtype=bool)
        for src in self._sources:
            if not unfilled.any():
                break
            if not src.supports_snow:
                continue
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            domain = src.domain_mask(lat, lon)
            mask = unfilled & domain
            if not mask.any():
                continue
            sub_lat = lat[mask]
            sub_lon = lon[mask]
            out[mask] = src.get_snow_mask(sub_lat, sub_lon, timestamp)
            unfilled &= ~domain
        return out
