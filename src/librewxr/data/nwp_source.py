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
            if not (remaining > 0.0).any():
                break
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
            if not unfilled.any():
                break
        return out
