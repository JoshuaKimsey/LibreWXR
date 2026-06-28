# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Geostationary satellite fixed-grid ↔ lat/lon projection.

Vectorized numpy implementation of the GOES-R Product Users' Guide
(PUG) Vol 5 §4.2.8 forward/inverse transforms.  The same math applies
to any geostationary imager that publishes on a fixed-grid (GOES ABI,
Himawari AHI, GEO-KOMPSAT-2A AMI) — the satellite-specific parameters
(sub-satellite longitude, orbital height, ellipsoid axes) are passed
in rather than hardcoded.
"""
from __future__ import annotations

import numpy as np

# WGS84 ellipsoid constants — shared across all geostationary satellites.
R_EQ = 6378137.0  # semi-major axis (m)
R_POL = 6356752.31414  # semi-minor axis (m)
_E2 = 1.0 - (R_POL / R_EQ) ** 2  # first eccentricity squared
_RPOL2_REQ2 = (R_POL / R_EQ) ** 2  # (r_pol / r_eq)²


def forward(
    lat: np.ndarray,
    lon: np.ndarray,
    sat_lon: float,
    sat_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert geodetic lat/lon to geostationary scan angles (x, y).

    Parameters
    ----------
    lat, lon : ndarray (float64)
        Geodetic latitude and longitude in **degrees**.
    sat_lon : float
        Sub-satellite longitude in degrees (e.g. -137.0 for GOES-18).
    sat_height : float
        Satellite height above the ellipsoid in metres
        (e.g. 35786023.0 for GOES-R series).

    Returns
    -------
    x, y : ndarray (float64)
        East/west and north/south scan angles in **radians**.
        Points on the far side of the earth (not visible to the
        satellite) are set to NaN.
    """
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon) - np.deg2rad(sat_lon)

    # Geocentric latitude
    phi_c = np.arctan(_RPOL2_REQ2 * np.tan(phi))

    cos_phi_c = np.cos(phi_c)
    sin_phi_c = np.sin(phi_c)
    cos_lam = np.cos(lam)

    # Geocentric distance from earth centre to the surface point
    r_c = R_POL / np.sqrt(1.0 - _E2 * cos_phi_c ** 2)

    H = sat_height + R_EQ  # distance from earth centre to satellite

    sx = H - r_c * cos_phi_c * cos_lam
    sy = -r_c * cos_phi_c * np.sin(lam)
    sz = r_c * sin_phi_c

    # Visibility check per GOES-R PUG: a surface point is visible from
    # the satellite when the line from the satellite to the point does
    # not pass through the earth.  Equivalent condition (from the PUG
    # §4.2.8.1): H·(H - sx) >= sy² + (r_eq/r_pol)² · sz²
    visible = H * (H - sx) >= sy ** 2 + (R_EQ / R_POL) ** 2 * sz ** 2

    r_s = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)

    x = np.where(visible, np.arcsin(-sy / r_s), np.nan)
    y = np.where(visible, np.arctan2(sz, sx), np.nan)

    return x, y


def inverse(
    x: np.ndarray,
    y: np.ndarray,
    sat_lon: float,
    sat_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert geostationary scan angles (x, y) to geodetic lat/lon.

    Parameters
    ----------
    x, y : ndarray (float64)
        East/west and north/south scan angles in **radians**.
    sat_lon : float
        Sub-satellite longitude in degrees.
    sat_height : float
        Satellite height above the ellipsoid in metres.

    Returns
    -------
    lat, lon : ndarray (float64)
        Geodetic latitude and longitude in **degrees**.
        Points outside the earth disk are NaN.
    """
    H = sat_height + R_EQ

    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)

    # Quadratic to find the distance along the line-of-sight to the
    # earth's surface: a·r² + b·r + c = 0 where r is the parameter.
    a = sin_x ** 2 + cos_x ** 2 * (
        cos_y ** 2 + (R_EQ / R_POL) ** 2 * sin_y ** 2
    )
    b = -2.0 * H * cos_x * cos_y
    c = H ** 2 - R_EQ ** 2

    discriminant = b ** 2 - 4.0 * a * c
    on_disk = discriminant >= 0.0

    # Pick the nearer root (smaller r).
    sqrt_disc = np.sqrt(np.where(on_disk, discriminant, 0.0))
    r_s = np.where(on_disk, (-b - sqrt_disc) / (2.0 * a), np.nan)

    # Cartesian position on the ellipsoid surface
    sx = r_s * cos_x * cos_y
    sy = -r_s * sin_x
    sz = r_s * cos_x * sin_y

    # Geodetic latitude (accounts for the ellipsoidal flattening)
    lat = np.rad2deg(
        np.arctan((R_EQ / R_POL) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2))
    )
    lon = np.rad2deg(np.deg2rad(sat_lon) - np.arctan2(sy, H - sx))

    lat = np.where(on_disk, lat, np.nan)
    lon = np.where(on_disk, lon, np.nan)

    return lat, lon
