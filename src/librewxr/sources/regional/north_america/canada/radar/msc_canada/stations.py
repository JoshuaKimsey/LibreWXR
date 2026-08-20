# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""ECCC Canadian weather radar network.

33 S-band dual-pol stations (METEOR 1700S) operated by Environment and
Climate Change Canada after the 2023 network modernization.  Coordinates
from the Canadian Weather Radar Network Wikipedia article,
cross-referenced with ECCC station pages.

Needed because MSC's RADAR_1KM_RRAI composite covers the whole CACOMP
bbox but only has actual data within ~240 km of these stations —
without a mask, the ECMWF fallback is suppressed over huge empty
regions (open Pacific, Arctic, Atlantic).
"""
from __future__ import annotations


STATIONS: list[tuple[float, float]] = [
    (49.01662, -122.48698),    # CASAG Aldergrove, BC
    (50.57118, -105.18290),    # CASBE Bethune, SK
    (45.70634, -73.85852),     # CASBV Blainville, QC
    (45.79317, -80.53385),     # CASBI Britt, ON
    (53.56056, -114.14495),    # CASCV Carvel, AB
    (46.22232, -65.69924),     # CASCM Chipman, NB
    (54.3785, -110.061378),    # CASCL Cold Lake, AB
    (49.85823, -92.79698),     # CASDR Dryden, ON
    (44.2305662, -79.78033),   # CASTS Egbert, ON
    (43.37243, -81.38070),     # CASET Exeter, ON
    (56.375642, -111.215177),  # CASFM Fort McMurray, AB
    (50.54887, -101.08570),    # CASFW Foxwarren, MB
    (45.04101, -76.11617),     # CASFT Franktown, ON
    (45.09850, -63.70433),     # CASGO Gore, NS
    (49.527017, -123.853583),  # CASHP Halfmoon Peak, BC
    (47.32644, -53.12658),     # CASHR Holyrood, NL
    (43.96393, -79.57388),     # CASKR King City, ON
    (48.55136, -77.80809),     # CASLA Landrienne, QC
    (48.93028, -57.83417),     # CASMM Marble Mountain, NL
    (45.94972, -60.20521),     # CASMB Marion Bridge, NS
    (47.977908, -71.430833),   # CASMA Mont Apica, QC
    (47.24773, -84.59652),     # CASMR Montreal River, ON
    (50.36950, -119.06436),    # CASSS Mount Silver Star, BC
    (53.61308, -122.95441),    # CASPG Prince George, BC
    (52.52048, -107.44269),    # CASRA Radisson, SK
    (46.449556, -71.913831),   # CASSF Sainte-Françoise, QC
    (50.31250, -110.19556),    # CASSU Schuler, AB
    (49.28146, -81.79406),     # CASRF Smooth Rock Falls, ON
    (55.69494, -119.23043),    # CASSR Spirit River, AB
    (51.20613, -113.39937),    # CASSM Strathmore, AB
    (48.595876, -89.100129),   # CASSN Shuniah, ON
    (48.48028, -67.60111),     # CASVD Val d'Irène, QC
    (50.15389, -97.77833),     # CASWL Woodlands, MB
]

STATION_MAP: dict[str, list[tuple[float, float]]] = {
    "CACOMP": STATIONS,
}
