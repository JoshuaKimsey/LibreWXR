# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Radar station coordinates per region.

Used to precompute coverage masks that distinguish "outside radar range"
from "clear sky within radar range" — both of which appear as value 0
in the source composites. Without this, the ECMWF fallback would either
bleed into real radar coverage (overlapping NEXRAD / Nordic data) or
leave rectangular gaps where radar is simply absent.

Coordinates are (lat, lon) in degrees. Range is in kilometers.

- NEXRAD WSR-88D (US): official NOAA station list
  https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt
  Effective precipitation range ~240 km.
- Nordic (MET Norway / SMHI / FMI / DMI): approximate station
  locations from the EUMETNET OPERA / NORDRAD networks. Coordinates
  are accurate to within a few km, which is more than sufficient
  given ~240 km range.
"""

# Effective precipitation detection range for C- and S-band weather radars.
RADAR_RANGE_KM = 240.0

# NEXRAD WSR-88D stations covering USCOMP (continental US).
NEXRAD_CONUS: list[tuple[float, float]] = [
    (45.455833, -98.413333),   # KABR
    (35.149722, -106.823889),  # KABX
    (36.98405, -77.007361),    # KAKQ
    (35.233333, -101.709278),  # KAMA
    (25.611083, -80.412667),   # KAMX
    (44.90635, -84.719533),    # KAPX
    (43.822778, -91.191111),   # KARX
    (48.194611, -122.495694),  # KATX
    (39.495639, -121.631611),  # KBBX
    (42.199694, -75.984722),   # KBGM
    (40.498583, -124.292167),  # KBHX
    (46.770833, -100.760556),  # KBIS
    (45.853778, -108.606806),  # KBLX
    (33.172417, -86.770167),   # KBMX
    (41.955778, -71.136861),   # KBOX
    (25.916, -97.418967),      # KBRO
    (42.948789, -78.736781),   # KBUF
    (24.5975, -81.703167),     # KBYX
    (33.948722, -81.118278),   # KCAE
    (46.03925, -67.806431),    # KCBW
    (43.490217, -116.236028),  # KCBX
    (40.923167, -78.003722),   # KCCX
    (41.413217, -81.859867),   # KCLE
    (32.655528, -81.042194),   # KCLX
    (35.238333, -97.46),       # KCRI
    (27.784017, -97.51125),    # KCRP
    (44.511, -73.166431),      # KCXX
    (41.151919, -104.806028),  # KCYS
    (38.501111, -121.677833),  # KDAX
    (37.760833, -99.968889),   # KDDC
    (29.273139, -100.280333),  # KDFX
    (32.279944, -89.984444),   # KDGX
    (39.947089, -74.410731),   # KDIX
    (46.836944, -92.209722),   # KDLH
    (41.7312, -93.722869),     # KDMX
    (38.825767, -75.440117),   # KDOX
    (42.7, -83.471667),        # KDTX
    (41.611667, -90.580833),   # KDVN
    (32.5385, -99.254333),     # KDYX
    (38.81025, -94.264472),    # KEAX
    (31.89365, -110.63025),    # KEMX
    (42.586556, -74.064083),   # KENX
    (31.460556, -85.459389),   # KEOX
    (31.873056, -106.698),     # KEPZ
    (35.70135, -114.89165),    # KESX
    (30.565033, -85.921667),   # KEVX
    (29.704056, -98.028611),   # KEWX
    (35.09785, -117.56075),    # KEYX
    (37.0244, -80.273969),     # KFCX
    (34.362194, -98.976667),   # KFDR
    (34.634167, -103.618889),  # KFDX
    (33.36355, -84.56595),     # KFFC
    (43.587778, -96.729444),   # KFSD
    (34.574333, -111.198444),  # KFSX
    (39.786639, -104.545806),  # KFTG
    (32.573, -97.30315),       # KFWS
    (48.206361, -106.624694),  # KGGW
    (39.062169, -108.213764),  # KGJX
    (39.366944, -101.700278),  # KGLD
    (44.498633, -88.111111),   # KGRB
    (30.721833, -97.382944),   # KGRK
    (42.893889, -85.544889),   # KGRR
    (34.883306, -82.219833),   # KGSP
    (33.896917, -88.329194),   # KGWX
    (43.891306, -70.256361),   # KGYX
    (30.5193, -90.4074),       # KHDC
    (33.077, -106.120028),     # KHDX
    (29.4719, -95.078733),     # KHGX
    (36.314181, -119.632131),  # KHNX
    (36.736972, -87.285583),   # KHPX
    (34.930556, -86.083611),   # KHTX
    (37.654444, -97.443056),   # KICT
    (37.59105, -112.862181),   # KICX
    (39.420483, -83.82145),    # KILN
    (40.1505, -89.336792),     # KILX
    (39.7075, -86.280278),     # KIND
    (36.175131, -95.564161),   # KINX
    (33.289233, -111.669911),  # KIWA
    (41.358611, -85.7),        # KIWX
    (30.484633, -81.7019),     # KJAX
    (32.675683, -83.350833),   # KJGX
    (37.590833, -83.313056),   # KJKL
    (33.654139, -101.814167),  # KLBB
    (30.125306, -93.215889),   # KLCH
    (47.116944, -124.106667),  # KLGX
    (30.336667, -89.825417),   # KLIX
    (41.957944, -100.576222),  # KLNX
    (41.604444, -88.084444),   # KLOT
    (40.73955, -116.8027),     # KLRX
    (38.698611, -90.682778),   # KLSX
    (33.98915, -78.429108),    # KLTX
    (37.975278, -85.943889),   # KLVX
    (38.976111, -77.4875),     # KLWX
    (34.8365, -92.262194),     # KLZK
    (31.943461, -102.189253),  # KMAF
    (42.081169, -122.717361),  # KMAX
    (48.393056, -100.864444),  # KMBX
    (34.775908, -76.876189),   # KMHX
    (42.9679, -88.550667),     # KMKX
    (28.113194, -80.654083),   # KMLB
    (30.679444, -88.24),       # KMOB
    (44.848889, -93.565528),   # KMPX
    (46.531111, -87.548333),   # KMQT
    (36.168611, -83.401944),   # KMRX
    (47.041, -113.986222),     # KMSX
    (41.262778, -112.447778),  # KMTX
    (37.155222, -121.898444),  # KMUX
    (47.527778, -97.325556),   # KMVX
    (32.53665, -85.78975),     # KMXX
    (32.919017, -117.0418),    # KNKX
    (35.344722, -89.873333),   # KNQA
    (41.320369, -96.366819),   # KOAX
    (36.247222, -86.5625),     # KOHX
    (40.865528, -72.863917),   # KOKX
    (47.680417, -117.626778),  # KOTX
    (35.236058, -97.46235),    # KOUN
    (37.068333, -88.771944),   # KPAH
    (40.531717, -80.217967),   # KPBZ
    (45.69065, -118.852931),   # KPDT
    (31.155278, -92.976111),   # KPOE
    (38.45955, -104.18135),    # KPUX
    (35.665519, -78.48975),    # KRAX
    (39.754056, -119.462028),  # KRGX
    (43.066089, -108.4773),    # KRIW
    (38.311111, -81.722778),   # KRLX
    (45.715039, -122.965),     # KRTX
    (43.1056, -112.686139),    # KSFX
    (37.235239, -93.400419),   # KSGF
    (32.450833, -93.84125),    # KSHV
    (31.371278, -100.4925),    # KSJT
    (33.817733, -117.636),     # KSOX
    (35.290417, -94.361889),   # KSRX
    (27.7055, -82.401778),     # KTBW
    (47.459583, -111.385333),  # KTFX
    (30.397583, -84.328944),   # KTLH
    (35.333361, -97.277761),   # KTLX
    (38.99695, -96.23255),     # KTWX
    (43.755694, -75.679861),   # KTYX
    (44.124722, -102.83),      # KUDX
    (40.320833, -98.441944),   # KUEX
    (30.890278, -83.001806),   # KVAX
    (34.83855, -120.397917),   # KVBX
    (36.740617, -98.127717),   # KVNX
    (34.412017, -119.17875),   # KVTX
    (38.26025, -87.724528),    # KVWX
    (32.495281, -114.656711),  # KYUX
]

# NEXRAD WSR-88D stations in Alaska (AKCOMP).
NEXRAD_ALASKA: list[tuple[float, float]] = [
    (60.791944, -161.876389),  # PABC Bethel
    (56.852778, -135.529167),  # PACG Biorka Island / Sitka
    (64.511389, -165.295),     # PAEC Nome
    (60.725914, -151.351456),  # PAHG Kenai
    (59.460767, -146.303444),  # PAIH Middleton Island
    (58.679444, -156.629444),  # PAKC King Salmon
    (65.035114, -147.501428),  # PAPD Pedro Dome / Fairbanks
]

# NEXRAD WSR-88D stations in Hawaii (HICOMP).
NEXRAD_HAWAII: list[tuple[float, float]] = [
    (21.893889, -159.5525),    # PHKI Kauai
    (20.125278, -155.777778),  # PHKM Kohala / Big Island
    (21.132778, -157.180278),  # PHMO Molokai
    (19.095, -155.568889),     # PHWA South Shore / Big Island
]

# NEXRAD WSR-88D station in Puerto Rico (PRCOMP).
NEXRAD_PUERTO_RICO: list[tuple[float, float]] = [
    (18.115667, -66.078167),   # TJUA
]

# NEXRAD WSR-88D station in Guam (GUCOMP).
NEXRAD_GUAM: list[tuple[float, float]] = [
    (13.455833, 144.811111),   # PGUA
]

# Nordic weather radar network (NORDRAD2 / OPERA).
# Norway (MET Norway), Sweden (SMHI), Finland (FMI), Denmark (DMI).
# Coordinates are approximate to within a few km.
NORDIC_STATIONS: list[tuple[float, float]] = [
    # Norway (MET Norway) - 11 radars
    (59.79, 5.23),    # Bømlo
    (61.21, 11.50),   # Hafjell
    (63.69, 10.20),   # Rissa
    (70.86, 29.02),   # Berlevåg
    (69.26, 16.01),   # Andøya
    (59.62, 10.55),   # Hurum
    (62.10, 5.11),    # Stad
    (59.38, 10.78),   # Rygge
    (67.53, 12.10),   # Røst
    (70.58, 22.13),   # Hasvik
    (65.37, 12.22),   # Sømna
    # Sweden (SMHI) - 12 radars
    (67.85, 20.42),   # Kiruna
    (65.43, 22.13),   # Luleå
    (63.30, 14.50),   # Östersund
    (61.58, 16.72),   # Hudiksvall
    (60.72, 14.88),   # Leksand
    (63.40, 18.60),   # Örnsköldsvik
    (59.65, 17.95),   # Arlanda
    (56.30, 15.61),   # Karlskrona
    (58.25, 12.83),   # Vara
    (58.11, 15.95),   # Vilebo
    (56.37, 12.85),   # Ängelholm
    # Finland (FMI) - 10 radars
    (60.27, 24.87),   # Vantaa
    (60.13, 21.64),   # Korppoo
    (60.90, 26.96),   # Anjalankoski
    (61.77, 23.08),   # Ikaalinen
    (62.86, 27.38),   # Kuopio
    (61.81, 22.40),   # Kankaanpää
    (63.10, 23.82),   # Vimpeli
    (64.78, 26.32),   # Utajärvi
    (67.14, 26.90),   # Luosto
    (62.30, 25.44),   # Petäjävesi
    # Denmark (DMI) - 5 radars
    (55.33, 12.45),   # Stevns
    (57.49, 10.14),   # Sindal
    (55.17, 8.55),    # Rømø
    (56.02, 10.02),   # Virring
    (55.11, 14.88),   # Bornholm
]

# Germany (DWD) radar network — 17 C-band stations.
# The DWD composite is delivered on a large polar-stereographic grid whose
# bbox extends into France, Benelux, Czech, Poland, Austria, etc., so a
# station-based coverage mask is needed to keep the ECMWF fallback from
# being suppressed across that rectangle.
GERMANY_STATIONS: list[tuple[float, float]] = [
    (54.004, 10.047),   # Boostedt
    (52.648, 13.858),   # Prötzel
    (52.160, 11.176),   # Ummendorf
    (51.124, 13.769),   # Dresden
    (51.405, 6.967),    # Essen
    (50.500, 11.135),   # Neuhaus
    (50.110, 8.714),    # Offenthal
    (49.985, 8.712),    # Offenbach (FBG candidate)
    (49.541, 12.403),   # Eisberg
    (49.541, 6.548),    # Neuheilenbach
    (48.175, 12.101),   # Isen
    (48.585, 9.783),    # Türkheim
    (47.874, 8.006),    # Feldberg
    (54.173, 12.058),   # Rostock
    (53.339, 7.024),    # Emden
    (52.460, 9.694),    # Hannover
    (51.311, 8.802),    # Flechtdorf
]


# Per-region station mapping for coverage mask generation.
# Regions not listed here skip mask generation (full-region coverage assumed).
REGION_STATIONS: dict[str, list[tuple[float, float]]] = {
    "USCOMP": NEXRAD_CONUS,
    "AKCOMP": NEXRAD_ALASKA,
    "HICOMP": NEXRAD_HAWAII,
    "PRCOMP": NEXRAD_PUERTO_RICO,
    "GUCOMP": NEXRAD_GUAM,
    "NORDIC": NORDIC_STATIONS,
    "GERMANY": GERMANY_STATIONS,
}
