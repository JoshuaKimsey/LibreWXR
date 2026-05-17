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
- OPERA (EUMETNET): pan-European radar network, ~155 stations
  across 24 countries. Coordinates from MeteoGate API and
  national met services.
- ECCC (Canada): 32 S-band dual-pol stations.
"""

# Default effective precipitation detection range (km).
RADAR_RANGE_KM = 240.0

# Per-region range overrides.  European C-band radars in the OPERA
# network detect precipitation out to ~300 km — using the default 240 km
# leaves gaps around peripheral stations (Iceland, Ireland) where OPERA
# still has data but the coverage mask says "not covered," causing
# visible ECMWF/OPERA overlap seams.
REGION_RADAR_RANGE: dict[str, float] = {
    "OPERA": 300.0,
    # SNET ``esar82`` product is explicitly the 120 km range overlay
    # (single S-band radar at San Andrés).  The default 240 km would
    # overstate coverage by 2× and bleed past the product's footprint
    # into Honduras / Nicaragua where no actual returns exist.
    "SVCOMP": 120.0,
    # CWA QPESUMS publishes coverage out to its full bbox edges
    # (~565 km from the nearest station) for typhoon tracking, as
    # verified from the ``-999`` sentinel pattern in a real frame.
    # We previously used 550 km to match that "claimed" reach, but
    # Taiwan's western and southern stations are close enough to
    # land that the resulting circles bleed visibly onto Fujian /
    # Hong Kong (Qigu's 550 km reach hit 114.7°E) and over the
    # northern tip of Luzon (Kenting's reach hit 17°N, past Cape
    # Bojeador).  At 450 km the mask still covers ~94% of CWA's
    # claimed in-range area and ~450 km of Pacific buffer east of
    # Hualien (well into the typhoon corridor); the trimmed halo
    # over the SCS / W. Pacific gets filled by IFS instead, which
    # resolves typhoons at 9 km perfectly well.  Per-station ranges
    # are the proper long-term fix (small western stations could go
    # to ~150 km, eastern stations could keep ~450 km) but
    # ``REGION_RADAR_RANGE`` is per-region today.  Tighten further
    # to 300 km if bleed onto Fujian / Luzon is still visible.
    "TWCOMP": 450.0,
    # MET Malaysia CAPPI composite extends well past the standard
    # 240 km Doppler range — empirically measured 372 km max extent
    # over Peninsular Malaysia and 332 km over East Malaysia from the
    # nearest station, with p99 distances of 313 km and 302 km
    # respectively.  The default 240 km would clip ~10% of legitimate
    # radar data (visible as IFS bleed at frame T+0 and a "chunk of
    # radar rain blinking out" at the first nowcast frame as the
    # blend switches to model outside the mask).  Ranges chosen to
    # snugly cover the measured max with small margin: a looser fit
    # would help if maxes fluctuate day-to-day, but it would also
    # hide IFS precipitation over the South China Sea / Celebes Sea
    # (which, unlike CWA's open Pacific halo, gets real rain).  If
    # clipping reappears on a heavy convective day, bump by 25 km.
    "MYPENINSULAR": 375.0,
    "MYEAST": 350.0,
}

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


# ECCC Canadian weather radar network.  32 S-band dual-pol stations
# (METEOR 1700S) operated by Environment and Climate Change Canada
# after the 2023 network modernization.  Coordinates from the Canadian
# Weather Radar Network Wikipedia article, cross-referenced with ECCC
# station pages.  Needed because MSC's RADAR_1KM_RRAI composite covers
# the whole CACOMP bbox but only has actual data within ~240 km of
# these stations — without a mask, the ECMWF fallback is suppressed
# over huge empty regions (open Pacific, Arctic, Atlantic).
CANADA_STATIONS: list[tuple[float, float]] = [
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


# OPERA pan-European CIRRUS composite radar network.
# Sourced verbatim from the official EUMETNET OPERA radar database
# (status=operational only):
#   https://www.eumetnet.eu/wp-content/themes/aeron-child/observations-programme/current-activities/opera/database/OPERA_Database/
# Used for station-circle coverage masks to prevent ECMWF fallback
# suppression across the entire LAEA grid bbox.  Regenerate this
# list by downloading Data/OPERA_RADARS_DB_<date>.json from the OPERA
# Database page and filtering on status="1".
OPERA_STATIONS: list[tuple[float, float]] = [
    # Belgium (BE) - 3 radars
    (51.0702, 5.4054),    # behel Helchteren
    (51.1919, 3.0641),    # bejab Jabbeke
    (49.9135, 5.5044),    # bewid Wideumont
    # Croatia (HR) - 6 radars
    (45.8835, 17.2005),   # hrbil Bilogora
    (44.0455, 15.3765),   # hrdeb Debeljak
    (45.0205, 14.1223),   # hrgol Goli
    (45.1592, 18.7033),   # hrgra Gradište
    (45.9078, 15.9683),   # hrpun Puntijarka
    (42.8944, 17.4783),   # hrulj Uljenje
    # Czechia (CZ) - 2 radars
    (49.6583, 13.8178),   # czbrd Brdy-Praha
    (49.5011, 16.7885),   # czska Skalky
    # Denmark (DK) - 4 radars
    (55.1127, 14.8875),   # dkbor Bornholm
    (55.1731, 8.5520),    # dkrom Römö
    (57.4893, 10.1365),   # dksin Sindal
    (55.3262, 12.4493),   # dkste Stevns
    # Estonia (EE) - 2 radars
    (59.3971, 24.6021),   # eehar Harku
    (58.4823, 25.5187),   # eesur Sürgavere
    # Finland (FI) - 12 radars
    (60.9039, 27.1081),   # fianj Anjalankoski
    (61.8108, 22.5020),   # fikan Kankaanpää
    (68.4344, 27.4440),   # fikau Kaunispää
    (61.9069, 29.7977),   # fikes Kesälahti
    (60.1285, 21.6434),   # fikor Korpo
    (62.8626, 27.3815),   # fikuo Kuopio
    (67.1391, 26.8969),   # filuo Luosto
    (63.8378, 29.4489),   # finur Nurmes
    (62.3045, 25.4401),   # fipet Petäjävesi
    (64.7749, 26.3189),   # fiuta Utajärvi
    (60.5562, 24.4956),   # fivih Vihti
    (63.1048, 23.8209),   # fivim Vimpeli
    # France (FR) - 25 radars
    (50.1360, 1.8347),    # frabb Abbeville
    (41.9531, 8.7005),    # fraja Ajaccio
    (42.1298, 9.4964),    # frale Aléria
    (50.1283, 3.8118),    # frave Avesnes
    (47.3552, 4.7759),    # frbla Blaisy-Haut
    (44.3230, 4.7621),    # frbol Bollène
    (44.8315, -0.6919),   # frbor Bordeaux
    (47.0586, 2.3595),    # frbou Bourges
    (48.9272, -0.1496),   # frcae Falaise
    (43.2166, 6.3729),    # frcol Collobrières
    (45.1044, 1.3697),    # frgre Grèzes
    (45.2892, 3.7095),    # frlep Sembadel
    (44.0128, 6.5292),    # frmau Mont Maurel
    (43.9905, 2.6096),    # frmcl Montclar
    (43.6245, -0.6094),   # frmom Momuy
    (47.3686, 7.0190),    # frmtc Montancy
    (48.7158, 6.5816),    # frnan Nancy
    (43.8061, 4.5027),    # frnim Nîmes
    (46.0678, 4.4453),    # frniz Saint Nizier
    (42.9184, 2.8650),    # fropo Opoul
    (48.4609, -4.4298),   # frpla Plabennec
    (43.5743, 1.3763),    # frtou Toulouse
    (48.7746, 2.0083),    # frtra Trappes
    (47.3374, -1.6563),   # frtre Treillères
    (48.4621, 4.3093),    # frtro Arcis-sur-Aube
    # Germany (DE) - 17 radars
    (53.5640, 6.7482),    # deasb Isle of Borkum
    (54.0043, 10.0468),   # deboo Boostedt
    (51.1246, 13.7686),   # dedrs Dresden
    (49.5407, 12.4028),   # deeis Eisberg
    (51.4055, 6.9669),    # deess Essen
    (47.8736, 8.0039),    # defbg Feldberg
    (51.3112, 8.8020),    # defld Flechtdorf
    (52.4600, 9.6945),    # dehnr Hannover
    (48.1747, 12.1017),   # deisn Isen/München
    (48.0421, 10.2192),   # demem Memmingen
    (50.5001, 11.1351),   # deneu Neuhaus
    (50.1097, 6.5483),    # denhb Neuheilenbach
    (49.9847, 8.7129),    # deoft Offenthal
    (52.6486, 13.8580),   # depro Protzel/Berlin
    (54.1757, 12.0580),   # deros Rostock
    (48.5853, 9.7828),    # detur Tuerkheim
    (52.1601, 11.1761),   # deumd Ummendorf
    # Greece (GR) - 5 radars
    (38.9166, 20.7528),   # grakt Aktio
    (37.9261, 21.2894),   # grand Andravida
    (39.6446, 22.4603),   # grlar Larisa
    (40.5282, 22.9757),   # grthe Thessaloniki
    (37.9461, 23.8138),   # grymi Ymittos
    # Hungary (HU) - 5 radars
    (47.4294, 19.1817),   # hubud Budapest
    (46.1775, 18.3372),   # huhar Harmashegy
    (47.9622, 21.8867),   # hunap Napkor
    (46.6604, 17.0624),   # hupog Poganyvar
    (46.6397, 20.4325),   # husze Szentes
    # Iceland (IS) - 3 radars
    (65.2658, -14.0618),  # isbjo Bjolfur
    (64.0257, -22.6353),  # iskef Keflavik
    (66.0557, -20.2680),  # isska Skagi
    # Ireland (IE) - 2 radars
    (53.4299, -6.2443),   # iedub Dublin
    (52.6928, -8.9201),   # iesha Shannon
    # Latvia (LV) - 1 radars
    (56.9143, 23.9897),   # lvrix Riga
    # Lithuania (LT) - 2 radars
    (55.6090, 22.2395),   # ltlau Laukuva
    (54.6262, 25.1067),   # ltvil Vilnius
    # Malta (MT) - 1 radars
    (35.8529, 14.4748),   # mtgud Gudja
    # Netherlands (NL) - 2 radars
    (52.9528, 4.7906),    # nldhl Den Helder
    (51.8371, 5.1380),    # nlhrw Herwijnen
    # Norway (NO) - 12 radars
    (69.2414, 16.0030),   # noand Andoya
    (70.5107, 29.0184),   # nober Berlevaag
    (59.8537, 5.0896),    # nobml Boemlo
    (70.6052, 22.4430),   # nohas Hasvik
    (61.2318, 10.5273),   # nohfj Hafjell
    (58.3601, 7.1648),    # nohgb Haegebostad
    (59.6272, 10.5645),   # nohur Hurum
    (63.6905, 10.2039),   # norsa Rissa
    (69.2186, 23.4398),   # norsg Rassegalvarri
    (67.5304, 12.0989),   # norst Rost
    (65.2199, 11.9925),   # nosmn Soemna
    (62.1871, 5.1275),    # nosta Stad
    # Poland (PL) - 10 radars
    (50.3942, 20.0832),   # plbrz Brzuchania
    (54.5009, 18.2718),   # plgdy Gdynia-Szemud
    (50.4639, 18.1532),   # plgsa Góra Św. Anny
    (52.4053, 20.9611),   # plleg Legionowo
    (50.8925, 16.0395),   # plpas Pastewnik
    (52.4133, 16.7970),   # plpoz Poznań
    (50.1513, 18.7251),   # plram Ramża
    (50.1141, 22.0370),   # plrze Rzeszów
    (53.7958, 15.8368),   # plswi Świdwin
    (53.8565, 21.4121),   # pluzr Uzranki
    # Portugal (PT) - 6 radars
    (37.3041, -7.9530),   # ptfar Loulé/Cavalos do Caldeirão
    (39.4634, -31.2200),  # ptflr Flores/Morro Alto
    (39.0714, -8.4001),   # ptlis Coruche/Cruz do Leão
    (40.8450, -8.2797),   # ptprt Arouca/Pico do Gralheiro
    (37.8191, -25.7516),  # ptsmg São Miguel/Pico Santos de Cima
    (38.7302, -27.3208),  # pttrc Terceira/ Santa Barbara
    # Romania (RO) - 7 radars
    (47.0118, 27.5826),   # robar Barnova
    (46.3602, 24.2252),   # robob Bobohalma
    (44.5127, 26.0773),   # robuc Bucuresti
    (44.3103, 23.8674),   # rocra Craiova
    (44.2434, 28.2506),   # romed Medgidia
    (47.0922, 21.9429),   # roora Oradea
    (45.7717, 21.2577),   # rotim Timisoara
    # Serbia (RS) - 3 radars
    (45.1573, 19.8109),   # rsfrg Fruska gora
    (43.3913, 21.4436),   # rsjas Jastrebac
    (45.1876, 20.7707),   # rssam Samos
    # Slovakia (SK) - 4 radars
    (48.2556, 17.1524),   # skjav Maly Javornik
    (48.7827, 20.9873),   # skkoj Kojsovska hola
    (49.2717, 19.2494),   # skkub Kubinska hola
    (48.2404, 19.2573),   # sklaz Spani laz
    # Slovenia (SI) - 2 radars
    (46.0680, 15.2850),   # silis Lisca
    (46.0980, 14.2283),   # sipas Pasja ravan
    # Spain (ES) - 15 radars
    (36.6133, -4.6593),   # esahr Alhaurin el Grande (Malaga)
    (36.8324, -2.0821),   # esalm Nijar (Almeria)
    (43.1690, -8.5269),   # escor Cerceda ( La Coruna)
    (41.4081, 1.8848),    # esgld Gelida (Barcelona)
    (41.9956, -4.6028),   # eslid Autilla Pino (Palencia)
    (28.0186, -15.6144),  # eslpa Artenara (Gran Canaria)
    (38.2644, -1.1897),   # esmur Fortuna (Murcia)
    (39.3797, 2.7851),    # espma Llucmajor (Baleares)
    (43.4625, -6.3019),   # essan Aguión (Asturias)
    (37.6887, -6.3331),   # essev Castillo las Guardas (Sevilla)
    (39.4288, -6.2853),   # essft Sierra de Fuentes (Caceres)
    (43.4033, -2.8419),   # essse Baquio (Vizcaya)
    (40.1759, -3.7136),   # estjv Torrejon de Velasco (Madrid)
    (39.1761, -0.2521),   # esval Cullera (Valencia)
    (41.7339, -0.5459),   # eszar Perdiguera (Zaragoza)
    # Sweden (SE) - 12 radars
    (56.3675, 12.8517),   # seang Ängelholm
    (58.1059, 15.9365),   # seatv Åtvidaberg (Vilebo)
    (59.6110, 17.5833),   # sebaa Bålsta
    (57.3035, 18.4001),   # sehem Hemse (Ase)
    (61.5771, 16.7144),   # sehuv Hudiksvall
    (56.2955, 15.6102),   # sekaa Karlskrona
    (67.7088, 20.6178),   # sekrn Kiruna
    (60.7230, 14.8776),   # selek Leksand
    (65.4309, 21.8650),   # sella Luleå (Rosvik)
    (63.6395, 18.4019),   # seoer Örnsköldsvik
    (63.2951, 14.7591),   # seosd Östersund
    (58.2556, 12.8260),   # sevax Vara
    # Switzerland (CH) - 5 radars
    (47.2843, 8.5120),    # chalb Albis
    (46.4251, 6.0994),    # chdol La Dole
    (46.0408, 8.8332),    # chlem Monte Lema
    (46.3706, 7.4866),    # chppm Plaine Monte
    (46.8350, 9.7945),    # chwei Weissfluhgiptel
    # United Kingdom (UK) - 16 radars
    (54.5020, -6.3429),   # ukcas Castor Bay
    (51.6895, -0.5302),   # ukche Chenies
    (52.3986, -2.5954),   # ukcle Clee Hill
    (50.9639, -3.4521),   # ukcob Cobbacombe Cross
    (51.9798, -4.4446),   # ukcyg Crug y Gorllwyn
    (51.0307, -1.6534),   # ukdea Dean Hill
    (57.4304, -2.0367),   # ukdud Hill Of Dudwick
    (53.7548, -2.2892),   # ukham Hameldon Hill
    (56.0180, -4.2171),   # ukhhd Holehead
    (54.8034, -1.4746),   # ukhmy High Moorsley
    (53.3347, -0.5594),   # uking Ingham
    (49.1772, -2.2239),   # ukjer Jersey
    (58.2117, -6.1821),   # uklew Druim A'Starraig
    (56.2148, -3.3118),   # ukmun Munduff Hill
    (50.0034, -5.2230),   # ukpre Predannack
    (51.2948, 0.6043),    # ukthu Thurnham
]


# Taiwan QPESUMS composite (CWA O-A0059-001) — 7 contributing radars
# operated by the Central Weather Administration.  Approximate
# coordinates from publicly-documented siting and CWA's radar
# inventory.  Two additional Taiwan radars exist (RCMK military,
# RCWF civil aviation) but are not part of this composite.
#
# The 240 km default range covers all of Taiwan + a substantial
# offshore buffer.  Exact lat/lons to the metre don't matter here
# because the radar circles overlap heavily over the island; the
# resulting union polygon is insensitive to small per-station shifts.
CWA_STATIONS: list[tuple[float, float]] = [
    (25.071, 121.773),   # 五分山 / Wufenshan  (NE Taiwan, original 4)
    (23.991, 121.622),   # 花蓮   / Hualien    (E Taiwan,  original 4)
    (23.146, 120.094),   # 七股   / Qigu/Cigu  (SW Taiwan, original 4)
    (21.902, 120.853),   # 墾丁   / Kenting    (S Taiwan,  original 4)
    (24.998, 121.420),   # 樹林   / Shulin     (N gap-fill)
    (24.140, 120.620),   # 南屯   / Nantun     (central gap-fill)
    (22.510, 120.397),   # 林園   / Linyuan    (S gap-fill, Kaohsiung)
]


# SNET (El Salvador) — single S-band radar at San Andrés volcano.
# Coordinates from the viewer's ``center = [13.687, -88.883]`` JS variable
# (snet.gob.sv/googlemaps/radares/radaresSV8.php).  120 km product, so the
# range override above applies to the SVCOMP mask.
SNET_STATIONS: list[tuple[float, float]] = [
    (13.687, -88.883),   # San Andrés
]

# MET Malaysia — 12-radar national network feeding the combined
# Peninsular + East composite GIF.  Coordinates are approximate (taken
# from each radar's host airport / city — the operational siting is
# usually co-located or within a few km).  Default 240 km Doppler range.
# Cross-checked against the station inventory on
# rainviewer.com/radars/malaysia.html (12 stations: MY2809-MY2819,
# MY2865).
MMD_PENINSULAR_STATIONS: list[tuple[float, float]] = [
    (6.20, 100.40),    # MY2810 Alor Setar
    (5.47, 100.39),    # MY2819 Butterworth
    (2.04, 103.32),    # MY2818 Kluang
    (6.17, 102.29),    # MY2815 Kota Bharu
    (3.78, 103.21),    # MY2817 Kuantan
    (3.13, 101.55),    # MY2816 Subang
    (2.74, 101.71),    # MY2865 TDR KLIA
]
MMD_EAST_STATIONS: list[tuple[float, float]] = [
    (3.16, 113.05),    # MY2812 Bintulu
    (5.94, 116.05),    # MY2809 Kota Kinabalu
    (1.48, 110.34),    # MY2814 Kuching
    (4.32, 113.99),    # MY2813 Miri
    (5.90, 118.06),    # MY2811 Sandakan
]
MMD_STATIONS: list[tuple[float, float]] = (
    MMD_PENINSULAR_STATIONS + MMD_EAST_STATIONS
)

# Per-region station mapping for coverage mask generation.
# Regions not listed here skip mask generation (full-region coverage assumed).
REGION_STATIONS: dict[str, list[tuple[float, float]]] = {
    "USCOMP": NEXRAD_CONUS,
    "AKCOMP": NEXRAD_ALASKA,
    "HICOMP": NEXRAD_HAWAII,
    "PRCOMP": NEXRAD_PUERTO_RICO,
    "GUCOMP": NEXRAD_GUAM,
    "CACOMP": CANADA_STATIONS,
    "SVCOMP": SNET_STATIONS,
    "OPERA": OPERA_STATIONS,
    "TWCOMP": CWA_STATIONS,
    "MYPENINSULAR": MMD_PENINSULAR_STATIONS,
    "MYEAST": MMD_EAST_STATIONS,
}

# MRMS ingests both NEXRAD (US) and ECCC (Canadian) radar networks.
# Per-region station lists for coverage masks when MRMS is the active source.
# Used when LIBREWXR_NA_SOURCE=mrms or mrms_fallback.
MRMS_STATIONS: dict[str, list[tuple[float, float]]] = {
    "USCOMP": NEXRAD_CONUS + CANADA_STATIONS,
    "CACOMP": NEXRAD_CONUS + CANADA_STATIONS,
    "AKCOMP": NEXRAD_ALASKA,
    "HICOMP": NEXRAD_HAWAII,
    "PRCOMP": NEXRAD_PUERTO_RICO,
    "GUCOMP": NEXRAD_GUAM,
}
