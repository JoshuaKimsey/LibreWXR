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


# OPERA pan-European CIRRUS composite radar network (~155 stations).
# Coordinates from MeteoGate /locations API and national met services.
# Used for station-circle coverage masks to prevent ECMWF fallback
# suppression across the entire LAEA grid bbox.
OPERA_STATIONS: list[tuple[float, float]] = [
    # Austria (AT) - 5 radars
    (47.0785, 15.4554),   # atfuh Feldkirchen/Graz
    (46.0428, 14.566),    # atljs Ljubljana (SI shared)
    (47.8383, 13.0061),   # atpat Patscherkofel/Salzburg
    (48.1133, 16.5517),   # atrau Wien/Rauchenwarth
    (47.3264, 11.3813),   # atvls Valluga
    # Belgium (BE) - 2 radars
    (51.1919, 3.0644),    # bejab Jabbeke
    (50.1138, 5.5049),    # bewid Wideumont
    # Bulgaria (BG) - 3 radars
    (42.653, 25.556),     # bgbot Botev Peak
    (42.65, 23.39),       # bgvar Varna approach
    (43.41, 27.89),       # bgvrs Varna/Shabla
    # Croatia (HR) - 3 radars
    (43.7525, 16.4622),   # hrbiok Biokovo
    (44.884, 13.921),     # hrpun Puntijarka
    (45.6, 16.05),        # hrlip Lipik
    # Cyprus (CY) - 1 radar
    (34.8833, 32.65),     # cynic Nicosia
    # Czech Republic (CZ) - 2 radars
    (49.658, 15.847),     # czbrd Brdy
    (49.7308, 13.818),    # czska Skalky
    # Denmark (DK) - 5 radars
    (55.3261, 12.4494),   # dkste Stevns
    (57.4872, 10.1389),   # dksin Sindal
    (55.1731, 8.5521),    # dkrom Rømø
    (56.0178, 10.0213),   # dkvir Virring
    (55.1128, 14.8821),   # dkbor Bornholm
    # Estonia (EE) - 2 radars
    (58.4819, 24.4831),   # eesur Sürgavere
    (57.8124, 26.6524),   # eehar Harku area
    # Finland (FI) - 10 radars
    (60.2706, 24.8694),   # fivan Vantaa
    (60.1285, 21.6434),   # fikor Korppoo
    (60.9039, 26.9611),   # fianj Anjalankoski
    (61.7673, 23.0764),   # fiika Ikaalinen
    (62.8626, 27.3815),   # fikuo Kuopio
    (63.104, 23.822),     # fivim Vimpeli
    (64.7749, 26.3189),   # fiuta Utajärvi
    (67.1386, 26.8969),   # filuo Luosto
    (62.3045, 25.4413),   # fipet Petäjävesi
    (69.3209, 25.7819),   # fikil Kilpisjärvi area
    # France (FR) - 7 radars
    (49.2147, 2.6336),    # frtro Trappes
    (48.7108, -3.5669),   # frtpz Plouzané
    (43.9403, 3.0178),    # frnms Nîmes
    (43.629, 1.3814),     # frtls Toulouse/Blagnac
    (44.83, -0.691),      # frbdx Bordeaux/Mérignac
    (47.3514, 2.2633),    # frbou Bourges
    (48.447, 7.636),      # frstg Strasbourg
    # Germany (DE) - 17 radars
    (54.004, 10.047),     # deBoo Boostedt
    (52.648, 13.858),     # dePro Prötzel
    (52.160, 11.176),     # deUmm Ummendorf
    (51.124, 13.769),     # deDrs Dresden
    (51.405, 6.967),      # deEss Essen
    (50.500, 11.135),     # deNeu Neuhaus
    (50.110, 8.714),      # deOff Offenthal
    (49.541, 12.403),     # deEis Eisberg
    (49.541, 6.548),      # deNhb Neuheilenbach
    (48.175, 12.101),     # deIse Isen
    (48.585, 9.783),      # deTur Türkheim
    (47.874, 8.006),      # deFdb Feldberg
    (54.173, 12.058),     # deRos Rostock
    (53.339, 7.024),      # deEmd Emden
    (52.460, 9.694),      # deHan Hannover
    (51.311, 8.802),      # deFle Flechtdorf
    (49.985, 8.712),      # deOfb Offenbach
    # Greece (GR) - 3 radars
    (40.5817, 22.9833),   # grthe Thessaloniki
    (38.035, 23.875),     # graeg Aegina area
    (35.3333, 24.4),      # grcre Crete
    # Hungary (HU) - 5 radars
    (47.4294, 19.1817),   # hubud Budapest
    (46.1775, 18.3372),   # huhar Harkány
    (47.9622, 21.8867),   # hunap Napkor
    (46.6604, 17.0624),   # hupog Pogányvár
    (46.6397, 20.4325),   # husze Szeged
    # Iceland (IS) - 1 radar
    (63.965, -22.455),    # iskef Keflavík
    # Ireland (IE) - 2 radars
    (53.4264, -6.2569),   # iedub Dublin
    (51.9411, -8.2031),   # iesha Shannon
    # Italy (IT) - 5 radars
    (44.6547, 11.6236),   # itspc S. Pietro Capofiume
    (40.625, 16.2656),    # itmtm Monte Macchia/Matera
    (38.3075, 16.0644),   # itlam Lamezia Terme
    (41.8736, 12.6506),   # itrom Roma Pratica
    (45.3458, 11.425),    # ittes Teolo area
    # Latvia (LV) - 1 radar
    (56.9628, 24.1208),   # lvrig Riga
    # Lithuania (LT) - 1 radar
    (55.7028, 23.8825),   # ltlau Laukuva
    # Netherlands (NL) - 2 radars
    (51.8371, 5.1381),    # nldbl De Bilt
    (52.9528, 4.7906),    # nldhl Den Helder
    # Norway (NO) - 11 radars
    (59.79, 5.23),        # nobml Bømlo
    (61.21, 11.50),       # nohhf Hafjell
    (63.69, 10.20),       # norsa Rissa
    (70.86, 29.02),       # nober Berlevåg
    (69.26, 16.01),       # noand Andøya
    (59.62, 10.55),       # nohur Hurum
    (62.10, 5.11),        # nostad Stad
    (59.38, 10.78),       # noryg Rygge
    (67.53, 12.10),       # norost Røst
    (70.58, 22.13),       # nohas Hasvik
    (65.37, 12.22),       # nosom Sømna
    # Poland (PL) - 8 radars
    (53.7914, 15.8311),   # plpas Pastewnik
    (51.1133, 16.0394),   # plram Ramża
    (50.1142, 20.9606),   # plbrz Brzuchania
    (51.7831, 19.8367),   # plleg Legionowo area
    (53.5314, 18.5297),   # plpoz Poznań area
    (50.1181, 22.7044),   # plrze Rzeszów
    (54.3828, 18.4561),   # plgda Gdańsk
    (52.4222, 20.9611),   # plwar Warsaw area
    # Portugal (PT) - 3 radars
    (37.27, -7.97),       # ptfar Faro
    (39.0714, -8.4001),   # ptlis Lisbon/Coruche
    (40.845, -8.2797),    # ptprt Porto/Arouca
    # Romania (RO) - 5 radars
    (45.503, 25.367),     # robog Bobohalma
    (44.486, 26.077),     # robuc Bucharest
    (47.247, 23.744),     # rotgm Târgu Mureș
    (44.713, 21.333),     # roors Oravița
    (47.533, 25.917),     # robar Bărăbanț
    # Serbia (RS) - 4 radars
    (43.5558, 20.7006),   # rsval Valjevo area
    (44.7558, 20.9378),   # rsbeo Belgrade
    (44.17, 22.56),       # rsnel Negotin area
    (45.2372, 19.7856),   # rsnsa Novi Sad area
    # Slovakia (SK) - 4 radars
    (48.2558, 17.1524),   # skjav Javorniky/Bratislava
    (48.7822, 20.9881),   # skkoj Kojšovská hoľa
    (49.2757, 19.2823),   # skkub Kubínska hoľa
    (48.2331, 19.2561),   # sklaz Lazy pod Makytou
    # Slovenia (SI) - 1 radar
    (46.0667, 14.2167),   # silis Lisca
    # Spain (ES) - 15 radars
    (43.49, -8.42),       # eslcd La Coruña
    (39.9311, -4.0356),   # esccl Cáceres area
    (41.8767, -3.27),     # eslab Labastida area
    (38.8919, -1.1756),   # esval Valencia area
    (37.88, -3.63),       # esalm Almería approach
    (36.61, -4.66),       # esahr Alhaurín el Grande (Málaga)
    (41.41, 1.88),        # esgld El Grao/Lleida area
    (43.44, -6.30),       # essan Santander area
    (39.43, -6.29),       # essft San Fernando de Henares
    (43.39, -2.84),       # essse San Sebastián
    (40.18, -3.71),       # estjv Torrejón de Velasco
    (39.16, -0.25),       # esval Valencia
    (28.5, -16.1),        # esizn Izaña (Tenerife)
    (39.56, 2.63),        # espal Palma de Mallorca
    (42.47, -7.86),       # esour Ourense
    # Sweden (SE) - 12 radars
    (67.85, 20.42),       # sekir Kiruna
    (65.43, 22.13),       # selul Luleå
    (63.30, 14.50),       # seosd Östersund
    (61.58, 16.72),       # sehud Hudiksvall
    (60.72, 14.88),       # selek Leksand
    (63.40, 18.60),       # seorn Örnsköldsvik
    (59.65, 17.95),       # searl Arlanda
    (56.30, 15.61),       # sekar Karlskrona
    (58.25, 12.83),       # sevar Vara
    (58.11, 15.95),       # sevil Vilebo
    (56.37, 12.85),       # seang Ängelholm
    (57.25, 16.15),       # seasi Ase
    # Switzerland (CH) - 5 radars
    (46.0408, 8.8331),    # chmon Monte Lema
    (46.425, 6.1006),     # chdol La Dôle
    (46.3706, 7.4869),    # chple Plaine Morte
    (46.8131, 9.7944),    # chwea Weissfluh
    (47.2842, 8.512),     # chalb Albis
    # Turkey (TR) - 4 radars
    (41.19, 32.95),       # trbal Ankara/Bala
    (40.96, 27.97),       # trist Istanbul
    (38.48, 27.15),       # trzm Izmir
    (36.95, 35.40),       # trade Adana
    # United Kingdom (UK) - 16 radars
    (54.50, -6.34),       # ukcas Castor Bay
    (51.6892, -0.5306),   # ukche Chenies
    (52.3981, -2.5969),   # ukcle Clee Hill
    (50.9633, -3.4528),   # ukcob Cobbacombe
    (51.9797, -4.4447),   # ukcyg Crug-y-Gorllwyn
    (51.0306, -1.6544),   # ukdea Dean Hill
    (57.4308, -2.0361),   # ukdud Dudwick
    (53.7547, -2.2886),   # ukham Hameldon Hill
    (56.0183, -4.2189),   # ukhhd High Moorsley
    (54.8056, -1.4756),   # ukhmy Holme Moss
    (53.335, -0.5592),    # uking Ingham
    (49.2094, -2.1989),   # ukjer Jersey
    (58.2111, -6.1831),   # uklew Stornoway
    (56.2147, -3.3106),   # ukmun Munduff Hill
    (50.0033, -5.2225),   # ukpre Predannack
    (51.2947, 0.6042),    # ukthu Thurnham
]


# Per-region station mapping for coverage mask generation.
# Regions not listed here skip mask generation (full-region coverage assumed).
REGION_STATIONS: dict[str, list[tuple[float, float]]] = {
    "USCOMP": NEXRAD_CONUS,
    "AKCOMP": NEXRAD_ALASKA,
    "HICOMP": NEXRAD_HAWAII,
    "PRCOMP": NEXRAD_PUERTO_RICO,
    "GUCOMP": NEXRAD_GUAM,
"CACOMP": CANADA_STATIONS,
    "OPERA": OPERA_STATIONS,
}
