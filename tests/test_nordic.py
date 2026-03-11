# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np
import pytest

from librewrx.data.regions import REGIONS, resolve_regions
from librewrx.data.sources import _dbz_float_to_uint8, _parse_nordic_catalog
from librewrx.tiles.coordinates import tile_overlaps_region


class TestNordicRegion:
    def test_nordic_in_regions(self):
        assert "NORDIC" in REGIONS

    def test_nordic_dimensions(self):
        r = REGIONS["NORDIC"]
        assert r.width == 1694
        assert r.height == 1951
        assert r.pixel_size == 0.028808
        assert r.pixel_size_y == 0.009585

    def test_nordic_group_resolution(self):
        assert resolve_regions("NORDIC") == ["NORDIC"]

    def test_all_includes_nordic(self):
        all_regions = resolve_regions("ALL")
        assert "NORDIC" in all_regions

    def test_mixed_us_and_nordic(self):
        result = resolve_regions("CONUS,NORDIC")
        assert "USCOMP" in result
        assert "NORDIC" in result

    def test_nordic_no_iem_dirs(self):
        r = REGIONS["NORDIC"]
        assert r.live_dir == ""
        assert r.archive_dir == ""


class TestDbzConversion:
    def test_nodata_maps_to_zero(self):
        arr = np.array([-32.5, -33.0, -100.0], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_minus32_maps_to_zero(self):
        arr = np.array([-32.0], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        assert result[0] == 0

    def test_zero_dbz(self):
        # 0 dBZ -> (0+32)*2 = 64
        arr = np.array([0.0], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        assert result[0] == 64

    def test_20_dbz(self):
        # 20 dBZ -> (20+32)*2 = 104
        arr = np.array([20.0], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        assert result[0] == 104

    def test_high_dbz_clips(self):
        # 95.5 dBZ -> (95.5+32)*2 = 255
        # 100 dBZ -> clips to 255
        arr = np.array([95.5, 100.0], dtype=np.float32)
        result = _dbz_float_to_uint8(arr)
        assert result[0] == 255
        assert result[1] == 255

    def test_matches_iem_encoding(self):
        """Verify our conversion matches IEM's uint8 encoding: dBZ = pixel*0.5 - 32."""
        for dbz in [-31.5, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
            arr = np.array([float(dbz)], dtype=np.float32)
            pixel = _dbz_float_to_uint8(arr)[0]
            # Reverse: dBZ = pixel * 0.5 - 32
            reconstructed = pixel * 0.5 - 32
            assert abs(reconstructed - dbz) < 0.5, f"dBZ={dbz}: pixel={pixel}, reconstructed={reconstructed}"


class TestCatalogParsing:
    def test_parse_valid_catalog(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="latest">
    <dataset name="yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20260310T120000Z.nc" />
    <dataset name="yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20260310T115500Z.nc" />
    <dataset name="yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20260310T115000Z.nc" />
  </dataset>
</catalog>"""
        entries = _parse_nordic_catalog(xml)
        assert len(entries) == 3
        # Should be sorted newest-first
        assert entries[0][0] > entries[1][0]
        assert entries[1][0] > entries[2][0]
        # Filenames should be preserved
        assert "20260310T120000Z" in entries[0][1]

    def test_parse_empty_catalog(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="latest" />
</catalog>"""
        entries = _parse_nordic_catalog(xml)
        assert entries == []

    def test_parse_ignores_non_nc_files(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <dataset name="latest">
    <dataset name="yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20260310T120000Z.nc" />
    <dataset name="some_other_file.txt" />
  </dataset>
</catalog>"""
        entries = _parse_nordic_catalog(xml)
        assert len(entries) == 1


class TestNordicTileOverlap:
    def test_tile_over_scandinavia_overlaps(self):
        region = REGIONS["NORDIC"]
        # Zoom 4, roughly over Scandinavia
        # At z=4, x=8, y=4 is roughly over Scandinavia
        assert tile_overlaps_region(region, z=4, x=8, y=4)

    def test_tile_over_us_does_not_overlap(self):
        region = REGIONS["NORDIC"]
        # z=4, x=3, y=5 is over the US
        assert not tile_overlaps_region(region, z=4, x=3, y=5)
