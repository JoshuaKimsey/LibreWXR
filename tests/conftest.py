# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np
import pytest

from librewxr.data.regions import REGIONS
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


@pytest.fixture
def sample_frame_data() -> np.ndarray:
    """Create a sample composite frame with some non-zero values."""
    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    # Add some "rain" in the middle (roughly over central US)
    center_row = COMPOSITE_HEIGHT // 2
    center_col = COMPOSITE_WIDTH // 2
    # Create a gradient block
    for i in range(200):
        for j in range(200):
            data[center_row - 100 + i, center_col - 100 + j] = min(
                255, int(((i + j) / 400) * 255)
            )
    return data


@pytest.fixture
def frame_store() -> FrameStore:
    return FrameStore(max_frames=12)


@pytest.fixture
def tile_cache() -> TileCache:
    return TileCache(max_size=100)


@pytest.fixture
def sample_radar_frame(sample_frame_data) -> RadarFrame:
    return RadarFrame(timestamp=1700000000, regions={"USCOMP": sample_frame_data})


@pytest.fixture
def sample_nordic_data() -> np.ndarray:
    """Create a sample Nordic frame with some non-zero values."""
    region = REGIONS["NORDIC"]
    data = np.zeros((region.height, region.width), dtype=np.uint8)
    # Add some "rain" in the middle (roughly over central Scandinavia)
    center_row = region.height // 2
    center_col = region.width // 2
    for i in range(100):
        for j in range(100):
            data[center_row - 50 + i, center_col - 50 + j] = min(
                255, int(((i + j) / 200) * 255)
            )
    return data
