# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


# Snapshot ``librewxr_*`` tempdirs at conftest import time — earlier
# than any session-scoped fixture, so this captures the state BEFORE
# pytest starts collecting (and importing) test modules.  That matters
# because some modules construct stores at module-load time
# (``test_api.py`` builds a FastAPI app + FrameStore at module scope),
# and those tempdirs would otherwise look "pre-existing" to any
# fixture-based snapshot.
_LIBREWXR_TMP_BEFORE_SESSION: set[Path] = set(
    Path(tempfile.gettempdir()).glob("librewxr_*")
)


@pytest.fixture(scope="session", autouse=True)
def _session_cleanup_librewxr_tempdirs():
    """Session-scoped final sweep — catches module-import-time leaks.

    Some test modules construct grids or stores at import / module
    load time, so the resulting ``librewxr_*`` tempdirs exist before
    any session-fixture snapshot can see them.  This finalizer uses
    the conftest-import-time snapshot above and removes anything new
    at session end.
    """
    yield
    tmp_root = Path(tempfile.gettempdir())
    for path in tmp_root.glob("librewxr_*"):
        if path not in _LIBREWXR_TMP_BEFORE_SESSION:
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cleanup_librewxr_tempdirs():
    """Sweep ``librewxr_*`` tempdirs created during this test.

    Grid classes (``WRFSMNGrid``, ``HRRRGrid``, …) construct via
    ``tempfile.mkdtemp(prefix="librewxr_…_")`` when ``cache_dir`` is
    not given.  The tempdir is normally cleaned in ``close()``, but
    plenty of sync unit tests construct a grid, inject frames, call
    ``sample`` / ``get_snow_mask``, and never reach ``await close()``.
    Each such test leaks a tempdir; over months of CI runs this can
    accumulate into thousands of empty dirs and trigger
    ``disk quota exceeded`` on the host.

    This fixture snapshots existing ``librewxr_*`` entries before the
    test, lets the test run, then removes any new ones afterwards.
    Anything that existed before the test is left alone (so the
    session-scoped fixture above catches anything created at module
    import time).
    """
    tmp_root = Path(tempfile.gettempdir())
    before = set(tmp_root.glob("librewxr_*"))
    yield
    for path in tmp_root.glob("librewxr_*"):
        if path not in before:
            shutil.rmtree(path, ignore_errors=True)


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
    return TileCache(max_mb=10)


@pytest.fixture
def sample_radar_frame(sample_frame_data) -> RadarFrame:
    return RadarFrame(timestamp=1700000000, regions={"USCOMP": sample_frame_data})


