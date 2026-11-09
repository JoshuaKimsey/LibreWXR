# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import os
import re
import subprocess
import sys
from importlib.resources import files

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api.dashboard import router as dashboard_router


def _dashboard_html() -> str:
    """Read the packaged asset exactly the way the module does."""
    return files("librewxr.api").joinpath("dashboard.html").read_text(
        encoding="utf-8"
    )


def test_dashboard_serves_packaged_asset():
    """The route serves the packaged asset verbatim as HTML."""
    app = FastAPI()
    app.include_router(dashboard_router)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "LibreWXR Dashboard" in resp.text
    assert resp.text == _dashboard_html()


def test_dashboard_asset_not_truncated():
    """Guard against an empty or truncated packaged asset."""
    assert len(_dashboard_html().encode("utf-8")) > 50000


_URL_RE = re.compile(r"https?://[^\s\"'<>)\\]+")
_ALLOWED_URL_PREFIXES = (
    "https://unpkg.com/maplibre-gl@",
    "https://tiles.openfreemap.org/",
)


def test_dashboard_asset_external_urls_allowlisted():
    """The asset is self-contained except the hot-tile map's external deps.

    The page is self-contained except the hot-tile map's MapLibre GL
    library (unpkg.com) and OpenFreeMap basemap (tiles.openfreemap.org);
    no other external URLs are permitted.
    """
    html = _dashboard_html()
    urls = _URL_RE.findall(html)
    insecure = [url for url in urls if url.startswith("http://")]
    assert not insecure, f"Dashboard asset uses insecure URLs: {insecure}"
    disallowed = [
        url for url in urls if not url.startswith(_ALLOWED_URL_PREFIXES)
    ]
    assert not disallowed, (
        "Dashboard asset references URLs outside the allowlist: "
        + ", ".join(sorted(set(disallowed)))
    )


def test_disabled_endpoints_mount_no_route():
    """Disabled dashboard/metrics mount no route at all.

    Runs in a subprocess because the ``if settings.dashboard_enabled:`` /
    ``if settings.metrics_enabled:`` wiring is evaluated once at import of
    ``librewxr.main``, against a module-level settings singleton.  A fresh
    interpreter re-reads the env overrides; importing ``librewxr.main``
    in-process here would also disturb the import state other tests rely on.
    """
    env = {
        **os.environ,
        "LIBREWXR_DASHBOARD_ENABLED": "false",
        "LIBREWXR_METRICS_ENABLED": "false",
        "LIBREWXR_TILE_HISTORY_ENABLED": "false",
    }
    code = (
        "from fastapi.testclient import TestClient\n"
        "from librewxr.main import app\n"
        "c = TestClient(app)\n"
        "assert c.get('/dashboard').status_code == 404, 'dashboard route mounted while disabled'\n"
        "assert c.get('/metrics').status_code == 404, 'metrics route mounted while disabled'\n"
        "assert c.get('/health/tile-history').status_code == 404, 'tile-history route mounted while disabled'\n"
        "assert c.get('/docs').status_code == 200, 'app built but docs missing?'\n"
        "print('mount-ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "mount-ok" in proc.stdout
