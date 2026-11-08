# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Self-contained operational health dashboard page.

``dashboard.html`` is a single offline HTML asset - no CDN, no external
URLs - that polls ``/health`` client-side and renders the payload.  It is
read once at import time and served verbatim, so a request costs one
bytes-in-memory response.

The router is only included in the app when ``settings.dashboard_enabled``
is true, so a disabled dashboard has no route at all (a plain 404).
"""

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from librewxr.config import settings

router = APIRouter()

# Read once at import: the asset is a static, packaged file and the page
# is a few hundred KB at most.
_DASHBOARD_HTML = files("librewxr.api").joinpath("dashboard.html").read_text(
    encoding="utf-8"
)


@router.get(settings.dashboard_path, response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the packaged dashboard page."""
    return HTMLResponse(_DASHBOARD_HTML)
