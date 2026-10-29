# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""WebSocket endpoint for the push channel.

The path comes from ``settings.websocket_path`` at import time, the same
way the MCP HTTP transport is mounted.  The router is only included in
the app when ``settings.websocket_enabled`` is true, so a disabled
channel has no route at all (a connect attempt is a plain 404 handshake
failure).
"""

from fastapi import APIRouter, WebSocket

from librewxr.api import routes
from librewxr.config import settings

router = APIRouter()


@router.websocket(settings.websocket_path)
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Attach the connection to the process-wide hub, if one is live.

    Accept-then-close is deliberate: a client always gets a readable close
    code instead of an opaque handshake rejection.
    """
    hub = routes.ws_hub
    if hub is None or hub.stopping:
        await websocket.accept()
        await websocket.close(code=1008)
        return
    await hub.handle(websocket)
