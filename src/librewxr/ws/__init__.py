# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""WebSocket push channel.

``WebSocketHub`` fans topic payloads (``weather-maps``, ``alerts``,
``storm-cells``, ``lightning``) out to connected clients; ``router``
mounts the endpoint at ``LIBREWXR_WEBSOCKET_PATH`` (default ``/v2/ws``).
"""

from librewxr.ws.endpoint import router
from librewxr.ws.hub import ClientState, WebSocketHub

__all__ = ["ClientState", "WebSocketHub", "router"]
