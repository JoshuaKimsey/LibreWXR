# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""WebSocket push hub.

One hub per render worker.  Connections subscribe to topics
(``weather-maps``, ``alerts``, ``storm-cells``, ``lightning``) and
receive a snapshot on subscribe, then ``update`` messages when the
underlying data changes.  The change signal comes from the existing
``state.json`` poller in ``main.py`` -- the hub never runs a poller of its
own; ``on_poll_tick`` / ``on_state_refreshed`` are called from
``_poll_state``.

Design notes:

* Every send goes through a bounded per-client queue drained by a
  dedicated sender task, so a slow client can never block the state
  poller (or any other client's delivery).
* Snapshots are cached by a cheap content key, so N subscribers asking
  for the same topic share one build.  A per-topic lock double-checks the
  key to collapse the concurrent rebuild stampede on a data change.
* Clients are duck-typed: only ``accept``, ``close``, ``send_text``,
  ``receive_text`` and ``query_params`` are used (tests pass fakes).
"""

import asyncio
import contextlib
import json
import logging

from dataclasses import dataclass

from librewxr.api import routes
from librewxr.config import settings
from librewxr.ws.topics import (
    ALL_TOPICS,
    STATE_TOPICS,
    TOPIC_LIGHTNING,
    TOPIC_WEATHER_MAPS,
    build_lightning_delta,
    build_lightning_snapshot,
    build_topic_snapshot,
    topic_available,
    topic_cache_key,
)

logger = logging.getLogger(__name__)


def _dumps(obj) -> str:
    """Compact JSON for the WS envelope (never pretty-printed)."""
    return json.dumps(obj, separators=(",", ":"))


def _payload_envelope(kind: str, topic: str, payload_json: str) -> str:
    """Frame ``{"type":kind,"topic":T,"data":<payload object>}``.

    ``data`` embeds the already-serialized payload object rather than a
    JSON-encoded string, so the envelope is assembled around it instead of
    re-parsing + re-dumping the payload.
    """
    return (
        '{"type":' + _dumps(kind)
        + ',"topic":' + _dumps(topic)
        + ',"data":' + payload_json
        + '}'
    )


@dataclass(eq=False)
class ClientState:
    """One live WebSocket connection and its delivery state.

    Identity-based hashing (``eq=False``) so clients can live in a set;
    two connections are never "equal" even if their fields match.
    """

    websocket: object  # duck-typed starlette WebSocket (tests use fakes)
    subscriptions: set[str]
    queue: asyncio.Queue  # pre-serialized JSON text frames
    sender: asyncio.Task | None = None
    lightning_watermark_s: int = 0  # 0 = no snapshot served yet
    dropped: bool = False


class WebSocketHub:
    """Topic fan-out for connected WebSocket clients."""

    _MAX_CONNECTIONS = 256  # per render worker; close 1013 beyond this
    _QUEUE_MAX = 32
    _SEND_TIMEOUT_S = 10.0

    def __init__(self) -> None:
        self._clients: set[ClientState] = set()
        # topic -> (cache key, payload_json, fingerprint_json)
        self._topic_state: dict[str, tuple[object, str, str]] = {}
        # topic -> last fingerprint pushed to subscribers
        self._topic_pushed: dict[str, str] = {}
        self._topic_locks: dict[str, asyncio.Lock] = {}
        # (store version, payload_json, watermark_s)
        self._lightning_cache: tuple[int, str, int] | None = None
        self._lightning_lock = asyncio.Lock()
        self._last_lightning_version: int | None = None
        # Keep best-effort close tasks referenced until they finish.
        self._close_tasks: set[asyncio.Task] = set()
        self.stopping = False

    # -- introspection -------------------------------------------------------

    def connection_count(self) -> int:
        return len(self._clients)

    def topic_counts(self) -> dict[str, int]:
        counts = {topic: 0 for topic in ALL_TOPICS}
        for client in self._clients:
            for topic in client.subscriptions:
                if topic in counts:
                    counts[topic] += 1
        return counts

    # -- connection lifecycle ------------------------------------------------

    async def handle(self, websocket) -> None:
        """Run one connection from accept to close."""
        await websocket.accept()
        if not self._origin_allowed(websocket):
            await self._safe_close(websocket, 1008)
            return
        if len(self._clients) >= self._MAX_CONNECTIONS:
            logger.debug("WebSocket connection limit reached; rejecting client")
            await self._safe_close(websocket, 1013)
            return

        client = ClientState(
            websocket=websocket,
            subscriptions=set(),
            queue=asyncio.Queue(maxsize=self._QUEUE_MAX),
        )
        self._clients.add(client)
        try:
            requested = self._initial_topics(websocket)
            invalid = [topic for topic in requested if topic not in ALL_TOPICS]
            # Mirror ``_handle_subscribe``: a known topic whose data is not
            # wired up must never be promised in ``hello.subscribed`` or get
            # a snapshot attempt on every later refresh.
            unavailable = [
                topic for topic in requested
                if topic in ALL_TOPICS and not topic_available(topic)
            ]
            subscribed = [
                topic for topic in requested
                if topic in ALL_TOPICS and topic not in unavailable
            ]
            client.subscriptions.update(subscribed)

            self._enqueue(client, {
                "type": "hello",
                "protocol": 1,
                "available": [t for t in ALL_TOPICS if topic_available(t)],
                "subscribed": subscribed,
            })
            if invalid:
                self._enqueue(client, {
                    "type": "error",
                    "message": "unknown topic(s): " + ", ".join(invalid),
                })
            if unavailable:
                self._enqueue(client, {
                    "type": "error",
                    "message": "topic unavailable: " + ", ".join(unavailable),
                })
            for topic in subscribed:
                await self._serve_snapshot(client, topic)

            client.sender = asyncio.create_task(self._sender_loop(client))
            await self._receive_loop(client)
        finally:
            sender = client.sender
            self._drop_client(client)
            if sender is not None and sender is not asyncio.current_task():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender

    async def stop(self) -> None:
        """Shut the hub down: close every connection and drain senders."""
        self.stopping = True
        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            client.dropped = True
            await self._safe_close(client.websocket, 1001)
        senders = [
            client.sender for client in clients
            if client.sender is not None
            and client.sender is not asyncio.current_task()
        ]
        for sender in senders:
            sender.cancel()
        if senders:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*senders, return_exceptions=True), 5,
                )
            except asyncio.TimeoutError:
                logger.debug("WebSocket sender tasks did not drain within 5s")

    # -- poller hooks --------------------------------------------------------

    async def on_poll_tick(self) -> None:
        """Cheap per-tick hook: push lightning deltas on a new artifact.

        The lightning store has no ``state.json`` section; its content
        version is the only reliable change signal (``maybe_reload``'s
        return value is also consumed by ``/health`` and ``/v2/lightning``,
        so it reports False when another caller already loaded the same
        artifact).
        """
        if not settings.lightning_enabled:
            return
        store = routes.lightning_store
        if store is None:
            return
        if not any(TOPIC_LIGHTNING in c.subscriptions for c in self._clients):
            return

        await store.maybe_reload()
        version = store.version
        if version == self._last_lightning_version:
            return
        self._last_lightning_version = version

        for client in list(self._clients):
            if TOPIC_LIGHTNING not in client.subscriptions:
                continue
            # No watermark guard: 0 means "snapshot served, store was empty".
            # ``build_lightning_delta(0)`` queries ``since_s=1``, which
            # correctly delivers the full (bounded) first window when strikes
            # arrive; afterwards the watermark is nonzero.  Every lightning
            # subscriber has a watermark because subscribe processing always
            # serves a snapshot before returning.
            try:
                delta = await build_lightning_delta(client.lightning_watermark_s)
            except Exception:
                logger.exception("WebSocket lightning delta build failed")
                continue
            if delta is None:
                continue
            payload_json, new_watermark = delta
            client.lightning_watermark_s = new_watermark
            self._enqueue(
                client,
                _payload_envelope("update", TOPIC_LIGHTNING, payload_json),
            )

    async def on_state_refreshed(self) -> None:
        """Called after the state poller successfully applied a snapshot."""
        if not self._clients:
            return

        for topic in STATE_TOPICS:
            if not any(topic in c.subscriptions for c in self._clients):
                continue
            try:
                await self._refresh_state_topic(topic)
            except Exception:
                logger.exception("WebSocket refresh failed for topic %s", topic)

    async def _refresh_state_topic(self, topic: str) -> None:
        key = await topic_cache_key(topic)
        cached = self._topic_state.get(topic)
        if cached is not None and cached[0] == key:
            return

        async with self._topic_lock(topic):
            # Double-check: another subscriber may have rebuilt while we
            # waited for the lock.
            cached = self._topic_state.get(topic)
            key = await topic_cache_key(topic)
            if cached is not None and cached[0] == key:
                return
            payload_json, fingerprint_json = await build_topic_snapshot(topic)
            self._topic_state[topic] = (key, payload_json, fingerprint_json)

        if self._topic_pushed.get(topic) == fingerprint_json:
            return
        self._topic_pushed[topic] = fingerprint_json
        self._broadcast(topic, _payload_envelope("update", topic, payload_json))

    # -- snapshots -----------------------------------------------------------

    async def _serve_snapshot(self, client: ClientState, topic: str) -> None:
        """Enqueue the current snapshot for ``topic`` on ``client``."""
        try:
            if topic == TOPIC_LIGHTNING:
                payload_json, watermark_s = await self._lightning_payload()
                client.lightning_watermark_s = watermark_s
            else:
                payload_json = await self._cached_state_payload(topic)
        except Exception:
            logger.exception("WebSocket snapshot build failed for topic %s", topic)
            self._enqueue(client, {
                "type": "error",
                "message": "snapshot unavailable: " + topic,
            })
            return
        self._enqueue(client, _payload_envelope("snapshot", topic, payload_json))

    async def _cached_state_payload(self, topic: str) -> str:
        key = await topic_cache_key(topic)
        cached = self._topic_state.get(topic)
        if cached is not None and cached[0] == key:
            return cached[1]

        async with self._topic_lock(topic):
            cached = self._topic_state.get(topic)
            key = await topic_cache_key(topic)
            if cached is not None and cached[0] == key:
                return cached[1]
            payload_json, fingerprint_json = await build_topic_snapshot(topic)
            self._topic_state[topic] = (key, payload_json, fingerprint_json)
            return payload_json

    async def _lightning_payload(self) -> tuple[str, int]:
        store = routes.lightning_store
        version = store.version if store is not None else 0
        cached = self._lightning_cache
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        async with self._lightning_lock:
            store = routes.lightning_store
            version = store.version if store is not None else 0
            cached = self._lightning_cache
            if cached is not None and cached[0] == version:
                return cached[1], cached[2]
            payload_json, watermark_s = await build_lightning_snapshot()
            version = store.version if store is not None else 0
            self._lightning_cache = (version, payload_json, watermark_s)
            return payload_json, watermark_s

    def _topic_lock(self, topic: str) -> asyncio.Lock:
        lock = self._topic_locks.get(topic)
        if lock is None:
            lock = asyncio.Lock()
            self._topic_locks[topic] = lock
        return lock

    # -- receive path --------------------------------------------------------

    async def _receive_loop(self, client: ClientState) -> None:
        while True:
            try:
                text = await client.websocket.receive_text()
            except Exception:
                # WebSocketDisconnect (or any transport failure) ends the
                # connection; the handler's finally does the cleanup.
                return
            try:
                message = json.loads(text)
            except (TypeError, ValueError):
                self._enqueue(client, {
                    "type": "error",
                    "message": "invalid JSON",
                })
                continue
            if not isinstance(message, dict):
                self._enqueue(client, {
                    "type": "error",
                    "message": "message must be a JSON object",
                })
                continue

            action = message.get("action")
            if action == "subscribe":
                await self._handle_subscribe(client, message)
            elif action == "unsubscribe":
                self._handle_unsubscribe(client, message)
            else:
                self._enqueue(client, {
                    "type": "error",
                    "message": "unknown action: " + str(action),
                })

    async def _handle_subscribe(self, client: ClientState, message: dict) -> None:
        requested = self._message_topics(message)
        if requested is None:
            self._enqueue(client, {
                "type": "error",
                "message": "topics must be a list of strings",
            })
            return

        invalid = [topic for topic in requested if topic not in ALL_TOPICS]
        if invalid:
            self._enqueue(client, {
                "type": "error",
                "message": "unknown topic(s): " + ", ".join(invalid),
            })

        unavailable = [
            topic for topic in requested
            if topic in ALL_TOPICS and not topic_available(topic)
        ]
        if unavailable:
            self._enqueue(client, {
                "type": "error",
                "message": "topic unavailable: " + ", ".join(unavailable),
            })

        subscribed = []
        for topic in requested:
            # Dedupe preserving first-occurrence order (the connect path
            # already does), so duplicates cannot produce duplicate acks.
            if (
                topic in ALL_TOPICS
                and topic not in unavailable
                and topic not in subscribed
            ):
                subscribed.append(topic)
        if not subscribed:
            return

        for topic in subscribed:
            client.subscriptions.add(topic)
        self._enqueue(client, {"type": "subscribed", "topics": subscribed})
        for topic in subscribed:
            await self._serve_snapshot(client, topic)

    def _handle_unsubscribe(self, client: ClientState, message: dict) -> None:
        requested = self._message_topics(message)
        if requested is None:
            self._enqueue(client, {
                "type": "error",
                "message": "topics must be a list of strings",
            })
            return
        removed = [topic for topic in requested if topic in client.subscriptions]
        for topic in removed:
            client.subscriptions.discard(topic)
        self._enqueue(client, {"type": "unsubscribed", "topics": removed})

    @staticmethod
    def _message_topics(message: dict) -> list[str] | None:
        """Client-sent topic list; ``None`` when the shape is wrong."""
        topics = message.get("topics")
        if isinstance(topics, list):
            if not all(isinstance(topic, str) for topic in topics):
                return None
            return list(topics)
        topic = message.get("topic")
        if isinstance(topic, str):
            return [topic]
        return None

    # -- delivery ------------------------------------------------------------

    def _broadcast(self, topic: str, text: str) -> None:
        for client in list(self._clients):
            if topic not in client.subscriptions:
                continue
            self._enqueue(client, text)

    def _enqueue(self, client: ClientState, message) -> None:
        """Queue a pre-serialized frame (or an object to serialize)."""
        if client.dropped:
            return
        text = message if isinstance(message, str) else _dumps(message)
        try:
            client.queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.debug("WebSocket client queue full; dropping slow client")
            self._drop_client(client, code=1013)

    async def _sender_loop(self, client: ClientState) -> None:
        while True:
            text = await client.queue.get()
            try:
                await asyncio.wait_for(
                    client.websocket.send_text(text), self._SEND_TIMEOUT_S,
                )
            except Exception:
                logger.debug("WebSocket send failed; dropping client")
                self._drop_client(client)
                return

    def _drop_client(self, client: ClientState, code: int = 1001) -> None:
        """Idempotently unregister ``client`` and tear its connection down."""
        if client.dropped:
            return
        client.dropped = True
        self._clients.discard(client)
        sender = client.sender
        if sender is not None and sender is not asyncio.current_task():
            sender.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._safe_close(client.websocket, code))
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)

    @staticmethod
    async def _safe_close(websocket, code: int) -> None:
        try:
            await websocket.close(code=code)
        except Exception:
            pass

    # -- helpers -------------------------------------------------------------

    def _initial_topics(self, websocket) -> list[str]:
        """Parsed ``?topics=`` request, defaulting to ``weather-maps``."""
        params = getattr(websocket, "query_params", None)
        raw = params.get("topics", "") if params is not None else ""
        requested: list[str] = []
        for name in str(raw or "").split(","):
            name = name.strip()
            if name and name not in requested:
                requested.append(name)
        if not requested:
            return [TOPIC_WEATHER_MAPS]
        return requested

    @staticmethod
    def _origin_allowed(websocket) -> bool:
        """Browser-origin check mirroring the CORS allow-list.

        Allowed when CORS is wide open (``*``), when no Origin header is
        presented (non-browser clients), or when the presented origin is in
        ``LIBREWXR_CORS_ORIGINS``.
        """
        origins = settings.cors_origins
        if "*" in origins:
            return True
        headers = getattr(websocket, "headers", None)
        origin = headers.get("origin") if headers is not None else None
        if not origin:
            return True
        return origin in origins
