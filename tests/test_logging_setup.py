# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the WARNING+ rotating log file (LIBREWXR_LOG_FILE).

``setup_logging`` uses ``force=True``, which wipes whatever handlers pytest
or a previous test left on the root logger, so each test ends by calling
``setup_logging(log_file="")`` to restore a console-only root handler for
the rest of the suite.  A bare ``setup_logging()`` is never used in
teardown because the log file is enabled by default now (``logs/
librewxr.log``) - it would create ``./logs`` in the current working
directory.  Likewise every test drives the log path through explicit
args, ``monkeypatch.setenv``, or ``monkeypatch.setattr`` on the settings
object with a ``tmp_path`` target so nothing ever writes into the repo
root.

Resolution order under test: explicit argument > live env var >
``settings.log_file`` (the .env plumbing).
"""

import logging
import logging.handlers

from librewxr.config import settings
from librewxr.logging_setup import _WebSocketLifecycleFilter, setup_logging

logger = logging.getLogger("librewxr.test_logging_setup")


def _root_handlers() -> list[logging.Handler]:
    return logging.getLogger().handlers


def _record(msg: str, *args: object, level: int = logging.INFO) -> logging.LogRecord:
    """Build an ``uvicorn.error`` record, optionally with %-args like uvicorn."""
    return logging.LogRecord(
        name="uvicorn.error",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def _file_handlers() -> list[logging.Handler]:
    return [
        h
        for h in _root_handlers()
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]


def test_explicit_empty_disables_file(monkeypatch, tmp_path):
    # Explicit log_file="" disables the file even though the default is
    # enabled - exactly one (Rich) root handler.
    monkeypatch.delenv("LIBREWXR_LOG_FILE", raising=False)
    setup_logging(log_file="")
    try:
        assert len(_root_handlers()) == 1
        assert _file_handlers() == []
    finally:
        setup_logging(log_file="")


def test_empty_env_var_disables_file(monkeypatch, tmp_path):
    # LIBREWXR_LOG_FILE="" set in the environment disables the file: the
    # empty live env value takes precedence over the enabled default.
    monkeypatch.setenv("LIBREWXR_LOG_FILE", "")
    setup_logging()
    try:
        assert len(_root_handlers()) == 1
        assert _file_handlers() == []
    finally:
        setup_logging(log_file="")


def test_explicit_arg_beats_env_var(monkeypatch, tmp_path):
    # Explicit argument is the top of the resolution order: even with
    # LIBREWXR_LOG_FILE set, an explicit log_file="" disables the file.
    monkeypatch.setenv("LIBREWXR_LOG_FILE", str(tmp_path / "ignored.log"))
    setup_logging(log_file="")
    try:
        assert len(_root_handlers()) == 1
        assert _file_handlers() == []
    finally:
        setup_logging(log_file="")


def test_env_var_enables_warning_file_handler(monkeypatch, tmp_path):
    log_path = tmp_path / "t.log"
    monkeypatch.setenv("LIBREWXR_LOG_FILE", str(log_path))
    setup_logging()
    try:
        handlers = _root_handlers()
        assert len(handlers) == 2
        file_handlers = _file_handlers()
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.WARNING
        # delay=True only creates the file on first emit, and the stream
        # buffers, so emit + flush before asserting the path was used.
        logger.warning("probe")
        file_handlers[0].flush()
        assert log_path.exists()
    finally:
        setup_logging(log_file="")


def test_settings_fallback_used(monkeypatch, tmp_path):
    # With LIBREWXR_LOG_FILE unset in the environment, the settings
    # object (the .env plumbing) provides the path.
    monkeypatch.delenv("LIBREWXR_LOG_FILE", raising=False)
    settings_path = tmp_path / "s.log"
    monkeypatch.setattr(settings, "log_file", str(settings_path))
    setup_logging()
    try:
        handlers = _root_handlers()
        assert len(handlers) == 2
        file_handlers = _file_handlers()
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.WARNING
        # delay=True only creates the file on first emit, and the stream
        # buffers, so emit + flush before asserting the path was used.
        logger.warning("probe")
        file_handlers[0].flush()
        assert settings_path.exists()
    finally:
        setup_logging(log_file="")


def test_file_captures_warning_and_traceback_not_info(monkeypatch, tmp_path):
    monkeypatch.delenv("LIBREWXR_LOG_FILE", raising=False)
    log_path = tmp_path / "test.log"
    setup_logging(log_file=str(log_path))
    try:
        logger.info("informational line")
        logger.warning("something went sideways")
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("exceptional failure")
        file_handlers = _file_handlers()
        assert file_handlers, "no RotatingFileHandler on the root logger"
        # delay=True only creates the file on first emit, and the stream
        # buffers, so flush before reading.
        file_handlers[0].flush()
        text = log_path.read_text(encoding="utf-8")
        assert "something went sideways" in text
        assert "exceptional failure" in text
        assert "ValueError: boom" in text
        assert "Traceback (most recent call last)" in text
        assert "informational line" not in text
    finally:
        setup_logging(log_file="")


# --- WebSocket lifecycle chatter -------------------------------------------
#
# uvicorn logs every WebSocket accept / reject / close at INFO on
# ``uvicorn.error`` (HTTP connection lifecycle is TRACE, so WS is the
# outlier), and the ``websockets`` library adds its own "connection open"
# / "connection closed" INFO records - both are routine per-client
# traffic and get filtered.  Everything else on that logger must survive.


def test_websocket_lifecycle_filter_drops_per_connection_records():
    ws_filter = _WebSocketLifecycleFilter()
    dropped = [
        # uvicorn's per-connection lines (plain and %-arg templated; the
        # latter is what the sansio impl actually emits).
        '176.225.29.246:0 - "WebSocket /v2/ws?topics=weather-maps" [accepted]',
        '176.225.29.246:0 - "WebSocket /v2/ws" 403',
        '176.225.29.246:0 - "WebSocket /v2/ws" 101',
        _record('%s - "WebSocket %s" [accepted]', "176.225.29.246:0", "/v2/ws"),
        # websockets library lifecycle records.
        "connection open",
        "connection closed",
        # websockets library rejection paths.
        "connection rejected (400 Bad Request)",
        "HTTP response sent (403 Forbidden)",
    ]
    for item in dropped:
        record = item if isinstance(item, logging.LogRecord) else _record(item)
        assert ws_filter.filter(record) is False, f"should drop: {record.getMessage()!r}"


def test_websocket_lifecycle_filter_passes_server_lifecycle_and_errors():
    ws_filter = _WebSocketLifecycleFilter()
    passed = [
        # uvicorn startup / shutdown INFO - the reason this is a filter and
        # not setLevel(WARNING) on uvicorn.error.
        "Started server process [123]",
        "Waiting for application startup.",
        "Application startup complete.",
        "Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)",
        "Shutting down",
        "Application shutdown complete.",
        "Finished server process [123]",
        # Errors / warnings / tracebacks from the handshake and ASGI layers.
        "opening handshake failed",
        "Exception in ASGI application",
        "Invalid HTTP request received.",
    ]
    for message in passed:
        for level in (logging.INFO, logging.WARNING, logging.ERROR):
            record = _record(message, level=level)
            assert ws_filter.filter(record) is True, (
                f"should pass: {message!r} at {logging.getLevelName(level)}"
            )


def test_setup_logging_attaches_ws_filter_exactly_once():
    # Repeated setup_logging() calls (main.py at module scope, then a
    # worker subprocess import, then a test's teardown) must never stack
    # duplicate filters.
    setup_logging(log_file="")
    try:
        setup_logging(log_file="")
        uvicorn_error = logging.getLogger("uvicorn.error")
        ws_filters = [
            f for f in uvicorn_error.filters if isinstance(f, _WebSocketLifecycleFilter)
        ]
        assert len(ws_filters) == 1
    finally:
        setup_logging(log_file="")


def test_ws_chatter_silenced_but_startup_info_captured(caplog):
    setup_logging(log_file="")
    # logging.basicConfig(force=True) inside setup_logging wipes every root
    # handler, including the one pytest installed to back caplog, so
    # re-attach it before exercising the logger.
    logging.getLogger().addHandler(caplog.handler)
    try:
        uvicorn_error = logging.getLogger("uvicorn.error")
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            caplog.clear()
            uvicorn_error.info("connection open")
            uvicorn_error.info("connection closed")
            uvicorn_error.info('1.2.3.4:0 - "WebSocket /v2/ws" [accepted]')
            uvicorn_error.info("Started server process [1]")
        messages = [r.getMessage() for r in caplog.records]
        # A logger-level filter blocks the record before it reaches any
        # handler, so caplog only ever sees what passed the filter.
        assert "Started server process [1]" in messages
        assert messages == ["Started server process [1]"]
    finally:
        setup_logging(log_file="")
