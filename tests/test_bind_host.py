# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Bind-host resolution for uvicorn.run.

uvicorn binds the shared socket itself when ``workers > 1`` and crashes
on ``host=None`` (``sock.bind((None, port))``), which put the Docker
renderer (16 workers by default) in a restart loop.
"""
from __future__ import annotations

import socket

from librewxr import main


def test_explicit_host_is_kept():
    assert main._resolve_bind_host("0.0.0.0", 16) == "0.0.0.0"
    assert main._resolve_bind_host("127.0.0.1", 1) == "127.0.0.1"


def test_single_worker_keeps_none_for_asyncio_dual_stack():
    assert main._resolve_bind_host(None, 1) is None


def test_multi_worker_unset_host_uses_ipv6_wildcard():
    assert main._resolve_bind_host(None, 16) == "::"


def test_multi_worker_falls_back_to_ipv4_without_ipv6(monkeypatch):
    real_socket = socket.socket

    def fake_socket(family=-1, *args, **kwargs):
        if family == socket.AF_INET6:
            raise OSError("Address family not supported by protocol")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(main.socket, "socket", fake_socket)
    assert main._resolve_bind_host(None, 16) == "0.0.0.0"


def test_resolved_host_is_bindable_by_uvicorn():
    from uvicorn.config import Config

    config = Config("librewxr.main:app", host=main._resolve_bind_host(None, 2), port=0)
    sock = config.bind_socket()
    sock.close()
