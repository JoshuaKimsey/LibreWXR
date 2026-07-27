# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP (Model Context Protocol) server for LibreWRX.

Exposes precipitation-nowcast and weather-alert query tools over both
an in-process HTTP transport (mounted inside the FastAPI app) and a
standalone stdio transport for local agents.
"""
