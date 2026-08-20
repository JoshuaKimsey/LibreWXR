# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Enable ``python -m librewxr.mcp`` for the standalone stdio transport."""

from librewxr.mcp.server import main

if __name__ == "__main__":
    main()