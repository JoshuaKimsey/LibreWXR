# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Runtime package version lookup.

``pyproject.toml`` is the single source of truth for the LibreWXR
version.  Code never hardcodes it: this module asks the installed
distribution metadata (which the build backend generates from
``pyproject.toml``) and only falls back to a constant when running from a
bare source tree with no installed metadata at all.
"""

from importlib.metadata import PackageNotFoundError, version

# Kept in sync with ``pyproject.toml`` by ``scripts/check_versions.py``
# (enforced in CI by ``.github/workflows/version-check.yml``).
FALLBACK_VERSION = "0.7.0"


def package_version() -> str:
    """Return the installed ``librewxr`` distribution version.

    Falls back to ``FALLBACK_VERSION`` when the package metadata is
    unavailable, e.g. running directly from a bare source tree.  In an
    editable dev install the version comes from the installed distribution
    metadata, so reinstall (``pip install -e .``) after bumping
    ``pyproject.toml``; Docker and runtime builds always pick it up from the
    fresh build.
    """
    try:
        return version("librewxr")
    except PackageNotFoundError:
        return FALLBACK_VERSION
