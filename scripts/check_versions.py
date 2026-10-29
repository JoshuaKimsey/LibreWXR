#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Enforce a single source of truth for the LibreWXR version.

``pyproject.toml`` holds the version.  Every other place where a version
number is written down - the runtime fallback constant, one MCP discovery
test, the roadmap prose, the server-card doc example - must agree with it,
and docs must never pin a numbered image tag (a pinned example goes stale
with every release).  CI runs this via
``.github/workflows/version-check.yml``, and ``docker-publish.yml`` also
runs it before building a release image.

Usage:

    .venv/bin/python scripts/check_versions.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = r"\d+\.\d+\.\d+"


@dataclass(frozen=True)
class Rule:
    """One consistency check: a regex applied to one or more files.

    Modes:

    - ``all``        every match of ``pattern`` (group ``v``) equals the version
    - ``one``        exactly one match of ``pattern`` (group ``v``), equal to it
    - ``forbid``     ``pattern`` never matches
    - ``forbid-line`` no line containing ``marker`` also contains a version
    """

    label: str
    paths: tuple[str, ...]
    mode: str
    pattern: str = ""
    marker: str = ""
    expected: str = "{version}"


RULES: list[Rule] = [
    Rule("version.py fallback", ("src/librewxr/version.py",), "all",
         pattern=rf'"(?P<v>{SEMVER})"'),
    Rule("mcp discovery test fallback", ("tests/test_mcp_discovery.py",), "one",
         pattern=rf'expected_version\s*=\s*"(?P<v>{SEMVER})"'),
    Rule("roadmap current version", ("ROADMAP.md",), "one",
         pattern=rf"The current version is (?P<v>{SEMVER})\."),
    Rule("roadmap publish-time version", ("ROADMAP.md",), "one",
         pattern=rf"at publish time \((?P<v>{SEMVER}) today"),
    Rule("MCP server card version", ("docs/mcp-server.md",), "all",
         pattern=rf'"version":\s*"(?P<v>{SEMVER})"'),
    Rule("README GHCR tag", ("README.md",), "forbid",
         pattern=r"ghcr\.io/joshuakimsey/librewxr:\d[\w.\-]*",
         expected="no numbered image tag"),
    Rule("README image tag", ("README.md",), "forbid-line",
         marker="LIBREWXR_IMAGE_TAG", expected="no version number"),
    Rule("image tag docs", (".env.example", "docs/configuration-reference.md"),
         "forbid-line", marker="LIBREWXR_IMAGE_TAG", expected="no version number"),
]


def pyproject_version() -> str:
    """Read ``[project].version`` from ``pyproject.toml``."""
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_rule(rule: Rule, version: str) -> list[tuple[str, int, str, str]]:
    """Return ``(path, line, found, expected)`` tuples for one rule."""
    expected = rule.expected.format(version=version)
    violations: list[tuple[str, int, str, str]] = []
    for rel in rule.paths:
        path = ROOT / rel
        if not path.exists():
            violations.append((rel, 1, "missing file", expected))
            continue
        text = path.read_text(encoding="utf-8")

        if rule.mode == "forbid-line":
            for lineno, line in enumerate(text.splitlines(), 1):
                if rule.marker in line:
                    hit = re.search(SEMVER, line)
                    if hit is not None:
                        violations.append((rel, lineno, hit.group(0), expected))
            continue

        matches = list(re.finditer(rule.pattern, text, re.MULTILINE))

        if rule.mode == "forbid":
            for match in matches:
                violations.append(
                    (rel, _line_of(text, match.start()), match.group(0), expected)
                )
            continue

        if rule.mode == "one" and len(matches) != 1:
            violations.append(
                (
                    rel,
                    1,
                    f"{len(matches)} occurrences of {rule.label}",
                    expected,
                )
            )
            continue

        if not matches:
            violations.append((rel, 1, f"missing {rule.label}", expected))
            continue

        for match in matches:
            found = match.group("v")
            if found != version:
                violations.append(
                    (rel, _line_of(text, match.start()), found, expected)
                )
    return violations


def main() -> int:
    version = pyproject_version()
    violations: list[tuple[str, int, str, str]] = []
    for rule in RULES:
        violations.extend(check_rule(rule, version))

    if violations:
        for path, lineno, found, expected in violations:
            print(f"{path}:{lineno}: {found} (expected {expected})")
        print(
            f"version check: FAILED (pyproject {version}, "
            f"{len(violations)} violation(s) across {len(RULES)} rules)",
            file=sys.stderr,
        )
        return 1

    print(f"version check: OK (pyproject {version}, {len(RULES)} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
