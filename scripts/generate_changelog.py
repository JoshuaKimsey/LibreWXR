#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Regenerate ``CHANGELOG.md`` from the published GitHub releases.

The GitHub release notes are the single source of truth for LibreWXR;
``CHANGELOG.md`` is a generated, read-only view of them. Never edit the
generated file by hand - edit the release on GitHub instead, then re-run:

    .venv/bin/python scripts/generate_changelog.py

The script reads every published release from the GitHub REST API (paginating
at 100 per page), sorts them newest-first by ``published_at``, and rewrites
``CHANGELOG.md`` in the repository root. Output is deterministic: the same API
response produces the same file bytes, so re-running after a release appends
one new section and changes nothing else.

Exit status is non-zero (and ``CHANGELOG.md`` is left untouched) when the API
request fails, including when the unauthenticated rate limit is exhausted.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "CHANGELOG.md"

RELEASES_URL = "https://api.github.com/repos/JoshuaKimsey/LibreWXR/releases"
PAGE_SIZE = 100
REQUEST_TIMEOUT = 30.0
# The GitHub API rejects requests without a User-Agent header.
USER_AGENT = "librewxr-changelog-generator"

HEADER = """# Changelog

All notable changes to LibreWXR, newest first. This file is generated from the
published GitHub releases and must not be edited by hand - regenerate it with
`.venv/bin/python scripts/generate_changelog.py` after publishing a release,
and commit the result alongside the version bump.
"""


def fetch_releases() -> list[dict]:
    """Fetch every published release from the GitHub REST API."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    releases: list[dict] = []
    page = 1
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers) as client:
            while True:
                response = client.get(
                    RELEASES_URL,
                    params={"per_page": PAGE_SIZE, "page": page},
                )
                if response.status_code == 403:
                    raise SystemExit(
                        "GitHub API request failed with HTTP 403 (forbidden). "
                        "This is almost certainly the unauthenticated rate limit "
                        "(60 requests/hour per IP) - wait for it to reset, then retry."
                    )
                response.raise_for_status()
                batch = response.json()
                releases.extend(batch)
                if len(batch) < PAGE_SIZE:
                    break
                page += 1
    except httpx.HTTPStatusError as exc:
        raise SystemExit(
            f"GitHub API request failed: HTTP {exc.response.status_code}. "
            "The unauthenticated API is rate-limited to 60 requests/hour per IP; "
            "if you hit that, wait for the limit to reset and retry."
        ) from exc
    except httpx.RequestError as exc:
        raise SystemExit(f"GitHub API request failed: {exc}") from exc
    # Draft releases are not published and do not belong in the changelog.
    return [release for release in releases if not release.get("draft")]


def format_date(published_at: str) -> str:
    """Render an API timestamp (``2026-10-10T00:59:56Z``) as ``2026-10-10``."""
    parsed = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    return parsed.strftime("%Y-%m-%d")


def render_document(releases: list[dict]) -> str:
    """Render the changelog text, newest release first."""
    ordered = sorted(
        releases,
        key=lambda rel: (rel.get("published_at") or "", rel.get("tag_name") or ""),
        reverse=True,
    )
    sections = [HEADER.rstrip("\n")]
    for rel in ordered:
        title = rel.get("name") or rel.get("tag_name") or ""
        url = rel.get("html_url") or ""
        date = format_date(rel.get("published_at") or "")
        # The body markdown is rendered verbatim in content, but its line
        # endings are normalized to LF (CRLF first, then any lone CR) so the
        # generated file matches the repo line-ending conventions. Trailing
        # newlines are then trimmed only so section spacing stays uniform.
        body = (
            (rel.get("body") or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .rstrip("\n")
        )
        if not body.strip():
            body = "(no release notes)"
        sections.append(f"---\n\n## [{title}]({url})\n\n{date}\n\n{body}")
    return "\n\n".join(sections) + "\n"


def main() -> int:
    releases = fetch_releases()
    if not releases:
        print("No published releases returned by the GitHub API - nothing to write.")
        return 1
    document = render_document(releases)
    # newline="\n" keeps the emitted bytes identical on every platform.
    OUTPUT.write_text(document, encoding="utf-8", newline="\n")
    noun = "release" if len(releases) == 1 else "releases"
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(releases)} {noun}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
