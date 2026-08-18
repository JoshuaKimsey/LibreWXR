# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import hashlib

from fastapi import Request, Response


def compute_etag(data: bytes) -> str:
    """Compute a strong quoted ETag from the SHA-256 digest of ``data``."""
    return '"' + hashlib.sha256(data).hexdigest()[:16] + '"'


def parse_if_none_match(header: str | None) -> tuple[set[str], bool]:
    """Parse an If-None-Match header into (normalized tag set, wildcard flag).

    ``*`` short-circuits the whole parse.  ``W/`` prefixes are stripped;
    surrounding quotes are kept as-is.
    """
    if not header:
        return (set(), False)
    tags: set[str] = set()
    for token in header.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "*":
            return (set(), True)
        if token.startswith("W/"):
            token = token[2:]
        tags.add(token)
    return (tags, False)


def etag_matches(etag: str, inm: tuple[set[str], bool]) -> bool:
    """Return True if ``etag`` satisfies the parsed If-None-Match condition.

    Weak and strong tags compare equivalently here because ``W/`` prefixes
    were already stripped into the set by :func:`parse_if_none_match`.
    """
    if inm[1]:
        return True
    return etag in inm[0]


def conditional_response(
    *,
    request: Request,
    body: bytes,
    etag: str,
    content_type: str,
    max_age: int,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    """Build a 304 or 200 response honoring If-None-Match.

    When the request's If-None-Match header matches ``etag`` a bodyless
    304 is returned; otherwise the full ``body`` is served with the ETag.
    ``extra_headers``, when given, are merged into the response headers
    and apply to both the 304 and the 200 branch.
    """
    inm_header = request.headers.get("if-none-match")
    inm = parse_if_none_match(inm_header)
    headers = {"ETag": etag, "Cache-Control": f"public, max-age={max_age}"}
    if extra_headers:
        headers.update(extra_headers)
    if etag_matches(etag, inm):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type=content_type, headers=headers)
