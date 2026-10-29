# LibreWXR Roadmap

This file tracks the path to version 1.0.0 and beyond. Each major addition ships as its own
release rather than in a single batch, but releases are not numbered ahead of time - work
happens in whatever order makes sense, and the version a feature lands in gets decided when
it ships. Checkboxes reflect current progress, unchecked items are candidates, and what
actually ships is recorded in CHANGELOG.md (generated from published GitHub releases). Each
actionable section also references its GitHub tracking issue (#NN) - check there for live
status, as this file and the issues are kept in sync. The lists are not exhaustive.

## The path to 1.0.0

The current version is 0.6.3. Every section below is an unordered theme: each ships as its
own 0.x release whenever it is next in line, with the version number assigned at release
time.

### 0.6.0 - Container image distribution

Tracked in issue #44.

- [x] Add a release-tagged GitHub Actions workflow building multi-arch images
  (linux/amd64 + linux/arm64) with docker/build-push-action.
- [x] Publish to GitHub Container Registry as the primary registry
  (`ghcr.io/joshuakimsey/librewxr`) - free, tied to this repository, OIDC-based with no
  extra stored secrets.
- [x] Optionally mirror to Docker Hub for broader discoverability; the two can coexist -
  decide once during this release. (decided during 0.6.0: GHCR only for now; the two can
  coexist later if there is demand)
- [x] Tag scheme: exact semver tags (e.g. `0.6.0`), `latest`, and a commit-SHA tag.
Note: images ship routinely for every future release from here on, so the
  publish pipeline is exercised many times before 1.0.
- [x] Extend the README quickstart with `docker pull` / `docker compose pull` instructions
  alongside the existing build-from-source flow.
- [x] Add standard OCI image labels (source, license AGPL-3.0-or-later, description).
- [ ] Retire the git-pull + source-rebuild self-update flow in `scripts/auto-update.sh` once
  published images exist: self-updating becomes `docker compose pull` + `up -d` against
  image tags instead of `git pull` + rebuild. Replace rather than patch; do not remove the
  old script until 0.6.0 images are live so source-based deployments keep a working update
  path in the meantime. (scripts/update.sh shipped in 0.6.0; removing the legacy script is
  deferred to a later release)
- [x] Carry over the one genuinely useful piece: `.env` vs `.env.example` reconciliation
  (append missing `LIBREWXR_*` vars with their explanatory comments, write `.env.bak`
  first, never modify existing values), then sweep references to the old flow (AGENTS.md,
  README, docs).

### Along the way - MCP Registry listing (hosted api.librewxr.net)

Tracked in issue #45.

A parallel workstream, not a feature release - it is mostly ops plus a small name-constant
alignment, so it can land alongside any release.

- [ ] Execute the runbook `mcp-registry-publishing.md` (repo root): pre-publish checklist,
  server.json template, validation rules, and source links all live there; re-verify the
  linked registry docs before running anything (registry is in PREVIEW, API frozen at v0.1).
- [ ] Decide the namespace: expected path is DNS verification (`net.librewxr/*` - apex TXT
  record + `mcp-publisher login dns`); the GitHub namespace is treated as unusable for the
  `api.librewxr.net` remote under the domain-matching rule.
- [ ] Create `server.json` at the repo root per the runbook template; `version` tracks
  `pyproject.toml` at publish time (0.6.3 today - re-check when executing).
- [ ] Publish the hosted instance; published `(name, version)` pairs are immutable, so the
  registry is only touched on real `pyproject.toml` version bumps.
- [ ] Follow-up: align the SEP-2127 server card name with the chosen namespace - one
  constant `_NAME` in `src/librewxr/mcp/discovery.py`, the matching constant in
  `tests/test_mcp_discovery.py`, and the example JSON in `docs/mcp-server.md` (currently
  `io.github.joshuakimsey/librewxr-mcp`).
- [ ] Optional: tag-triggered publish workflow `.github/workflows/publish-mcp.yml` (see
  runbook; reconcile OIDC/GitHub-namespace auth with the DNS domain-matching requirement
  before relying on it - a DNS-secret variant exists).

### Along the way - Landing site refresh (librewxr-site/)

Tracked in issue #46.

A parallel workstream like the MCP registry listing: polish the gitignored landing site so
it is visually appealing and eye-catching rather than purely utilitarian, ready before the
1.0 announcement.

- [ ] Refresh the visual design of `librewxr-site/` (gitignored, outside version control):
  better graphical polish, more visually appealing and eye-catching instead of purely
  utilitarian.
- [ ] Aim to land the refresh before the 1.0 announcement so the landing site matches the
  public launch (Docker images, README quickstart).

### External color palettes

Tracked in issue #47.

- [ ] Add `LIBREWXR_COLOR_SCHEMES_FILE`: load scheme definitions from an external CSV in the
  existing `color_table.csv` format, merged over the built-in schemes (built-ins currently
  load at runtime from package data in `src/librewxr/colors/schemes.py`, so this is an
  extension, not a rewrite).
- [ ] Optional hot reload (stat the file's mtime) so palette updates need no backend restart
  or version bump.
- [ ] Design constraints: LUT caches are built lazily and cached process-wide, and
  colorization happens in the present-time tile tail - a palette change must invalidate LUT
  caches and be fingerprinted into tile-cache and shared-tile-store keys; the geometry cache
  is unaffected thanks to the compute/present split.
- [ ] Keep `docs-check.yml` green: regenerate scheme docs/previews via
  `scripts/generate_scheme_docs.py` + `scripts/generate_color_scheme_previews.py` when the
  format or defaults change.

### Health dashboard

Tracked in issue #48.

- [ ] Ship a static dashboard (e.g. served at `/dashboard`) rendering the existing `/health`
  payload: cluster section (per-worker RSS, hit ratios, cgroup split), per-NWP-grid health,
  sources, frames, tile cache.
- [ ] Surface "most loaded areas" from the `tile_requests` hot-tile counters already
  collected by the request tracker.
- [ ] Env-toggleable; document reverse-proxy auth instead of hand-rolling authentication.
- [ ] Optional companion in this release: Prometheus `/metrics` export derived from
  `/health` for users who already run Grafana.

### WebSocket support

Tracked in issue #49.

- [x] v1 scope: a WebSocket endpoint that pushes the weather-maps.json payload when the
  frame list changes, built on the snapshot change-detection the render workers already use.
- [x] HTTP JSON API remains canonical for Rain Viewer compatibility; WebSocket is additive,
  never a replacement.
- [x] Document in `docs/web-integration-guide.md`.

Note: the alerts, storm-cell, and lightning topics originally deferred to post-1.0 (#52)
were pulled forward and shipped as part of this release.

### 1.0.0 - Stability, split-mode docs, and the freeze

Tracked in issue #50.

- [ ] Split mode, phase 1: support and document render workers on separate hosts mounting one
  shared filesystem (NFS or object-store-backed volume), including memmap-over-network
  caveats; update `docs/self-host-sizing.md`. The full network-separated pipeline/renderer is
  a post-1.0 design project - today's multi-mode and this phase share the same
  shared-filesystem assumption.
- [ ] Declare the public surface stable at 1.0.0: Rain Viewer-compatible HTTP endpoints, tile
  query parameters, JSON response shapes, and MCP tools. Semver applies to this surface;
  internal paths may change.
- [ ] Documentation pass: `docs/configuration-reference.md` covers every knob added across
  the pre-1.0 feature releases.
- [ ] Cut 1.0.0 per repo conventions: `gh release create`, then
  `scripts/generate_changelog.py` and commit the regenerated `CHANGELOG.md` with the version
  bump; GHCR images built from the release tag; MCP registry republish at 1.0.0 (registry
  version tracks `pyproject.toml`).

## Post-1.0 (1.1+ candidates)

Unordered candidates; nothing here is committed.

### Network split mode (true "split")

Tracked in issue #51.

- [ ] Design spike first: replace the shared-filesystem handoff with a network artifact
  handoff so the pipeline and renderers can run on entirely separate servers (candidates:
  object storage or an HTTP snapshot service the renderers pull from).

Note: this redesigns the memmap + shared-store coupling - state.json,
`coord/`, `tiles_shared/`, `mask/`, `lightning/`, `radar/`, `workers/` all assume a shared
`cache_dir` today.

### WebSocket topic extensions

Tracked in issue #52.

- [x] Alerts, storm-cell, and lightning streams as additional topics alongside the
  weather-maps channel.

Pulled forward from the post-1.0 list and shipped alongside the initial WebSocket channel
(see #49): all three topics are live in `docs/web-integration-guide.md`.

### Dashboard v2

Tracked in issue #53.

- [ ] Historical views: ring-buffered health snapshots for time-series trends and per-region
  load views.

### Palette ecosystem

Tracked in issue #54.

- [ ] First-class support for community palettes: external-palette documentation and preview
  generation for user-supplied CSVs.

### Your ideas here

- [ ] Placeholder: community feedback items TBD.
