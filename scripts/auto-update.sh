#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
#
# LibreWXR auto-update helper.
#
# Checks the tracked branch for new commits and, if any exist, pulls them
# and rebuilds the running docker compose stack. Safe to run from cron or
# a systemd timer — exits quietly when there is nothing to do, uses an
# flock to prevent concurrent runs, and refuses to touch a dirty working
# tree.
#
# Safety: this script is a no-op unless the host explicitly opts in by
# creating a sentinel file at ${REPO_DIR}/.auto-update-enabled (or by
# setting LIBREWXR_AUTO_UPDATE=1 in the environment). The sentinel is in
# .gitignore so cloning the repo on your development machine will *not*
# enable auto-updates — you have to `touch .auto-update-enabled` on the
# production host you actually want to run this on.
#
# Usage:
#   scripts/auto-update.sh              # run once (requires opt-in)
#   scripts/auto-update.sh --dry-run    # check for updates without applying
#                                       # (safe, runs without opt-in)
#
# Enable on a production host:
#   touch /path/to/LibreWXR/.auto-update-enabled
#
# Cron example (hourly, logging to /var/log/librewxr-update.log):
#   0 * * * * /path/to/LibreWXR/scripts/auto-update.sh \
#             >> /var/log/librewxr-update.log 2>&1

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# Resolve the repo root from the script location so this works no matter
# where cron invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

# Opt-in guard. Refuse to do anything (except --dry-run, which only reads)
# unless the host has explicitly enabled auto-updates.
SENTINEL="${REPO_DIR}/.auto-update-enabled"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${SENTINEL}" && "${LIBREWXR_AUTO_UPDATE:-0}" != "1" ]]; then
        log "auto-update not enabled on this host (no ${SENTINEL}, and"
        log "LIBREWXR_AUTO_UPDATE is not set). Exiting without action."
        log "To enable on a production host: touch ${SENTINEL}"
        exit 0
    fi
fi

# Prevent concurrent runs (useful if an update takes longer than the
# cron interval).
LOCK_FILE="${TMPDIR:-/tmp}/librewxr-auto-update.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "another auto-update is already running; exiting"
    exit 0
fi

if [[ ! -d .git ]]; then
    log "error: ${REPO_DIR} is not a git checkout"
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == "HEAD" ]]; then
    log "error: repo is in detached HEAD state, refusing to update"
    exit 1
fi

# Refuse to update if the working tree has local modifications. Users
# hacking on the code should not have their changes clobbered by a cron
# job. (Dry-run is a harmless query, so we allow it on a dirty tree.)
if [[ "${DRY_RUN}" -eq 0 ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log "working tree is dirty; refusing to update"
        exit 1
    fi
fi

log "fetching origin for branch ${BRANCH}"
git fetch --quiet origin "${BRANCH}"

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/${BRANCH}")"

if [[ "${LOCAL}" == "${REMOTE}" ]]; then
    log "already up to date (${LOCAL:0:7})"
    exit 0
fi

# Only act when local is strictly behind remote (remote contains all of
# local's commits, plus more). If local is ahead or diverged, something
# unusual is going on and we should stay out of the way.
if ! git merge-base --is-ancestor "${LOCAL}" "${REMOTE}"; then
    log "local (${LOCAL:0:7}) is not an ancestor of origin/${BRANCH} (${REMOTE:0:7});"
    log "refusing to update (branch is ahead or has diverged)"
    exit 0
fi

log "update available: ${LOCAL:0:7} -> ${REMOTE:0:7}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "dry-run mode, not pulling"
    exit 0
fi

# Pick the docker compose command variant that's available.
if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    log "error: neither 'docker compose' nor 'docker-compose' is available"
    exit 1
fi

log "pulling ${BRANCH}"
git pull --ff-only --quiet origin "${BRANCH}"

log "rebuilding and restarting stack"
"${COMPOSE[@]}" up -d --build

log "update complete: now at $(git rev-parse --short HEAD)"
