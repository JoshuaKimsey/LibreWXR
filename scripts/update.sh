#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
#
# LibreWXR image-based update helper.
#
# Image-mode successor to scripts/auto-update.sh. Instead of rebuilding the
# compose stack from source, this pulls the published container images from
# ghcr.io/joshuakimsey/librewxr and restarts the stack. The git checkout is
# still used for docker-compose.yml, .env.example and the scripts, so the
# git rails below are unchanged: fast-forward-only pulls, dirty-tree
# refusal, and the .env reconciliation step.
#
# Safe to run from cron or a systemd timer -- exits quietly when there is
# nothing to do, uses an flock to prevent concurrent runs, and refuses to
# touch a dirty working tree.
#
# Safety: like auto-update.sh, the default mode is a no-op unless the host
# explicitly opts in by creating a sentinel file at
# ${REPO_DIR}/.auto-update-enabled (or by setting LIBREWXR_AUTO_UPDATE=1 in
# the environment). The sentinel name, the environment variable and the
# flock path are shared with auto-update.sh, so existing cron entries and
# the .gitignore entry keep working unchanged. The sentinel is in
# .gitignore so cloning the repo on your development machine will *not*
# enable auto-updates -- you have to `touch .auto-update-enabled` on the
# production host you actually want to run this on.
#
# Usage:
#   scripts/update.sh              # pull the latest images and restart
#                                  # (requires opt-in)
#   scripts/update.sh --dry-run    # report what would happen, mutate nothing
#                                  # (runs without opt-in)
#   scripts/update.sh --env        # reconcile .env with .env.example
#                                  # (no git, no docker)
#   scripts/update.sh --build      # legacy source-build path for deployments
#                                  # that build from this checkout:
#                                  # `up -d --build`, no image pull
#   scripts/update.sh --help       # print this usage summary
#
# --dry-run, --env and --build are manual-invocation flags and bypass the
# opt-in sentinel -- they only do what you explicitly asked for. They still
# respect the concurrent-run flock so they can't race with a cron-driven
# full update.
#
# Enable on a production host:
#   touch /path/to/LibreWXR/.auto-update-enabled
#
# Cron example (hourly, logging to /var/log/librewxr-update.log):
#   0 * * * * /path/to/LibreWXR/scripts/update.sh \
#             >> /var/log/librewxr-update.log 2>&1
#
# Note: if you need to customise docker-compose.yml (e.g. memory limits),
# set the corresponding LIBREWXR_* variables in .env instead of editing
# the file directly -- the updater refuses to run on a dirty working tree,
# and .env is already in .gitignore.
#
# After a successful pull, the script also reconciles your .env against
# .env.example: any LIBREWXR_* variables that have appeared upstream but
# are missing locally get appended (with their default value and the
# explanatory comments from .env.example) so you don't have to manually
# track new settings between releases. Existing values are never touched,
# and a backup is written to .env.bak before any modification. Set
# LIBREWXR_AUTO_ENV_SYNC=0 to disable this step.

set -euo pipefail

usage() {
    cat <<'EOF'
usage: update.sh [MODE]

Modes:
  (none)      Pull the latest images from ghcr.io/joshuakimsey/librewxr and
              restart the compose stack. Requires the opt-in sentinel
              (.auto-update-enabled) or LIBREWXR_AUTO_UPDATE=1.
  --dry-run   Check upstream for new commits and report what would happen.
              Mutates nothing; runs without opt-in.
  --env       Reconcile .env with .env.example (append missing variables with
              their comments). No git, no docker.
  --build     Legacy source-build path for deployments that build the stack
              from this checkout: git sync + env sync, then `up -d --build`.
              No image pull.
  --help      Print this message.
EOF
}

MODE=update
case "${1:-}" in
    --dry-run) MODE=dry-run ;;
    --env)     MODE=env-sync ;;
    --build)   MODE=build ;;
    --help|-h)
        usage
        exit 0
        ;;
    "")        ;;
    *)
        printf 'unknown flag: %s\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac

# Resolve the repo root from the script location so this works no matter
# where cron invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

# Resolve the docker compose command variant (v2 plugin or v1 standalone).
# Sets the global COMPOSE array; exits on failure.
pick_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE=(docker-compose)
    else
        log "error: neither 'docker compose' nor 'docker-compose' is available"
        exit 1
    fi
}

# Append any LIBREWXR_* variables that exist in .env.example but are
# missing from .env, copying along the explanatory comment block that
# precedes each one. Existing values are never modified. A backup is
# written to .env.bak before changes.
sync_env_with_example() {
    local env_file="${REPO_DIR}/.env"
    local example="${REPO_DIR}/.env.example"

    if [[ "${LIBREWXR_AUTO_ENV_SYNC:-1}" != "1" ]]; then
        log "env sync disabled (LIBREWXR_AUTO_ENV_SYNC=0); skipping"
        return 0
    fi
    if [[ ! -f "${env_file}" ]]; then
        log "no .env present; skipping env sync"
        return 0
    fi
    if [[ ! -f "${example}" ]]; then
        log ".env.example missing from repo; skipping env sync"
        return 0
    fi

    # Pull every LIBREWXR_* name declared in .env.example, including
    # commented-default forms like "#LIBREWXR_MEMORY=7G".
    local example_vars
    example_vars=$(grep -oE '^#?[[:space:]]*LIBREWXR_[A-Z0-9_]+' "${example}" \
                   | sed -E 's/^#?[[:space:]]*//' \
                   | sort -u)

    local missing=()
    local var
    while IFS= read -r var; do
        [[ -z "${var}" ]] && continue
        # Only names that actually have an assignment line in .env.example
        # are addable. A bare prose mention (e.g. "# LIBREWXR_FOO is still
        # accepted as an alias.") has no line to copy, and treating it as
        # missing would re-append it on every run.
        if ! grep -qE "^#?[[:space:]]*${var}=" "${example}"; then
            continue
        fi
        # A var that's present commented or uncommented counts as
        # "the user has seen it" -- don't re-add.
        if ! grep -qE "^#?[[:space:]]*${var}=" "${env_file}"; then
            missing+=("${var}")
        fi
    done <<< "${example_vars}"

    if [[ ${#missing[@]} -eq 0 ]]; then
        log ".env already has every variable from .env.example"
        return 0
    fi

    log "adding ${#missing[@]} new variable(s) to .env: ${missing[*]}"
    cp -p "${env_file}" "${env_file}.bak"

    {
        echo
        echo "# === Added by auto-update on $(date -u +'%Y-%m-%d') ==="
        echo "# Previous .env preserved at .env.bak. Edit values as needed."
        for var in "${missing[@]}"; do
            echo
            # Walk .env.example accumulating consecutive comment lines,
            # resetting on blank/non-comment lines, and emitting the buffer
            # along with the matching var line when found.
            awk -v target="${var}" '
                $0 ~ "^#?[[:space:]]*" target "=" {
                    printf "%s%s\n", buf, $0
                    exit
                }
                /^[[:space:]]*$/ { buf = ""; next }
                /^#/ { buf = buf $0 ORS; next }
                { buf = "" }
            ' "${example}"
        done
    } >> "${env_file}"

    log ".env updated; backup saved to ${env_file}.bak"
}

# Opt-in guard. The default update path can be cron-driven, so it must
# be explicitly enabled. --dry-run is read-only; --env and --build are
# manual-invocation flags that the user is asking for explicitly. None
# of those need the sentinel.
SENTINEL="${REPO_DIR}/.auto-update-enabled"
if [[ "${MODE}" == "update" ]]; then
    if [[ ! -f "${SENTINEL}" && "${LIBREWXR_AUTO_UPDATE:-0}" != "1" ]]; then
        log "auto-update not enabled on this host (no ${SENTINEL}, and"
        log "LIBREWXR_AUTO_UPDATE is not set). Exiting without action."
        log "To enable on a production host: touch ${SENTINEL}"
        exit 0
    fi
fi

# Prevent concurrent runs (useful if an update takes longer than the
# cron interval). All modes that mutate share the lock so a manual
# --env or --build can't race a cron-driven update.
LOCK_FILE="${TMPDIR:-/tmp}/librewxr-auto-update.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "another auto-update is already running; exiting"
    exit 0
fi

# The env-only mode never touches git or docker, so handle it here and
# exit before the git checks below.
if [[ "${MODE}" == "env-sync" ]]; then
    sync_env_with_example
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
if [[ "${MODE}" == "update" || "${MODE}" == "build" ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log "working tree is dirty; refusing to update"
        exit 1
    fi
fi

log "fetching origin for branch ${BRANCH}"
git fetch --quiet origin "${BRANCH}"

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/${BRANCH}")"

NEED_PULL=1
if [[ "${LOCAL}" == "${REMOTE}" ]]; then
    log "already up to date (${LOCAL:0:7})"
    NEED_PULL=0
elif ! git merge-base --is-ancestor "${LOCAL}" "${REMOTE}"; then
    # Only act when local is strictly behind remote (remote contains all
    # of local's commits, plus more). If local is ahead or diverged,
    # something unusual is going on and we should stay out of the way.
    log "local (${LOCAL:0:7}) is not an ancestor of origin/${BRANCH} (${REMOTE:0:7});"
    log "refusing to update (branch is ahead or has diverged)"
    exit 0
else
    log "update available: ${LOCAL:0:7} -> ${REMOTE:0:7}"
fi

if [[ "${MODE}" == "dry-run" ]]; then
    log "dry-run mode: would pull ${BRANCH} (if needed), reconcile .env,"
    log "pull images from ghcr.io/joshuakimsey/librewxr, and run 'up -d'"
    exit 0
fi

pick_compose

if [[ "${NEED_PULL}" -eq 1 ]]; then
    log "pulling ${BRANCH}"
    git pull --ff-only --quiet origin "${BRANCH}"
else
    log "no new commits to pull"
fi

# Reconcile .env against the freshly-pulled .env.example so any new
# upstream settings land in the local config before the stack restarts.
sync_env_with_example

if [[ "${MODE}" == "build" ]]; then
    # Legacy path: source-based deployments that build the stack locally
    # instead of running the published images.
    log "rebuilding and restarting stack from local checkout (legacy --build path)"
    "${COMPOSE[@]}" up -d --build
    log "build complete: at $(git rev-parse --short HEAD)"
    exit 0
fi

log "pulling latest images from ghcr.io/joshuakimsey/librewxr"
"${COMPOSE[@]}" pull

log "restarting stack"
"${COMPOSE[@]}" up -d

log "update complete: now at $(git rev-parse --short HEAD)"
