# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""External hooks: operator commands the data pipeline runs at handoff points.

The pipeline hands its results to other processes through files: the
``state.json`` snapshot after every fetch cycle, the nowcast memmaps it
names, and the lightning artifact.  An operator who wants to post-process
or mirror those (export the nowcast to another format, sync the cache to
object storage for render nodes elsewhere, ...) would otherwise have to
poll for them.  A hook runs a user-supplied command instead, at the moment
the files are complete:

- ``LIBREWXR_HOOK_STATE_CMD`` after every ``state.json`` write;
- ``LIBREWXR_HOOK_NOWCAST_CMD`` after the first ``state.json`` write that
  carries a nowcast for the latest radar frame (once per nowcast, never
  for the mid-cycle snapshots the pipeline writes on frame merges);
- ``LIBREWXR_HOOK_LIGHTNING_CMD`` after every lightning artifact write.

The command is a shell-style line split with ``shlex`` (no shell is
involved), run as a child process detached from the fetch loop, with the
event's context in ``LIBREWXR_HOOK_*`` environment variables on top of the
pipeline's own environment (so ``LIBREWXR_CACHE_DIR`` and the rest of the
configuration are visible).  A hook still running when its next trigger
arrives is left alone and the trigger is skipped; one that overruns
``LIBREWXR_HOOK_TIMEOUT`` is killed.  A failing hook is logged and never
affects the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Last lines of a failed hook's output kept in the log
_OUTPUT_TAIL_LINES = 20


class HookRunner:
    """One configured hook command, run at most once at a time."""

    def __init__(self, name: str, command: str, timeout_s: float) -> None:
        self.name = name
        self._argv = shlex.split(command) if command and command.strip() else []
        self._timeout_s = float(timeout_s)
        self._task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._argv)

    def fire(self, env: dict[str, str]) -> bool:
        """Start the hook with ``env`` added to the environment.

        Returns ``True`` when a run was started; ``False`` when the hook is
        not configured or its previous run is still going (that trigger
        is skipped rather than queued: the next one will carry newer data).
        """
        if not self._argv:
            return False
        if self._task is not None and not self._task.done():
            logger.warning(
                "%s hook still running from the previous trigger; skipping this one",
                self.name,
            )
            return False
        self._task = asyncio.create_task(self._run(dict(env)))
        return True

    async def _run(self, env: dict[str, str]) -> None:
        started = time.monotonic()
        full_env = {**os.environ, **env}
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv,
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            logger.error("%s hook could not start (%s): %s", self.name, self._argv[0], exc)
            return
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), self._timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(
                "%s hook killed after %.0f s (LIBREWXR_HOOK_TIMEOUT)",
                self.name, self._timeout_s,
            )
            return
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        elapsed = time.monotonic() - started
        if proc.returncode == 0:
            logger.info("%s hook finished in %.1f s", self.name, elapsed)
        else:
            tail = output.decode("utf-8", "replace").splitlines()[-_OUTPUT_TAIL_LINES:]
            logger.error(
                "%s hook exited %d after %.1f s; last output:\n%s",
                self.name, proc.returncode, elapsed, "\n".join(tail),
            )

    async def wait(self) -> None:
        """Let a running hook finish (shutdown; the timeout still applies)."""
        if self._task is not None and not self._task.done():
            await self._task


def state_event_env(payload: dict, cache_dir: Path) -> dict[str, str]:
    """Environment for the state hook: where the snapshot is and when it was written."""
    return {
        "LIBREWXR_HOOK_EVENT": "state",
        "LIBREWXR_CACHE_DIR": str(cache_dir),
        "LIBREWXR_HOOK_STATE_PATH": str(Path(cache_dir) / "state.json"),
        "LIBREWXR_HOOK_WRITTEN_AT": str(int(payload.get("written_at", 0))),
    }


def nowcast_event_env(payload: dict, cache_dir: Path) -> dict[str, str] | None:
    """Environment for the nowcast hook, or ``None`` when this snapshot has no
    nowcast for its latest radar frame.

    A snapshot written mid-cycle (on a radar frame merge) can already list
    the new radar frame while the nowcast still belongs to the previous one;
    its first step is then not after the latest frame.  Such a snapshot is
    not a nowcast event: the one written after nowcast generation is.
    """
    stores = payload.get("stores", {})
    frame_store = stores.get("frame_store") or {}
    nowcast = stores.get("nowcast_store") or {}
    radar_frames = frame_store.get("frames") or []
    nowcast_frames = nowcast.get("frames") or []
    if not radar_frames or not nowcast_frames:
        return None
    latest_radar_ts = max(int(f["timestamp"]) for f in radar_frames)
    nowcast_ts = sorted(int(f["timestamp"]) for f in nowcast_frames)
    if nowcast_ts[0] <= latest_radar_ts:
        return None
    return {
        **state_event_env(payload, cache_dir),
        "LIBREWXR_HOOK_EVENT": "nowcast",
        "LIBREWXR_HOOK_RADAR_TS": str(latest_radar_ts),
        "LIBREWXR_HOOK_NOWCAST_TS": ",".join(str(ts) for ts in nowcast_ts),
        "LIBREWXR_HOOK_NOWCAST_DIR": str(nowcast.get("memmap_dir", "")),
    }


def lightning_event_env(artifact_path: Path, version: int, cache_dir: Path) -> dict[str, str]:
    """Environment for the lightning hook: the artifact just written."""
    return {
        "LIBREWXR_HOOK_EVENT": "lightning",
        "LIBREWXR_CACHE_DIR": str(cache_dir),
        "LIBREWXR_HOOK_LIGHTNING_PATH": str(artifact_path),
        "LIBREWXR_HOOK_LIGHTNING_VERSION": str(int(version)),
    }
