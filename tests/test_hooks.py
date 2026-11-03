# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""External hooks: operator commands run at the pipeline's handoff points."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from librewxr.hooks import (
    HookRunner,
    lightning_event_env,
    nowcast_event_env,
    state_event_env,
)

# A hook that records its environment in a file and exits with a given code
_RECORDER = (
    "import os, sys; "
    "open(sys.argv[1], 'w').write(os.environ.get('LIBREWXR_HOOK_EVENT', '') + ' ' "
    "+ os.environ.get('LIBREWXR_HOOK_RADAR_TS', '')); "
    "sys.exit(int(sys.argv[2]))"
)


def _recorder_cmd(out: Path, exit_code: int = 0) -> str:
    return f'"{sys.executable}" -c "{_RECORDER}" "{out}" {exit_code}'


async def test_hook_runs_command_with_event_environment(tmp_path):
    out = tmp_path / "seen"
    hook = HookRunner("nowcast", _recorder_cmd(out), timeout_s=30)
    assert hook.enabled
    assert hook.fire({"LIBREWXR_HOOK_EVENT": "nowcast", "LIBREWXR_HOOK_RADAR_TS": "42"})
    await hook.wait()
    assert out.read_text() == "nowcast 42"


async def test_unconfigured_hook_is_a_noop():
    hook = HookRunner("state", "", timeout_s=30)
    assert not hook.enabled
    assert not hook.fire({"LIBREWXR_HOOK_EVENT": "state"})
    await hook.wait()


async def test_running_hook_skips_the_next_trigger(tmp_path, caplog):
    out = tmp_path / "seen"
    hook = HookRunner(
        "state",
        f'"{sys.executable}" -c "import time, sys; time.sleep(1); '
        f"open(sys.argv[1], 'w').write('done')\" \"{out}\"",
        timeout_s=30,
    )
    assert hook.fire({})
    await asyncio.sleep(0.1)
    assert not hook.fire({})
    assert "still running" in caplog.text
    await hook.wait()
    assert out.read_text() == "done"


async def test_failing_hook_is_logged_not_raised(tmp_path, caplog):
    out = tmp_path / "seen"
    hook = HookRunner("lightning", _recorder_cmd(out, exit_code=3), timeout_s=30)
    hook.fire({"LIBREWXR_HOOK_EVENT": "lightning"})
    await hook.wait()
    assert "exited 3" in caplog.text
    assert out.read_text() == "lightning "


async def test_overrunning_hook_is_killed(caplog):
    hook = HookRunner(
        "state", f'"{sys.executable}" -c "import time; time.sleep(30)"', timeout_s=0.5
    )
    hook.fire({})
    await hook.wait()
    assert "killed after" in caplog.text


def test_nowcast_event_only_for_a_nowcast_of_the_latest_radar_frame(tmp_path):
    """state.json is also written mid-cycle (on radar frame merges), when the
    new radar frame is in but the nowcast still belongs to the previous one.
    That snapshot is a state event but not a nowcast event.
    """
    t0 = 1_793_650_200
    payload = {
        "written_at": t0 + 30,
        "stores": {
            "frame_store": {"frames": [{"timestamp": t0 - 600}, {"timestamp": t0}]},
            "nowcast_store": {
                "memmap_dir": str(tmp_path / "nowcast"),
                # Steps of the previous frame: the first coincides with t0
                "frames": [{"timestamp": t0 + 600 * k} for k in range(0, 3)],
            },
        },
    }
    assert nowcast_event_env(payload, tmp_path) is None
    assert state_event_env(payload, tmp_path)["LIBREWXR_HOOK_EVENT"] == "state"

    payload["stores"]["nowcast_store"]["frames"] = [
        {"timestamp": t0 + 600 * k} for k in range(1, 4)
    ]
    env = nowcast_event_env(payload, tmp_path)
    assert env["LIBREWXR_HOOK_EVENT"] == "nowcast"
    assert env["LIBREWXR_HOOK_RADAR_TS"] == str(t0)
    assert env["LIBREWXR_HOOK_NOWCAST_TS"] == ",".join(
        str(t0 + 600 * k) for k in range(1, 4)
    )
    assert env["LIBREWXR_HOOK_STATE_PATH"] == str(tmp_path / "state.json")
    assert env["LIBREWXR_HOOK_NOWCAST_DIR"] == str(tmp_path / "nowcast")

    # No nowcast store at all (nowcast disabled): never a nowcast event
    del payload["stores"]["nowcast_store"]
    assert nowcast_event_env(payload, tmp_path) is None


def test_lightning_event_env(tmp_path):
    env = lightning_event_env(tmp_path / "lightning" / "current.npz", 7, tmp_path)
    assert env["LIBREWXR_HOOK_EVENT"] == "lightning"
    assert env["LIBREWXR_HOOK_LIGHTNING_VERSION"] == "7"
    assert env["LIBREWXR_HOOK_LIGHTNING_PATH"].endswith("current.npz")


@pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics")
async def test_hook_env_extends_the_pipeline_environment(tmp_path, monkeypatch):
    """The pipeline's own configuration (LIBREWXR_CACHE_DIR and friends) must
    stay visible to the hook so an external script can find the cache."""
    monkeypatch.setenv("LIBREWXR_EXAMPLE_SETTING", "kept")
    out = tmp_path / "seen"
    hook = HookRunner(
        "state",
        f'"{sys.executable}" -c "import os, sys; '
        f"open(sys.argv[1], 'w').write(os.environ['LIBREWXR_EXAMPLE_SETTING'])\" \"{out}\"",
        timeout_s=30,
    )
    hook.fire({"LIBREWXR_HOOK_EVENT": "state"})
    await hook.wait()
    assert out.read_text() == "kept"
