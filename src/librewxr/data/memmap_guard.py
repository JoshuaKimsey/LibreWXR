# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Guarded memmap writes for the project's per-cycle stores.

A memmap write is not a normal file write: the pages are allocated by
faulting them in during ``mm[:] = data``, and when the backing filesystem
has no room the kernel converts that fault into ``SIGBUS``.  No Python
handler can catch it, so an undersized tmpfs (Docker's default 64 MiB
``/dev/shm`` being the classic case) kills the process mid-cycle instead
of raising -- GitHub issue #43.

This module puts the pre-flight space check and the project's safe write
recipe (pid+uuid tmp name -> fill -> flush -> ``os.replace``) in one
place so every grid writer shares them.
"""

import os
from pathlib import Path
from uuid import uuid4

import numpy as np

# Headroom multiplier on top of the payload: the fill is not the only
# consumer (filesystem metadata, other writers racing for the same tmpfs),
# so require ~10% slack before starting.
_SPACE_SAFETY_FACTOR = 1.1


def ensure_space(directory: Path, nbytes: int, *, what: str) -> None:
    """Raise ``RuntimeError`` unless ``directory`` can hold the write.

    Checks ``os.statvfs`` (one syscall) for at least ``nbytes * 1.1`` of
    free space.  A pass does not guarantee the write succeeds -- another
    writer can claim the space in between -- but a failure here is an
    actionable error instead of a SIGBUS.
    """
    directory = Path(directory).resolve()
    try:
        st = os.statvfs(directory)
    except OSError as exc:
        raise RuntimeError(
            f"{what}: cannot stat {directory} for the memmap space pre-flight "
            f"({exc}); increase shm_size / bind a host tmpfs with enough "
            f"capacity at this exact path / unset the volatile cache dir to "
            f"fall back to the durable cache"
        ) from exc
    free = st.f_bavail * st.f_frsize
    needed = int(nbytes * _SPACE_SAFETY_FACTOR)
    if free < needed:
        raise RuntimeError(
            f"{what}: not enough space in {directory} ({free} bytes free, "
            f"{needed} bytes needed for a {nbytes}-byte memmap write); "
            f"increase shm_size / bind a host tmpfs with enough capacity at "
            f"this exact path / unset the volatile cache dir to fall back to "
            f"the durable cache"
        )


def atomic_memmap_write(
    final: Path, arr: np.ndarray, *, what: str
) -> np.ndarray:
    """Write ``arr`` to ``final`` safely, returning a read-only memmap.

    Pre-flights the space, then writes through a unique tmp file (pid+uuid
    so two overlapping pipeline processes during a deploy cannot steal
    each other's in-flight file) and ``os.replace``s it into place, which
    is atomic -- readers only ever see a complete file.  A failed fill is
    raised as ``RuntimeError`` with guidance instead of surfacing as a
    bare ``OSError`` (or, on tmpfs, a SIGBUS that kills the process).
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(final.parent, arr.nbytes, what=what)

    tmp = final.with_suffix(final.suffix + f".{os.getpid()}.{uuid4().hex}.tmp")
    try:
        mm = np.memmap(tmp, dtype=arr.dtype, mode="w+", shape=arr.shape)
        try:
            mm[:] = arr
            mm.flush()
        finally:
            del mm
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{what}: memmap write failed ({exc}); the volatile cache dir may "
            f"be full - increase shm_size / bind a host tmpfs with enough "
            f"capacity at this exact path / unset the volatile cache dir to "
            f"fall back to the durable cache"
        ) from exc

    os.replace(tmp, final)
    return np.memmap(final, dtype=arr.dtype, mode="r", shape=arr.shape)
