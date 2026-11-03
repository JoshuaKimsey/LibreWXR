# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Snapshot sync: the pipeline/renderer handoff over object storage.

In the multi-worker split the pipeline and the render workers share one
cache directory: the pipeline writes memmap files and names them in
``state.json``, render workers poll that file and memory-map what it
names.  This module carries the same handoff over a network artifact
store so the two halves can run on different servers (ROADMAP: network
split mode).

Pipeline side, :class:`SnapshotPublisher` (``LIBREWXR_SNAPSHOT_SYNC_URL``):
after each ``state.json`` write it uploads every file the snapshot names
that the store does not have yet (frame files are immutable per
timestamp, so a cycle uploads only what changed), then the snapshot
itself under a unique name, then one ``manifest.json`` that lists the
snapshot and its files.  The manifest is the single atomic object a
reader keys off; files a snapshot no longer names are deleted once they
are out of the previous manifest too, the same one-cycle grace the
pipeline's own eviction keeps.  The lightning artifact, which lives
outside ``state.json``, is uploaded whenever the pipeline writes it.

Render side, :class:`SnapshotPuller` (``python -m librewxr.snapshot_pull``
on a render node, next to render-only workers that share its cache
directory): it polls the manifest, downloads the files it lists into the
local cache under the same relative paths, rewrites the pipeline's
absolute directories in the snapshot to the local cache, and only then
writes ``state.json``, which the render workers pick up exactly as they
would from a local pipeline.  It also takes over the maintenance the
pipeline would do on that node: pruning the shared coordinate and tile
stores, priming the page cache, and removing files no snapshot names.

Any ``fsspec`` URL works (``s3://bucket/prefix``, ``file:///path`` for a
shared or local disk); LibreWXR already depends on ``s3fs``.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec

from librewxr.data.master_state import STATE_FILENAME
from librewxr.data.pagecache import _collect_memmap_files

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
# Remote prefix of the uploaded snapshots (one object per published state.json)
SNAPSHOT_PREFIX = "snapshots"
# Local record of the last manifest a puller applied
LOCAL_MANIFEST_NAME = "snapshot_manifest.json"
# Files the pipeline regenerates every cycle live under the volatile dir
# (LIBREWXR_VOLATILE_CACHE_DIR); on the render node they go under this
# subdirectory of the cache instead, so one directory holds everything.
VOLATILE_SUBDIR = "volatile"
# Directories whose whole content is part of a snapshot, next to the
# memmap files the snapshot names (see snapshot_files)
_WHOLE_DIRS = ("coverage_masks", "palettes")
LIGHTNING_ARTIFACT = "lightning/current.npz"
# Per-publish hard-link directories beside the files being uploaded
STAGE_DIRNAME = ".publish"
# Snapshot payload keys that hold a directory on the pipeline's host
_DIRECTORY_KEYS = ("memmap_dir", "cache_root")


# ---------------------------------------------------------------------------
# Helpers shared by both sides
# ---------------------------------------------------------------------------

def open_store(url: str, endpoint_url: str = "") -> tuple[Any, str]:
    """An fsspec filesystem and root path for ``url``.

    Credentials come from the usual environment (``AWS_*`` for S3);
    ``endpoint_url`` points S3 at a compatible store (MinIO, OVH, ...).
    """
    options: dict[str, Any] = {}
    if endpoint_url and url.startswith("s3://"):
        options["client_kwargs"] = {"endpoint_url": endpoint_url}
    fs, root = fsspec.core.url_to_fs(url, **options)
    return fs, root.rstrip("/")


def _is_local(fs) -> bool:
    """Whether ``fs`` is a directory-based (local) filesystem."""
    protocol = fs.protocol if isinstance(fs.protocol, (tuple, list)) else (fs.protocol,)
    return "file" in protocol or "local" in protocol


def relative_name(
    path: Path, cache_dir: Path, volatile_dir: Path | None
) -> str | None:
    """A file's name in the store, or ``None`` when it is outside both dirs."""
    try:
        return path.relative_to(cache_dir).as_posix()
    except ValueError:
        pass
    if volatile_dir is not None:
        try:
            return f"{VOLATILE_SUBDIR}/{path.relative_to(volatile_dir).as_posix()}"
        except ValueError:
            pass
    return None


def _store_directories(node, found: set[Path]) -> None:
    """Every ``memmap_dir`` a store payload names, at any depth."""
    if isinstance(node, dict):
        value = node.get("memmap_dir")
        if isinstance(value, str) and value:
            found.add(Path(value))
        for child in node.values():
            _store_directories(child, found)
    elif isinstance(node, (list, tuple)):
        for child in node:
            _store_directories(child, found)


def snapshot_files(
    payload: dict, cache_dir: Path, volatile_dir: Path | None
) -> dict[str, Path]:
    """Every file a snapshot depends on, keyed by its name in the store.

    The content of every directory a store names as its ``memmap_dir``:
    stores describe their files differently (radar and nowcast list
    ``[basename, dtype, shape]`` triples, the regional NWP grids list
    ``[run, lead]`` keys and derive the names, the storm-cell store
    rescans its directory), and a store's directory holds exactly what
    that store serves plus files kept one cycle for eviction grace.  The
    satellite frames (resolved from ``cache_root`` by the page-cache
    walker), the directories read whole (coverage masks, the palette
    artifact) and the lightning artifact complete the set.
    """
    files: dict[str, Path] = {}

    def add(path: Path) -> None:
        name = relative_name(path, cache_dir, volatile_dir)
        if name is None:
            logger.warning("Snapshot file %s is outside the cache; not synced", path)
            return
        if path.is_file():
            files[name] = path

    directories: set[Path] = set()
    for store_payload in payload.get("stores", {}).values():
        if isinstance(store_payload, dict):
            _store_directories(store_payload, directories)
            for path in _collect_memmap_files(store_payload):
                add(path)
    directories.update(cache_dir / name for name in _WHOLE_DIRS)
    for directory in directories:
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_file() and ".tmp" not in path.name:
                    add(path)
    add(cache_dir / LIGHTNING_ARTIFACT)
    return files


def _stamp(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def rewrite_directories(
    payload: dict, remote_cache_dir: str, remote_volatile_dir: str | None, local: Path
) -> dict:
    """The snapshot with the pipeline's directories replaced by the local cache.

    Every ``memmap_dir`` / ``cache_root`` value under the pipeline's cache
    directory maps to the same place under ``local``; one under its
    volatile directory maps under ``local/volatile`` (where the puller put
    those files).  Anything else is left alone.
    """
    remote_cache = remote_cache_dir.rstrip("/")
    remote_volatile = (remote_volatile_dir or "").rstrip("/")

    def map_dir(value: str) -> str:
        if remote_volatile and (
            value == remote_volatile or value.startswith(remote_volatile + "/")
        ):
            return str(local / VOLATILE_SUBDIR / value[len(remote_volatile) :].lstrip("/"))
        if value == remote_cache or value.startswith(remote_cache + "/"):
            return str(local / value[len(remote_cache) :].lstrip("/"))
        return value

    def walk(node):
        if isinstance(node, dict):
            return {
                key: (
                    map_dir(value)
                    if key in _DIRECTORY_KEYS and isinstance(value, str)
                    else walk(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(value) for value in node]
        return node

    return walk(payload)


def _replace_file(final: Path, write) -> None:
    """Write through a unique temp file and rename, so readers never see a partial file."""
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Pipeline side
# ---------------------------------------------------------------------------

class SnapshotPublisher:
    """Upload each snapshot and the files it names to the artifact store."""

    def __init__(
        self,
        cache_dir: Path,
        url: str,
        *,
        volatile_dir: Path | None = None,
        endpoint_url: str = "",
        workers: int = 8,
        stale_grace_s: float = 60.0,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._volatile_dir = Path(volatile_dir) if volatile_dir else None
        # How long a file stays in the store after a snapshot stops naming
        # it.  The pipeline publishes several snapshots per cycle (frame
        # merges, satellite fetches), so "the next publish" can be seconds
        # away while a reader is still downloading the previous manifest's
        # files; a time-based grace matches the pipeline's own eviction.
        self._stale_grace_s = float(stale_grace_s)
        self._url = url
        self._fs, self._root = open_store(url, endpoint_url)
        self._workers = max(1, int(workers))
        # One publish at a time; a trigger arriving mid-publish is skipped
        # (the next snapshot carries newer data)
        self._lock = threading.Lock()
        self._previous: dict | None = None
        self._sweep_stale_links()

    def _sweep_stale_links(self) -> None:
        """Remove link directories a publish interrupted by a crash left behind.

        Each holds inodes of files the pipeline may have deleted since, so
        they would otherwise keep that disk space for ever.
        """
        roots = [self._cache_dir] + ([self._volatile_dir] if self._volatile_dir else [])
        removed = 0
        for root in roots:
            if not root.is_dir():
                continue
            for stale_dir in root.rglob(f"{STAGE_DIRNAME}-*"):
                if not stale_dir.is_dir():
                    continue
                for link in stale_dir.iterdir():
                    link.unlink(missing_ok=True)
                stale_dir.rmdir()
                removed += 1
        if removed:
            logger.info("Removed %d leftover snapshot publish link directories", removed)

    def _remote(self, name: str) -> str:
        return f"{self._root}/{name}"

    def _ensure_parent(self, remote: str) -> None:
        """A directory-based store needs the parent directory to exist.

        Object stores have no directories, and asking one to create the
        bucket's "directory" turns into a HeadBucket / GetBucketLocation
        call that a key limited to object access may be forbidden.
        """
        if _is_local(self._fs):
            self._fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)

    def _put(self, local: Path, name: str) -> str | None:
        """Upload one file; ``None`` when it vanished before it could be read.

        ``local`` is normally the publish's own hard link to the file (see
        ``_stage``), so the pipeline can delete or replace the original
        meanwhile.  Where linking was not possible it is the original, and
        a file removed before the upload is left out of the manifest; the
        next snapshot (which no longer names it) carries the replacement.
        """
        remote = self._remote(name)
        self._ensure_parent(remote)
        try:
            self._fs.put_file(str(local), remote)
        except FileNotFoundError:
            return None
        return name

    def _stage(self, files: dict[str, Path], sequence: int) -> dict[str, Path]:
        """Hard-link the snapshot's files so the publish reads a fixed set.

        The pipeline keeps writing while a publish runs: a nowcast
        regeneration deletes the previous frames and an atomic replace
        gives a name a new inode.  A link made now keeps the inode the
        snapshot described until the publish is done, at no copy cost.
        Links go in a ``.publish-<sequence>`` directory beside each file
        (same filesystem, outside every store's glob).  A file that cannot
        be linked (already gone, or a filesystem without hard links) keeps
        its original path and is uploaded from there.
        """
        staged: dict[str, Path] = {}
        for name, path in files.items():
            link = path.parent / f"{STAGE_DIRNAME}-{sequence}" / path.name
            try:
                link.parent.mkdir(exist_ok=True)
                os.link(path, link)
                staged[name] = link
            except FileNotFoundError:
                continue
            except OSError:
                staged[name] = path
        return staged

    @staticmethod
    def _unstage(staged: dict[str, Path], sequence: int) -> None:
        for path in staged.values():
            if path.parent.name == f"{STAGE_DIRNAME}-{sequence}":
                path.unlink(missing_ok=True)
                try:
                    path.parent.rmdir()
                except OSError:
                    pass

    def _pipe(self, name: str, data: bytes) -> None:
        remote = self._remote(name)
        self._ensure_parent(remote)
        self._fs.pipe_file(remote, data)

    def _read_manifest(self) -> dict | None:
        try:
            return json.loads(self._fs.cat_file(self._remote(MANIFEST_NAME)))
        except FileNotFoundError:
            return None

    def publish(self, payload: dict) -> dict | None:
        """Upload the files ``payload`` names, the snapshot, then the manifest.

        Returns the manifest, or ``None`` when a publish is still running.
        Raises on store errors so the caller logs them; a failed publish
        leaves the previous manifest in place, which a reader can still use.
        """
        if not self._lock.acquire(blocking=False):
            logger.warning("Snapshot publish still running; skipping this snapshot")
            return None
        try:
            return self._publish(payload)
        finally:
            self._lock.release()

    def _publish(self, payload: dict) -> dict:
        started = time.monotonic()
        if self._previous is None:
            self._previous = self._read_manifest() or {"files": {}, "snapshot": None}
        previous = self._previous
        written_at = int(payload.get("written_at", time.time()))
        sequence = int(previous.get("sequence", 0)) + 1
        files = self._stage(
            snapshot_files(payload, self._cache_dir, self._volatile_dir), sequence
        )
        try:
            stamps: dict[str, dict] = {}
            for name, path in files.items():
                try:
                    stamps[name] = _stamp(path)
                except FileNotFoundError:
                    continue
            changed = [
                name for name, stamp in stamps.items()
                if previous["files"].get(name) != stamp
            ]
            uploaded = 0
            vanished = 0
            with ThreadPoolExecutor(self._workers) as pool:
                for name, done in zip(
                    changed, pool.map(lambda n: self._put(files[n], n), changed)
                ):
                    if done is None:
                        vanished += 1
                        del stamps[name]
                    else:
                        uploaded += stamps[name]["size"]
        finally:
            self._unstage(files, sequence)
        if vanished:
            logger.warning(
                "%d file(s) vanished during the publish (the pipeline moved on); "
                "left out of this manifest", vanished,
            )

        snapshot_name = f"{SNAPSHOT_PREFIX}/{written_at}-{sequence}.json.gz"
        self._pipe(
            snapshot_name,
            gzip.compress(json.dumps(payload, default=str).encode("utf-8")),
        )
        # Files and snapshots this manifest no longer names stay in the store
        # for stale_grace_s (recorded in the manifest with the time they
        # became stale, so a restarted pipeline still deletes them): a
        # reader mid-pull on an older manifest can finish, the same grace
        # the pipeline's local eviction keeps.
        now = int(time.time())
        # Only what an earlier manifest already marked stale can expire: a
        # file dropped by this very publish always survives it, whatever
        # the grace, so a reader on the previous manifest can finish.
        stale: dict[str, int] = {
            name: since for name, since in previous.get("stale", {}).items()
            if name not in stamps
        }
        stale_snapshots: dict[str, int] = dict(previous.get("stale_snapshots", {}))
        expired = {
            name for name, since in {**stale, **stale_snapshots}.items()
            if now - since >= self._stale_grace_s
        }
        for name in set(previous["files"]) - set(stamps):
            stale.setdefault(name, now)
        if previous.get("snapshot") and previous["snapshot"] != snapshot_name:
            stale_snapshots.setdefault(previous["snapshot"], now)
        manifest = {
            "version": MANIFEST_VERSION,
            "sequence": sequence,
            "written_at": written_at,
            "published_at": now,
            "cache_dir": str(self._cache_dir),
            "volatile_dir": str(self._volatile_dir) if self._volatile_dir else None,
            "snapshot": snapshot_name,
            "files": stamps,
            "stale": {n: t for n, t in stale.items() if n not in expired},
            "stale_snapshots": {
                n: t for n, t in stale_snapshots.items() if n not in expired
            },
        }
        self._pipe(MANIFEST_NAME, json.dumps(manifest, sort_keys=True).encode())
        if expired:
            try:
                self._fs.rm([self._remote(name) for name in sorted(expired)])
            except FileNotFoundError:
                pass
        self._previous = manifest
        logger.info(
            "Published snapshot %d: %d of %d files uploaded (%.0f MB) in %.1f s",
            sequence, len(changed), len(files), uploaded / 1e6,
            time.monotonic() - started,
        )
        return manifest

    def publish_lightning(self) -> bool:
        """Upload the lightning artifact (written outside the snapshot cycle)."""
        path = self._cache_dir / LIGHTNING_ARTIFACT
        if not path.is_file():
            return False
        self._put(path, LIGHTNING_ARTIFACT)
        return True


# ---------------------------------------------------------------------------
# Render side
# ---------------------------------------------------------------------------

class SnapshotPuller:
    """Bring the published snapshot into a local cache for render workers."""

    def __init__(
        self,
        cache_dir: Path,
        url: str,
        *,
        endpoint_url: str = "",
        workers: int = 8,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._fs, self._root = open_store(url, endpoint_url)
        self._workers = max(1, int(workers))
        self._local_manifest_path = self._cache_dir / LOCAL_MANIFEST_NAME
        try:
            self._applied: dict = json.loads(self._local_manifest_path.read_text())
        except (OSError, ValueError):
            self._applied = {"sequence": 0, "files": {}, "previous_files": {}}
        self._lightning_stamp: tuple | None = None

    def _remote(self, name: str) -> str:
        return f"{self._root}/{name}"

    @property
    def applied_sequence(self) -> int:
        return int(self._applied.get("sequence", 0))

    def pull(self) -> bool:
        """Apply the published manifest if it is newer; ``True`` when it was.

        Downloads every listed file the local cache lacks (or has at a
        different size/mtime), writes the rewritten snapshot as
        ``state.json`` last, then deletes tracked files that neither this
        nor the previously applied manifest names.  The lightning artifact
        is refreshed on every call.  Raises on store errors; the local
        cache is left as it was (a partial download never reaches
        ``state.json``).
        """
        self._fs.invalidate_cache()
        try:
            manifest = json.loads(self._fs.cat_file(self._remote(MANIFEST_NAME)))
        except FileNotFoundError:
            logger.info("No manifest at %s yet", self._root)
            self._pull_lightning()
            return False
        self._pull_lightning()
        if int(manifest.get("sequence", 0)) <= self.applied_sequence:
            return False

        started = time.monotonic()
        files: dict[str, dict] = manifest["files"]
        needed = [
            name for name, stamp in files.items()
            if name != LIGHTNING_ARTIFACT and self._applied["files"].get(name) != stamp
        ]
        downloaded = 0
        with ThreadPoolExecutor(self._workers) as pool:
            for name in pool.map(self._download, needed):
                downloaded += files[name]["size"]

        raw = gzip.decompress(self._fs.cat_file(self._remote(manifest["snapshot"])))
        payload = rewrite_directories(
            json.loads(raw), manifest["cache_dir"], manifest.get("volatile_dir"),
            self._cache_dir,
        )
        _replace_file(
            self._cache_dir / STATE_FILENAME,
            lambda tmp: tmp.write_bytes(
                gzip.compress(json.dumps(payload, default=str).encode("utf-8"))
            ),
        )

        keep = set(files) | set(self._applied.get("files", {}))
        removed = 0
        for name in set(self._applied.get("previous_files", {})) - keep:
            try:
                (self._cache_dir / name).unlink()
                removed += 1
            except FileNotFoundError:
                pass
        self._applied = {
            "sequence": int(manifest["sequence"]),
            "written_at": manifest.get("written_at"),
            "files": files,
            "previous_files": self._applied.get("files", {}),
        }
        _replace_file(
            self._local_manifest_path,
            lambda tmp: tmp.write_text(json.dumps(self._applied)),
        )
        logger.info(
            "Applied snapshot %d (written %d s ago): %d files downloaded (%.0f MB), "
            "%d removed, in %.1f s",
            manifest["sequence"], int(time.time()) - int(manifest.get("written_at", 0)),
            len(needed), downloaded / 1e6, removed, time.monotonic() - started,
        )
        return True

    def _download(self, name: str) -> str:
        _replace_file(
            self._cache_dir / name,
            lambda tmp: self._fs.get_file(self._remote(name), str(tmp)),
        )
        return name

    def _pull_lightning(self) -> None:
        """Refresh the lightning artifact when the store's copy changed."""
        try:
            info = self._fs.info(self._remote(LIGHTNING_ARTIFACT))
        except FileNotFoundError:
            return
        stamp = (info.get("size"), info.get("ETag") or info.get("mtime") or info.get("LastModified"))
        if stamp == self._lightning_stamp:
            return
        self._download(LIGHTNING_ARTIFACT)
        self._lightning_stamp = stamp
