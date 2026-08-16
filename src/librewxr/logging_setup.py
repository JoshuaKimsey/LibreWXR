# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import faulthandler
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# Map dotted logger names to short subsystem tags so concurrent startup
# (radar / IFS / NWP / GMGSI all firing in parallel) reads cleanly in the log.
# Anything not in the map falls back to the last segment of the module
# path (e.g. an unmapped third-party logger keeps its own short name).
_LOG_TAGS = {
    "librewxr.main": "main",
    "librewxr.data_pipeline": "pipeline",
    "librewxr.config": "config",
    "librewxr.memory": "memory",
    "librewxr.api.routes": "api",
    "librewxr.data.sources": "radar",
    "librewxr.data.fetcher": "fetcher",
    "librewxr.data.store": "store",
    "librewxr.data.regions": "regions",
    "librewxr.data.coverage": "coverage",
    "librewxr.data.master_state": "state",
    "librewxr.sources.world.ifs.grid": "ifs",
    "librewxr.sources.world.ifs.interpolation": "ifs",
    "librewxr.sources.satellite.gmgsi.source": "gmgsi",
    "librewxr.sources.regional.north_america.usa.nwp.hrrr.grid": "hrrr",
    "librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska.grid": "hrrr-ak",
    "librewxr.sources.regional.europe.nwp.icon_eu.grid": "icon-eu",
    "librewxr.sources.regional.europe.nwp.dmi_dini.grid": "dmi-dini",
    "librewxr.sources.regional.north_america.canada.nwp.hrdps.grid": "hrdps",
    "librewxr.sources.regional.caribbean.nwp.arome_antilles.grid": "arome-ant",
    "librewxr.sources.regional.south_america.nwp.wrf_smn.grid": "wrf-smn",
    "librewxr.data.nowcast": "nowcast",
    "librewxr.tiles.warmer": "warmer",
    "librewxr.tiles.cache": "tiles",
    "librewxr.tiles.renderer": "tiles",
    "librewxr.tiles.satellite_renderer": "tiles",
    "librewxr.tiles.coordinates": "tiles",
    "librewxr.data.alerts_fetcher": "alerts",
    "librewxr.data.alerts_store": "alerts",
    # uvicorn's own loggers share the same tag format so its startup /
    # shutdown lines (and the access log, if ever enabled) match ours.
    "uvicorn": "uvicorn",
    "uvicorn.error": "uvicorn",
    "uvicorn.access": "uvicorn",
}


class _TagFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.tag = _LOG_TAGS.get(record.name, record.name.rsplit(".", 1)[-1])
        return super().format(record)


VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def normalize_level(value: str) -> str:
    """Normalize a LIBREWXR_LOG_LEVEL value to its uppercase canonical form."""
    normalized = value.strip().upper()
    if normalized not in VALID_LEVELS:
        raise ValueError(
            f"Invalid LIBREWXR_LOG_LEVEL: {value!r} "
            f"(expected one of {', '.join(VALID_LEVELS)})"
        )
    return normalized


logger = logging.getLogger(__name__)


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Install the shared Rich-tagged root handler at the given level.

    Both parameters resolve in the same order: an explicit argument
    wins, then the live ``LIBREWXR_*`` env var, then the pydantic
    settings object (which itself resolves real env vars over ``.env``
    over built-in defaults).  ``level`` therefore defaults to the
    ``LIBREWXR_LOG_LEVEL`` env var, then to ``settings.log_level``
    (INFO when unset anywhere).  ``log_file`` mirrors WARNING+ records
    (warnings, errors, exception tracebacks) to a rotating file (5 MB x
    3 backups); it is enabled by default at ``logs/librewxr.log``
    (relative to the process CWD), and an empty/whitespace value
    disables the file entirely.  Only httpx/httpcore are additionally
    quieted; every other logger propagates to the root handler.
    """
    # Dump Python tracebacks to stderr if a render worker dies from a fatal
    # signal (SIGSEGV/SIGABRT/SIGFPE/SIGBUS), so native crashes leave a record
    # alongside the supervisor's exit-code log line.
    faulthandler.enable(all_threads=True)
    # Imported here, not at module top: pydantic-settings validates
    # defaults at instantiation, so importing config at module load runs
    # Settings() -> the log_level validator -> back into this module
    # before it has finished defining normalize_level (circular import).
    from librewxr.config import settings

    if level is None:
        level = os.getenv("LIBREWXR_LOG_LEVEL") or settings.log_level
    normalized = normalize_level(level)
    if log_file is None:
        log_file = os.getenv("LIBREWXR_LOG_FILE")
    if log_file is None:
        log_file = settings.log_file
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(_TagFormatter("[%(tag)s] %(message)s"))
    handlers: list[logging.Handler] = [handler]

    log_path: Path | None = None
    if log_file.strip():
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(
            _TagFormatter(
                "%(asctime)s %(levelname)-8s [%(tag)s] %(message)s",
                datefmt="[%x %X]",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, normalized),
        handlers=handlers,
        force=True,
    )
    if log_path is not None:
        logger.info("Warning/error log file: %s", log_path)
    # Suppress noisy per-request INFO logs from httpx/httpcore — sources
    # already log fetch results themselves in fetcher.py / the sources.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Quiet the third-party MCP SDK namespace (e.g. the
    # mcp.server.streamable_http_manager "StreamableHTTP session manager
    # started" INFO line).  Our own librewxr.mcp.* loggers are unaffected.
    logging.getLogger("mcp").setLevel(logging.WARNING)
