# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
import os

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


def setup_logging(level: str | None = None) -> None:
    """Install the shared Rich-tagged root handler at the given level.

    ``level`` defaults to the ``LIBREWXR_LOG_LEVEL`` env var (INFO when
    unset).  Only httpx/httpcore are additionally quieted; every other
    logger propagates to the root handler.
    """
    if level is None:
        level = os.getenv("LIBREWXR_LOG_LEVEL", "INFO")
    normalized = normalize_level(level)
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(_TagFormatter("[%(tag)s] %(message)s"))
    logging.basicConfig(
        level=getattr(logging, normalized),
        handlers=[handler],
        force=True,
    )
    # Suppress noisy per-request INFO logs from httpx/httpcore — sources
    # already log fetch results themselves in fetcher.py / the sources.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
