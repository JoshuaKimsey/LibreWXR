# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from typing import Any, Literal

from pydantic import BaseModel


class AlertProperties(BaseModel):
    title: str
    severity: str
    time: int | None
    expires: int | None
    description: str
    regions: list[str]
    uri: str


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    properties: AlertProperties
    geometry: dict[str, Any] | None


class AlertsResponse(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


class StormCellProperties(BaseModel):
    area_km2: float
    max_dbz: float
    motion_speed_kmh: float | None
    motion_heading_deg: float | None
    region: str


class StormCellFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: StormCellProperties
    geometry: dict


class StormCellsResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[StormCellFeature]


class StormCellsData(BaseModel):
    generated_at: int | None
    cells: list[dict]


class RadarTimestamp(BaseModel):
    time: int
    path: str


class ColorScheme(BaseModel):
    id: int
    name: str


class RadarData(BaseModel):
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]
    colorSchemes: list[ColorScheme]


class SatelliteData(BaseModel):
    infrared: list[RadarTimestamp]


class WeatherMapsResponse(BaseModel):
    version: str
    generated: int
    host: str
    radar: RadarData
    satellite: SatelliteData
