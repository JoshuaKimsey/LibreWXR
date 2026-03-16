# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from pydantic import BaseModel


class RadarTimestamp(BaseModel):
    time: int
    path: str


class RadarData(BaseModel):
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]


class SatelliteData(BaseModel):
    infrared: list[RadarTimestamp]


class WeatherMapsResponse(BaseModel):
    version: str
    generated: int
    host: str
    radar: RadarData
    satellite: SatelliteData
