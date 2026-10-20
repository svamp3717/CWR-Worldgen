# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional OFP/CWA config.cpp map metadata for Island Studio.

WRP 4WVR/OPRW headers do not store CfgWorlds LandGrid or Names. Read these
only from a matching world class in a nearby or embedded source config.cpp.
Do not execute config code or guess locations from model/object records.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

from .rap_config import parse_rap_world

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_LOCATIONS = 5000
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CLASS = re.compile(r"\bclass\s+([A-Za-z_]\w*)\b[^;{}]*\{", re.I)
_GRID = re.compile(r"\bLandGrid\s*=\s*(" + _NUM + r")\s*;", re.I)
_NAME = re.compile(r'\bname\s*=\s*"([^"\r\n]{1,160})"\s*;', re.I)
_POSITION = re.compile(
    r"\bposition\s*\[\s*\]\s*=\s*\{\s*(" + _NUM
    + r")\s*,\s*(" + _NUM + r")\s*(?:,\s*" + _NUM + r")?\s*\}\s*;",
    re.I,
)


@dataclass(frozen=True, slots=True)
class MapLocation:
    name: str
    x: float
    z: float


@dataclass(frozen=True, slots=True)
class WorldMetadata:
    land_grid: float | None = None
    locations: tuple[MapLocation, ...] = ()
    source: str = ""


def _clean_comments(text: str) -> str:
    # Preserve offsets and ignore braces in comments. Strings are handled
    # when finding braces so labels containing delimiters remain safe.
    pattern = re.compile(r'"(?:\\.|[^"\\])*"|/\*[\s\S]*?\*/|//[^\r\n]*')
    def sub(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.startswith('"') else " " * len(token)
    return pattern.sub(sub, text)


def _block(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for at in range(start, len(text)):
        char = text[at]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:at]
    return None


def _class_body(text: str, name: str) -> str | None:
    for match in _CLASS.finditer(text):
        if match.group(1).casefold() == name.casefold():
            return _block(text, match.end() - 1)
    return None


def parse_world_config(
    source: str | bytes, world_name: str, *, label: str = "config.cpp",
) -> WorldMetadata:
    if isinstance(source, bytes):
        if len(source) > _MAX_CONFIG_BYTES:
            return WorldMetadata()
        if source.startswith(b"\x00raP"):
            try:
                result = parse_rap_world(source, world_name)
            except ValueError:
                return WorldMetadata()
            if result is None:
                return WorldMetadata()
            grid, locations = result
            return WorldMetadata(
                grid, tuple(MapLocation(*place) for place in locations), label,
            )
        source = source.decode("latin-1", "replace")
    if len(source) > _MAX_CONFIG_BYTES:
        return WorldMetadata()
    cleaned = _clean_comments(source)
    worlds = _class_body(cleaned, "CfgWorlds")
    world = _class_body(worlds, world_name) if worlds is not None else None
    if world is None:
        return WorldMetadata()
    match = _GRID.search(world)
    grid: float | None = None
    if match:
        value = float(match.group(1))
        if math.isfinite(value) and 0 < value <= 10000:
            grid = value
    locations: list[MapLocation] = []
    names = _class_body(world, "Names")
    if names is not None:
        for entry in _CLASS.finditer(names):
            body = _block(names, entry.end() - 1)
            if body is None:
                continue
            place = _NAME.search(body)
            position = _POSITION.search(body)
            if place and position:
                x, z = float(position.group(1)), float(position.group(2))
                if math.isfinite(x) and math.isfinite(z):
                    locations.append(MapLocation(place.group(1), x, z))
            if len(locations) >= _MAX_LOCATIONS:
                break
    return WorldMetadata(grid, tuple(locations), label)


def load_loose_world_metadata(world: Path) -> WorldMetadata:
    # Source configs take precedence when both formats exist, matching
    # CWA addon loading. Read RAP content by its signature, not extension.
    candidates = tuple(
        base / name
        for base in (world.parent, world.parent.parent)
        for name in ("config.cpp", "config.bin")
    )
    for candidate in candidates:
        try:
            if candidate.stat().st_size <= _MAX_CONFIG_BYTES:
                metadata = parse_world_config(
                    candidate.read_bytes(), world.stem, label=str(candidate),
                )
                if metadata.source:
                    return metadata
        except OSError:
            continue
    return WorldMetadata()
