# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded P3D metadata classification for CWA Island Studio.

File/path heuristics remain the fallback when a model cannot be found or has no
relevant metadata. Positive metadata comes from actual MLOD TAGG properties
or legacy ODOL v6/v7 LOD properties / model-level MapType, not mesh geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
import struct
from typing import Sequence

from .asset_sources import roots
from .pbo_textures import PboTextures, decode_lzss_stream
from .textures import _safe_relative

_MAX_MODEL = 32 * 1024 * 1024
_MAX_LODS = 128
_MAX_POINTS = 1_000_000
_MAX_TAGS = 100_000


@dataclass(frozen=True, slots=True)
class ModelCategory:
    road: bool = False
    forest: bool = False


# Engine's CWA MapTypes, see cwr_worldgen/odol_v7.py.
_FOREST_MAP_IDS = frozenset((0, 1, 2, 5, 6, 7))
_MAP_FOREST = frozenset((
    "tree", "small tree", "bush", "forest border", "forest triangle",
    "forest square",
))
_MAP_ROAD = frozenset(("road",))
_CLASS_FOREST = frozenset(("tree", "bush", "bushsoft", "forest"))
_CLASS_ROAD = frozenset(("road",))


def _classify_properties(properties: dict[str, str]) -> ModelCategory | None:
    map_type = properties.get("map", "").strip().casefold()
    object_class = properties.get("class", "").strip().casefold()
    if map_type in _MAP_FOREST or object_class in _CLASS_FOREST:
        return ModelCategory(forest=True)
    if map_type in _MAP_ROAD or object_class in _CLASS_ROAD:
        return ModelCategory(road=True)
    # MapHide suppresses the 2D game-map icon. It does not mean the
    # underlying model is not a road or vegetation object. For example,
    # generated roads use map=hide plus class=road in another LOD.
    if map_type and map_type != "hide" or object_class:
        return ModelCategory()
    return None


def _count(stream: io.BytesIO, cap: int = _MAX_POINTS) -> int:
    data = stream.read(4)
    if len(data) != 4:
        raise ValueError("truncated P3D count")
    count = struct.unpack("<I", data)[0]
    if count > cap:
        raise ValueError("implausible P3D array count")
    return count


def _read(stream: io.BytesIO, size: int) -> bytes:
    if size < 0 or size > _MAX_MODEL or stream.tell() + size > len(stream.getbuffer()):
        raise ValueError("truncated or oversized P3D field")
    return stream.read(size)


def _skip(stream: io.BytesIO, size: int) -> None:
    if size < 0 or size > _MAX_MODEL or stream.tell() + size > len(stream.getbuffer()):
        raise ValueError("truncated or oversized P3D field")
    stream.seek(size, io.SEEK_CUR)


def _cstring(stream: io.BytesIO) -> str:
    start = stream.tell()
    end = stream.getvalue().find(b"\x00", start, min(start + 1024, len(stream.getbuffer())))
    if end < 0:
        raise ValueError("unterminated P3D property")
    stream.seek(end + 1)
    return _read_text(stream.getbuffer()[start:end])


def _read_text(data: bytes | memoryview) -> str:
    return bytes(data).decode("latin-1", "replace").strip()


def _array(stream: io.BytesIO, width: int) -> None:
    count = _count(stream)
    size = count * width
    if size > _MAX_MODEL:
        raise ValueError("oversized ODOL array")
    if size >= 1024:
        decode_lzss_stream(stream, size)
    else:
        _skip(stream, size)


def _read_mlod(data: bytes) -> ModelCategory | None:
    stream = io.BytesIO(data)
    _skip(stream, 8)
    lod_count = _count(stream, _MAX_LODS)
    choices: list[tuple[float, dict[str, str]]] = []
    for _ in range(lod_count):
        if _read(stream, 4) != b"SP3X":
            raise ValueError("unsupported MLOD LOD header")
        header_size, _version, points, normals, faces, _flags = struct.unpack(
            "<6I", _read(stream, 24)
        )
        if not 28 <= header_size <= 4096 or max(points, normals, faces) > _MAX_POINTS:
            raise ValueError("invalid MLOD table sizes")
        _skip(stream, header_size - 28 + points * 16 + normals * 12 + faces * 104)
        if _read(stream, 4) != b"TAGG":
            raise ValueError("missing MLOD TAGG")
        properties: dict[str, str] = {}
        for _ in range(_MAX_TAGS):
            name, size = struct.unpack("<64sI", _read(stream, 68))
            token = name.split(b"\0", 1)[0]
            payload = _read(stream, size)
            if token == b"#Property#" and size == 128:
                key = _read_text(payload[:64].split(b"\0", 1)[0]).casefold()
                value = _read_text(payload[64:].split(b"\0", 1)[0])
                properties[key] = value
            if token == b"#EndOfFile#":
                break
        else:
            raise ValueError("unterminated MLOD TAGG")
        (resolution,) = struct.unpack("<f", _read(stream, 4))
        if not math.isfinite(resolution):
            raise ValueError("invalid MLOD LOD resolution")
        choices.append((resolution, properties))
    # Geometry LOD wins, as in Poseidon's MLOD property resolver.
    ordered = [props for resolution, props in choices if math.isclose(resolution, 1e13, rel_tol=1e-6)]
    ordered.extend(props for _resolution, props in choices[:1])
    for props in ordered:
        found = _classify_properties(props)
        if found is not None:
            return found
    return None


def _read_odol(data: bytes) -> ModelCategory | None:
    stream = io.BytesIO(data)
    _skip(stream, 8)
    lod_count = _count(stream, _MAX_LODS)
    lod_properties: list[dict[str, str]] = []
    for _ in range(lod_count):
        _array(stream, 4)  # point flags
        _array(stream, 8)  # UVs
        _skip(stream, _count(stream) * 12)  # vertices
        _skip(stream, _count(stream) * 12 + 48)  # normals, bounds
        for _ in range(_count(stream, _MAX_TAGS)):
            _cstring(stream)  # textures
        _array(stream, 2)  # vertex index arrays
        _array(stream, 2)
        face_count = _count(stream)
        _skip(stream, 4)  # face allocation size
        for _ in range(face_count):
            _skip(stream, 6)
            corners = _read(stream, 1)[0]
            if corners > 4:
                raise ValueError("invalid ODOL face")
            _skip(stream, corners * 2)
        _skip(stream, _count(stream) * 18)  # draw sections
        for _ in range(_count(stream, _MAX_TAGS)):
            _cstring(stream)  # named selection
            for width in (2, 1, 4):
                _array(stream, width)
            _skip(stream, 1)
            for width in (4, 2, 1):
                _array(stream, width)
        props: dict[str, str] = {}
        for _ in range(_count(stream, _MAX_TAGS)):
            key = _cstring(stream).casefold()
            props[key] = _cstring(stream)
        lod_properties.append(props)
        for _ in range(_count(stream, _MAX_TAGS)):
            _skip(stream, 4)
            _skip(stream, _count(stream) * 12)
        _skip(stream, 12)  # colours + flags
        for _ in range(_count(stream, _MAX_TAGS)):
            _cstring(stream)
            _skip(stream, 56)  # proxy matrix + indices
    resolutions = struct.unpack("<" + "f" * lod_count, _read(stream, 4 * lod_count))
    if not all(math.isfinite(value) for value in resolutions):
        raise ValueError("invalid ODOL resolutions")
    ordered = [p for res, p in zip(resolutions, lod_properties)
               if math.isclose(res, 1e13, rel_tol=1e-6)]
    ordered.extend(lod_properties[:1])
    for props in ordered:
        found = _classify_properties(props)
        if found is not None:
            return found
    # After the resolution table, the global MapType byte follows Shape
    # metadata (see CWA ShapeFile.cpp / cwr_worldgen.odol_v7).
    _skip(stream, 4 + 8 + 12 + 12 + 8 + 4 + 12 * 5 + 36 + 5)
    (map_type,) = struct.unpack("<b", _read(stream, 1))
    if map_type in _FOREST_MAP_IDS:
        return ModelCategory(forest=True)
    if map_type == 22:  # MapHide: no explicit category
        return None
    if 0 <= map_type <= 23:
        return ModelCategory()
    return None


def read_p3d_category(data: bytes) -> ModelCategory | None:
    """Return explicit road/forest metadata or None when not available."""
    if not 12 <= len(data) <= _MAX_MODEL:
        return None
    try:
        if data[:4] == b"MLOD":
            return _read_mlod(data)
        if data[:4] == b"ODOL" and struct.unpack_from("<I", data, 4)[0] in (6, 7):
            return _read_odol(data)
    except (ValueError, OverflowError, struct.error, BufferError):
        return None
    return None


def load_p3d_category(
    virtual_path: str, root: Path | str | Sequence[Path | str] | None,
    archives: PboTextures | None,
) -> ModelCategory | None:
    """Look up only selected model assets, respecting loose-folder priority."""
    relative = _safe_relative(virtual_path)
    if relative is None or relative.suffix.casefold() != ".p3d":
        return None
    for base in reversed(roots(root)):
        candidate = base / relative
        try:
            if candidate.is_file() and candidate.stat().st_size <= _MAX_MODEL:
                return read_p3d_category(candidate.read_bytes())
        except OSError:
            continue
    if archives is not None:
        try:
            contents = archives.read_model(virtual_path)
        except (OSError, ValueError):
            return None
        if contents is not None:
            return read_p3d_category(contents)
    return None
