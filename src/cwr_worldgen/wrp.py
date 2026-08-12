# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import struct
from typing import Iterable, Sequence

from .model import WorldObject, encode_wire_path

_RVW4_HEADER = struct.Struct("<4sii")
_RVW4_OBJECT = struct.Struct("<12fi76s")
_TEXTURE_RECORDS = 512
_TEXTURE_PATH_BYTES = 32


@dataclass(frozen=True, slots=True)
class WrpSummary:
    width: int
    height: int
    minimum_height: float
    maximum_height: float
    texture_paths: tuple[str, ...]
    texture_slots: tuple[str, ...]
    texture_index_counts: tuple[int, ...]
    object_count: int
    object_ids: tuple[int, ...]
    object_models: tuple[str, ...]
    object_positions: tuple[tuple[float, float, float], ...]
    has_object_terminator: bool


def quantize_height(height: float, scale: float) -> int:
    if scale <= 0:
        raise ValueError("height scale must be positive")
    raw = int(round(height / scale))
    if not -32768 <= raw <= 32767:
        raise ValueError(f"height {height} m cannot be represented at scale {scale}")
    return raw


def quantize_elevations(elevations: Sequence[float], scale: float) -> tuple[float, ...]:
    """Return the exact metre values that RVW4 will store.

    Object grounding must use this grid rather than the higher-precision terrain
    solver output. Otherwise the WRP can round the ground up after an object has
    already been placed, which is a remarkably efficient way to bury small props.
    """

    return tuple(quantize_height(float(value), scale) * scale for value in elevations)


def write_rvw4(
    path: Path,
    width: int,
    height: int,
    elevations: Sequence[float],
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
    objects: Iterable[WorldObject],
    *,
    height_scale: float,
) -> None:
    expected = width * height
    if width <= 0 or height <= 0:
        raise ValueError("terrain dimensions must be positive")
    if len(elevations) != expected:
        raise ValueError(f"expected {expected} elevations, got {len(elevations)}")
    if len(texture_indices) != expected:
        raise ValueError(f"expected {expected} texture indices, got {len(texture_indices)}")
    if not texture_paths or len(texture_paths) > _TEXTURE_RECORDS:
        raise ValueError("RVW4 requires between 1 and 512 texture table entries")

    encoded_textures = [
        encode_wire_path(value, _TEXTURE_PATH_BYTES - 1, "terrain texture path")
        for value in texture_paths
    ]
    maximum_index = len(texture_paths) - 1
    for index in texture_indices:
        if not 0 <= index <= maximum_index:
            raise ValueError(f"texture index {index} is outside 0..{maximum_index}")

    checked_objects = list(objects)
    seen_ids: set[int] = set()
    for obj in checked_objects:
        obj.validate()
        if obj.object_id in seen_ids:
            raise ValueError(f"duplicate object ID {obj.object_id}")
        seen_ids.add(obj.object_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_RVW4_HEADER.pack(b"4WVR", width, height))
        for elevation in elevations:
            stream.write(struct.pack("<h", quantize_height(elevation, height_scale)))
        for index in texture_indices:
            # The engine reads this field as a signed short. Valid texture slots
            # are non-negative, so the wire representation is identical.
            stream.write(struct.pack("<h", index))
        for slot in range(_TEXTURE_RECORDS):
            encoded = encoded_textures[slot] if slot < len(encoded_textures) else b""
            stream.write(encoded.ljust(_TEXTURE_PATH_BYTES, b"\0"))
        for obj in checked_objects:
            model = encode_wire_path(obj.model_path, 75, "model path").ljust(76, b"\0")
            stream.write(_RVW4_OBJECT.pack(*obj.matrix_4x3(), obj.object_id, model))

        # Landscape::SaveData always writes a complete empty SingleObject4
        # record to terminate the object list, even when the map has no objects.
        # Reaching physical EOF instead leaves the input stream in a failed
        # state and is not a serializer-compatible 4WVR file.
        stream.write(bytes(_RVW4_OBJECT.size))


def inspect_rvw4(path: Path, *, height_scale: float) -> WrpSummary:
    data = path.read_bytes()
    stream = io.BytesIO(data)
    header = stream.read(_RVW4_HEADER.size)
    if len(header) != _RVW4_HEADER.size:
        raise ValueError("truncated RVW4 header")
    magic, width, height = _RVW4_HEADER.unpack(header)
    if magic != b"4WVR":
        raise ValueError("not an RVW4 file")
    if width <= 0 or height <= 0:
        raise ValueError("invalid RVW4 dimensions")

    cells = width * height
    raw_heights = stream.read(cells * 2)
    if len(raw_heights) != cells * 2:
        raise ValueError("truncated RVW4 height grid")
    unpacked = struct.unpack(f"<{cells}h", raw_heights)
    elevations = [value * height_scale for value in unpacked]

    raw_indices = stream.read(cells * 2)
    if len(raw_indices) != cells * 2:
        raise ValueError("truncated RVW4 texture grid")
    unpacked_indices = struct.unpack(f"<{cells}h", raw_indices)
    if any(index < 0 for index in unpacked_indices):
        raise ValueError("RVW4 contains a negative texture index")

    texture_slots: list[str] = []
    texture_paths: list[str] = []
    for _ in range(_TEXTURE_RECORDS):
        record = stream.read(_TEXTURE_PATH_BYTES)
        if len(record) != _TEXTURE_PATH_BYTES:
            raise ValueError("truncated RVW4 texture table")
        value = record.split(b"\0", 1)[0]
        decoded = value.decode("ascii") if value else ""
        texture_slots.append(decoded)
        if decoded:
            texture_paths.append(decoded)

    maximum_texture_index = max(unpacked_indices, default=0)
    texture_index_counts = [0] * (maximum_texture_index + 1)
    for index in unpacked_indices:
        texture_index_counts[index] += 1

    object_ids: list[int] = []
    object_models: list[str] = []
    object_positions: list[tuple[float, float, float]] = []
    found_terminator = False
    while stream.tell() < len(data):
        remaining = len(data) - stream.tell()
        if remaining < _RVW4_OBJECT.size:
            raise ValueError("trailing partial RVW4 object record")
        record = stream.read(_RVW4_OBJECT.size)
        values = _RVW4_OBJECT.unpack(record)
        model = values[13].split(b"\0", 1)[0]
        if not model:
            found_terminator = True
            if stream.tell() != len(data):
                raise ValueError("data appears after the RVW4 object terminator")
            break
        object_ids.append(values[12])
        object_models.append(model.decode("ascii"))
        object_positions.append((values[9], values[10], values[11]))

    if not found_terminator:
        raise ValueError("RVW4 object list is missing its 128-byte terminator")

    return WrpSummary(
        width=width,
        height=height,
        minimum_height=min(elevations),
        maximum_height=max(elevations),
        texture_paths=tuple(texture_paths),
        texture_slots=tuple(texture_slots),
        texture_index_counts=tuple(texture_index_counts),
        object_count=len(object_ids),
        object_ids=tuple(object_ids),
        object_models=tuple(object_models),
        object_positions=tuple(object_positions),
        has_object_terminator=found_terminator,
    )
