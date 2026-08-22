# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import math
import os
import struct
import tempfile
from typing import Iterable, Sequence

import numpy as np

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


def _height_grid_bytes(elevations: Sequence[float], scale: float) -> bytes:
    """Vectorized RVW4 int16 quantization with ``round``-compatible ties-to-even."""

    if scale <= 0:
        raise ValueError("height scale must be positive")
    values = np.asarray(elevations, dtype=np.float64)
    if values.ndim != 1:
        values = values.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("terrain elevations must be finite")
    raw = np.rint(values / float(scale))
    if raw.size and (float(raw.min()) < -32768.0 or float(raw.max()) > 32767.0):
        # Preserve the useful per-value error from the scalar contract.
        for value in elevations:
            quantize_height(float(value), scale)
        raise ValueError("terrain elevation cannot be represented at requested scale")
    return raw.astype("<i2", copy=False).tobytes(order="C")


def _texture_grid_bytes(texture_indices: Sequence[int], maximum_index: int) -> bytes:
    values = np.asarray(texture_indices)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.dtype.kind not in {"b", "i", "u"}:
        # Do not silently truncate float/object values; ``struct.pack('<h', x)``
        # historically rejected them as well.
        for index in texture_indices:
            if not isinstance(index, (int, np.integer, bool)):
                raise TypeError("RVW4 texture indices must be integers")
        values = np.asarray(texture_indices, dtype=np.int64)
    if values.size:
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < 0 or maximum > maximum_index:
            bad = minimum if minimum < 0 else maximum
            raise ValueError(f"texture index {bad} is outside 0..{maximum_index}")
    return values.astype("<i2", copy=False).tobytes(order="C")


def _object_matrix_4x3_fast(obj: WorldObject) -> tuple[float, ...]:
    """Inline transform fast path used by the million-record WRP serializer."""

    heading = math.radians(obj.heading_degrees)
    cosine_heading = math.cos(heading)
    sine_heading = math.sin(heading)
    if obj.pitch_degrees == 0.0:
        # Preserve the exact signed-zero bytes produced by WorldObject.matrix_4x3.
        # RVW4 does not care about +/-0.0, but byte-identical regeneration does.
        sine_pitch = 0.0
        return (
            cosine_heading, 0.0, -sine_heading,
            -sine_heading * sine_pitch, 1.0, -cosine_heading * sine_pitch,
            sine_heading, sine_pitch, cosine_heading,
            obj.x, obj.y, obj.z,
        )
    pitch = math.radians(obj.pitch_degrees)
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    return (
        cosine_heading,
        0.0,
        -sine_heading,
        -sine_heading * sine_pitch,
        cosine_pitch,
        -cosine_heading * sine_pitch,
        sine_heading * cosine_pitch,
        sine_pitch,
        cosine_heading * cosine_pitch,
        obj.x,
        obj.y,
        obj.z,
    )


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
    renumber_object_ids: bool = False,
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
    height_bytes = _height_grid_bytes(elevations, height_scale)
    texture_bytes = _texture_grid_bytes(texture_indices, maximum_index)
    texture_table = b"".join(
        (encoded_textures[slot] if slot < len(encoded_textures) else b"").ljust(
            _TEXTURE_PATH_BYTES, b"\0"
        )
        for slot in range(_TEXTURE_RECORDS)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(_RVW4_HEADER.pack(b"4WVR", width, height))
            stream.write(height_bytes)
            stream.write(texture_bytes)
            stream.write(texture_table)

            # One megabyte keeps syscall count tiny without allocating a second
            # 128 MB object stream for a million-object world.
            records_per_chunk = 8192
            record_buffer = bytearray(_RVW4_OBJECT.size * records_per_chunk)
            offset = 0
            model_cache: dict[str, bytes] = {}
            seen_ids: set[int] | None = set() if not renumber_object_ids else None

            for wire_id, obj in enumerate(objects, 1):
                if obj.object_id < 0:
                    raise ValueError("object IDs must be non-negative")
                if seen_ids is not None:
                    if obj.object_id in seen_ids:
                        raise ValueError(f"duplicate object ID {obj.object_id}")
                    seen_ids.add(obj.object_id)
                    object_id = obj.object_id
                else:
                    object_id = wire_id

                for label, value in (("x", obj.x), ("y", obj.y), ("z", obj.z)):
                    if not math.isfinite(value):
                        raise ValueError(f"object {label} coordinate must be finite")
                if not math.isfinite(obj.heading_degrees):
                    raise ValueError("object heading must be finite")
                if not math.isfinite(obj.pitch_degrees) or not -89.0 < obj.pitch_degrees < 89.0:
                    raise ValueError("object pitch must be finite and within -89..89 degrees")

                model = model_cache.get(obj.model_path)
                if model is None:
                    model = encode_wire_path(obj.model_path, 75, "model path").ljust(76, b"\0")
                    # Generated worlds normally have hundreds or a few thousand
                    # models. Cap pathological unique-model input so the cache
                    # itself cannot become a million-entry memory problem.
                    if len(model_cache) < 16384:
                        model_cache[obj.model_path] = model

                _RVW4_OBJECT.pack_into(
                    record_buffer,
                    offset,
                    *_object_matrix_4x3_fast(obj),
                    object_id,
                    model,
                )
                offset += _RVW4_OBJECT.size
                if offset == len(record_buffer):
                    stream.write(record_buffer)
                    offset = 0
            if offset:
                stream.write(memoryview(record_buffer)[:offset])

            # Landscape::SaveData always writes a complete empty SingleObject4
            # record to terminate the object list, even when the map has no objects.
            stream.write(bytes(_RVW4_OBJECT.size))

        assert temporary_name is not None
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


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
