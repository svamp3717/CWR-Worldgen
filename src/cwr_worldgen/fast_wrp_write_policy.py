# SPDX-License-Identifier: GPL-3.0-or-later
"""Accelerate the generator's large RVW4 object stream without bypassing policies.

The terrain grids and texture table are already vectorized/small. Dense worlds spend
most of the RVW4 stage serializing 128-byte SingleObject4 records one at a time in
Python. For the normal generator path, where object IDs are renumbered on write and
objects are already materialized as a sequence, build those records in NumPy chunks.

Runway support wraps ``generator.write_rvw4`` so it can prepare the final terrain
texture table immediately before serialization. The fast serializer must therefore
sit *inside* that wrapper, not replace it. Direct/library calls keep the original
scalar writer and its exact validation contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import os
import tempfile

import numpy as np


_INSTALLED = False
_CHUNK_OBJECTS = 32_768
_RECORD_DTYPE = np.dtype([
    ("rotation", "<f4", (9,)),
    ("position", "<f4", (3,)),
    ("object_id", "<i4"),
    ("model", "S76"),
])
assert _RECORD_DTYPE.itemsize == 128


def _fast_write_rvw4(
    original,
    wrp,
    path: Path,
    width: int,
    height: int,
    elevations,
    texture_indices,
    texture_paths,
    objects,
    *,
    height_scale: float,
    renumber_object_ids: bool = False,
) -> None:
    """Vectorized generator-only serializer; fall back for general iterables/contracts."""
    if not renumber_object_ids or not isinstance(objects, Sequence):
        return original(
            path, width, height, elevations, texture_indices, texture_paths, objects,
            height_scale=height_scale, renumber_object_ids=renumber_object_ids,
        )

    expected = int(width) * int(height)
    if width <= 0 or height <= 0:
        raise ValueError("terrain dimensions must be positive")
    if len(elevations) != expected:
        raise ValueError(f"expected {expected} elevations, got {len(elevations)}")
    if len(texture_indices) != expected:
        raise ValueError(f"expected {expected} texture indices, got {len(texture_indices)}")
    if not texture_paths or len(texture_paths) > wrp._TEXTURE_RECORDS:
        raise ValueError("RVW4 requires between 1 and 512 texture table entries")

    object_count = len(objects)
    if object_count > 0x7FFFFFFF:
        raise ValueError("RVW4 object count exceeds signed 32-bit object IDs")

    encoded_textures = [
        wrp.encode_wire_path(value, wrp._TEXTURE_PATH_BYTES - 1, "terrain texture path")
        for value in texture_paths
    ]
    height_bytes = wrp._height_grid_bytes(elevations, height_scale)
    texture_bytes = wrp._texture_grid_bytes(texture_indices, len(texture_paths) - 1)
    texture_table = b"".join(
        (encoded_textures[slot] if slot < len(encoded_textures) else b"").ljust(
            wrp._TEXTURE_PATH_BYTES, b"\0"
        )
        for slot in range(wrp._TEXTURE_RECORDS)
    )

    from .progress import report_progress

    report_progress(86, f"Writing RVW4 world file ({object_count:,} objects)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    model_cache: dict[str, bytes] = {}
    next_report = 10

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(wrp._RVW4_HEADER.pack(b"4WVR", width, height))
            stream.write(height_bytes)
            stream.write(texture_bytes)
            stream.write(texture_table)

            for start in range(0, object_count, _CHUNK_OBJECTS):
                end = min(object_count, start + _CHUNK_OBJECTS)
                chunk = objects[start:end]
                rows: list[tuple[float, float, float, float, float]] = []
                models: list[bytes] = []

                for obj in chunk:
                    rows.append((
                        float(obj.x), float(obj.y), float(obj.z),
                        float(obj.heading_degrees), float(obj.pitch_degrees),
                    ))
                    model = model_cache.get(obj.model_path)
                    if model is None:
                        model = wrp.encode_wire_path(
                            obj.model_path, 75, "model path"
                        ).ljust(76, b"\0")
                        if len(model_cache) < 16_384:
                            model_cache[obj.model_path] = model
                    models.append(model)

                numeric = np.asarray(rows, dtype=np.float64)
                if numeric.size and not np.all(np.isfinite(numeric)):
                    raise ValueError("object coordinates and orientation must be finite")
                pitches = numeric[:, 4] if numeric.size else np.empty(0, dtype=np.float64)
                if pitches.size and np.any((pitches <= -89.0) | (pitches >= 89.0)):
                    raise ValueError("object pitch must be within -89..89 degrees")

                records = np.empty(end - start, dtype=_RECORD_DTYPE)
                heading = np.deg2rad(numeric[:, 3])
                pitch = np.deg2rad(pitches)
                ch = np.cos(heading)
                sh = np.sin(heading)
                cp = np.cos(pitch)
                sp = np.sin(pitch)
                rotation = records["rotation"]
                rotation[:, 0] = ch
                rotation[:, 1] = 0.0
                rotation[:, 2] = -sh
                rotation[:, 3] = -sh * sp
                rotation[:, 4] = cp
                rotation[:, 5] = -ch * sp
                rotation[:, 6] = sh * cp
                rotation[:, 7] = sp
                rotation[:, 8] = ch * cp
                records["position"] = numeric[:, :3]
                records["object_id"] = np.arange(start + 1, end + 1, dtype=np.int32)
                records["model"] = np.asarray(models, dtype="S76")
                stream.write(records.tobytes(order="C"))

                if object_count >= 100_000:
                    percent = int(end * 100 / object_count)
                    if percent >= next_report:
                        report_progress(
                            86,
                            f"Writing RVW4 objects {end:,}/{object_count:,} ({percent}%)",
                        )
                        next_report = ((percent // 10) + 1) * 10

            stream.write(bytes(wrp._RVW4_OBJECT.size))

        assert temporary_name is not None
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _make_fast_writer(original, wrp):
    def write_rvw4_fast(*args, **kwargs):
        return _fast_write_rvw4(original, wrp, *args, **kwargs)

    write_rvw4_fast.__name__ = "write_rvw4_fast"
    return write_rvw4_fast


def _install_writer_binding(generator, wrp, runway=None) -> str:
    """Install beneath the runway wrapper when it already owns the generator hook."""
    if (
        runway is not None
        and bool(getattr(runway, "_INSTALLED", False))
        and callable(getattr(runway, "_ORIGINAL_WRITE_RVW4", None))
    ):
        runway._ORIGINAL_WRITE_RVW4 = _make_fast_writer(
            runway._ORIGINAL_WRITE_RVW4, wrp
        )
        # Leave generator.write_rvw4 pointing at the runway wrapper. It prepares
        # revised texture indices/paths, then delegates here for fast serialization.
        return "runway-inner"

    generator.write_rvw4 = _make_fast_writer(generator.write_rvw4, wrp)
    return "generator"


def install_fast_wrp_write_policy() -> None:
    """Accelerate serialization while preserving any installed runway pre-write hook."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator
    from . import wrp
    try:
        from . import runway_surface_policy as runway
    except ImportError:
        runway = None

    _install_writer_binding(generator, wrp, runway)
    _INSTALLED = True
