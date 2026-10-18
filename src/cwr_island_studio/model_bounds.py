# SPDX-License-Identifier: GPL-3.0-or-later
"""On-demand first-LOD horizontal bounds for native 4WVR object markers.

Only models actually visible in the map viewport are measured. The supported
MLOD/SP3X and OFP/CWA ODOL v6/v7 layouts provide origin-relative X/Z bounds;
no guessed dimensions or entire-world model decoding are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import math
import struct

import numpy as np

from .pbo_textures import PboTextures, decode_lzss_stream
from .textures import _safe_relative

_MAX_POINTS = 1_000_000
_MAX_MODEL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def length(self) -> float:
        return self.max_z - self.min_z


def _exact(stream: io.BytesIO, count: int) -> bytes:
    if count < 0 or count > _MAX_MODEL_BYTES:
        raise ValueError("model array exceeds preview limits")
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated model data")
    return data


def _u32(stream: io.BytesIO) -> int:
    return struct.unpack("<I", _exact(stream, 4))[0]


def _vertices_bounds(points: np.ndarray) -> ModelBounds:
    if len(points) == 0:
        raise ValueError("model contains no vertices")
    xs = points[:, 0]
    zs = points[:, 2]
    if not bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(zs))):
        raise ValueError("non-finite model bounds")
    bounds = ModelBounds(float(xs.min()), float(xs.max()),
                         float(zs.min()), float(zs.max()))
    if bounds.width < 0 or bounds.length < 0:
        raise ValueError("invalid model bounds")
    return bounds


def read_model_bounds(data: bytes) -> ModelBounds:
    """Read first visual LOD footprint in metres relative to the model origin."""
    if len(data) < 16 or len(data) > _MAX_MODEL_BYTES:
        raise ValueError("unsupported model size")
    stream = io.BytesIO(data)
    signature = _exact(stream, 4)
    if signature == b"MLOD":
        _version = _u32(stream)
        lod_count = _u32(stream)
        if not 1 <= lod_count <= 128 or _exact(stream, 4) != b"SP3X":
            raise ValueError("invalid first MLOD LOD")
        major = _u32(stream)
        _minor = _u32(stream)
        count = _u32(stream)
        _normals = _u32(stream)
        _faces = _u32(stream)
        _flags = _u32(stream)
        if major not in (27, 28) or not 0 < count <= _MAX_POINTS:
            raise ValueError("unsupported MLOD point table")
        raw = _exact(stream, count * 16)
        points = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)[:, :3]
        return _vertices_bounds(points)

    if signature == b"ODOL":
        version = _u32(stream)
        lod_count = _u32(stream)
        if version not in (6, 7) or not 1 <= lod_count <= 128:
            raise ValueError("only CWA ODOL v6/v7 is supported")
        flags = _u32(stream)
        if flags > _MAX_POINTS:
            raise ValueError("invalid point flag count")
        flag_size = flags * 4
        if flag_size >= 1024:
            decode_lzss_stream(stream, flag_size)
        else:
            _exact(stream, flag_size)
        uvs = _u32(stream)
        if uvs > _MAX_POINTS:
            raise ValueError("invalid UV count")
        uv_size = uvs * 8
        if uv_size >= 1024:
            decode_lzss_stream(stream, uv_size)
        else:
            _exact(stream, uv_size)
        count = _u32(stream)
        if count != flags or count != uvs or not 0 < count <= _MAX_POINTS:
            raise ValueError("ODOL vertex table count mismatch")
        raw = _exact(stream, count * 12)
        points = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        return _vertices_bounds(points)

    raise ValueError("unsupported P3D model format")


def load_model_bounds(
    virtual_path: str, root: Path | str | None,
    archives: PboTextures | None,
) -> ModelBounds | None:
    """Resolve and measure a loose P3D or selected PBO member if supported."""
    relative = _safe_relative(virtual_path)
    if relative is None or relative.suffix.casefold() != ".p3d":
        return None
    data: bytes | None = None
    if root is not None:
        base = Path(root).resolve()
        candidate = base / relative
        try:
            if candidate.is_file() and candidate.stat().st_size <= _MAX_MODEL_BYTES:
                data = candidate.read_bytes()
        except OSError:
            pass
    if data is None and archives is not None:
        try:
            data = archives.read_model(virtual_path)
        except (OSError, ValueError):
            return None
    if data is None:
        return None
    try:
        return read_model_bounds(data)
    except (ValueError, struct.error, OverflowError):
        return None


def footprint_corners(
    bounds: ModelBounds, x: float, z: float, heading: float,
    *, basis: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Project the origin-relative bounds with the object's actual WRP basis."""
    if basis is None:
        angle = math.radians(heading)
        co, si = math.cos(angle), math.sin(angle)
        basis = (co, si, -si, co)
    xx, xz, zx, zz = basis
    return tuple(
        (x + xx * lx + xz * lz, z + zx * lx + zz * lz)
        for lx, lz in (
            (bounds.min_x, bounds.min_z),
            (bounds.max_x, bounds.min_z),
            (bounds.max_x, bounds.max_z),
            (bounds.min_x, bounds.max_z),
        )
    )


def footprint_contains(
    corners: tuple[tuple[float, float], ...],
    x: float, z: float,
) -> bool:
    """Hit-test the rotated four-corner preview rectangle."""
    crosses = [
        (corners[(i + 1) % 4][0] - corners[i][0]) * (z - corners[i][1])
        - (corners[(i + 1) % 4][1] - corners[i][1]) * (x - corners[i][0])
        for i in range(4)
    ]
    return min(crosses) >= -1e-7 or max(crosses) <= 1e-7
