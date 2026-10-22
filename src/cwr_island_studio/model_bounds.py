# SPDX-License-Identifier: GPL-3.0-or-later
"""On-demand first-LOD ground-plane outlines for Island Studio previews.

Only models visible in the viewport are measured. Supported MLOD/SP3X and
OFP/CWA ODOL v6/v7 supply origin-relative X/Z vertices for compact convex
outlines, with rectangular bounds retained as a safe fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import io
import math
import struct

import numpy as np

from .pbo_textures import PboTextures, decode_lzss_stream
from .textures import _safe_relative
from .asset_sources import roots
from typing import Sequence

_MAX_POINTS = 1_000_000
_MAX_MODEL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    # First visual LOD's projected, simplified convex hull. Keep legacy
    # rectangular bounds in equality comparisons and for missing geometry.
    outline: tuple[tuple[float, float], ...] = field(
        default=(), compare=False, repr=False,
    )

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



def _projected_outline(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    """A small, convex top-down outline of the first visual LOD's X/Z points.

    Sampling limits expensive hull construction on very dense models. Preserve
    the four axis extrema when sampling so the outline stays in the measured
    footprint. This is a silhouette approximation, not a game collision mesh.
    """
    projected = points[:, (0, 2)]
    if len(projected) > 8192:
        indices = np.linspace(0, len(projected) - 1, 8192, dtype=np.intp)
        extrema = [
            int(np.argmin(projected[:, axis])) for axis in (0, 1)
        ] + [
            int(np.argmax(projected[:, axis])) for axis in (0, 1)
        ]
        projected = projected[np.concatenate((indices, extrema))]
    positions = sorted({(float(x), float(z)) for x, z in projected})
    if len(positions) < 3:
        return ()

    def cross(
        a: tuple[float, float], b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower: list[tuple[float, float]] = []
    for point in positions:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(positions):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) > 48:
        # Preserve the outer support silhouette using no more than 48 Tk
        # vertices; thousands of visible objects must remain inexpensive.
        picked: set[int] = set()
        for index in range(48):
            theta = math.tau * index / 48
            co, si = math.cos(theta), math.sin(theta)
            picked.add(max(
                range(len(hull)),
                key=lambda i: hull[i][0] * co + hull[i][1] * si,
            ))
        hull = [hull[i] for i in sorted(picked)]
    return tuple(hull) if len(hull) >= 3 else ()


def _vertices_bounds(points: np.ndarray) -> ModelBounds:
    if len(points) == 0:
        raise ValueError("model contains no vertices")
    xs = points[:, 0]
    zs = points[:, 2]
    if not bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(zs))):
        raise ValueError("non-finite model bounds")
    bounds = ModelBounds(
        float(xs.min()), float(xs.max()), float(zs.min()), float(zs.max()),
        outline=_projected_outline(points),
    )
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
    virtual_path: str,
    root: Path | str | Sequence[Path | str] | None,
    archives: PboTextures | None,
) -> ModelBounds | None:
    """Resolve and measure a loose P3D or selected PBO member if supported."""
    relative = _safe_relative(virtual_path)
    if relative is None or relative.suffix.casefold() != ".p3d":
        return None
    data: bytes | None = None
    for base in reversed(roots(root)):
        candidate = base / relative
        try:
            if candidate.is_file() and candidate.stat().st_size <= _MAX_MODEL_BYTES:
                data = candidate.read_bytes()
                break
        except OSError:
            continue
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



def footprint_outline(
    bounds: ModelBounds, x: float, z: float, heading: float,
    *, basis: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Transform actual top-down P3D outline by the original WRP X/Z basis.

    Unsupported or degenerate models use their old origin-relative rectangle.
    """
    if len(bounds.outline) < 3:
        return footprint_corners(bounds, x, z, heading, basis=basis)
    if basis is None:
        angle = math.radians(heading)
        co, si = math.cos(angle), math.sin(angle)
        basis = (co, si, -si, co)
    xx, xz, zx, zz = basis
    return tuple(
        (x + xx * lx + xz * lz, z + zx * lx + zz * lz)
        for lx, lz in bounds.outline
    )


def footprint_contains(
    corners: tuple[tuple[float, float], ...],
    x: float, z: float,
) -> bool:
    """Hit-test a convex projected outline or legacy four-corner rectangle."""
    # A line-like or point-like LOD has no selectable polygon area.
    if len(corners) < 3:
        return False
    first, second, third = corners[:3]
    area = (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )
    if abs(area) <= 1e-9:
        return False
    # A projected visual LOD can have any number of hull vertices (3..48).
    # The previous fixed-size loop crashed on triangles and ignored points
    # after the fourth vertex on larger model silhouettes.
    size = len(corners)
    crosses = [
        (corners[(i + 1) % size][0] - corners[i][0]) * (z - corners[i][1])
        - (corners[(i + 1) % size][1] - corners[i][1]) * (x - corners[i][0])
        for i in range(size)
    ]
    return min(crosses) >= -1e-7 or max(crosses) <= 1e-7
