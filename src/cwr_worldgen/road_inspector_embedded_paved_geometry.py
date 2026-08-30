# SPDX-License-Identifier: GPL-3.0-or-later
"""Use the PBO's actual generated paved helper geometry during inspection.

Generated paved wedges are world-local P3Ds.  Their filename only records the
quantized turn angle, not the generator revision that authored the triangle.
Using today's ``paved_wedge_local_points`` for an older PBO can therefore make
Road Inspector believe a narrow historical helper is as wide as the current
model and suppress a grass wedge that CWA still renders.

This outer inspector layer reads the first (visual) MLOD triangle from every
embedded ``paved_wedge_qNNN.p3d`` and temporarily makes the existing coverage
layers test that real footprint.  Dirt/gravel is deliberately untouched.
"""
from __future__ import annotations

import math
from pathlib import Path
import struct

from . import road_inspector as _core
from .pbo import read_pbo


_MLOD_HEADER = struct.Struct("<4sBBHI")
_SP3X_HEADER = struct.Struct("<4siiiiii")
_POINT = struct.Struct("<fffi")

Triangle = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]

_ORIGINAL_INSPECT = None
_ORIGINAL_PAVED_WEDGE_CONTAINS = None
_INSTALLED = False
_MISSING = object()


def _basename(model_path: str) -> str:
    return str(model_path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(first[0]) * float(second[1]) - float(first[1]) * float(second[0])


def _embedded_wedge_triangle(data: bytes) -> Triangle | None:
    """Return the first visual MLOD triangle from one generated wedge P3D."""

    if len(data) < _MLOD_HEADER.size + _SP3X_HEADER.size:
        return None
    signature, _major, _minor, _padding, lod_count = _MLOD_HEADER.unpack_from(data, 0)
    if signature != b"MLOD" or lod_count < 1:
        return None

    offset = _MLOD_HEADER.size
    sp3x, head_size, _version, point_count, _normals, _faces, _flags = (
        _SP3X_HEADER.unpack_from(data, offset)
    )
    if (
        sp3x != b"SP3X"
        or head_size < _SP3X_HEADER.size
        or point_count != 3
    ):
        return None
    offset += head_size
    if offset + point_count * _POINT.size > len(data):
        return None

    points: list[tuple[float, float]] = []
    for index in range(point_count):
        x, _y, z, _flags = _POINT.unpack_from(data, offset + index * _POINT.size)
        if not math.isfinite(x) or not math.isfinite(z):
            return None
        points.append((float(x), float(z)))

    area_twice = abs(
        _cross(
            (points[1][0] - points[0][0], points[1][1] - points[0][1]),
            (points[2][0] - points[0][0], points[2][1] - points[0][1]),
        )
    )
    if area_twice <= 1.0e-9:
        return None
    return points[0], points[1], points[2]


def _embedded_wedge_footprints(input_path: Path) -> dict[str, Triangle | None]:
    """Return actual world-local paved-wedge footprints keyed by P3D basename."""

    path = Path(input_path)
    if path.suffix.casefold() != ".pbo":
        return {}
    try:
        entries = read_pbo(path)
    except (OSError, ValueError):
        return {}

    footprints: dict[str, Triangle | None] = {}
    for entry in entries:
        filename = _basename(entry.name)
        if _core.paved_wedge_angle_degrees(filename) is None:
            continue
        # Record even a malformed model.  If a helper exists in the PBO but its
        # visual footprint cannot be proven, coverage must fail closed rather
        # than silently substituting today's wider procedural model.
        footprints[filename] = _embedded_wedge_triangle(entry.data)
    return footprints


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return math.dist(point, start)
    fraction = (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[1]) - float(start[1])) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    nearest = (
        float(start[0]) + dx * fraction,
        float(start[1]) + dz * fraction,
    )
    return math.dist(point, nearest)


def _triangle_contains_with_margin(
    triangle: Triangle,
    point: tuple[float, float],
    margin: float,
) -> bool:
    """Return whether a point lies in/on a triangle within a tiny edge margin."""

    a, b, c = triangle

    def side(start: tuple[float, float], end: tuple[float, float]) -> float:
        return _cross(
            (float(end[0]) - float(start[0]), float(end[1]) - float(start[1])),
            (float(point[0]) - float(start[0]), float(point[1]) - float(start[1])),
        )

    values = (side(a, b), side(b, c), side(c, a))
    epsilon = 1.0e-9
    inside = not (
        any(value < -epsilon for value in values)
        and any(value > epsilon for value in values)
    )
    if inside:
        return True

    allowance = max(0.0, float(margin))
    if allowance <= 0.0:
        return False
    return min(
        _point_segment_distance(point, a, b),
        _point_segment_distance(point, b, c),
        _point_segment_distance(point, c, a),
    ) <= allowance


def _embedded_wedge_contains(
    road,
    point: tuple[float, float],
    triangle: Triangle,
    *,
    margin: float = 0.0,
) -> bool:
    """Test a world point against the actual embedded wedge visual triangle."""

    dx = float(point[0]) - float(road.logical_center[0])
    dz = float(point[1]) - float(road.logical_center[1])
    heading = math.radians(float(road.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(road.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return False
    local_z = (
        dx * math.sin(heading) + dz * math.cos(heading)
    ) / cosine_pitch
    return _triangle_contains_with_margin(
        triangle,
        (local_x, local_z),
        margin,
    )


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    if _ORIGINAL_INSPECT is None or _ORIGINAL_PAVED_WEDGE_CONTAINS is None:
        raise RuntimeError("embedded paved geometry audit is not installed")

    footprints = _embedded_wedge_footprints(Path(input_path))
    original_contains = _core._paved_wedge_contains

    def contains_actual_embedded_wedge(road, point, *, margin: float = 0.0):
        filename = _basename(road.model_path)
        triangle = footprints.get(filename, _MISSING)
        if triangle is _MISSING:
            return _ORIGINAL_PAVED_WEDGE_CONTAINS(road, point, margin=margin)
        if triangle is None:
            return False
        return _embedded_wedge_contains(
            road,
            point,
            triangle,
            margin=margin,
        )

    # The existing coverage and final paved-wedge audit both resolve this helper
    # dynamically.  Patch it only for this inspection call so every layer sees
    # the PBO's real helper mesh rather than the current source-tree recipe.
    _core._paved_wedge_contains = contains_actual_embedded_wedge
    try:
        return _ORIGINAL_INSPECT(
            input_path,
            roads_geojson=roads_geojson,
            endpoint_tolerance=endpoint_tolerance,
            minimum_edge_gap=minimum_edge_gap,
            minimum_tangent_error=minimum_tangent_error,
            junction_match_tolerance=junction_match_tolerance,
        )
    finally:
        _core._paved_wedge_contains = original_contains


def install() -> None:
    """Install the version-accurate embedded paved-helper coverage layer."""

    global _ORIGINAL_INSPECT, _ORIGINAL_PAVED_WEDGE_CONTAINS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_PAVED_WEDGE_CONTAINS = _core._paved_wedge_contains
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
