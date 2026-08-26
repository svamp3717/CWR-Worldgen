# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified model-space connector geometry for OFP/CWA stock road P3Ds.

These values come from the ODOL7 Memory LOD connector selections rather than
from filename conventions or visual bounding-box guesses. WRP placement does
not scale a P3D, so fitting must use these physical dimensions exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

STOCK_CURVE_ANGLE_DEGREES = 10.0
STOCK_JUNCTION_CONNECTOR_RADIUS_METRES = 6.25
STOCK_STRAIGHT_LENGTHS_METRES = {25: 25.0, 12: 12.5, 6: 6.25}
STOCK_HALF_WIDTHS_METRES = {
    "sil": 4.55,
    "kos": 4.55,
    "asf": 3.50,
    "ces": 1.75,
}

_STRAIGHT = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)(?P<length>25|12|6)\.p3d$",
    re.IGNORECASE,
)
_CURVE = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)10 (?P<radius>25|50|75|100)\.p3d$",
    re.IGNORECASE,
)
_T_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?:sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_X_JUNCTION = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StockCurveConnectors:
    family: str
    radius_metres: float
    begin: tuple[float, float]
    end: tuple[float, float]
    chord_length_metres: float
    local_chord_heading_degrees: float


def stock_straight_match(model_path: str) -> re.Match[str] | None:
    return _STRAIGHT.fullmatch(str(model_path).replace("/", "\\"))


def stock_curve_match(model_path: str) -> re.Match[str] | None:
    return _CURVE.fullmatch(str(model_path).replace("/", "\\"))


def stock_straight_length(model_path: str) -> float | None:
    match = stock_straight_match(model_path)
    if match is None:
        return None
    return STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))]


def stock_curve_connectors(model_path: str) -> StockCurveConnectors | None:
    """Return the actual Memory-LOD centerline connectors for a stock curve.

    Stock curves turn right by ten degrees when traversed from ``begin`` to
    ``end``. Their Memory LOD confirms the centerline chord itself is headed five
    degrees relative to model +Z. Because the ODOL is autocentered around the
    full road strip rather than the centerline chord, the chord midpoint is not
    at the model origin. The offset is determined by the road-family half width.
    """

    match = stock_curve_match(model_path)
    if match is None:
        return None
    family = match.group("family").casefold()
    radius = float(match.group("radius"))
    half_width = STOCK_HALF_WIDTHS_METRES[family]
    angle = math.radians(STOCK_CURVE_ANGLE_DEGREES)
    half_angle = angle * 0.5
    chord = 2.0 * radius * math.sin(half_angle)
    chord_unit = (math.sin(half_angle), math.cos(half_angle))

    # Verified against the Memory-LOD lb/pb and le/pe selection midpoints.
    midpoint = (
        half_width * (1.0 - math.cos(angle)) * 0.5,
        -half_width * math.sin(angle) * 0.5,
    )
    begin = (
        midpoint[0] - chord_unit[0] * chord * 0.5,
        midpoint[1] - chord_unit[1] * chord * 0.5,
    )
    end = (
        midpoint[0] + chord_unit[0] * chord * 0.5,
        midpoint[1] + chord_unit[1] * chord * 0.5,
    )
    return StockCurveConnectors(
        family=family,
        radius_metres=radius,
        begin=begin,
        end=end,
        chord_length_metres=chord,
        local_chord_heading_degrees=STOCK_CURVE_ANGLE_DEGREES * 0.5,
    )


def rotate_local(point: tuple[float, float], heading_degrees: float) -> tuple[float, float]:
    """Rotate a model-local X/Z point into world X/Z using WRP yaw semantics."""

    angle = math.radians(float(heading_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, z = point
    return cosine * x + sine * z, -sine * x + cosine * z


def transform_local(
    point: tuple[float, float],
    origin: tuple[float, float],
    heading_degrees: float,
) -> tuple[float, float]:
    offset = rotate_local(point, heading_degrees)
    return origin[0] + offset[0], origin[1] + offset[1]


def solve_planar_connector_transform(
    world_begin: tuple[float, float],
    world_end: tuple[float, float],
    local_begin: tuple[float, float],
    local_end: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Return object origin and yaw mapping local connectors onto world endpoints."""

    local_dx = local_end[0] - local_begin[0]
    local_dz = local_end[1] - local_begin[1]
    world_dx = world_end[0] - world_begin[0]
    world_dz = world_end[1] - world_begin[1]
    local_length = math.hypot(local_dx, local_dz)
    world_length = math.hypot(world_dx, world_dz)
    if local_length <= 1.0e-9 or world_length <= 1.0e-9:
        raise ValueError("road connectors must have non-zero length")
    if not math.isclose(local_length, world_length, rel_tol=0.0, abs_tol=1.0e-3):
        raise ValueError(
            f"world connector chord {world_length:.6f} m does not match model chord "
            f"{local_length:.6f} m"
        )

    local_heading = math.degrees(math.atan2(local_dx, local_dz))
    world_heading = math.degrees(math.atan2(world_dx, world_dz))
    heading = (world_heading - local_heading) % 360.0
    begin_offset = rotate_local(local_begin, heading)
    origin = (
        world_begin[0] - begin_offset[0],
        world_begin[1] - begin_offset[1],
    )

    mapped_end = transform_local(local_end, origin, heading)
    if math.dist(mapped_end, world_end) > 1.0e-3:
        raise ValueError("failed to solve stock-road connector transform")
    return origin, heading


def native_junction_intersection_offset(model_path: str) -> tuple[float, float] | None:
    """Return model-local position of the logical road-intersection center."""

    path = str(model_path).replace("/", "\\")
    match = _T_JUNCTION.fullmatch(path)
    if match is not None:
        main = match.group("main").casefold()
        # T models are autocentered around their asymmetric mesh. Memory-LOD
        # connector centers lie 6.25 m from this logical intersection point.
        x = (
            STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
            - STOCK_HALF_WIDTHS_METRES[main]
        ) * 0.5
        return x, 0.0
    if _X_JUNCTION.fullmatch(path) is not None:
        return 0.0, 0.0
    return None
