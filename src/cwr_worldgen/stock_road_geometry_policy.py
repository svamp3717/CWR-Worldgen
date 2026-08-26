# SPDX-License-Identifier: GPL-3.0-or-later
"""Authoritative physical geometry for OFP/CWA stock road pieces.

Stock P3Ds are not scaled when they are placed in a WRP. Their fitted connector
spacing therefore has to come from the actual model geometry, not from the
configurable long-piece spacing used by custom roads. Ordinary bends are rounded
with bounded constant-radius fillets so the native ten-degree curves can be used
without pulling the road far away from the source centerline.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_curve_policy as _curve
from . import stock_road_model_geometry as _model_geometry

STOCK_CURVE_ANGLE_DEGREES = _model_geometry.STOCK_CURVE_ANGLE_DEGREES
_MAXIMUM_TANGENT_TURN_ERROR_DEGREES = 3.0
_MAXIMUM_STOCK_FILLET_DEVIATION_METRES = 0.45
_MAXIMUM_MICRO_BEND_DEVIATION_METRES = 0.35
_MAXIMUM_MICRO_BEND_DEGREES = 5.0
_STOCK_RADII_METRES = (100.0, 75.0, 50.0, 25.0)
_STOCK_STRAIGHT = re.compile(
    r"^(?:.*[\\/])(?P<family>sil|ces|asf|kos)(?P<length>25|12|6)\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_CURVE_CHAIN = _curve._stock_curve_chain
_STRAIGHT_FALLBACK_CHAIN = _curve._ORIGINAL_CHAIN
_ORIGINAL_VARIANTS = _p.road_model_variants
_ORIGINAL_ROUNDED_ROAD_RUN = _p._rounded_road_run
_ORIGINAL_QUALITY_PIECE_LENGTH = _quality._piece_length
_INSTALLED = False


def stock_curve_geometry(radius_nominal: int, scale: float = 1.0) -> tuple[float, float, float]:
    """Return the measured connector chord, turn angle and sagitta of a stock curve.

    ``scale`` is retained for compatibility with older callers but deliberately
    ignored. WRP transforms do not scale P3Ds.
    """

    if radius_nominal <= 0:
        raise ValueError("stock road curve radius must be positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("stock road curve scale must be positive and finite")
    radius = float(radius_nominal)
    angle = math.radians(STOCK_CURVE_ANGLE_DEGREES)
    chord = 2.0 * radius * math.sin(angle * 0.5)
    sagitta = radius * (1.0 - math.cos(angle * 0.5))
    return chord, STOCK_CURVE_ANGLE_DEGREES, sagitta


def _is_stock_straight(model_path: str) -> re.Match[str] | None:
    return _STOCK_STRAIGHT.fullmatch(str(model_path).replace("/", "\\"))


def _exact_stock_variants(model_path: str, configured_long_length: float):
    pieces = _ORIGINAL_VARIANTS(model_path, configured_long_length)
    if _is_stock_straight(model_path) is None:
        return pieces
    return tuple(
        replace(
            piece,
            length_metres=_model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[piece.nominal_length],
        )
        for piece in pieces
    )


def _exact_piece_length(model_path: str, configured_long_length: float) -> float:
    straight_length = _model_geometry.stock_straight_length(model_path)
    if straight_length is not None:
        return straight_length
    curve = _curve._curve_match(str(model_path))
    if curve is not None:
        chord, _turn, _sagitta = stock_curve_geometry(int(curve.group("radius")))
        return chord
    return _ORIGINAL_QUALITY_PIECE_LENGTH(model_path, configured_long_length)


def _curve_turn_error_degrees(run, start, end) -> float:
    start_heading = _p._nearest_polyline_heading(run, start)
    end_heading = _p._nearest_polyline_heading(run, end)
    source_turn = abs(_p._signed_heading_delta(start_heading, end_heading))
    return abs(source_turn - STOCK_CURVE_ANGLE_DEGREES)


def _chain_is_seam_safe(measure, fitted) -> bool:
    """Require exact connector positions and a close tangent match for curves."""

    previous_end = None
    for piece, start, end in fitted:
        if previous_end is not None and math.dist(previous_end, start) > 1.0e-4:
            return False
        previous_end = end
        if _curve._curve_match(piece.model_path) is None:
            continue
        if not math.isclose(
            math.dist(start, end), float(piece.length_metres), rel_tol=0.0, abs_tol=1.0e-4
        ):
            return False
        if (
            _curve_turn_error_degrees(measure.points, start, end)
            > _MAXIMUM_TANGENT_TURN_ERROR_DEGREES
        ):
            return False
    return True


def _seam_safe_stock_curve_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    fitted = _ORIGINAL_CURVE_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _chain_is_seam_safe(measure, fitted):
        return fitted
    return _STRAIGHT_FALLBACK_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _point_segment_distance(point, start, end) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator <= 1.0e-12:
        return math.dist(point, start)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dz
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
    return math.dist(point, nearest)


def _simplify_micro_bends(points):
    """Remove tiny source heading noise instead of serialising rotated road slabs."""

    result = list(_p._clean_road_points(points))
    changed = True
    while changed and len(result) >= 3:
        changed = False
        simplified = [result[0]]
        for index in range(1, len(result) - 1):
            previous, point, following = result[index - 1], result[index], result[index + 1]
            turn = _p._turn_degrees(previous, point, following)
            deviation = _point_segment_distance(point, previous, following)
            if (
                turn <= _MAXIMUM_MICRO_BEND_DEGREES
                and deviation <= _MAXIMUM_MICRO_BEND_DEVIATION_METRES
            ):
                changed = True
                continue
            simplified.append(point)
        simplified.append(result[-1])
        result = simplified
    return tuple(result)


def _corner_fillet_radius(turn_radians: float, available_tangent: float) -> float | None:
    if turn_radians <= 1.0e-9:
        return None
    tangent_factor = math.tan(turn_radians * 0.5)
    if tangent_factor <= 1.0e-9:
        return None
    for radius in _STOCK_RADII_METRES:
        tangent = radius * tangent_factor
        deviation = radius * (1.0 / math.cos(turn_radians * 0.5) - 1.0)
        if (
            tangent <= available_tangent + 1.0e-9
            and deviation <= _MAXIMUM_STOCK_FILLET_DEVIATION_METRES + 1.0e-9
        ):
            return radius
    return None


def _circular_road_run(
    points,
    *,
    minimum_turn_degrees: float = 7.5,
    maximum_turn_degrees: float = 135.0,
    maximum_tangent_metres: float = 9.0,
    tangent_fraction: float = 0.30,
    samples_per_corner: int = 4,
):
    """Round bends with constant-radius arcs compatible with native CWA curves."""

    cleaned = _simplify_micro_bends(points)
    if len(cleaned) < 3:
        return cleaned
    rounded = [cleaned[0]]
    samples_per_corner = max(2, int(samples_per_corner))

    for index in range(1, len(cleaned) - 1):
        previous, corner, following = cleaned[index - 1], cleaned[index], cleaned[index + 1]
        incoming = (corner[0] - previous[0], corner[1] - previous[1])
        outgoing = (following[0] - corner[0], following[1] - corner[1])
        incoming_length = math.hypot(*incoming)
        outgoing_length = math.hypot(*outgoing)
        turn_degrees = _p._turn_degrees(previous, corner, following)
        if (
            incoming_length <= 0.10
            or outgoing_length <= 0.10
            or turn_degrees < minimum_turn_degrees
            or turn_degrees > maximum_turn_degrees
        ):
            if math.dist(rounded[-1], corner) > 0.05:
                rounded.append(corner)
            continue

        turn = math.radians(turn_degrees)
        available = min(
            maximum_tangent_metres,
            incoming_length * tangent_fraction,
            outgoing_length * tangent_fraction,
        )
        radius = _corner_fillet_radius(turn, available)
        if radius is None:
            fallback = _ORIGINAL_ROUNDED_ROAD_RUN(
                (previous, corner, following),
                minimum_turn_degrees=minimum_turn_degrees,
                maximum_turn_degrees=maximum_turn_degrees,
                maximum_tangent_metres=maximum_tangent_metres,
                tangent_fraction=tangent_fraction,
                samples_per_corner=samples_per_corner,
            )
            for point in fallback[1:-1]:
                if math.dist(rounded[-1], point) > 0.05:
                    rounded.append(point)
            continue

        tangent = radius * math.tan(turn * 0.5)
        in_unit = (incoming[0] / incoming_length, incoming[1] / incoming_length)
        out_unit = (outgoing[0] / outgoing_length, outgoing[1] / outgoing_length)
        entry = (corner[0] - in_unit[0] * tangent, corner[1] - in_unit[1] * tangent)
        exit = (corner[0] + out_unit[0] * tangent, corner[1] + out_unit[1] * tangent)
        cross = in_unit[0] * out_unit[1] - in_unit[1] * out_unit[0]
        turn_sign = 1.0 if cross > 0.0 else -1.0
        left_normal = (-in_unit[1], in_unit[0])
        centre = (
            entry[0] + left_normal[0] * radius * turn_sign,
            entry[1] + left_normal[1] * radius * turn_sign,
        )
        start_angle = math.atan2(entry[1] - centre[1], entry[0] - centre[0])

        if math.dist(rounded[-1], entry) > 0.05:
            rounded.append(entry)
        sections = max(samples_per_corner, int(math.ceil(turn_degrees / 2.5)))
        for sample in range(1, sections):
            fraction = sample / sections
            angle = start_angle + turn_sign * turn * fraction
            point = (
                centre[0] + math.cos(angle) * radius,
                centre[1] + math.sin(angle) * radius,
            )
            if math.dist(rounded[-1], point) > 0.05:
                rounded.append(point)
        if math.dist(rounded[-1], exit) > 0.05:
            rounded.append(exit)

    if math.dist(rounded[-1], cleaned[-1]) > 0.05:
        rounded.append(cleaned[-1])
    return tuple(rounded)


def install_stock_road_geometry_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _curve._curve_geometry = stock_curve_geometry
    _curve._piece_length = _exact_piece_length
    _p.road_model_variants = _exact_stock_variants
    _quality._piece_length = _exact_piece_length
    _p._rounded_road_run = _circular_road_run
    _p._stock_piece_chain = _seam_safe_stock_curve_chain
    _INSTALLED = True
