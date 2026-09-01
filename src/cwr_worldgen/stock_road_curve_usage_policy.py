# SPDX-License-Identifier: GPL-3.0-or-later
"""Own connector-locked stock curve usage after the sharp-turn search.

Two installation moments share one curve-selection responsibility:

* the micro-bend phase lets the shared beam retain one native ten-degree curve
  for gentle 7.5-15 degree bends that the stricter sharp-turn search rejects;
* the later curve-usage phase tries a connector-locked stock curve chain before
  the inherited straight-oriented fitter for broader coherent bends.

Both phases use the same verified ``sil/asf/kos/ces`` stock assets, the same
0.60 m source corridor, and the same sharp-turn geometry helpers. They remain
separate installers because the exact S-bend phase runs between them. Generated
gravel and custom road families remain outside this owner.
"""
from __future__ import annotations

import bisect
from functools import lru_cache
import math

from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_relaxation_policy as _relax
from . import stock_road_sharp_turn_policy as _sharp

MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES = 7.5
MAXIMUM_MICRO_BEND_TOTAL_TURN_DEGREES = 15.0
MAXIMUM_MICRO_BEND_BOUNDARY_TANGENT_ERROR_DEGREES = 4.5
MAXIMUM_MICRO_EXACT_RUN_METRES = 120.0
MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES = 0.40
MINIMUM_MICRO_EXACT_SHORT_STRAIGHTS = 2
MINIMUM_MICRO_EXACT_CURVES = 1
MAXIMUM_MICRO_EXACT_EXTRA_PIECES = 2
MAXIMUM_MICRO_VERTEX_TURN_DEGREES = 18.0
MINIMUM_MICRO_VERTEX_TURN_DEGREES = 0.45
MAXIMUM_MICRO_REVERSE_NOISE_DEGREES = 1.0
MICRO_EXACT_END_PROGRESS_TOLERANCE_METRES = 0.20

_MAXIMUM_PROMOTION_RUN_METRES = 180.0
_MINIMUM_TOTAL_TURN_DEGREES = 15.0
_MAXIMUM_TOTAL_TURN_DEGREES = 70.0
_MINIMUM_PROMOTED_CURVES = 1
_MINIMUM_ENDPOINT_COVER_METRES = 0.40
_MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES = 1.50
_END_PROGRESS_TOLERANCE_METRES = 0.20
_MINIMUM_SIGNIFICANT_VERTEX_TURN_DEGREES = 0.45
_MAXIMUM_LOCAL_VERTEX_TURN_DEGREES = 35.0
_MAXIMUM_REVERSE_NOISE_DEGREES = 1.50
_BEAM_CACHE_SIZE = 512
_SEGMENT_TABLE_CACHE_SIZE = 1024
_STRICT_MINIMUM_CURVES = 2
_STOCK_CURVE_TURN_DEGREES = 10.0

_ORIGINAL_BEAM = None
_ORIGINAL_NEAREST_FORWARD = _sharp._nearest_forward
_ORIGINAL_CHORD_ENDPOINT = _p._PolylineMeasure.chord_endpoint
_ORIGINAL_MAXIMUM_CHORD_DEVIATION = _p._PolylineMeasure.maximum_chord_deviation
_ORIGINAL_MICRO_CHAIN = None
_MICRO_INSTALLED = False
_ORIGINAL_CHAIN = None
_INSTALLED = False


def _canonical_pieces(pieces) -> tuple[_p._RoadPiece, ...]:
    """Return hashable beam inputs using only geometry-relevant piece fields."""

    return tuple(
        _p._RoadPiece(
            str(piece.model_path),
            float(piece.length_metres),
            int(piece.nominal_length),
        )
        for piece in pieces
    )


@lru_cache(maxsize=_SEGMENT_TABLE_CACHE_SIZE)
def _segment_table(measure) -> tuple[tuple[float, ...], ...]:
    """Precompute immutable source-segment geometry reused by every beam state."""

    result = []
    for index, (start, end) in enumerate(zip(measure.points, measure.points[1:])):
        segment_start = float(measure.cumulative[index])
        segment_end = float(measure.cumulative[index + 1])
        ax, az = float(start[0]), float(start[1])
        dx = float(end[0]) - ax
        dz = float(end[1]) - az
        length = max(1.0e-9, segment_end - segment_start)
        denominator = dx * dx + dz * dz
        result.append(
            (
                segment_start,
                segment_end,
                ax,
                az,
                dx,
                dz,
                length,
                denominator,
            )
        )
    return tuple(result)


def _fast_nearest_forward(measure, point, minimum_distance: float, maximum_distance: float):
    """Equivalent bounded projection using cached segments and one final square root."""

    minimum = max(0.0, min(float(measure.total), float(minimum_distance)))
    maximum = max(minimum, min(float(measure.total), float(maximum_distance)))
    first = max(0, bisect.bisect_right(measure.cumulative, minimum) - 2)
    last = min(
        len(measure.points) - 2,
        bisect.bisect_right(measure.cumulative, maximum),
    )
    px, pz = float(point[0]), float(point[1])
    best_distance_squared = math.inf
    best_along = math.inf
    found = False
    segments = _segment_table(measure)

    for index in range(first, last + 1):
        (
            segment_start,
            segment_end,
            ax,
            az,
            dx,
            dz,
            length,
            denominator,
        ) = segments[index]
        low = max(minimum, segment_start)
        high = min(maximum, segment_end)
        if high < low - 1.0e-9 or denominator <= 1.0e-12:
            continue
        low_t = max(0.0, min(1.0, (low - segment_start) / length))
        high_t = max(low_t, min(1.0, (high - segment_start) / length))
        t = ((px - ax) * dx + (pz - az) * dz) / denominator
        t = max(low_t, min(high_t, t))
        offset_x = px - (ax + dx * t)
        offset_z = pz - (az + dz * t)
        distance_squared = offset_x * offset_x + offset_z * offset_z
        along = segment_start + length * t
        if (
            not found
            or distance_squared < best_distance_squared
            or (
                distance_squared == best_distance_squared
                and along < best_along
            )
        ):
            found = True
            best_distance_squared = distance_squared
            best_along = along

    if not found:
        return None
    return math.sqrt(best_distance_squared), best_along


def _fast_chord_endpoint(
    measure,
    start_distance: float,
    chord_length: float,
    maximum_distance: float,
):
    """Find the original chord endpoint without scanning unrelated breakpoints."""

    if chord_length <= 0.0 or maximum_distance <= start_distance + 1.0e-9:
        return None
    origin_x, origin_z, _ = measure.point(start_distance)
    first = bisect.bisect_right(measure.cumulative, start_distance + 1.0e-9)
    last = bisect.bisect_left(measure.cumulative, maximum_distance - 1.0e-9)
    breakpoints = [start_distance]
    breakpoints.extend(measure.cumulative[first:last])
    breakpoints.append(maximum_distance)
    radius_squared = chord_length * chord_length

    for distance0, distance1 in zip(breakpoints, breakpoints[1:]):
        ax, az, _ = measure.point(distance0)
        bx, bz, _ = measure.point(distance1)
        vx, vz = bx - ax, bz - az
        denominator = vx * vx + vz * vz
        if denominator <= 1.0e-12:
            continue
        ox, oz = ax - origin_x, az - origin_z
        linear = 2.0 * (ox * vx + oz * vz)
        constant = ox * ox + oz * oz - radius_squared
        discriminant = linear * linear - 4.0 * denominator * constant
        if discriminant < -1.0e-8:
            continue
        root = math.sqrt(max(0.0, discriminant))
        fractions = sorted(
            (
                (-linear - root) / (2.0 * denominator),
                (-linear + root) / (2.0 * denominator),
            )
        )
        for fraction in fractions:
            if fraction < -1.0e-8 or fraction > 1.0 + 1.0e-8:
                continue
            fraction = max(0.0, min(1.0, fraction))
            distance = distance0 + (distance1 - distance0) * fraction
            if distance <= start_distance + 1.0e-7:
                continue
            x = ax + vx * fraction
            z = az + vz * fraction
            heading = math.degrees(math.atan2(x - origin_x, z - origin_z)) % 360.0
            return distance, x, z, heading
    return None


def _fast_maximum_chord_deviation(
    measure,
    start_distance: float,
    end_distance: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Test only source breakpoints that lie inside the candidate chord span."""

    first = bisect.bisect_right(measure.cumulative, start_distance + 1.0e-7)
    last = bisect.bisect_left(measure.cumulative, end_distance - 1.0e-7)
    maximum = 0.0
    for point in measure.points[first:last]:
        maximum = max(maximum, _p._point_segment_distance(point, start, end))
    return maximum


def _strict_beam_can_reach_boundary(entry_heading: float, exit_heading: float) -> bool:
    """Return whether two same-direction stock curves can reach the requested exit."""

    target_turn = _p._heading_difference(float(entry_heading), float(exit_heading))
    minimum_strict_turn = _STRICT_MINIMUM_CURVES * _STOCK_CURVE_TURN_DEGREES
    return (
        target_turn + _sharp._MAXIMUM_LOCKED_BOUNDARY_TANGENT_ERROR_DEGREES
        >= minimum_strict_turn - 1.0e-9
    )


@lru_cache(maxsize=_BEAM_CACHE_SIZE)
def _cached_micro_beam_stock_path(
    source_points: tuple[tuple[float, float], ...],
    turn_sign: int,
    entry_heading: float,
    exit_heading: float,
    pieces: tuple[_p._RoadPiece, ...],
):
    """Memoize the stock beam and skip a strict pass that cannot satisfy its boundary."""

    if _ORIGINAL_BEAM is None:
        raise RuntimeError("stock road micro-bend policy is not installed")

    if _strict_beam_can_reach_boundary(entry_heading, exit_heading):
        result = _ORIGINAL_BEAM(
            source_points,
            turn_sign,
            entry_heading,
            exit_heading,
            pieces,
        )
        if result is not None:
            return result

    return _ORIGINAL_BEAM(
        source_points,
        turn_sign,
        entry_heading,
        exit_heading,
        pieces,
        minimum_curve_count=1,
        maximum_boundary_tangent_error_degrees=(
            MAXIMUM_MICRO_BEND_BOUNDARY_TANGENT_ERROR_DEGREES
        ),
    )


def _micro_beam_stock_path(
    source_points,
    turn_sign: int,
    entry_heading: float,
    exit_heading: float,
    pieces,
):
    """Use the strict stock beam when feasible, otherwise allow one curve section."""

    return _cached_micro_beam_stock_path(
        tuple((float(point[0]), float(point[1])) for point in source_points),
        int(turn_sign),
        float(entry_heading),
        float(exit_heading),
        _canonical_pieces(pieces),
    )


def _dominant_micro_bend(points):
    """Return one coherent gentle bend sign and accumulated source turn."""

    return _sharp._coherent_bend(
        points,
        minimum_vertex_turn_degrees=MINIMUM_MICRO_VERTEX_TURN_DEGREES,
        maximum_vertex_turn_degrees=MAXIMUM_MICRO_VERTEX_TURN_DEGREES,
        maximum_reverse_noise_degrees=MAXIMUM_MICRO_REVERSE_NOISE_DEGREES,
        minimum_total_turn_degrees=MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES,
        maximum_total_turn_degrees=MAXIMUM_MICRO_BEND_TOTAL_TURN_DEGREES,
        minimum_significant_vertices=2,
    )


def _micro_exact_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Retain a one-curve beam result instead of greedily re-faceting it."""

    if _ORIGINAL_MICRO_CHAIN is None:
        raise RuntimeError("stock road micro-bend exact policy is not installed")

    baseline = _ORIGINAL_MICRO_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._curveable_family(pieces) is None:
        return baseline
    if measure.total > MAXIMUM_MICRO_EXACT_RUN_METRES:
        return baseline
    if _sharp._baseline_short_straights(baseline) < MINIMUM_MICRO_EXACT_SHORT_STRAIGHTS:
        return baseline

    bend = _dominant_micro_bend(measure.points)
    if bend is None:
        return baseline
    turn_sign, _total_turn = bend

    start_cover = float(start_distance)
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        start_cover < MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES
        or end_cover < MINIMUM_MICRO_EXACT_ENDPOINT_COVER_METRES
    ):
        return baseline

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return baseline

    source_points, entry_heading, source_exit_heading = _sharp._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _sharp._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )
    locked_path = _micro_beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None:
        return baseline

    exact = _sharp._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None:
        return baseline
    exact_curves = _sharp._curve_count(exact)
    baseline_curves = _sharp._curve_count(baseline)
    if exact_curves < MINIMUM_MICRO_EXACT_CURVES or exact_curves <= baseline_curves:
        return baseline
    if len(exact) > len(baseline) + MAXIMUM_MICRO_EXACT_EXTRA_PIECES:
        return baseline

    end_projection = _sharp._nearest_forward(
        measure,
        locked_path[-1],
        start,
        float(maximum_end_distance),
    )
    if end_projection is None:
        return baseline
    if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
        return baseline
    if (
        end_projection[1]
        < float(minimum_end_distance) - MICRO_EXACT_END_PROGRESS_TOLERANCE_METRES
    ):
        return baseline

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return baseline

    for previous, current in zip(exact, exact[1:]):
        if math.dist(previous[2], current[1]) > 1.0e-4:
            return baseline
    return exact


def install_stock_road_micro_bend_policy() -> None:
    """Install one-curve recovery before the exact S-bend phase."""

    global _ORIGINAL_BEAM, _ORIGINAL_MICRO_CHAIN, _MICRO_INSTALLED
    if _MICRO_INSTALLED:
        return

    _ORIGINAL_BEAM = _sharp._beam_stock_path
    _cached_micro_beam_stock_path.cache_clear()
    _segment_table.cache_clear()
    _sharp._nearest_forward = _fast_nearest_forward
    _p._PolylineMeasure.chord_endpoint = _fast_chord_endpoint
    _p._PolylineMeasure.maximum_chord_deviation = _fast_maximum_chord_deviation
    _sharp._beam_stock_path = _micro_beam_stock_path

    _ORIGINAL_MICRO_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _micro_exact_chain
    _MICRO_INSTALLED = True


def _dominant_bend(points) -> tuple[int, float] | None:
    """Return one coherent bend sign and accumulated turn for a stock run."""

    return _sharp._coherent_bend(
        points,
        minimum_vertex_turn_degrees=_MINIMUM_SIGNIFICANT_VERTEX_TURN_DEGREES,
        maximum_vertex_turn_degrees=_MAXIMUM_LOCAL_VERTEX_TURN_DEGREES,
        maximum_reverse_noise_degrees=_MAXIMUM_REVERSE_NOISE_DEGREES,
        minimum_total_turn_degrees=_MINIMUM_TOTAL_TURN_DEGREES,
        maximum_total_turn_degrees=_MAXIMUM_TOTAL_TURN_DEGREES,
    )


def _piece_tangents(item, turn_sign: int) -> tuple[float, float]:
    piece, start, end = item
    chord = _sharp._heading(start, end)
    if _geometry.stock_curve_match(str(piece.model_path)) is None:
        return chord, chord
    half_turn = _geometry.STOCK_CURVE_ANGLE_DEGREES * 0.5
    if turn_sign > 0:
        return (chord - half_turn) % 360.0, (chord + half_turn) % 360.0
    return (chord + half_turn) % 360.0, (chord - half_turn) % 360.0


def _maximum_internal_tangent_error(fitted, turn_sign: int) -> float:
    maximum = 0.0
    for previous, current in zip(fitted, fitted[1:]):
        previous_end = _piece_tangents(previous, turn_sign)[1]
        current_start = _piece_tangents(current, turn_sign)[0]
        maximum = max(maximum, _p._heading_difference(previous_end, current_start))
    return maximum


def _fallback_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    return _ORIGINAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _path_is_obstacle_safe(path) -> bool:
    """Check the smoothed stock alignment against source-backed obstacles."""

    context = _relax._CONTEXT.get()
    if context is None:
        return True
    return all(
        _relax._shortcut_clear(context.obstacles, first, second)
        for first, second in zip(path, path[1:])
    )


def _curve_promotion_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Return an exact native-curve chain first, then fall back to old fitting."""

    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road curve-usage policy is not installed")

    fallback_args = dict(
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._curveable_family(pieces) is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    if measure.total > _MAXIMUM_PROMOTION_RUN_METRES:
        return _fallback_chain(measure, pieces, **fallback_args)

    bend = _dominant_bend(measure.points)
    if bend is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    turn_sign, _total_turn = bend

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return _fallback_chain(measure, pieces, **fallback_args)

    source_points, entry_heading, source_exit_heading = _sharp._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _sharp._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )

    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        end_cover < _MINIMUM_ENDPOINT_COVER_METRES
        and _p._heading_difference(stock_exit_heading, source_exit_heading)
        > _MAXIMUM_UNCOVERED_EXIT_ERROR_DEGREES
    ):
        return _fallback_chain(measure, pieces, **fallback_args)

    locked_path = _sharp._beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None or not _path_is_obstacle_safe(locked_path):
        return _fallback_chain(measure, pieces, **fallback_args)

    exact = _sharp._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None or _sharp._curve_count(exact) < _MINIMUM_PROMOTED_CURVES:
        return _fallback_chain(measure, pieces, **fallback_args)

    if _maximum_internal_tangent_error(exact, turn_sign) > 1.0e-4:
        return _fallback_chain(measure, pieces, **fallback_args)

    end_projection = _sharp._nearest_forward(
        measure,
        locked_path[-1],
        start,
        float(maximum_end_distance),
    )
    if end_projection is None:
        return _fallback_chain(measure, pieces, **fallback_args)
    if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
        return _fallback_chain(measure, pieces, **fallback_args)
    if end_projection[1] < float(minimum_end_distance) - _END_PROGRESS_TOLERANCE_METRES:
        return _fallback_chain(measure, pieces, **fallback_args)

    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return _fallback_chain(measure, pieces, **fallback_args)
    return exact


def install_stock_road_curve_usage_policy() -> None:
    """Install broader curve-first fitting after the exact S-bend phase."""

    global _ORIGINAL_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    if not _MICRO_INSTALLED:
        raise RuntimeError("stock road micro-bend policy must install first")
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _curve_promotion_chain
    _INSTALLED = True
