# SPDX-License-Identifier: GPL-3.0-or-later
"""Final continuity rules for stock curves and strongly skewed paved T nodes.

Stock curve P3Ds have fixed ten-degree connector geometry. A sampled source arc
therefore has two distinct quantities that matter: its true tangent change and
its radius. Earlier selection used nearest segment headings and a permissive
sagitta tolerance, which let a smooth 100 m arc become a mixture of 75 m, 50 m
and straight pieces. Those pieces meet at their centre connectors but their
painted borders do not form one continuous curve.

Reconstruct vertex tangents from adjacent source chords and score a native curve
against the radius implied by those tangents. A coherent 100 m source arc now
stays on the 100 m stock family; a curve whose radius cannot be represented
closely falls back to ordinary short straights rather than mixing incompatible
native radii. Final physical seam repair is owned later by the emitted-seam
stage, so this policy does not mutate the retired intermediate visual seam hook.

For a fallback same-family paved T, keep the dominant through-road axis exact and
use the native T mesh only when its branch connector centre still lies inside the
actual branch road width. The ordinary fitted approaches already continue under
the cap, so this overlap closes a 45-degree skew connector without inventing a
lateral repair piece.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_curve_policy as _curve
from . import stock_road_geometry_policy as _geometry
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES = 1.75
MAXIMUM_NATIVE_RADIUS_ERROR_RATIO = 0.12
SKEW_T_CONNECTOR_EDGE_MARGIN_METRES = 0.05
MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES = 7.5

_ORIGINAL_REALIGN_LEGACY_CAPS = None
_INSTALLED = False


class _SmoothedHeadingMeasure:
    """Delegate polyline geometry while returning a reconstructed tangent."""

    def __init__(self, measure):
        self._measure = measure

    def __getattr__(self, name):
        return getattr(self._measure, name)

    def point(self, distance: float):
        x, z, _heading = self._measure.point(distance)
        heading = _smoothed_measure_heading(self._measure, distance)
        return x, z, heading


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _vertex_tangent_headings(measure) -> tuple[float, ...]:
    segment_headings = tuple(
        _heading(start, end) for start, end in zip(measure.points, measure.points[1:])
    )
    if not segment_headings:
        return (0.0,)
    if len(segment_headings) == 1:
        return (segment_headings[0], segment_headings[0])

    first_delta = _p._signed_heading_delta(segment_headings[0], segment_headings[1])
    result = [(segment_headings[0] - first_delta * 0.5) % 360.0]
    for previous, following in zip(segment_headings, segment_headings[1:]):
        delta = _p._signed_heading_delta(previous, following)
        result.append((previous + delta * 0.5) % 360.0)
    last_delta = _p._signed_heading_delta(segment_headings[-2], segment_headings[-1])
    result.append((segment_headings[-1] + last_delta * 0.5) % 360.0)
    return tuple(result)


def _smoothed_measure_heading(measure, distance: float) -> float:
    """Interpolate reconstructed vertex tangents along one source segment."""

    tangents = _vertex_tangent_headings(measure)
    if len(measure.points) < 2:
        return tangents[0]
    if distance <= 0.0:
        return tangents[0]
    if distance >= float(measure.total):
        return tangents[-1]

    segment = 0
    for index in range(len(measure.cumulative) - 1):
        if float(measure.cumulative[index + 1]) >= float(distance) - 1.0e-12:
            segment = index
            break
    start_distance = float(measure.cumulative[segment])
    end_distance = float(measure.cumulative[segment + 1])
    length = max(1.0e-9, end_distance - start_distance)
    fraction = max(0.0, min(1.0, (float(distance) - start_distance) / length))
    delta = _p._signed_heading_delta(tangents[segment], tangents[segment + 1])
    return (tangents[segment] + delta * fraction) % 360.0


def _distance_on_measure(measure, point) -> float:
    """Return cumulative distance of the nearest projection of ``point``."""

    best = None
    for index, (start, end) in enumerate(zip(measure.points, measure.points[1:])):
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        denominator = dx * dx + dz * dz
        if denominator <= 1.0e-12:
            continue
        t = (
            (float(point[0]) - float(start[0])) * dx
            + (float(point[1]) - float(start[1])) * dz
        ) / denominator
        t = max(0.0, min(1.0, t))
        projected = (
            float(start[0]) + dx * t,
            float(start[1]) + dz * t,
        )
        distance = math.dist((float(point[0]), float(point[1])), projected)
        along = float(measure.cumulative[index]) + math.hypot(dx, dz) * t
        candidate = (distance, index, along)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return 0.0 if best is None else float(best[2])


def _smoothed_curve_turn_error_degrees(run, start, end) -> float:
    measure = _p._PolylineMeasure.create(run)
    start_distance = _distance_on_measure(measure, start)
    end_distance = _distance_on_measure(measure, end)
    start_heading = _smoothed_measure_heading(measure, start_distance)
    end_heading = _smoothed_measure_heading(measure, end_distance)
    source_turn = abs(_p._signed_heading_delta(start_heading, end_heading))
    return abs(source_turn - _model_geometry.STOCK_CURVE_ANGLE_DEGREES)


def _implied_radius(chord_length: float, turn_degrees: float) -> float:
    if turn_degrees <= 0.10:
        return math.inf
    sine = math.sin(math.radians(turn_degrees) * 0.5)
    if abs(sine) <= 1.0e-9:
        return math.inf
    return float(chord_length) / (2.0 * sine)


def _coherent_curve_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Fit stock pieces while requiring native curves to match source radius."""

    _curve._REVERSED_CURVE_KEYS.clear()
    curves = _curve._curve_pieces(pieces)
    if not curves:
        return _curve._ORIGINAL_CHAIN(
            measure,
            pieces,
            start_distance=start_distance,
            preferred_end_distance=preferred_end_distance,
            minimum_end_distance=minimum_end_distance,
            maximum_end_distance=maximum_end_distance,
        )

    context = _quality._CONTEXT.get()
    if context is not None:
        (
            start_distance,
            preferred_end_distance,
            minimum_end_distance,
            maximum_end_distance,
        ) = _quality._quality_window(
            measure,
            pieces,
            start_distance,
            preferred_end_distance,
            minimum_end_distance,
            maximum_end_distance,
            context,
        )

    if not pieces or preferred_end_distance <= start_distance + 0.05:
        return ()

    measured = _SmoothedHeadingMeasure(measure)
    ordered = tuple(
        sorted(pieces, key=lambda piece: (-piece.length_metres, piece.model_path.casefold()))
    )
    shortest = min(piece.length_metres for piece in ordered)
    longest = max(
        max(piece.length_metres for piece in ordered),
        max(piece.length_metres for piece, _turn, _sagitta in curves),
    )
    lengths = tuple(
        dict.fromkeys(
            [piece.length_metres for piece in ordered]
            + [piece.length_metres for piece, _turn, _sagitta in curves]
        )
    )
    current = start_distance
    fitted = []
    maximum_objects = max(
        1, int(math.ceil((maximum_end_distance - start_distance) / shortest)) + 2
    )

    for _ in range(maximum_objects):
        if current >= preferred_end_distance - 0.05:
            break
        remaining = preferred_end_distance - current
        if current >= minimum_end_distance - 0.05 and remaining < shortest * 0.45:
            break

        preferred = _p._road_piece_sequence(remaining, ordered)
        preferred_piece = preferred[0] if preferred else ordered[-1]
        start_x, start_z, start_heading = measured.point(current)
        near_end = remaining <= longest * 2.25
        candidates = []

        for piece in ordered:
            endpoint = measure.chord_endpoint(
                current, piece.length_metres, maximum_end_distance
            )
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measured.point(end_distance)[2]
            signed_turn = _p._signed_heading_delta(start_heading, end_heading)
            turn = max(
                _p._heading_difference(chord_heading, start_heading),
                _p._heading_difference(chord_heading, end_heading),
            )
            deviation = measure.maximum_chord_deviation(
                current, end_distance, (start_x, start_z), (end_x, end_z)
            )
            if piece.nominal_length >= 25:
                turn_limit, deviation_limit = 7.0, 0.45
            elif piece.nominal_length >= 12:
                turn_limit, deviation_limit = 11.0, 0.30
            else:
                turn_limit, deviation_limit = 18.0, 0.22
            fidelity_penalty = int(turn > turn_limit or deviation > deviation_limit)
            curve_preference = int(abs(signed_turn) >= _curve._MINIMUM_CURVE_TURN_DEGREES)
            geometry_ratio = max(turn / turn_limit, deviation / deviation_limit)
            terrain_penalty = 0
            terrain_ratio = 0.0
            if context is not None:
                bulge = _quality._terrain_bulge(
                    context, (start_x, start_z), (end_x, end_z), piece.nominal_length
                )
                terrain_penalty = int(bulge > _quality._STOCK_BULGE_LIMIT)
                terrain_ratio = bulge / max(0.001, _quality._STOCK_BULGE_LIMIT)
            tail_error = (
                _curve._tail_error(
                    measure,
                    lengths,
                    end_distance,
                    preferred_end_distance,
                    maximum_end_distance,
                    _quality._LOOKAHEAD_DEPTH,
                )
                if near_end
                else 0.0
            )
            tail_tolerance = (
                max(
                    0.20,
                    float(getattr(context.spec, "road_connection_tolerance", 0.35)),
                )
                if context is not None
                else 0.35
            )
            tail_penalty = int(near_end and tail_error > tail_tolerance)
            score = (
                fidelity_penalty,
                tail_penalty,
                terrain_penalty,
                curve_preference,
                geometry_ratio,
                terrain_ratio,
                tail_error,
                0 if piece == preferred_piece else 1,
                abs(preferred_end_distance - end_distance),
                -piece.length_metres,
            )
            candidates.append((score, piece, endpoint))

        for piece, expected_turn, expected_sagitta in curves:
            endpoint = measure.chord_endpoint(
                current, piece.length_metres, maximum_end_distance
            )
            if endpoint is None:
                continue
            end_distance, end_x, end_z, _chord_heading = endpoint
            end_heading = measured.point(end_distance)[2]
            turn = abs(_p._signed_heading_delta(start_heading, end_heading))
            deviation = measure.maximum_chord_deviation(
                current, end_distance, (start_x, start_z), (end_x, end_z)
            )
            model_match = _model_geometry.stock_curve_match(piece.model_path)
            if model_match is None:
                continue
            model_radius = float(model_match.group("radius"))
            source_radius = _implied_radius(piece.length_metres, turn)
            radius_error_ratio = (
                abs(source_radius - model_radius) / model_radius
                if math.isfinite(source_radius)
                else math.inf
            )
            turn_error = abs(turn - expected_turn)
            sagitta_tolerance = max(0.12, expected_sagitta * 0.45)
            sagitta_error = abs(deviation - expected_sagitta)
            fidelity_penalty = int(
                turn < _curve._MINIMUM_CURVE_TURN_DEGREES
                or turn_error > MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES
                or radius_error_ratio > MAXIMUM_NATIVE_RADIUS_ERROR_RATIO
                or sagitta_error > sagitta_tolerance
            )
            geometry_ratio = max(
                turn_error / MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES,
                radius_error_ratio / MAXIMUM_NATIVE_RADIUS_ERROR_RATIO,
                sagitta_error / sagitta_tolerance,
            )
            terrain_penalty = 0
            terrain_ratio = 0.0
            if context is not None:
                bulge = _quality._terrain_bulge(
                    context, (start_x, start_z), (end_x, end_z), 10
                )
                terrain_penalty = int(bulge > _quality._STOCK_BULGE_LIMIT)
                terrain_ratio = bulge / max(0.001, _quality._STOCK_BULGE_LIMIT)
            tail_error = (
                _curve._tail_error(
                    measure,
                    lengths,
                    end_distance,
                    preferred_end_distance,
                    maximum_end_distance,
                    _quality._LOOKAHEAD_DEPTH,
                )
                if near_end
                else 0.0
            )
            tail_tolerance = (
                max(
                    0.20,
                    float(getattr(context.spec, "road_connection_tolerance", 0.35)),
                )
                if context is not None
                else 0.35
            )
            tail_penalty = int(near_end and tail_error > tail_tolerance)
            score = (
                fidelity_penalty,
                tail_penalty,
                terrain_penalty,
                0,
                geometry_ratio,
                terrain_ratio,
                tail_error,
                0,
                abs(preferred_end_distance - end_distance),
                -piece.length_metres,
            )
            candidates.append((score, piece, endpoint))

        if not candidates:
            if current >= minimum_end_distance - 0.05:
                break
            piece = ordered[-1]
            target_distance = min(preferred_end_distance, measure.total)
            target_x, target_z, target_heading = measured.point(target_distance)
            dx, dz = target_x - start_x, target_z - start_z
            length = math.hypot(dx, dz)
            if length <= 1.0e-9:
                angle = math.radians(target_heading)
                dx, dz, length = math.sin(angle), math.cos(angle), 1.0
            fitted.append(
                (
                    piece,
                    (start_x, start_z),
                    (
                        start_x + dx / length * piece.length_metres,
                        start_z + dz / length * piece.length_metres,
                    ),
                )
            )
            break

        _score, piece, endpoint = min(candidates, key=lambda item: item[0])
        end_distance, end_x, end_z, _heading_value = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1.0e-7:
            break
        current = end_distance

    return tuple(fitted)


def _same_family_paved_skew_t(incidents, family: str):
    if len(incidents) != 3 or family not in {"sil", "asf", "kos"}:
        return None
    if any(incident.family != family for incident in incidents):
        return None
    model = _junction._T_JUNCTION_MODELS.get((family, family))
    if model is None:
        return None

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    branch_heading = _junction._heading(incidents[branch].direction)

    candidates = []
    for zero, opposite in ((first, second), (second, first)):
        rotation, main_error = _junction._best_rotation(
            (
                (0.0, _junction._heading(incidents[zero].direction)),
                (180.0, _junction._heading(incidents[opposite].direction)),
            )
        )
        branch_error = _junction._angular_distance(
            (rotation + 90.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
        return None

    half_width = float(_model_geometry.STOCK_HALF_WIDTHS_METRES[family])
    lateral = (
        _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
        * math.sin(math.radians(branch_error))
    )
    if lateral > half_width - SKEW_T_CONNECTOR_EDGE_MARGIN_METRES:
        return None

    return _junction._NativeJunction(
        model_path=model,
        heading_degrees=rotation % 360.0,
        maximum_heading_error_degrees=max(main_error, branch_error),
        cap_family=family,
    )


def _replace_physically_covered_skew_t_caps(report, dataset, projection, elevations, spec):
    if _ORIGINAL_REALIGN_LEGACY_CAPS is None:
        raise RuntimeError("final stock-road continuity policy is not installed")
    report = _ORIGINAL_REALIGN_LEGACY_CAPS(
        report, dataset, projection, elevations, spec
    )

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report
    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        old = objects[index]
        match = _model_geometry.stock_straight_match(str(old.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        key = _p._road_node_key((float(old.x), float(old.z)))
        junction = incident_map.get(key)
        if junction is None:
            continue
        node, incidents = junction
        if math.dist((float(old.x), float(old.z)), node) > 0.25:
            continue
        native = _same_family_paved_skew_t(incidents, family)
        if native is None:
            continue
        objects[index] = _junction._native_junction_object(
            old, native, elevations, spec
        )
        changed = True

    if not changed:
        return report
    from dataclasses import replace

    return replace(report, objects=tuple(objects))


def install_stock_road_final_continuity_policy() -> None:
    global _ORIGINAL_REALIGN_LEGACY_CAPS, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_REALIGN_LEGACY_CAPS = _finish._realign_legacy_caps
    if _ORIGINAL_REALIGN_LEGACY_CAPS is None:
        raise RuntimeError("stock road visual-finish policy must install first")

    _geometry._ORIGINAL_CURVE_CHAIN = _coherent_curve_chain
    _geometry._curve_turn_error_degrees = _smoothed_curve_turn_error_degrees
    _geometry._MAXIMUM_TANGENT_TURN_ERROR_DEGREES = (
        MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES
    )

    _finish._realign_legacy_caps = _replace_physically_covered_skew_t_caps
    _INSTALLED = True
