# SPDX-License-Identifier: GPL-3.0-or-later
"""Use native CWA stock-road curve models on bends instead of short straight faceting."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
import math
import re
from typing import Sequence

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _quality

_CURVE_RADII = (100, 75, 50, 25)
_STRAIGHT_FAMILY = re.compile(
    r"^(?P<prefix>.*[\\/])(?P<family>sil|ces|asf|kos)(?:25|12|6)\.p3d$",
    re.IGNORECASE,
)
_CURVE_MODEL = re.compile(
    r"^(?P<prefix>.*[\\/])(?P<family>sil|ces|asf|kos)10 (?P<radius>25|50|75|100)\.p3d$",
    re.IGNORECASE,
)
_MINIMUM_CURVE_TURN_DEGREES = 3.0
_CURVE_REVERSE: ContextVar[bool] = ContextVar("cwr_stock_curve_reverse", default=False)

_ORIGINAL_CHAIN = _p._stock_piece_chain
_ORIGINAL_VARIANT_PATHS = _p.road_model_variant_paths
_ORIGINAL_CURVE_MODEL_FOR_RUN = _p._curved_gravel_model_for_run
_ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
_ORIGINAL_MODEL_AXIS = _p._model_axis
_ORIGINAL_QUALITY_PIECE_LENGTH = _quality._piece_length
_REVERSED_CURVE_KEYS: set[tuple[int, str, float, float, float]] = set()
_INSTALLED = False


def _straight_family(model_path: str) -> tuple[str, str] | None:
    match = _STRAIGHT_FAMILY.fullmatch(model_path.replace("/", "\\"))
    if match is None:
        return None
    return match.group("prefix"), match.group("family")


def _curve_match(model_path: str):
    return _CURVE_MODEL.fullmatch(model_path.replace("/", "\\"))


def _stock_curve_model_paths(model_path: str) -> tuple[str, ...]:
    family = _straight_family(model_path)
    if family is None:
        return ()
    prefix, stem = family
    return tuple(f"{prefix}{stem}10 {radius}.p3d" for radius in (25, 50, 75, 100))


def _curve_geometry(radius_nominal: int, scale: float) -> tuple[float, float, float]:
    """Return chord length, turn degrees and sagitta for a stock 10 m arc."""
    radius = float(radius_nominal) * scale
    arc_length = 10.0 * scale
    angle = arc_length / radius
    chord = 2.0 * radius * math.sin(angle * 0.5)
    sagitta = radius * (1.0 - math.cos(angle * 0.5))
    return chord, math.degrees(angle), sagitta


def _curve_pieces(pieces: Sequence[object]) -> tuple[tuple[object, float, float], ...]:
    stock = [piece for piece in pieces if _straight_family(piece.model_path) is not None]
    if not stock:
        return ()
    families = {_straight_family(piece.model_path) for piece in stock}
    if len(families) != 1:
        return ()
    family = next(iter(families))
    assert family is not None
    scale_values = [
        piece.length_metres / float(piece.nominal_length)
        for piece in stock
        if piece.nominal_length > 0
    ]
    if not scale_values:
        return ()
    scale = sum(scale_values) / len(scale_values)
    prefix, stem = family
    result = []
    for radius in _CURVE_RADII:
        chord, turn_degrees, sagitta = _curve_geometry(radius, scale)
        piece = _p._RoadPiece(f"{prefix}{stem}10 {radius}.p3d", chord, 10)
        result.append((piece, turn_degrees, sagitta))
    return tuple(result)


def _tail_error(measure, lengths, current, preferred_end, maximum_end, depth: int) -> float:
    best = abs(preferred_end - current)
    if depth <= 0 or current >= preferred_end - 0.05:
        return best
    for length in lengths:
        endpoint = measure.chord_endpoint(current, length, maximum_end)
        if endpoint is None or endpoint[0] <= current + 1e-7:
            continue
        best = min(
            best,
            _tail_error(
                measure,
                lengths,
                endpoint[0],
                preferred_end,
                maximum_end,
                depth - 1,
            ),
        )
    return best


def _stock_curve_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    _REVERSED_CURVE_KEYS.clear()
    curves = _curve_pieces(pieces)
    if not curves:
        return _ORIGINAL_CHAIN(
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
        start_x, start_z, start_heading = measure.point(current)
        near_end = remaining <= longest * 2.25
        candidates = []

        for piece in ordered:
            endpoint = measure.chord_endpoint(
                current, piece.length_metres, maximum_end_distance
            )
            if endpoint is None:
                continue
            end_distance, end_x, end_z, chord_heading = endpoint
            end_heading = measure.point(end_distance)[2]
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
            curve_preference = int(abs(signed_turn) >= _MINIMUM_CURVE_TURN_DEGREES)
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
                _tail_error(
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
            end_heading = measure.point(end_distance)[2]
            signed_turn = _p._signed_heading_delta(start_heading, end_heading)
            turn = abs(signed_turn)
            deviation = measure.maximum_chord_deviation(
                current, end_distance, (start_x, start_z), (end_x, end_z)
            )
            turn_tolerance = max(2.0, expected_turn * 0.30)
            sagitta_tolerance = max(0.18, expected_sagitta * 0.75)
            turn_error = abs(turn - expected_turn)
            sagitta_error = abs(deviation - expected_sagitta)
            fidelity_penalty = int(
                turn < _MINIMUM_CURVE_TURN_DEGREES
                or turn_error > turn_tolerance
                or sagitta_error > sagitta_tolerance
            )
            geometry_ratio = max(
                turn_error / turn_tolerance,
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
                _tail_error(
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
            target_x, target_z, target_heading = measure.point(target_distance)
            dx, dz = target_x - start_x, target_z - start_z
            length = math.hypot(dx, dz)
            if length <= 1e-9:
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
        end_distance, end_x, end_z, _heading = endpoint
        fitted.append((piece, (start_x, start_z), (end_x, end_z)))
        if end_distance <= current + 1e-7:
            break
        current = end_distance

    return tuple(fitted)


def _curve_reverse_for_run(run, start, end) -> bool:
    start_heading = _p._nearest_polyline_heading(run, start)
    end_heading = _p._nearest_polyline_heading(run, end)
    return _p._signed_heading_delta(start_heading, end_heading) < 0.0


def _curve_model_for_run(model_path, run, start, end):
    if _curve_match(model_path) is not None:
        _CURVE_REVERSE.set(_curve_reverse_for_run(run, start, end))
        return model_path
    _CURVE_REVERSE.set(False)
    return _ORIGINAL_CURVE_MODEL_FOR_RUN(model_path, run, start, end)


def _curve_object_key(obj) -> tuple[int, str, float, float, float]:
    return (
        int(obj.object_id),
        obj.model_path.casefold(),
        float(obj.x),
        float(obj.z),
        float(obj.heading_degrees),
    )


def _road_object_on_slope(*args, **kwargs):
    model_path = args[1] if len(args) > 1 else kwargs.get("model_path", "")
    reverse_curve = _CURVE_REVERSE.get()
    try:
        obj = _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)
    finally:
        _CURVE_REVERSE.set(False)
    if reverse_curve and _curve_match(str(model_path)) is not None:
        obj = replace(obj, heading_degrees=(obj.heading_degrees + 180.0) % 360.0)
        _REVERSED_CURVE_KEYS.add(_curve_object_key(obj))
    return obj


def _model_axis(obj, length: float):
    axis = _ORIGINAL_MODEL_AXIS(obj, length)
    if _curve_match(obj.model_path) is not None and _curve_object_key(obj) in _REVERSED_CURVE_KEYS:
        return axis[1], axis[0]
    return axis


def _road_model_variant_paths(model_path: str, configured_long_length: float) -> tuple[str, ...]:
    paths = list(_ORIGINAL_VARIANT_PATHS(model_path, configured_long_length))
    paths.extend(_stock_curve_model_paths(model_path))
    return tuple(dict.fromkeys(paths))


def _piece_length(model_path: str, configured_long_length: float) -> float:
    match = _curve_match(model_path)
    if match is None:
        return _ORIGINAL_QUALITY_PIECE_LENGTH(model_path, configured_long_length)
    scale = configured_long_length / 25.0
    chord, _turn, _sagitta = _curve_geometry(int(match.group("radius")), scale)
    return chord


def install_stock_road_curve_policy() -> None:
    """Install native sil/ces/asf/kos curve selection after road-quality policy."""
    global _INSTALLED
    if _INSTALLED:
        return
    _p._stock_piece_chain = _stock_curve_chain
    _p._curved_gravel_model_for_run = _curve_model_for_run
    _p._road_object_on_slope = _road_object_on_slope
    _p._model_axis = _model_axis
    _p.road_model_variant_paths = _road_model_variant_paths
    _generator.road_model_variant_paths = _road_model_variant_paths
    _quality._piece_length = _piece_length
    _INSTALLED = True
