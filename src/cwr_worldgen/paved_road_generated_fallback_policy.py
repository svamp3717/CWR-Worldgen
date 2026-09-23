# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate paved road pieces only when the stock P3D family cannot fit.

The stock road fitter remains authoritative. The quality scorer first tries the
vanilla 25/12/6 P3D family, including a visible edge-discontinuity check against
the preceding stock slab. This policy substitutes a generated world-local paved ribbon only
when the winning stock piece still exceeds its turn/deviation limits, still makes
a clipping stock-to-stock joint, or cannot cover a required short tail.

Generated names are canonicalized by width, decimetre chord length and a
five-degree curve bucket, so identical failures reuse one P3D and the existing
procedural infrastructure cache reuses it across builds.

Dirt and gravel chains are deliberately excluded from this first implementation.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

from . import playability as _p
from . import procedural_infrastructure as _pi
from . import road_quality_policy as _quality
from . import road_quality_parallel_compat_policy as _quality_parallel

_INSTALLED = False
_ORIGINAL_SERIAL_CHAIN: Any = None
_ORIGINAL_PARALLEL_CHAIN: Any = None

_PAVED_HALF_WIDTHS = {
    "sil": 4.55,
    "kos": 4.55,
    "asf": 3.50,
}
_FAMILY = re.compile(r"^([a-z]+)", re.IGNORECASE)


def _canonical(path: str) -> str:
    return str(path).replace("/", "\\").casefold()


def _family(path: str) -> str:
    filename = str(path).replace("/", "\\").rsplit("\\", 1)[-1]
    match = _FAMILY.match(filename)
    return match.group(1).casefold() if match else ""


def _generated_width(pieces: Sequence[Any], spec: Any) -> float:
    for piece in pieces:
        family = _family(piece.model_path)
        if family in _PAVED_HALF_WIDTHS:
            return _PAVED_HALF_WIDTHS[family] * 2.0
    family = _family(getattr(spec, "paved_road_model", ""))
    return _PAVED_HALF_WIDTHS.get(
        family, _pi.GENERATED_PAVED_HALF_WIDTH_METRES
    ) * 2.0


def _eligible_paved_chain(pieces: Sequence[Any], spec: Any) -> bool:
    if not pieces or not bool(getattr(spec, "procedural_paved_road_fallback", False)):
        return False
    if any(
        _p.is_generated_gravel_road_model(piece.model_path)
        or _pi.is_generated_paved_road_model(piece.model_path)
        for piece in pieces
    ):
        return False

    configured_length = float(getattr(spec, "road_segment_length", 25.0))
    dirt_model = getattr(spec, "dirt_road_model", "")
    dirt_variants = {
        _canonical(piece.model_path)
        for piece in _p.road_model_variants(dirt_model, configured_length)
    } if dirt_model else set()
    if any(_canonical(piece.model_path) in dirt_variants for piece in pieces):
        return False

    paved_model = getattr(spec, "paved_road_model", "")
    paved_variants = {
        _canonical(piece.model_path)
        for piece in _p.road_model_variants(paved_model, configured_length)
    } if paved_model else set()
    return any(_canonical(piece.model_path) in paved_variants for piece in pieces)


def _stock_limits(piece: Any) -> tuple[float, float]:
    nominal = int(getattr(piece, "nominal_length", 0))
    if nominal >= 25:
        return 7.0, 0.45
    if nominal >= 12:
        return 11.0, 0.30
    return 18.0, 0.22


def _signed_curve_degrees(
    measure: Any,
    start_distance: float,
    end_distance: float,
    start: tuple[float, float],
    end: tuple[float, float],
    deviation: float,
) -> float:
    chord_x = end[0] - start[0]
    chord_z = end[1] - start[1]
    chord = math.hypot(chord_x, chord_z)
    if chord <= 1.0e-6 or deviation <= 1.0e-5:
        return 0.0

    mid_x = (start[0] + end[0]) * 0.5
    mid_z = (start[1] + end[1]) * 0.5
    best_lateral = 0.0
    span = max(0.0, end_distance - start_distance)
    for fraction in (0.25, 0.50, 0.75):
        point_x, point_z, _heading = measure.point(start_distance + span * fraction)
        # Positive is local right for a chord whose model forward axis is +Z.
        lateral = (
            (point_x - mid_x) * chord_z
            - (point_z - mid_z) * chord_x
        ) / chord
        if abs(lateral) > abs(best_lateral):
            best_lateral = lateral

    if abs(best_lateral) <= 1.0e-6:
        return 0.0
    # For a circular arc, theta = 4*atan(2*sagitta/chord). The generated ribbon
    # uses a quadratic Bezier whose midpoint has the same sagitta.
    magnitude = math.degrees(4.0 * math.atan2(2.0 * deviation, chord))
    magnitude = min(45.0, max(0.0, magnitude))
    return magnitude if best_lateral > 0.0 else -magnitude


def _generated_piece(
    context: Any,
    pieces: Sequence[Any],
    measure: Any,
    *,
    start_distance: float,
    end_distance: float,
    start: tuple[float, float],
    end: tuple[float, float],
    deviation: float,
) -> Any:
    length = math.dist(start, end)
    curve = _signed_curve_degrees(
        measure,
        start_distance,
        end_distance,
        start,
        end,
        deviation,
    )
    model_path = _pi.paved_fallback_model_path(
        context.spec.name,
        _generated_width(pieces, context.spec),
        length,
        curve,
    )
    return _p._RoadPiece(
        model_path,
        length,
        max(1, int(round(length))),
    )


def _upgrade_stock_result(
    result: Sequence[Any],
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
) -> tuple[Any, ...]:
    context = _quality._CONTEXT.get()
    if context is None or not _eligible_paved_chain(pieces, context.spec):
        return tuple(result)

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
    current = float(start_distance)
    upgraded: list[Any] = []
    paved_half_width = _generated_width(pieces, context.spec) * 0.5
    previous_edge_heading: float | None = None
    previous_edge_half_width: float | None = None
    previous_entry_start_distance: float | None = None
    previous_entry_generated = False

    def set_generated_boundary(
        generated_start_distance: float,
        generated_end_distance: float,
    ) -> None:
        nonlocal previous_edge_heading
        nonlocal previous_edge_half_width
        nonlocal previous_entry_start_distance
        nonlocal previous_entry_generated
        previous_edge_heading = measure.point(generated_end_distance)[2]
        previous_edge_half_width = paved_half_width
        previous_entry_start_distance = generated_start_distance
        previous_entry_generated = True

    def set_stock_boundary(
        stock_piece: Any,
        heading: float,
        entry_start_distance: float,
    ) -> None:
        nonlocal previous_edge_heading
        nonlocal previous_edge_half_width
        nonlocal previous_entry_start_distance
        nonlocal previous_entry_generated
        family = _quality._stock_paved_family(stock_piece)
        previous_edge_heading = heading
        previous_edge_half_width = (
            _quality._STOCK_PAVED_HALF_WIDTH_METRES[family]
            if family is not None
            else None
        )
        previous_entry_start_distance = entry_start_distance
        previous_entry_generated = False

    for piece, start_point, end_point in result:
        endpoint = measure.chord_endpoint(
            current,
            float(piece.length_metres),
            maximum_end_distance,
        )
        if endpoint is None:
            # The stock fitter's last-resort behavior extends the shortest P3D
            # beyond a remainder that cannot contain it. Generate only the
            # required paved tail instead.
            target_distance = min(float(preferred_end_distance), float(measure.total))
            if target_distance > current + 0.05:
                sx, sz, _ = measure.point(current)
                ex, ez, _ = measure.point(target_distance)
                start = (sx, sz)
                end = (ex, ez)
                deviation = measure.maximum_chord_deviation(
                    current,
                    target_distance,
                    start,
                    end,
                )
                generated = _generated_piece(
                    context,
                    pieces,
                    measure,
                    start_distance=current,
                    end_distance=target_distance,
                    start=start,
                    end=end,
                    deviation=deviation,
                )
                upgraded.append((generated, start, end))
                set_generated_boundary(current, target_distance)
                current = target_distance
            else:
                upgraded.append((piece, start_point, end_point))
                if _quality._is_stock_paved_piece(piece):
                    set_stock_boundary(
                        piece,
                        _quality._piece_chord_heading(start_point, end_point),
                        current,
                    )
                else:
                    previous_edge_heading = None
                    previous_edge_half_width = None
                    previous_entry_start_distance = None
                    previous_entry_generated = False
            break

        end_distance, end_x, end_z, chord_heading = endpoint
        start_x, start_z, start_heading = measure.point(current)
        end_heading = measure.point(end_distance)[2]
        turn = max(
            _p._heading_difference(chord_heading, start_heading),
            _p._heading_difference(chord_heading, end_heading),
        )
        deviation = measure.maximum_chord_deviation(
            current,
            end_distance,
            (start_x, start_z),
            (end_x, end_z),
        )
        turn_limit, deviation_limit = _stock_limits(piece)
        current_heading = _quality._piece_chord_heading(
            (start_x, start_z), (end_x, end_z)
        )
        current_family = _quality._stock_paved_family(piece)
        current_half_width = (
            _quality._STOCK_PAVED_HALF_WIDTH_METRES[current_family]
            if current_family is not None
            else paved_half_width
        )
        clipping_joint = bool(
            previous_edge_heading is not None
            and previous_edge_half_width is not None
            and current_family is not None
            and _quality._paved_edge_discontinuity(
                previous_edge_heading,
                previous_edge_half_width,
                current_heading,
                current_half_width,
            )
            > _quality._STOCK_PAVED_MAX_EDGE_DISCONTINUITY_METRES
        )

        fidelity_failed = (
            turn > turn_limit
            or deviation > deviation_limit
        )

        if clipping_joint and upgraded and previous_entry_start_distance is not None:
            # A bad seam is a property of the pair, not merely the second slab.
            # Remove the preceding stock/generated entry and replace the minimal
            # contiguous bend region with one ribbon, eliminating the seam rather
            # than drawing another surface over it.
            merged_start_distance = previous_entry_start_distance
            merged_start_x, merged_start_z, _ = measure.point(merged_start_distance)
            merged_start = (merged_start_x, merged_start_z)
            merged_end = (end_x, end_z)
            merged_deviation = measure.maximum_chord_deviation(
                merged_start_distance,
                end_distance,
                merged_start,
                merged_end,
            )
            upgraded.pop()
            generated = _generated_piece(
                context,
                pieces,
                measure,
                start_distance=merged_start_distance,
                end_distance=end_distance,
                start=merged_start,
                end=merged_end,
                deviation=merged_deviation,
            )
            upgraded.append((generated, merged_start, merged_end))
            set_generated_boundary(merged_start_distance, end_distance)
        elif fidelity_failed:
            generated = _generated_piece(
                context,
                pieces,
                measure,
                start_distance=current,
                end_distance=end_distance,
                start=(start_x, start_z),
                end=(end_x, end_z),
                deviation=deviation,
            )
            upgraded.append((generated, (start_x, start_z), (end_x, end_z)))
            set_generated_boundary(current, end_distance)
        else:
            upgraded.append((piece, start_point, end_point))
            if current_family is not None:
                set_stock_boundary(piece, current_heading, current)
            else:
                previous_edge_heading = None
                previous_edge_half_width = None
                previous_entry_start_distance = None
                previous_entry_generated = False
        current = end_distance

    # Do not fill intentionally hub-covered tails. Only intervene when the
    # original chain still failed its required minimum coverage.
    if current < float(minimum_end_distance) - 0.05:
        target_distance = min(float(preferred_end_distance), float(measure.total))
        if target_distance > current + 0.05:
            sx, sz, _ = measure.point(current)
            ex, ez, _ = measure.point(target_distance)
            start = (sx, sz)
            end = (ex, ez)
            deviation = measure.maximum_chord_deviation(
                current,
                target_distance,
                start,
                end,
            )
            generated = _generated_piece(
                context,
                pieces,
                measure,
                start_distance=current,
                end_distance=target_distance,
                start=start,
                end=end,
                deviation=deviation,
            )
            upgraded.append((generated, start, end))
            set_generated_boundary(current, target_distance)

    return tuple(upgraded)


def _serial_chain(
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    result = _ORIGINAL_SERIAL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    return _upgrade_stock_result(
        result,
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def _parallel_chain(
    measure: Any,
    pieces: Sequence[Any],
    *,
    start_distance: float,
    preferred_end_distance: float,
    minimum_end_distance: float,
    maximum_end_distance: float,
):
    result = _ORIGINAL_PARALLEL_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    return _upgrade_stock_result(
        result,
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )


def install_paved_road_generated_fallback_policy() -> None:
    global _INSTALLED, _ORIGINAL_SERIAL_CHAIN, _ORIGINAL_PARALLEL_CHAIN
    if _INSTALLED:
        return

    _ORIGINAL_SERIAL_CHAIN = _quality._quality_chain
    _ORIGINAL_PARALLEL_CHAIN = _quality_parallel._batched_quality_chain

    _quality._quality_chain = _serial_chain
    _quality_parallel._batched_quality_chain = _parallel_chain

    # road_quality_policy was already installed during package import, so its
    # function object may already be bound directly into playability.
    if _p._stock_piece_chain is _ORIGINAL_SERIAL_CHAIN:
        _p._stock_piece_chain = _serial_chain

    _INSTALLED = True
