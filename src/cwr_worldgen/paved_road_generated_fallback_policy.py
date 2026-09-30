# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate road geometry when the stock/fixed P3D family cannot fit.

The stock road fitter remains authoritative. For legacy specs this retains the
historical paved-only fallback. Milestone 9 can additionally opt into custom
road shapes, allowing paved, gravel and dirt chains to substitute world-local
ribbons whose width, chord length and turn angle are encoded in the model name.

Custom turns use one-degree buckets up to the safe procedural-ribbon limit, so
mapped geometry is no longer restricted to the finite stock curve catalogue.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

from . import playability as _p
from . import paved_junction_policy as _junctions
from . import procedural_infrastructure as _pi
from . import road_quality_policy as _quality
from . import road_quality_parallel_compat_policy as _quality_parallel

_INSTALLED = False
_ORIGINAL_SERIAL_CHAIN: Any = None
_ORIGINAL_PARALLEL_CHAIN: Any = None

_PAVED_HALF_WIDTHS = {
    "sil": 4.55,
    "silnice": 4.55,
    "kos": 4.55,
    "asf": 3.50,
    "asfaltka": 3.50,
}
_DIRT_HALF_WIDTHS = {
    "ces": 1.75,
    "cesta": 1.75,
}
_FAMILY = re.compile(r"^([a-z]+)", re.IGNORECASE)


def _canonical(path: str) -> str:
    return str(path).replace("/", "\\").casefold()


def _family(path: str) -> str:
    filename = str(path).replace("/", "\\").rsplit("\\", 1)[-1]
    match = _FAMILY.match(filename)
    return match.group(1).casefold() if match else ""


def _configured_variants(spec: Any, attribute: str) -> set[str]:
    model = str(getattr(spec, attribute, "") or "")
    if not model:
        return set()
    configured_length = float(getattr(spec, "road_segment_length", 25.0))
    return {
        _canonical(piece.model_path)
        for piece in _p.road_model_variants(model, configured_length)
    }


def _chain_surface(pieces: Sequence[Any], spec: Any) -> str | None:
    if not pieces:
        return None

    paths = tuple(_canonical(piece.model_path) for piece in pieces)
    if all(_p.is_generated_gravel_road_model(piece.model_path) for piece in pieces):
        return "gravel"
    if all(_pi.is_generated_dirt_road_model(piece.model_path) for piece in pieces):
        return "dirt"
    if all(_pi.is_generated_paved_road_model(piece.model_path) for piece in pieces):
        return "paved"

    gravel_variants = _configured_variants(spec, "gravel_road_model")
    if gravel_variants and all(path in gravel_variants for path in paths):
        return "gravel"

    dirt_variants = _configured_variants(spec, "dirt_road_model")
    if dirt_variants and all(path in dirt_variants for path in paths):
        return "dirt"

    paved_variants = _configured_variants(spec, "paved_road_model")
    if paved_variants and all(path in paved_variants for path in paths):
        return "paved"
    return None


def _generated_width(
    pieces: Sequence[Any],
    spec: Any,
    surface: str,
) -> float:
    if bool(getattr(spec, "custom_road_shapes", False)):
        donor_attribute = {
            "paved": "paved_road_model",
            "gravel": "gravel_road_model",
            "dirt": "dirt_road_model",
        }.get(surface)
        donor_model = (
            str(getattr(spec, donor_attribute, "") or "")
            if donor_attribute is not None
            else ""
        )
        measured = _p.road_model_dimensions(donor_model) if donor_model else None
        if measured is not None:
            width = max(1.5, float(measured[0]))
            curve_model = str(
                getattr(spec, f"{surface}_road_curve_model", "") or ""
            ).strip()
            curve_measured = (
                _p.road_model_dimensions(curve_model)
                if curve_model
                else None
            )
            if curve_measured is not None:
                curve_width = max(1.5, float(curve_measured[0]))
                tolerance = max(0.15, width * 0.05)
                if abs(curve_width - width) > tolerance:
                    raise ValueError(
                        f"{surface} road donor connector widths do not match: "
                        f"{donor_model!r} is {width:.2f} m, while "
                        f"{curve_model!r} is {curve_width:.2f} m. "
                        "Choose a straight/curve pair from the same modular road "
                        "family or fix the selected P3Ds."
                    )
            return width

        # Known stock families have stable widths even when their P3Ds are not
        # present under the configured asset roots. Unknown/modded families do
        # not. Falling back to OSM width here creates visibly different-width
        # generated bends beside the real mod straight pieces.
        donor_family = _family(donor_model) if donor_model else ""
        if donor_family in _PAVED_HALF_WIDTHS:
            return _PAVED_HALF_WIDTHS[donor_family] * 2.0
        if donor_family in _DIRT_HALF_WIDTHS:
            return _DIRT_HALF_WIDTHS[donor_family] * 2.0
        if donor_model and not _p.is_generated_gravel_road_model(donor_model):
            raise ValueError(
                f"could not measure the {surface} road donor {donor_model!r}; "
                "generated road shapes cannot safely match its width. Add the "
                "PBO/PBO.ZST containing the straight donor to Asset roots, or "
                "select a measurable straight P3D."
            )

        active_tags = _p._ACTIVE_ROAD_TAGS.get()
        if active_tags is not None:
            return max(1.5, float(_p.road_width_metres(active_tags)))
    if surface == "gravel":
        return _pi.GENERATED_GRAVEL_HALF_WIDTH_METRES * 2.0
    if surface == "dirt":
        for piece in pieces:
            family = _family(piece.model_path)
            if family in _DIRT_HALF_WIDTHS:
                return _DIRT_HALF_WIDTHS[family] * 2.0
        family = _family(getattr(spec, "dirt_road_model", ""))
        return _DIRT_HALF_WIDTHS.get(family, 1.75) * 2.0

    for piece in pieces:
        family = _family(piece.model_path)
        if family in _PAVED_HALF_WIDTHS:
            return _PAVED_HALF_WIDTHS[family] * 2.0
    family = _family(getattr(spec, "paved_road_model", ""))
    return _PAVED_HALF_WIDTHS.get(
        family, _pi.GENERATED_PAVED_HALF_WIDTH_METRES
    ) * 2.0


def _eligible_chain(pieces: Sequence[Any], spec: Any) -> str | None:
    surface = _chain_surface(pieces, spec)
    if surface is None:
        return None
    if bool(getattr(spec, "custom_road_shapes", False)):
        return surface
    # Preserve the original API/behavior for specs that predate custom roads.
    if (
        surface == "paved"
        and bool(getattr(spec, "procedural_paved_road_fallback", False))
        and not any(
            _p.is_generated_gravel_road_model(piece.model_path)
            or _pi.is_generated_paved_road_model(piece.model_path)
            for piece in pieces
        )
    ):
        return "paved"
    return None


def _eligible_paved_chain(pieces: Sequence[Any], spec: Any) -> bool:
    """Compatibility helper retained for the paved-fallback test surface."""
    return _eligible_chain(pieces, spec) == "paved"

def _stock_junction_protects_interval(
    measure: Any,
    start_distance: float,
    end_distance: float,
) -> bool:
    """Keep the stock-junction approach reserve free of generated micro-slabs."""

    plans = _junctions._PLANS.get() or {}
    if not plans:
        return False

    reserve = float(_junctions._APPROACH_RESERVE)
    start_plan = plans.get(_p._road_node_key(measure.points[0]))
    end_plan = plans.get(_p._road_node_key(measure.points[-1]))
    start_stock = (
        start_plan is not None
        and not _pi.is_generated_paved_junction_model(start_plan.model_path)
    )
    end_stock = (
        end_plan is not None
        and not _pi.is_generated_paved_junction_model(end_plan.model_path)
    )

    if start_stock and float(start_distance) < reserve + 0.05:
        return True
    if end_stock and (
        float(measure.total) - float(end_distance)
    ) < reserve + 0.05:
        return True
    return False


def _piece_limits(piece: Any, surface: str) -> tuple[float, float]:
    nominal = int(getattr(piece, "nominal_length", 0))
    if surface == "gravel":
        if nominal >= 25:
            return 15.0, 0.85
        if nominal >= 12:
            return 22.0, 0.55
        if nominal >= 6:
            return 30.0, 0.35
        return 42.0, 0.20
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
    magnitude = min(
        float(_pi.CUSTOM_ROAD_MAX_CURVE_DEGREES),
        max(0.0, magnitude),
    )
    return magnitude if best_lateral > 0.0 else -magnitude


def _generated_piece(
    context: Any,
    pieces: Sequence[Any],
    measure: Any,
    *,
    surface: str,
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
    if bool(getattr(context.spec, "custom_road_shapes", False)):
        model_path = _pi.custom_road_model_path(
            context.spec.name,
            surface,
            _generated_width(pieces, context.spec, surface),
            length,
            curve,
        )
    else:
        model_path = _pi.paved_fallback_model_path(
            context.spec.name,
            _generated_width(pieces, context.spec, "paved"),
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
    surface = (
        _eligible_chain(pieces, context.spec)
        if context is not None
        else None
    )
    if context is None or surface is None:
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
    for piece, start_point, end_point in result:
        endpoint = measure.chord_endpoint(
            current,
            float(piece.length_metres),
            maximum_end_distance,
        )
        if endpoint is None:
            # The stock fitter's last-resort behavior extends the shortest P3D
            # beyond a remainder that cannot contain it. Around a successful
            # vanilla junction, preserve that stock behavior inside the 32 m
            # approach reserve instead of inserting a tiny generated slab.
            target_distance = min(float(preferred_end_distance), float(measure.total))
            if (
                surface == "paved"
                and _stock_junction_protects_interval(
                    measure,
                    current,
                    target_distance,
                )
            ):
                upgraded.append((piece, start_point, end_point))
                break
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
                    surface=surface,
                    start_distance=current,
                    end_distance=target_distance,
                    start=start,
                    end=end,
                    deviation=deviation,
                )
                upgraded.append((generated, start, end))
                current = target_distance
            else:
                upgraded.append((piece, start_point, end_point))
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
        turn_limit, deviation_limit = _piece_limits(piece, surface)

        # The quality scorer puts fidelity_penalty first. If its selected stock
        # piece still fails this test, every available stock candidate at this
        # chain step failed the same geometric-fit class.
        custom_shape_needed = (
            bool(getattr(context.spec, "custom_road_shapes", False))
            and (turn >= 2.0 or deviation >= 0.05)
        )
        if (
            (custom_shape_needed or turn > turn_limit or deviation > deviation_limit)
            and not (
                surface == "paved"
                and _stock_junction_protects_interval(
                    measure,
                    current,
                    end_distance,
                )
            )
        ):
            generated = _generated_piece(
                context,
                pieces,
                measure,
                surface=surface,
                start_distance=current,
                end_distance=end_distance,
                start=(start_x, start_z),
                end=(end_x, end_z),
                deviation=deviation,
            )
            upgraded.append((generated, (start_x, start_z), (end_x, end_z)))
        else:
            upgraded.append((piece, start_point, end_point))
        current = end_distance

    # Do not fill intentionally hub-covered tails. Only intervene when the
    # original chain still failed its required minimum coverage.
    if current < float(minimum_end_distance) - 0.05:
        target_distance = min(float(preferred_end_distance), float(measure.total))
        if (
            target_distance > current + 0.05
            and not (
                surface == "paved"
                and _stock_junction_protects_interval(
                    measure,
                    current,
                    target_distance,
                )
            )
        ):
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
                surface=surface,
                start_distance=current,
                end_distance=target_distance,
                start=start,
                end=end,
                deviation=deviation,
            )
            upgraded.append((generated, start, end))

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
