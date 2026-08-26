# SPDX-License-Identifier: GPL-3.0-or-later
"""Hide visual stock-road cracks with bounded, intentional surface overlap.

Memory-LOD connectors describe where road models connect logically, but a pair of
straight rectangular road slabs that changes heading at one connector can still
leave a triangular hole between the visible edges. The same problem appears at
native junctions when an OSM approach differs slightly from the junction model's
fixed connector heading.

Use two deliberately small visual-overlap rules:

* measured stock junction approaches continue underneath the central junction
  object all the way to the logical node instead of stopping at its connector;
  the junction object is already rendered slightly above the branch roads;
* when two *straight paved* pieces meet with a visible heading change, place one
  ordinary 6.25 m stock straight centred over that seam on the heading bisector.
  Curved P3Ds are never covered because their tangent connectors already model
  the bend correctly.

The seam cover carries a tiny logical audit length while its placement span is
nearly the physical six-metre model length. This keeps road-quality connection
metrics about the underlying chain meaningful without scaling the P3D.
"""
from __future__ import annotations

import math

from . import gravel_junction_policy as _gravel_junction
from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_model_geometry as _geometry

MINIMUM_STRAIGHT_SEAM_TURN_DEGREES = 2.5
SEAM_COVER_LOGICAL_LENGTH_METRES = 0.08
SEAM_COVER_PLACEMENT_SPAN_METRES = 5.80
SEAM_COVER_VERTICAL_BIAS_METRES = 0.007
MAXIMUM_SEAM_CONNECTOR_GAP_METRES = 0.50

_PAVED_FAMILIES = {"sil", "asf", "kos"}
_ORIGINAL_CHAIN = None
_ORIGINAL_QUALITY_WINDOW = None
_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_INSTALLED = False


def _heading(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0


def _signed_delta(first: float, second: float) -> float:
    return (second - first + 180.0) % 360.0 - 180.0


def _straight_family(model_path: str) -> str | None:
    match = _geometry.stock_straight_match(model_path)
    if match is None:
        return None
    family = match.group("family").casefold()
    return family if family in _PAVED_FAMILIES else None


def _six_metre_model(model_path: str, family: str) -> str:
    normalised = str(model_path).replace("/", "\\")
    prefix = normalised.rsplit("\\", 1)[0] if "\\" in normalised else ""
    return f"{prefix}\\{family}6.p3d" if prefix else f"{family}6.p3d"


def _measured_stock_junction(junction) -> bool:
    if junction is None or _gravel_junction._is_gravel_junction(junction):
        return False
    extent = _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    return (
        math.isclose(float(junction.half_length), extent, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(float(junction.half_width), extent, rel_tol=0.0, abs_tol=1.0e-6)
    )


def _quality_window(
    measure,
    pieces,
    start_distance,
    preferred_end,
    minimum_end,
    maximum_end,
    context,
):
    if _ORIGINAL_QUALITY_WINDOW is None:
        raise RuntimeError("stock road surface overlap policy is not installed")
    start_distance, preferred_end, minimum_end, maximum_end = _ORIGINAL_QUALITY_WINDOW(
        measure,
        pieces,
        start_distance,
        preferred_end,
        minimum_end,
        maximum_end,
        context,
    )
    if not pieces:
        return start_distance, preferred_end, minimum_end, maximum_end

    start_junction = context.junctions.get(_p._road_node_key(measure.points[0]))
    end_junction = context.junctions.get(_p._road_node_key(measure.points[-1]))
    shortest = min(float(piece.length_metres) for piece in pieces)

    # Do not make a visible butt joint at a measured junction connector. Let the
    # branch road continue beneath the central object to the actual OSM node.
    # The native/overlay junction object already receives a small positive Y
    # bias, so it remains the visible top surface where the two overlap.
    if _measured_stock_junction(start_junction):
        start_distance = 0.0
    if _measured_stock_junction(end_junction):
        preferred_end = max(start_distance, measure.total)
        minimum_end = max(start_distance, measure.total - 0.10)
        maximum_end = max(maximum_end, measure.total + shortest * 0.5)
    return start_distance, preferred_end, minimum_end, maximum_end


def _seam_cover(previous, current):
    previous_piece, previous_start, previous_end = previous
    current_piece, current_start, current_end = current
    previous_family = _straight_family(previous_piece.model_path)
    current_family = _straight_family(current_piece.model_path)
    if previous_family is None or current_family != previous_family:
        return None

    connector_gap = math.dist(previous_end, current_start)
    if connector_gap > MAXIMUM_SEAM_CONNECTOR_GAP_METRES:
        return None

    previous_heading = _heading(previous_start, previous_end)
    current_heading = _heading(current_start, current_end)
    turn = _signed_delta(previous_heading, current_heading)
    if abs(turn) < MINIMUM_STRAIGHT_SEAM_TURN_DEGREES:
        return None

    seam = (
        (previous_end[0] + current_start[0]) * 0.5,
        (previous_end[1] + current_start[1]) * 0.5,
    )
    bisector = (previous_heading + turn * 0.5) % 360.0
    angle = math.radians(bisector)
    direction = (math.sin(angle), math.cos(angle))
    half_span = SEAM_COVER_PLACEMENT_SPAN_METRES * 0.5
    start = (
        seam[0] - direction[0] * half_span,
        seam[1] - direction[1] * half_span,
    )
    end = (
        seam[0] + direction[0] * half_span,
        seam[1] + direction[1] * half_span,
    )
    piece = _p._RoadPiece(
        _six_metre_model(previous_piece.model_path, previous_family),
        SEAM_COVER_LOGICAL_LENGTH_METRES,
        6,
    )
    return piece, start, end


def _with_straight_seam_covers(fitted):
    fitted = tuple(fitted)
    if len(fitted) < 2:
        return fitted
    result = []
    for index, item in enumerate(fitted):
        if index:
            cover = _seam_cover(fitted[index - 1], item)
            if cover is not None:
                result.append(cover)
        result.append(item)
    return tuple(result)


def _stock_piece_chain(measure, pieces, **kwargs):
    if _ORIGINAL_CHAIN is None:
        raise RuntimeError("stock road surface overlap policy is not installed")
    return _with_straight_seam_covers(_ORIGINAL_CHAIN(measure, pieces, **kwargs))


def _is_seam_cover_placement(model_path: str, start, end) -> bool:
    match = _geometry.stock_straight_match(model_path)
    if match is None or match.group("family").casefold() not in _PAVED_FAMILIES:
        return False
    if int(match.group("length")) != 6:
        return False
    return math.isclose(
        math.dist(tuple(start), tuple(end)),
        SEAM_COVER_PLACEMENT_SPAN_METRES,
        rel_tol=0.0,
        abs_tol=0.02,
    )


def _road_object_on_slope(*args, **kwargs):
    if _ORIGINAL_ROAD_OBJECT_ON_SLOPE is None:
        raise RuntimeError("stock road surface overlap policy is not installed")
    model_path = str(args[1] if len(args) > 1 else kwargs.get("model_path", ""))
    start = args[2] if len(args) > 2 else kwargs.get("start")
    end = args[3] if len(args) > 3 else kwargs.get("end")
    if start is not None and end is not None and _is_seam_cover_placement(model_path, start, end):
        updated = dict(kwargs)
        updated["vertical_offset"] = float(updated.get("vertical_offset", 0.0)) + SEAM_COVER_VERTICAL_BIAS_METRES
        kwargs = updated
    return _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)


def install_stock_road_surface_overlap_policy() -> None:
    global _ORIGINAL_CHAIN, _ORIGINAL_QUALITY_WINDOW
    global _ORIGINAL_ROAD_OBJECT_ON_SLOPE, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_CHAIN = _p._stock_piece_chain
    _ORIGINAL_QUALITY_WINDOW = _quality._quality_window
    _ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
    _quality._quality_window = _quality_window
    _p._stock_piece_chain = _stock_piece_chain
    _p._road_object_on_slope = _road_object_on_slope
    _INSTALLED = True
