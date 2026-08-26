# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen.road_quality_policy import _Context, _Junction
from cwr_worldgen.stock_road_model_geometry import STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
from cwr_worldgen.stock_road_surface_overlap_policy import (
    MINIMUM_STRAIGHT_SEAM_TURN_DEGREES,
    SEAM_COVER_LOGICAL_LENGTH_METRES,
    SEAM_COVER_PLACEMENT_SPAN_METRES,
    _with_straight_seam_covers,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _endpoint(start: tuple[float, float], heading_degrees: float, length: float):
    direction = _direction(heading_degrees)
    return start[0] + direction[0] * length, start[1] + direction[1] * length


def test_angled_straight_asphalt_seam_gets_normal_six_metre_cover():
    model = r"O\Road\sil6.p3d"
    piece = _p._RoadPiece(model, 6.25, 6)
    seam = (0.0, 6.25)
    first = (piece, (0.0, 0.0), seam)
    second = (piece, seam, _endpoint(seam, 8.0, 6.25))

    covered = _with_straight_seam_covers((first, second))

    assert len(covered) == 3
    cover_piece, cover_start, cover_end = covered[1]
    assert cover_piece.model_path.casefold().endswith(r"\sil6.p3d")
    assert math.isclose(
        cover_piece.length_metres,
        SEAM_COVER_LOGICAL_LENGTH_METRES,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        math.dist(cover_start, cover_end),
        SEAM_COVER_PLACEMENT_SPAN_METRES,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    midpoint = (
        (cover_start[0] + cover_end[0]) * 0.5,
        (cover_start[1] + cover_end[1]) * 0.5,
    )
    assert math.dist(midpoint, seam) < 1.0e-9


def test_nearly_straight_seam_does_not_add_patch():
    model = r"O\Road\sil6.p3d"
    piece = _p._RoadPiece(model, 6.25, 6)
    seam = (0.0, 6.25)
    first = (piece, (0.0, 0.0), seam)
    second = (
        piece,
        seam,
        _endpoint(seam, MINIMUM_STRAIGHT_SEAM_TURN_DEGREES * 0.25, 6.25),
    )

    assert _with_straight_seam_covers((first, second)) == (first, second)


def test_native_curve_connector_is_not_covered_by_straight_patch():
    straight = _p._RoadPiece(r"O\Road\sil6.p3d", 6.25, 6)
    curve = _p._RoadPiece(r"O\Road\sil10 50.p3d", 8.715574, 10)
    seam = (0.0, 6.25)
    first = (straight, (0.0, 0.0), seam)
    second = (curve, seam, _endpoint(seam, 8.0, curve.length_metres))

    assert _with_straight_seam_covers((first, second)) == (first, second)


def test_measured_stock_junction_approach_continues_under_cap_to_node():
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 20.0)))
    end_key = _p._road_node_key(measure.points[-1])
    extent = STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
    junction = _Junction(
        point=measure.points[-1],
        axis=(0.0, 1.0),
        half_length=extent,
        half_width=extent,
        directions=((0.0, -1.0), (0.0, 1.0), (1.0, 0.0)),
    )
    context = _Context(
        elevations=(),
        spec=SimpleNamespace(cells=1, cell_size=1.0),
        junctions={end_key: junction},
    )
    pieces = (_p._RoadPiece(r"O\Road\sil6.p3d", 6.25, 6),)

    start, preferred_end, minimum_end, maximum_end = _quality._quality_window(
        measure,
        pieces,
        0.0,
        20.0,
        20.0,
        20.0,
        context,
    )

    assert start == 0.0
    assert math.isclose(preferred_end, measure.total, abs_tol=1.0e-9)
    assert minimum_end >= measure.total - 0.10 - 1.0e-9
    assert maximum_end >= measure.total + 3.125 - 1.0e-9
