# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen.gravel_asphalt_transition_policy import (
    _native_t_junction,
    _quality_window,
    _relaxation_eligible,
)
from cwr_worldgen.model import WorldObject
from cwr_worldgen.road_quality_policy import _Context, _Junction
from cwr_worldgen.stock_road_3d_connector_policy import (
    _TerrainMeasure,
    _curve_world_point,
    _solve_curve_transform,
    _uses_measured_rigid_connectors,
)
from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen.stock_road_model_geometry import (
    STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
    stock_curve_connectors,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_flat_straight_keeps_full_horizontal_connector_length():
    cells = 64
    spec = SimpleNamespace(cells=cells, cell_size=1.0)
    context = _Context((0.0,) * (cells * cells), spec, {})
    measure = _p._PolylineMeasure.create(((10.0, 10.0), (10.0, 50.0)))

    endpoint = _TerrainMeasure(measure, context).chord_endpoint(0.0, 25.0, 30.0)

    assert endpoint is not None
    _distance, end_x, end_z, _heading = endpoint
    assert math.isclose(math.dist((10.0, 10.0), (end_x, end_z)), 25.0, abs_tol=1.0e-6)


def test_sloped_straight_uses_three_dimensional_connector_length():
    cells = 128
    cell_size = 1.0
    elevations = tuple(z * 0.05 for z in range(cells) for _x in range(cells))
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)
    context = _Context(elevations, spec, {})
    measure = _p._PolylineMeasure.create(((10.0, 10.0), (10.0, 100.0)))
    proxy = _TerrainMeasure(measure, context)

    endpoint = proxy.chord_endpoint(0.0, 25.0, 40.0)

    assert endpoint is not None
    _distance, end_x, end_z, _heading = endpoint
    start_height = _p._sample_elevation(elevations, cells, cell_size, 10.0, 10.0)
    end_height = _p._sample_elevation(elevations, cells, cell_size, end_x, end_z)
    horizontal = math.dist((10.0, 10.0), (end_x, end_z))
    assert horizontal < 25.0
    assert math.isclose(
        math.hypot(horizontal, end_height - start_height),
        25.0,
        rel_tol=0.0,
        abs_tol=2.0e-4,
    )


def test_pitched_stock_straight_axis_uses_horizontal_connector_projection():
    obj = WorldObject(
        1,
        r"o\road\sil25.p3d",
        100.0,
        5.0,
        200.0,
        0.0,
        10.0,
    )

    start, end = _p._model_axis(obj, 25.0)

    assert math.isclose(
        math.dist(start, end),
        25.0 * math.cos(math.radians(10.0)),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )


def test_3d_connector_semantics_are_limited_to_dirt_gravel_and_known_rigid_cases():
    custom = (
        _p._RoadPiece(r"custom\road25.p3d", 25.0, 25),
        _p._RoadPiece(r"custom\road12.p3d", 12.5, 12),
    )
    paved = (_p._RoadPiece(r"o\road\sil25.p3d", 25.0, 25),)
    dirt = (_p._RoadPiece(r"o\road\ces25.p3d", 25.0, 25),)
    gravel = (_p._RoadPiece(r"synthetic\i\gravel6.p3d", 6.0, 6),)

    assert not _uses_measured_rigid_connectors(custom)
    # Reference WrpTool placement keeps paved stock pieces planar in X/Z.
    assert not _uses_measured_rigid_connectors(paved)
    assert _uses_measured_rigid_connectors(dirt)
    assert _uses_measured_rigid_connectors(gravel)


def test_native_curve_transform_maps_both_connectors_in_3d():
    geometry = stock_curve_connectors(r"O\Road\sil10 50.p3d")
    assert geometry is not None
    local_begin, local_end = geometry.begin, geometry.end
    length = geometry.chord_length_metres
    delta_height = 0.45
    horizontal = math.sqrt(length * length - delta_height * delta_height)
    heading = math.radians(32.0)
    world_begin = (20.0, 7.0, 30.0)
    world_end = (
        world_begin[0] + math.sin(heading) * horizontal,
        world_begin[1] + delta_height,
        world_begin[2] + math.cos(heading) * horizontal,
    )

    solved = _solve_curve_transform(local_begin, local_end, world_begin, world_end)

    assert solved is not None
    origin, object_heading, pitch = solved
    mapped_begin = _curve_world_point(local_begin, origin, object_heading, pitch)
    mapped_end = _curve_world_point(local_end, origin, object_heading, pitch)
    for actual, expected in zip(mapped_begin, world_begin):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6)
    for actual, expected in zip(mapped_end, world_end):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6)


def test_generated_gravel_branch_uses_short_normal_paved_overlay_without_connector_snapping():
    incidents = (
        _Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(180.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(270.0), "ces", r"synthetic\i\gravel6.p3d"),
    )

    native = _native_t_junction(incidents)

    assert native is not None
    assert native.model_path == r"o\road\sil6.p3d"
    assert "kr_new" not in native.model_path.casefold()
    assert not _relaxation_eligible(incidents)


def test_generated_gravel_continues_to_node_under_paved_overlay():
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
    pieces = (_p._RoadPiece(r"synthetic\i\gravel6.p3d", 6.0, 6),)

    start, preferred_end, minimum_end, maximum_end = _quality_window(
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
    assert minimum_end >= measure.total - 0.25 - 1.0e-9
    assert maximum_end >= measure.total + 3.0 - 1.0e-9
