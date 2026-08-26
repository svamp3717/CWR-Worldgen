# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen.gravel_asphalt_transition_policy import (
    _native_t_junction,
    _relaxation_eligible,
)
from cwr_worldgen.road_quality_policy import _Context
from cwr_worldgen.stock_road_3d_connector_policy import (
    _TerrainMeasure,
    _curve_world_point,
    _solve_curve_transform,
)
from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen.stock_road_model_geometry import stock_curve_connectors


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


def test_generated_gravel_branch_uses_normal_paved_overlay_without_connector_snapping():
    incidents = (
        _Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(180.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(270.0), "ces", r"synthetic\i\gravel6.p3d"),
    )

    native = _native_t_junction(incidents)

    assert native is not None
    assert native.model_path == r"o\road\sil12.p3d"
    assert "kr_new" not in native.model_path.casefold()
    assert not _relaxation_eligible(incidents)
