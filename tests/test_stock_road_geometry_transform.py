# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_geometry_policy as _geometry_policy
from cwr_worldgen.stock_road_model_geometry import stock_curve_connectors


def test_installed_curve_transform_places_real_connectors_on_fitted_chord():
    model = r"O\Road\sil10 50.p3d"
    geometry = stock_curve_connectors(model)
    assert geometry is not None

    start = (10.0, 20.0)
    chord_heading = 35.0
    angle = math.radians(chord_heading)
    end = (
        start[0] + math.sin(angle) * geometry.chord_length_metres,
        start[1] + math.cos(angle) * geometry.chord_length_metres,
    )
    spec = SimpleNamespace(cells=2, cell_size=100.0)
    obj = _p._road_object_on_slope(
        7,
        model,
        start,
        end,
        (0.0, 0.0, 0.0, 0.0),
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    begin, finish = _p._model_axis(obj, geometry.chord_length_metres)

    assert math.dist(begin, start) < 1.0e-6
    assert math.dist(finish, end) < 1.0e-6
    assert math.isclose(
        obj.heading_degrees,
        (chord_heading - geometry.local_chord_heading_degrees) % 360.0,
        abs_tol=1.0e-6,
    )


def test_planar_wrapper_defers_slope_shortened_curve_chord():
    model = r"O\Road\sil10 100.p3d"
    geometry = stock_curve_connectors(model)
    assert geometry is not None

    start = (0.0, 0.0)
    horizontal = geometry.chord_length_metres - 0.060356
    end = (0.0, horizontal)
    spec = SimpleNamespace(cells=2, cell_size=100.0)

    obj = _geometry_policy._transform_road_object_on_slope(
        70,
        model,
        start,
        end,
        (0.0, 0.0, 0.0, 0.0),
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )

    assert obj.model_path == model
    assert math.isclose(math.dist(start, end), horizontal, abs_tol=1.0e-9)


def test_installed_paved_curve_keeps_full_planar_connectors_on_sloped_terrain():
    model = r"O\Road\sil10 50.p3d"
    geometry = stock_curve_connectors(model)
    assert geometry is not None

    cells = 128
    cell_size = 1.0
    elevations = tuple(z * 0.04 for z in range(cells) for _x in range(cells))
    start = (20.0, 20.0)
    end = (20.0, 20.0 + geometry.chord_length_metres)
    spec = SimpleNamespace(cells=cells, cell_size=cell_size)

    obj = _p._road_object_on_slope(
        8,
        model,
        start,
        end,
        elevations,
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    begin, finish = _p._model_axis(obj, geometry.chord_length_metres)

    assert math.dist(begin, start) < 1.0e-6
    assert math.dist(finish, end) < 1.0e-6
    assert math.isclose(obj.pitch_degrees, 0.0, abs_tol=1.0e-12)


def test_transform_stage_is_owned_by_geometry_module():
    assert _geometry_policy._TRANSFORM_INSTALLED
    assert _p._model_axis is not _geometry_policy._ORIGINAL_TRANSFORM_MODEL_AXIS
