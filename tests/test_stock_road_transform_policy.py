# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
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
