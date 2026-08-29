# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_model_geometry as _geometry


def _object(object_id, model, x, z, heading, *, pitch=0.0, y=0.0):
    return SimpleNamespace(
        object_id=int(object_id),
        model_path=model,
        x=float(x),
        y=float(y),
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=float(pitch),
    )


def _straight_from_start(object_id, start, heading, *, pitch=0.0):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    horizontal = length * math.cos(math.radians(float(pitch)))
    half = horizontal * 0.5
    angle = math.radians(float(heading))
    direction = (math.sin(angle), math.cos(angle))
    centre = (
        float(start[0]) + direction[0] * half,
        float(start[1]) + direction[1] * half,
    )
    return _object(
        object_id,
        r"o\road\sil6.p3d",
        centre[0],
        centre[1],
        heading,
        pitch=pitch,
    )


def test_final_pass_sees_pitch_projected_straight_gap():
    # Build the objects from their *physical* WRP horizontal spans. The older
    # intermediate hook could miss this class because it ran before the final
    # pitched objects existed.
    first_pitch = 8.0
    first_length = 6.25 * math.cos(math.radians(first_pitch))
    first = _object(
        1,
        r"o\road\sil6.p3d",
        0.0,
        -first_length * 0.5,
        0.0,
        pitch=first_pitch,
    )
    second = _straight_from_start(2, (0.18, 0.0), 8.0, pitch=5.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

    plans = _emitted._emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert plans[0].model_path.casefold() == r"o\road\sil6.p3d"
    assert math.dist(plans[0].centre, (0.09, 0.0)) < 0.01


def test_final_curve_gap_uses_straight_side_tangent():
    model = r"o\road\sil10 50.p3d"
    geometry = _geometry.stock_curve_connectors(model)
    assert geometry is not None
    curve = _object(1, model, 0.0, 0.0, 0.0)
    curve_axis = _p._model_axis(curve, geometry.chord_length_metres)
    seam = curve_axis[1]

    straight_heading = 15.0
    direction = (
        math.sin(math.radians(straight_heading)),
        math.cos(math.radians(straight_heading)),
    )
    straight_start = (
        seam[0] + direction[0] * 1.0,
        seam[1] + direction[1] * 1.0,
    )
    straight = _straight_from_start(2, straight_start, straight_heading)
    report = SimpleNamespace(objects=(curve, straight), junction_cap_objects=0)

    plans = _emitted._emitted_seam_cover_plans(report)

    assert len(plans) == 1
    assert math.isclose(
        plans[0].tangent_axis_degrees,
        straight_heading,
        abs_tol=1.0e-6,
    )


def test_existing_paved_underlay_prevents_duplicate_cover():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, (0.10, 0.0), 8.0)
    # A low, same-family straight already spans the whole seam area.
    existing = _object(3, r"o\road\sil6.p3d", 0.05, 0.0, 4.0, y=-0.01)
    report = SimpleNamespace(
        objects=(first, second, existing),
        junction_cap_objects=0,
    )

    assert _emitted._emitted_seam_cover_plans(report) == ()
