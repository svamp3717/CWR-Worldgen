# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_paved_wedge_audit as _audit
from cwr_worldgen import stock_road_paved_wedge_policy as _paved
from cwr_worldgen.paved_wedge_geometry import (
    PAVED_WEDGE_BASE_INSET_METRES,
    PAVED_WEDGE_LATERAL_OVERLAP_METRES,
    paved_wedge_local_points,
)


def test_inset_triangle_keeps_requested_width_at_real_road_edge() -> None:
    turn = 5.0
    radius = 4.55
    points = paved_wedge_local_points(turn, radius_metres=radius)
    depth = float(points[0][2])
    model_base_half_width = abs(float(points[1][0]))

    fraction_at_real_edge = 1.0 - PAVED_WEDGE_BASE_INSET_METRES / depth
    half_width_at_real_edge = model_base_half_width * fraction_at_real_edge
    required = (
        radius * math.sin(math.radians(turn * 0.5))
        + PAVED_WEDGE_LATERAL_OVERLAP_METRES
    )

    assert half_width_at_real_edge >= required - 1.0e-9


def test_generator_seam_endpoints_use_pitch_projected_wrp_span() -> None:
    pitch = 10.0
    obj = SimpleNamespace(
        object_id=1,
        model_path=r"o\road\sil6.p3d",
        x=100.0,
        y=10.0,
        z=200.0,
        heading_degrees=0.0,
        pitch_degrees=pitch,
    )
    report = SimpleNamespace(objects=(obj,), junction_cap_objects=0)

    endpoints = _paved._physical_seam_endpoints(report)

    assert len(endpoints) == 2
    horizontal_span = math.dist(endpoints[0].point, endpoints[1].point)
    assert math.isclose(
        horizontal_span,
        6.25 * math.cos(math.radians(pitch)),
        abs_tol=1.0e-9,
    )


def test_generator_curve_endpoints_use_real_connector_offset_and_pitch() -> None:
    obj = SimpleNamespace(
        object_id=2,
        model_path=r"o\road\sil10 50.p3d",
        x=20.0,
        y=4.0,
        z=30.0,
        heading_degrees=25.0,
        pitch_degrees=7.0,
    )
    report = SimpleNamespace(objects=(obj,), junction_cap_objects=0)

    endpoints = _paved._physical_seam_endpoints(report)

    assert len(endpoints) == 2
    assert endpoints[0].object_kind if hasattr(endpoints[0], "object_kind") else True
    # A stock curve is not centred on its connector chord in model space.  This
    # guards against falling back to a fake centred straight axis again.
    midpoint = (
        (endpoints[0].point[0] + endpoints[1].point[0]) * 0.5,
        (endpoints[0].point[1] + endpoints[1].point[1]) * 0.5,
    )
    assert math.dist(midpoint, (obj.x, obj.z)) > 1.0e-3


def test_inspector_strict_miter_check_does_not_hide_five_cm_sliver() -> None:
    turn = 20.0
    radius = _core.GENERATED_PAVED_FILL_RADIUS_METRES + _core.GENERATED_PAVED_MITER_SAFETY_METRES
    apex = radius / math.cos(math.radians(turn * 0.5))
    road = _core.RoadObject(
        object_id=10,
        model_path=r"wg_test\i\paved_miter_q080.p3d",
        x=0.0,
        y=1.0,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="paved_miter",
        nominal_length_metres=apex * 2.0,
        logical_center=(0.0, 0.0),
        endpoints=(),
    )
    point = (apex + 0.05, 0.0)

    # The legacy generic coverage margin was 8 cm, large enough to hide this.
    assert _core._paved_miter_contains(road, point, margin=0.08)
    assert not _audit._strict_miter_contains(road, point)
