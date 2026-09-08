from __future__ import annotations

import math

from cwr_worldgen import bridge_final_alignment_policy as final
from cwr_worldgen import bridge_tangent_transition_policy as tangent
from cwr_worldgen.model import WorldObject


def _direction(heading: float) -> tuple[float, float]:
    angle = math.radians(heading)
    return math.sin(angle), math.cos(angle)


def test_stock_straight_lengths_use_actual_cwa_connector_spans() -> None:
    tangent.install_bridge_tangent_transition_policy()
    assert final._straight_road_nominal_length(r"o\road\sil6.p3d") == 6.25
    assert final._straight_road_nominal_length(r"o\road\sil12.p3d") == 12.5
    assert final._straight_road_nominal_length(r"o\road\sil25.p3d") == 25.0


def test_tinybridgetest12_start_uses_ten_degree_curve_into_real_approach() -> None:
    tangent.install_bridge_tangent_transition_policy()
    bridge_point = (591.4072481, 666.3218960)
    bridge_heading = 57.0188262
    outward = _direction((bridge_heading + 180.0) % 360.0)
    approach = WorldObject(
        853,
        r"o\road\sil6.p3d",
        586.0164185,
        7.035,
        662.0222168,
        47.8162586,
        0.0,
    )

    curved = tangent._tangent_curve_filler(
        2000,
        bridge_point,
        outward,
        approach,
        7.035,
    )
    assert curved is not None
    obj, _finish, final_heading, join_distance, _fraction, road_heading = curved
    assert obj.model_path.casefold().endswith(r"\sil10 25.p3d")
    assert join_distance < 0.20
    assert abs(tangent._signed_heading_delta(final_heading, road_heading)) < 1.0


def test_tinybridgetest12_end_uses_ten_degree_curve_into_real_approach() -> None:
    tangent.install_bridge_tangent_transition_policy()
    bridge_point = (928.2240011, 884.8959699)
    bridge_heading = 57.0188262
    outward = _direction(bridge_heading)
    approach = WorldObject(
        1068,
        r"o\road\sil6.p3d",
        932.2178345,
        7.1381631,
        886.4214478,
        249.0762802,
        0.2536515,
    )

    curved = tangent._tangent_curve_filler(
        2001,
        bridge_point,
        outward,
        approach,
        7.1565833,
    )
    assert curved is not None
    obj, _finish, final_heading, join_distance, _fraction, road_heading = curved
    assert obj.model_path.casefold().endswith(r"\sil10 25.p3d")
    assert join_distance < 0.60
    assert abs(tangent._signed_heading_delta(final_heading, road_heading)) < 2.10


def test_large_unmatched_turn_does_not_insert_a_straight_wall() -> None:
    tangent.install_bridge_tangent_transition_policy()
    bridge_point = (0.0, 0.0)
    outward = _direction(0.0)
    approach = WorldObject(
        1,
        r"o\road\sil6.p3d",
        3.0,
        7.0,
        0.0,
        90.0,
        0.0,
    )
    endpoints = final._road_endpoints(approach)
    assert endpoints is not None
    near = min(
        endpoints,
        key=lambda point: math.dist((point[0], point[2]), bridge_point),
    )
    gap = math.dist((near[0], near[2]), bridge_point)

    assert tangent._small_heading_straight_filler(
        2,
        bridge_point,
        outward,
        0.0,
        approach,
        near,
        gap,
        7.0,
    ) is None
