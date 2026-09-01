# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_surface_height as _height


def _road(
    object_id,
    *,
    x,
    y,
    z,
    heading,
    pitch,
    kind="straight",
    nominal=6.25,
    logical_center=None,
    endpoint=None,
):
    point = (float(x), float(z)) if endpoint is None else tuple(endpoint)
    return SimpleNamespace(
        object_id=int(object_id),
        x=float(x),
        y=float(y),
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=float(pitch),
        kind=kind,
        nominal_length_metres=float(nominal),
        logical_center=(float(x), float(z)) if logical_center is None else tuple(logical_center),
        endpoints=(SimpleNamespace(point=point),),
    )


def _turning_issue(node, *, approach_error: float):
    return _core.RoadIssue(
        issue_id="RI-test",
        severity="critical",
        score=95.0,
        category="turning_intersection_cap",
        x=float(node[0]),
        z=float(node[1]),
        object_ids=(1, 2),
        models=(r"o\road\sil6.p3d", r"o\road\sil12.p3d"),
        message="old turning-cap message",
        candidate_fix="old turning-cap fix",
        metrics={
            "through_turn_degrees": 20.0,
            "maximum_approach_heading_error_degrees": float(approach_error),
            "estimated_edge_offset_metres": 1.5,
            "cap_below_approach_margin_metres": -0.005,
        },
    )


def _hidden_cap_and_approach(node):
    cap = _road(
        1,
        x=node[0],
        y=0.025,
        z=node[1],
        heading=0.0,
        pitch=0.0,
        logical_center=node,
        endpoint=node,
    )
    pitch = math.degrees(math.atan2(0.015, 5.0))
    approach = _road(
        2,
        x=node[0],
        y=0.020,
        z=node[1] - 5.0,
        heading=0.0,
        pitch=pitch,
        nominal=12.5,
        logical_center=(node[0], node[1] - 5.0),
        endpoint=node,
    )
    return cap, approach


def test_surface_height_projects_sloped_approach_to_logical_node():
    node = (10.0, 10.0)
    # The approach origin is lower than the cap, but it slopes upward so its
    # actual road plane at the node is 35 mm high.
    pitch = math.degrees(math.atan2(0.015, 5.0))
    approach = _road(
        2,
        x=10.0,
        y=0.020,
        z=5.0,
        heading=0.0,
        pitch=pitch,
        endpoint=node,
    )

    assert math.isclose(
        _height._surface_height_at(approach, node),
        0.035,
        abs_tol=1.0e-6,
    )


def test_hidden_turning_cap_is_not_reported_from_origin_heights():
    node = (10.0, 10.0)
    cap, approach = _hidden_cap_and_approach(node)
    issue = _turning_issue(node, approach_error=0.0)

    assert approach.y < cap.y
    assert _height._correct_turning_cap_issue(issue, {1: cap, 2: approach}) is None


def test_hidden_turning_cap_with_bad_approach_is_reclassified():
    node = (10.0, 10.0)
    cap, approach = _hidden_cap_and_approach(node)
    issue = _turning_issue(node, approach_error=12.0)

    corrected = _height._correct_turning_cap_issue(
        issue,
        {1: cap, 2: approach},
    )

    assert corrected is not None
    assert corrected.category == "intersection_approach_mismatch"
    assert corrected.category != "turning_intersection_cap"
    assert corrected.metrics["cap_height_detector"] == "wrp_pitch_projected_surface"
    assert corrected.metrics["cap_below_approach_margin_metres"] >= 0.002
    assert "safely below" in corrected.message
