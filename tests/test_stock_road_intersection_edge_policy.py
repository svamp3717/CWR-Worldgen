# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_intersection_edge_policy as _edge
from cwr_worldgen import stock_road_junction_policy as _junction


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str = "sil"):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def test_turning_t_gets_branch_and_bent_main_underlay_axes():
    # The through road turns twelve degrees at the intersection.  A single
    # straight cap aligned to the north arm cannot match both visible edges.
    incidents = (
        _incident(0.0),
        _incident(168.0),
        _incident(270.0),
    )

    headings = _edge._uncovered_incident_headings(incidents, 0.0)

    assert len(headings) == 2
    assert any(_edge._axis_heading_difference(value, 168.0) < 1.0e-9 for value in headings)
    assert any(_edge._axis_heading_difference(value, 270.0) < 1.0e-9 for value in headings)


def test_straight_t_only_needs_the_branch_underlay_axis():
    incidents = (
        _incident(0.0),
        _incident(180.0),
        _incident(270.0),
    )

    headings = _edge._uncovered_incident_headings(incidents, 0.0)

    assert len(headings) == 1
    assert _edge._axis_heading_difference(headings[0], 270.0) < 1.0e-9


def test_mixed_surface_intersection_is_not_rewritten():
    incidents = (
        _incident(0.0, "sil"),
        _incident(180.0, "sil"),
        _incident(270.0, "ces"),
    )

    assert not _edge._same_family_paved_incidents(incidents, "sil")


def test_legacy_cap_is_lower_than_visible_road_surface():
    spec = SimpleNamespace(cells=2, cell_size=10.0)
    node = (10.0, 10.0)
    cap = _p.WorldObject(7, r"o\road\sil6.p3d", 10.0, 0.041, 10.0, 0.0)

    lowered = _edge._lower_legacy_cap(
        cap,
        node,
        (0.0, 0.0, 0.0, 0.0),
        spec,
    )

    assert math.isclose(lowered.x, node[0], abs_tol=1.0e-9)
    assert math.isclose(lowered.z, node[1], abs_tol=1.0e-9)
    assert math.isclose(
        lowered.y,
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
        + _edge.INTERSECTION_CAP_UNDERLAY_BIAS_METRES,
        abs_tol=1.0e-9,
    )
    assert lowered.y < _p._STOCK_ROAD_VERTICAL_OFFSET_METRES


def test_intersection_tongue_overlaps_node_and_incoming_approach():
    spec = SimpleNamespace(cells=2, cell_size=10.0)
    node = (10.0, 10.0)
    tongue = _edge._tongue_object(
        8,
        "sil",
        node,
        90.0,
        (0.0, 0.0, 0.0, 0.0),
        spec,
    )
    start, end = _p._model_axis(tongue, 6.25)

    # Heading 90 is +X.  The underlay reaches behind the logical node and much
    # farther outward, overlapping the real approach instead of stopping at the
    # same edge and opening another wedge.
    assert math.isclose(start[0], node[0] - 1.75, abs_tol=1.0e-6)
    assert math.isclose(end[0], node[0] + 4.50, abs_tol=1.0e-6)
    assert math.isclose(start[1], node[1], abs_tol=1.0e-6)
    assert math.isclose(end[1], node[1], abs_tol=1.0e-6)
    assert tongue.y < _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
