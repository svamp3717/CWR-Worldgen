# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_inspector_candidate_policy as _candidate
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_native_junction_ownership_policy as _ownership
from cwr_worldgen import stock_road_surface_overlap_policy as _surface
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def _flat_spec(*, name: str = "candidate_test"):
    return SimpleNamespace(
        name=name,
        cells=8,
        cell_size=10.0,
        max_road_objects=1000,
        advisory_object_limits=False,
    )


def _flat_elevations():
    return (0.0,) * 64


def _mixed_native_cap(node=(30.0, 30.0)):
    native = _candidate._measured_native_t_junction(
        (
            _incident(0.0, "sil"),
            _incident(180.0, "sil"),
            _incident(270.0, "ces"),
        )
    )
    assert native is not None
    old = _p.WorldObject(
        1,
        r"o\road\sil6.p3d",
        float(node[0]),
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
        float(node[1]),
        0.0,
    )
    return _candidate._measured_native_junction_object(
        old, native, _flat_elevations(), _flat_spec()
    )


def test_measured_t_uses_negative_x_branch_connector() -> None:
    native = _candidate._measured_native_t_junction(
        (
            _incident(0.0, "sil"),
            _incident(180.0, "sil"),
            _incident(270.0, "ces"),
        )
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_ces_t.p3d"
    assert math.isclose(native.heading_degrees % 360.0, 0.0, abs_tol=1.0e-9)
    assert native.maximum_heading_error_degrees < 1.0e-9


def test_measured_native_origin_keeps_logical_center_on_source_node() -> None:
    node = (30.0, 30.0)
    cap = _mixed_native_cap(node)
    local_center = _geometry.native_junction_intersection_offset(cap.model_path)
    assert local_center is not None
    mapped_center = _geometry.transform_local(
        local_center,
        (float(cap.x), float(cap.z)),
        float(cap.heading_degrees),
    )

    assert math.dist(mapped_center, node) < 1.0e-9
    connectors = _surface._native_cap_connectors(cap)
    assert len(connectors) == 3
    assert all(
        math.isclose(
            math.dist(connector.point, node),
            _geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES,
            abs_tol=1.0e-9,
        )
        for connector in connectors
    )


def test_mixed_native_center_removes_short_ces_intruder() -> None:
    node = (30.0, 30.0)
    cap = _mixed_native_cap(node)
    ces = _p.WorldObject(
        2,
        r"o\road\ces6.p3d",
        node[0],
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
        node[1],
        270.0,
    )

    objects, _next_id, changed = _candidate._trim_one_native_center(
        [cap, ces],
        cap,
        cap_count=1,
        elevations=_flat_elevations(),
        spec=_flat_spec(),
        next_id=3,
    )

    assert changed
    assert objects == [cap]


def test_mixed_native_center_keeps_only_real_ces_branch_side() -> None:
    node = (30.0, 30.0)
    cap = _mixed_native_cap(node)
    ces = _p.WorldObject(
        2,
        r"o\road\ces25.p3d",
        node[0],
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
        node[1],
        270.0,
    )

    objects, _next_id, changed = _candidate._trim_one_native_center(
        [cap, ces],
        cap,
        cap_count=1,
        elevations=_flat_elevations(),
        spec=_flat_spec(),
        next_id=3,
    )

    assert changed
    assert len(objects) == 2
    rebuilt = objects[1]
    assert rebuilt.model_path.casefold() == r"o\road\ces6.p3d"

    branch = next(
        connector
        for connector in _surface._native_cap_connectors(cap)
        if connector.family == "ces"
    )
    axis = _ownership._physical_straight_axis(rebuilt)
    assert axis is not None
    assert min(math.dist(branch.point, endpoint) for endpoint in axis) < 1.0e-6
    assert max(math.dist(node, endpoint) for endpoint in axis) > 12.0


def test_wedge_candidate_adds_borderless_overlay_not_stock_strip(monkeypatch) -> None:
    plan = _finish._SeamCoverPlan(
        model_path=r"o\road\sil6.p3d",
        centre=(30.0, 30.0),
        tangent_axis_degrees=0.0,
        turn_degrees=10.0,
        outer_miter_apex=(34.0, 30.0),
    )
    report = _p.RoadFitReport(
        objects=(),
        chain_count=0,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )

    monkeypatch.setattr(
        _candidate,
        "_ORIGINAL_STOCK_APPLY",
        lambda current, elevations, spec: current,
    )
    monkeypatch.setattr(
        _emitted,
        "_terrain_wedge_cover_plans",
        lambda current, elevations=None, spec=None: (plan,),
    )

    fixed = _candidate._apply_wedge_candidates(
        report,
        _flat_elevations(),
        _flat_spec(name="candidate_wedge"),
    )

    assert len(fixed.objects) == 1
    model = fixed.objects[0].model_path.replace("/", "\\").casefold()
    assert "\\paved_wedge_q" in model
    assert model != r"o\road\sil6.p3d"


def test_candidate_policy_is_wired_into_production_hooks() -> None:
    assert _candidate._INSTALLED
    assert _junction._native_t_junction is _candidate._measured_native_t_junction
    assert _junction._native_junction_object is _candidate._measured_native_junction_object
    assert _ownership._trim_one_native_center is _candidate._trim_one_native_center
    assert _emitted._apply_emitted_seam_covers is _candidate._apply_wedge_candidates
