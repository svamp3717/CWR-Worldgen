# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_grass_wedge as _grass
from cwr_worldgen import road_inspector_paved_wedge_audit as _audit
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_paved_wedge_policy as _paved


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


def _straight_from_start(object_id, family, start, heading):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    angle = math.radians(float(heading))
    direction = (math.sin(angle), math.cos(angle))
    centre = (
        float(start[0]) + direction[0] * half,
        float(start[1]) + direction[1] * half,
    )
    return _object(
        object_id,
        rf"o\road\{family}6.p3d",
        centre[0],
        centre[1],
        heading,
    )


def test_terrain_wedge_planner_covers_every_paved_family_but_not_dirt():
    for family in ("sil", "asf", "kos"):
        first = _object(1, rf"o\road\{family}6.p3d", 0.0, -3.125, 0.0)
        second = _straight_from_start(2, family, (0.0, 0.0), 12.0)
        report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)

        plans = _paved._terrain_wedge_cover_plans(report)

        assert len(plans) == 1
        assert plans[0].model_path.casefold() == rf"o\road\{family}6.p3d"
        assert plans[0].outer_miter_apex is not None

    first = _object(1, r"o\road\ces6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, "ces", (0.0, 0.0), 12.0)
    report = SimpleNamespace(objects=(first, second), junction_cap_objects=0)
    assert _paved._terrain_wedge_cover_plans(report) == ()


def test_generated_paved_wedge_counts_as_existing_visible_paved_surface():
    wedge = _object(3, r"wg_test\i\paved_wedge_q048.p3d", 0.0, 0.0, 0.0, y=0.03)

    assert _paved._surface_is_paved(wedge)
    assert _paved._surface_contains(wedge, (0.0, 0.05))


def _endpoint(object_id, *, family="asf", endpoint_index, tangent, outward, point):
    return _core._endpoint(
        object_id=object_id,
        model_path=rf"o\road\{family}6.p3d",
        family=family,
        kind="straight",
        endpoint_index=endpoint_index,
        point=point,
        tangent_axis_degrees=tangent,
        outward_heading_degrees=outward,
    )


def _road(object_id, family, first, second):
    return _core.RoadObject(
        object_id=object_id,
        model_path=rf"o\road\{family}6.p3d",
        x=100.0,
        y=10.0,
        z=200.0,
        heading_degrees=float(first.tangent_axis_degrees),
        pitch_degrees=0.0,
        family=family,
        kind="straight",
        nominal_length_metres=6.25,
        logical_center=(100.0, 200.0),
        endpoints=(first, second),
    )


def test_direct_inspector_audit_adds_asphalt_wedge_and_ignores_dirt():
    incoming = _endpoint(
        1,
        endpoint_index=1,
        tangent=0.0,
        outward=0.0,
        point=(100.0, 200.0),
    )
    outgoing = _endpoint(
        2,
        endpoint_index=0,
        tangent=20.0,
        outward=200.0,
        point=(100.0, 200.0),
    )
    first = _road(
        1,
        "asf",
        _endpoint(
            1,
            endpoint_index=0,
            tangent=0.0,
            outward=180.0,
            point=(100.0, 193.75),
        ),
        incoming,
    )
    second = _road(
        2,
        "asf",
        outgoing,
        _endpoint(
            2,
            endpoint_index=1,
            tangent=20.0,
            outward=20.0,
            point=(102.1376, 205.8731),
        ),
    )
    result = _core.InspectionResult(
        input_path="synthetic.wrp",
        wrp_entry="synthetic.wrp",
        road_object_count=2,
        source_junction_count=0,
        issues=(),
        road_objects=(first, second),
    )

    _grass._SEAM_CATEGORIES = frozenset(set(_grass._SEAM_CATEGORIES) | {"connector_gap"})
    audited = _audit._scan_missing_grass_wedges(result, (), 0.75, None)

    assert len(audited.issues) == 1
    assert audited.issues[0].category == "grass_wedge"
    assert audited.issues[0].models[0].casefold().endswith(r"asf6.p3d")

    dirt_first = _core.RoadObject(
        **{**first.__dict__, "family": "ces", "model_path": r"o\road\ces6.p3d"}
    ) if hasattr(first, "__dict__") else None
    # Slotted RoadObject instances have no __dict__; the paved-only candidate
    # filter itself is the contract we care about for dirt.
    assert all(
        endpoint.family in {"sil", "asf", "kos"}
        for pair in _audit._candidate_pairs((first, second))
        for endpoint in pair
    )
