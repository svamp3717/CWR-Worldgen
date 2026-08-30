# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_wrptool_catalogue as _catalogue_inspector


def _endpoint(object_id: int, family: str, heading: float):
    angle = math.radians(float(heading))
    point = (math.sin(angle) * 6.0, math.cos(angle) * 6.0)
    endpoint = _core.RoadEndpoint(
        object_id=object_id,
        model_path=rf"o\road\{family}6.p3d",
        family=family,
        object_kind="straight",
        endpoint_index=0,
        point=point,
        tangent_axis_degrees=heading % 180.0,
        outward_heading_degrees=heading,
        half_width_metres=4.55 if family in {"sil", "kos"} else 3.50 if family == "asf" else 1.75,
    )
    road = _core.RoadObject(
        object_id=object_id,
        model_path=endpoint.model_path,
        x=point[0],
        y=0.035,
        z=point[1],
        heading_degrees=heading,
        pitch_degrees=0.0,
        family=family,
        kind="straight",
        nominal_length_metres=6.25,
        logical_center=point,
        endpoints=(endpoint,),
    )
    return road


def test_inspector_recommends_exact_wrptool_mixed_t_model(monkeypatch) -> None:
    roads = (
        _endpoint(1, "asf", 0.0),
        _endpoint(2, "asf", 180.0),
        _endpoint(3, "sil", 90.0),
        _core.RoadObject(
            object_id=4,
            model_path=r"o\road\sil6.p3d",
            x=0.0,
            y=0.035,
            z=0.0,
            heading_degrees=0.0,
            pitch_degrees=0.0,
            family="sil",
            kind="straight",
            nominal_length_metres=6.25,
            logical_center=(0.0, 0.0),
            endpoints=(),
        ),
    )
    source = _core.SourceJunction((0.0, 0.0), (0.0, 90.0, 180.0))
    monkeypatch.setattr(
        _catalogue_inspector,
        "_ORIGINAL_SOURCE_INTERSECTION_ISSUES",
        lambda roads, junctions, match_tolerance: [],
    )

    issues = _catalogue_inspector._source_intersection_issues(
        roads,
        (source,),
        match_tolerance=0.75,
    )

    issue = next(
        item for item in issues if item.category == "intersection_stock_asset_mismatch"
    )
    assert issue.metrics["expected_stock_model"] == r"o\road\kr_new_asf_sil_t.p3d"
    assert issue.metrics["actual_central_model"] == r"o\road\sil6.p3d"


def test_inspector_accepts_correct_wrptool_native_t(monkeypatch) -> None:
    roads = (
        _endpoint(1, "asf", 0.0),
        _endpoint(2, "asf", 180.0),
        _endpoint(3, "sil", 90.0),
        _core.RoadObject(
            object_id=4,
            model_path=r"o\road\kr_new_asf_sil_t.p3d",
            x=0.0,
            y=0.041,
            z=0.0,
            heading_degrees=0.0,
            pitch_degrees=0.0,
            family="asf",
            kind="junction_t",
            nominal_length_metres=6.25,
            logical_center=(0.0, 0.0),
            endpoints=(),
        ),
    )
    source = _core.SourceJunction((0.0, 0.0), (0.0, 90.0, 180.0))
    monkeypatch.setattr(
        _catalogue_inspector,
        "_ORIGINAL_SOURCE_INTERSECTION_ISSUES",
        lambda roads, junctions, match_tolerance: [],
    )

    issues = _catalogue_inspector._source_intersection_issues(
        roads,
        (source,),
        match_tolerance=0.75,
    )

    assert not any(
        item.category == "intersection_stock_asset_mismatch" for item in issues
    )


def test_wrptool_special_kos_road_is_not_silently_discarded(monkeypatch) -> None:
    values = [0.0] * 14
    values[0] = 1.0
    values[2] = 0.0
    values[7] = 0.0
    values[9] = 12.0
    values[10] = 1.5
    values[11] = 34.0
    values[12] = 77
    values[13] = b"o\\road\\kr_new_kos.p3d\0"
    monkeypatch.setattr(
        _catalogue_inspector,
        "_ORIGINAL_ROAD_OBJECT_FROM_RECORD",
        lambda values: None,
    )

    road = _catalogue_inspector._road_object_from_record(tuple(values))

    assert road is not None
    assert road.object_id == 77
    assert road.family == "kos"
    assert road.kind == "stock_special"
    assert road.endpoints == ()
