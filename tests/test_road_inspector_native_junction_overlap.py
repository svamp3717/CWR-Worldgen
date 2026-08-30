# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_native_junction_overlap as _overlap


def _endpoint(object_id, index, point, heading=0.0):
    return _core.RoadEndpoint(
        object_id=object_id,
        model_path=r"o\road\sil6.p3d",
        family="sil",
        object_kind="straight",
        endpoint_index=index,
        point=point,
        tangent_axis_degrees=heading % 180.0,
        outward_heading_degrees=heading % 360.0,
        half_width_metres=4.55,
    )


def _straight(object_id, first, second):
    return _core.RoadObject(
        object_id=object_id,
        model_path=r"o\road\sil6.p3d",
        x=(first[0] + second[0]) * 0.5,
        y=0.035,
        z=(first[1] + second[1]) * 0.5,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="straight",
        nominal_length_metres=6.25,
        logical_center=((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5),
        endpoints=(
            _endpoint(object_id, 0, first, 180.0),
            _endpoint(object_id, 1, second, 0.0),
        ),
    )


def _native_t():
    return _core.RoadObject(
        object_id=1,
        model_path=r"o\road\kr_new_sil_sil_t.p3d",
        x=0.0,
        y=0.041,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="junction_t",
        nominal_length_metres=6.25,
        logical_center=(0.0, 0.0),
        endpoints=(),
    )


def test_inspector_reports_ordinary_road_crossing_native_t_center(monkeypatch) -> None:
    source = _core.SourceJunction((0.0, 0.0), (0.0, 90.0, 180.0))
    intruder = _straight(2, (0.0, -3.125), (0.0, 3.125))
    monkeypatch.setattr(
        _overlap,
        "_ORIGINAL_SOURCE_INTERSECTION_ISSUES",
        lambda roads, junctions, match_tolerance: [],
    )

    issues = _overlap._source_intersection_issues(
        (_native_t(), intruder),
        (source,),
        match_tolerance=0.75,
    )

    issue = next(item for item in issues if item.category == "intersection_native_overlap")
    assert issue.metrics["native_model"] == r"o\road\kr_new_sil_sil_t.p3d"
    assert issue.metrics["intruding_road_count"] == 1
    assert set(issue.object_ids) == {1, 2}


def test_inspector_accepts_approach_that_stops_at_native_connector(monkeypatch) -> None:
    source = _core.SourceJunction((0.0, 0.0), (0.0, 90.0, 180.0))
    approach = _straight(2, (0.0, 6.25), (0.0, 12.50))
    monkeypatch.setattr(
        _overlap,
        "_ORIGINAL_SOURCE_INTERSECTION_ISSUES",
        lambda roads, junctions, match_tolerance: [],
    )

    issues = _overlap._source_intersection_issues(
        (_native_t(), approach),
        (source,),
        match_tolerance=0.75,
    )

    assert not any(item.category == "intersection_native_overlap" for item in issues)
