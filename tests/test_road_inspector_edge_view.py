from __future__ import annotations

# Import the public entry first so all read-only inspector layers are installed.
import cwr_worldgen.road_inspector_entry  # noqa: F401
from cwr_worldgen import road_inspector as _core
from cwr_worldgen import stock_road_model_geometry as _geometry


def _straight_road() -> _core.RoadObject:
    first = _core.RoadEndpoint(
        object_id=7,
        model_path=r"o\road\sil6.p3d",
        family="sil",
        object_kind="straight",
        endpoint_index=0,
        point=(100.0, 100.0),
        tangent_axis_degrees=0.0,
        outward_heading_degrees=180.0,
        half_width_metres=4.55,
    )
    second = _core.RoadEndpoint(
        object_id=7,
        model_path=r"o\road\sil6.p3d",
        family="sil",
        object_kind="straight",
        endpoint_index=1,
        point=(100.0, 106.25),
        tangent_axis_degrees=0.0,
        outward_heading_degrees=0.0,
        half_width_metres=4.55,
    )
    return _core.RoadObject(
        object_id=7,
        model_path=r"o\road\sil6.p3d",
        x=100.0,
        y=0.035,
        z=103.125,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="straight",
        nominal_length_metres=6.25,
        logical_center=(100.0, 103.125),
        endpoints=(first, second),
    )


def test_html_report_contains_stock_road_edge_geometry_and_search_controls() -> None:
    road = _straight_road()
    issue = _core.RoadIssue(
        issue_id="RI-00001",
        severity="high",
        score=60.0,
        category="straight_miter",
        x=100.0,
        z=106.25,
        object_ids=(7, 8),
        models=(r"o\road\sil6.p3d", r"o\road\sil6.p3d"),
        message="test seam",
        candidate_fix="test fix",
        metrics={"source_road_ids": "road-000280", "edge_gap_max_metres": 0.42},
    )
    result = _core.InspectionResult(
        input_path="sample.wrp",
        wrp_entry="sample.wrp",
        road_object_count=1,
        source_junction_count=0,
        issues=(issue,),
        road_objects=(road,),
    )

    payload = _core._road_payload(road)
    document = _core._html_document(result)

    assert payload["half_width"] == 4.55
    assert ".road.edge" in document
    assert "r.half_width" in document
    assert "svg.insertBefore(line,svg.firstChild)" in document
    assert "issue-search" in document
    assert "Issue metrics" in document
    assert "source_road_ids" in document


def test_native_curve_map_geometry_is_sampled_as_an_arc() -> None:
    geometry = _geometry.stock_curve_connectors(r"o\road\sil10 50.p3d")
    assert geometry is not None
    begin = _geometry.transform_local(geometry.begin, (200.0, 300.0), 20.0)
    end = _geometry.transform_local(geometry.end, (200.0, 300.0), 20.0)
    road = _core.RoadObject(
        object_id=9,
        model_path=r"o\road\sil10 50.p3d",
        x=200.0,
        y=0.035,
        z=300.0,
        heading_degrees=20.0,
        pitch_degrees=0.0,
        family="sil",
        kind="curve",
        nominal_length_metres=geometry.chord_length_metres,
        logical_center=((begin[0] + end[0]) * 0.5, (begin[1] + end[1]) * 0.5),
        endpoints=(
            _core.RoadEndpoint(9, r"o\road\sil10 50.p3d", "sil", "curve", 0, begin, 20.0, 200.0, 4.55),
            _core.RoadEndpoint(9, r"o\road\sil10 50.p3d", "sil", "curve", 1, end, 30.0, 30.0, 4.55),
        ),
    )

    payload = _core._road_payload(road)

    assert len(payload["segments"]) == 8
    assert payload["segments"][0][:2] == [round(begin[0], 4), round(begin[1], 4)]
    assert payload["segments"][-1][2:] == [round(end[0], 4), round(end[1], 4)]
