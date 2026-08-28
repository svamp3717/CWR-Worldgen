from __future__ import annotations

# Import the public entry first so all read-only inspector layers are installed.
import cwr_worldgen.road_inspector_entry  # noqa: F401
from cwr_worldgen import road_inspector as _core


def test_html_report_contains_stock_road_edge_geometry() -> None:
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
    road = _core.RoadObject(
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
    result = _core.InspectionResult(
        input_path="sample.wrp",
        wrp_entry="sample.wrp",
        road_object_count=1,
        source_junction_count=0,
        issues=(),
        road_objects=(road,),
    )

    payload = _core._road_payload(road)
    document = _core._html_document(result)

    assert payload["half_width"] == 4.55
    assert ".road.edge" in document
    assert "r.half_width" in document
    assert "svg.insertBefore(line,svg.firstChild)" in document
