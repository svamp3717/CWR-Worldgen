from __future__ import annotations

import json
import math
from pathlib import Path

from cwr_worldgen.model import WorldObject
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.road_inspector_entry import inspect_road_geometry, write_inspection_report
from cwr_worldgen.wrp import write_rvw4


def _write_world(path: Path, objects) -> None:
    write_rvw4(
        path,
        2,
        2,
        [0.0] * 4,
        [0] * 4,
        [r"data\more_anim.01.pac"],
        objects,
        height_scale=0.05,
    )


def _straight_from_begin(object_id: int, begin, heading: float, *, length: float = 6.25) -> WorldObject:
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    return WorldObject(
        object_id,
        r"o\road\sil6.p3d",
        begin[0] + direction[0] * length * 0.5,
        0.035,
        begin[1] + direction[1] * length * 0.5,
        heading,
    )


def _straight_to_end(object_id: int, end, outward_heading: float, *, length: float = 6.25) -> WorldObject:
    # A piece whose endpoint 1 lands on ``end`` and whose outward direction from
    # the junction is ``outward_heading`` therefore points toward the node.
    heading = (outward_heading + 180.0) % 360.0
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    return WorldObject(
        object_id,
        r"o\road\sil6.p3d",
        end[0] - direction[0] * length * 0.5,
        0.035,
        end[1] - direction[1] * length * 0.5,
        heading,
    )


def _local_to_wgs84(point, bbox, world_size: float):
    west, south, east, north = bbox
    return [
        west + float(point[0]) / world_size * (east - west),
        south + float(point[1]) / world_size * (north - south),
    ]


def test_detects_visible_straight_miter_edge_discontinuity(tmp_path: Path) -> None:
    wrp = tmp_path / "miter.wrp"
    seam = (0.0, 6.25)
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, seam, 6.0)
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    issue = next(issue for issue in result.issues if issue.category == "grass_wedge")
    assert issue.object_ids == (1, 2)
    assert issue.metrics["center_gap_metres"] < 0.001
    assert issue.metrics["tangent_error_degrees"] > 5.9
    assert issue.metrics["edge_gap_max_metres"] > 0.45
    assert issue.metrics["grass_wedge_area_square_metres"] > 0.001
    assert "connector-locked" in issue.candidate_fix


def test_circular_paved_fill_does_not_hide_straight_miter(tmp_path: Path) -> None:
    wrp = tmp_path / "circular-filled-miter.wrp"
    seam = (0.0, 6.25)
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, seam, 6.0)
    fill = WorldObject(
        3,
        r"wg_test\i\paved_fill.p3d",
        seam[0],
        0.025,
        seam[1],
        3.0,
    )
    _write_world(wrp, (first, second, fill))

    result = inspect_road_geometry(wrp)

    assert result.road_object_count == 3
    assert [
        issue
        for issue in result.issues
        if issue.category == "grass_wedge"
        and set(issue.object_ids) == {1, 2}
    ]


def test_angle_matched_paved_miter_covers_straight_miter(tmp_path: Path) -> None:
    wrp = tmp_path / "miter-filled-turn.wrp"
    seam = (0.0, 6.25)
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, seam, 6.0)
    fill = WorldObject(
        3,
        r"wg_test\i\paved_miter_q024.p3d",
        seam[0],
        0.025,
        seam[1],
        3.0,
    )
    _write_world(wrp, (first, second, fill))

    result = inspect_road_geometry(wrp)

    assert result.road_object_count == 3
    assert not [
        issue
        for issue in result.issues
        if issue.category in {"straight_miter", "grass_wedge"}
        and set(issue.object_ids) == {1, 2}
    ]


def test_tangent_continuous_straights_do_not_raise_seam_issue(tmp_path: Path) -> None:
    wrp = tmp_path / "straight.wrp"
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = WorldObject(2, r"o\road\sil6.p3d", 0.0, 0.035, 9.375, 0.0)
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    assert not [issue for issue in result.issues if issue.category in {"straight_miter", "grass_wedge", "connector_gap", "curve_transition"}]


def test_reads_real_wrp_entry_from_generated_pbo(tmp_path: Path) -> None:
    wrp = tmp_path / "sample.wrp"
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, (0.0, 6.25), 5.0)
    _write_world(wrp, (first, second))
    pbo = tmp_path / "sample.pbo"
    write_pbo(pbo, (PboEntry("sample.wrp", wrp.read_bytes()), PboEntry("config.cpp", b"class CfgWorlds {};\n")))

    result = inspect_road_geometry(pbo)

    assert result.wrp_entry == "sample.wrp"
    assert result.road_object_count == 2
    assert any(issue.category == "grass_wedge" for issue in result.issues)


def test_normalized_roads_flag_visible_straight_cap_on_turning_intersection(tmp_path: Path) -> None:
    wrp = tmp_path / "junction.wrp"
    node = (100.0, 100.0)
    # Legacy visible cap follows north/south while the southern approach is
    # already turning southwest. The three approaches terminate on the node.
    cap = WorldObject(1, r"o\road\sil6.p3d", node[0], 0.041, node[1], 0.0)
    north = _straight_to_end(2, node, 0.0)
    southwest = _straight_to_end(3, node, 190.0)
    east = _straight_to_end(4, node, 90.0)
    _write_world(wrp, (cap, north, southwest, east))

    roads = tmp_path / "roads.geojson"
    roads.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"road_id": "main"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[98.2635, 90.1519], [100.0, 100.0], [100.0, 120.0]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"road_id": "branch"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[100.0, 100.0], [120.0, 100.0]],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = inspect_road_geometry(wrp, roads_geojson=roads)

    issue = next(issue for issue in result.issues if issue.category == "turning_intersection_cap")
    assert issue.x == 100.0
    assert issue.z == 100.0
    assert issue.metrics["through_turn_degrees"] > 9.0
    assert issue.metrics["cap_below_approach_margin_metres"] < 0.0
    assert issue.metrics["maximum_approach_heading_error_degrees"] < 0.5


def test_normalized_turning_intersection_accepts_borderless_paved_fill(
    tmp_path: Path,
) -> None:
    wrp = tmp_path / "filled-junction.wrp"
    node = (100.0, 100.0)
    fill = WorldObject(
        1,
        r"wg_test\i\paved_fill.p3d",
        node[0],
        0.031,
        node[1],
        0.0,
    )
    north = _straight_to_end(2, node, 0.0)
    southwest = _straight_to_end(3, node, 190.0)
    east = _straight_to_end(4, node, 90.0)
    _write_world(wrp, (fill, north, southwest, east))

    roads = tmp_path / "filled-roads.geojson"
    roads.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"road_id": "main"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [98.2635, 90.1519],
                                [100.0, 100.0],
                                [100.0, 120.0],
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"road_id": "branch"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[100.0, 100.0], [120.0, 100.0]],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = inspect_road_geometry(wrp, roads_geojson=roads)

    assert result.source_junction_count == 1
    assert not [
        issue
        for issue in result.issues
        if issue.category
        in {"turning_intersection_cap", "intersection_connector_orientation"}
    ]


def test_normalized_wgs84_roads_are_projected_into_wrp_metres(tmp_path: Path) -> None:
    wrp = tmp_path / "junction_wgs84.wrp"
    node = (500.0, 500.0)
    cap = WorldObject(1, r"o\road\sil6.p3d", node[0], 0.041, node[1], 0.0)
    north = _straight_to_end(2, node, 0.0)
    southwest = _straight_to_end(3, node, 190.0)
    east = _straight_to_end(4, node, 90.0)
    _write_world(wrp, (cap, north, southwest, east))

    bbox = [10.0, 20.0, 12.0, 22.0]
    roads = tmp_path / "roads-wgs84.geojson"
    roads.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "bbox": bbox,
                "cwr_world": {
                    "coordinate_reference": "WGS84 longitude/latitude",
                    "world_size_metres": 1000.0,
                    "grid_cells": 40,
                    "cell_size_metres": 25.0,
                },
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"road_id": "main"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                _local_to_wgs84((498.2635, 490.1519), bbox, 1000.0),
                                _local_to_wgs84(node, bbox, 1000.0),
                                _local_to_wgs84((500.0, 620.0), bbox, 1000.0),
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"road_id": "branch"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                _local_to_wgs84(node, bbox, 1000.0),
                                _local_to_wgs84((620.0, 500.0), bbox, 1000.0),
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = inspect_road_geometry(wrp, roads_geojson=roads)

    assert result.source_junction_count == 1
    issue = next(issue for issue in result.issues if issue.category == "turning_intersection_cap")
    assert math.isclose(issue.x, 500.0, abs_tol=0.01)
    assert math.isclose(issue.z, 500.0, abs_tol=0.01)
    assert issue.metrics["maximum_approach_heading_error_degrees"] < 0.5


def test_pitched_straights_use_rvw4_horizontal_connector_positions(tmp_path: Path) -> None:
    wrp = tmp_path / "pitched.wrp"
    pitch = 10.0
    horizontal = 6.25 * math.cos(math.radians(pitch))
    first = WorldObject(
        1,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        horizontal * 0.5,
        0.0,
        pitch,
    )
    second = WorldObject(
        2,
        r"o\road\sil6.p3d",
        0.0,
        0.035,
        horizontal * 1.5,
        0.0,
        pitch,
    )
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    assert not [
        issue
        for issue in result.issues
        if issue.category in {"straight_miter", "grass_wedge", "connector_gap", "curve_transition"}
    ]


def test_nearby_facing_connectors_are_reported_even_outside_cluster_tolerance(tmp_path: Path) -> None:
    wrp = tmp_path / "connector-gap.wrp"
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, (0.0, 6.70), 0.0)
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    issue = next(issue for issue in result.issues if issue.category == "connector_gap")
    assert 0.44 < issue.metrics["center_gap_metres"] < 0.46
    assert issue.metrics["detector"] == "nearby_unmatched_connector"
    assert issue.metrics["gap_alignment_first_degrees"] < 0.1
    assert issue.metrics["gap_alignment_second_degrees"] < 0.1


def test_writes_self_contained_html_json_csv_and_summary(tmp_path: Path) -> None:
    wrp = tmp_path / "report.wrp"
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, (0.0, 6.25), 6.0)
    _write_world(wrp, (first, second))
    result = inspect_road_geometry(wrp)

    paths = write_inspection_report(result, tmp_path / "inspection")

    assert set(paths) == {"issues_json", "issues_csv", "summary_json", "html"}
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["road_objects"] == 2
    assert summary["issue_count"] >= 1
    html_text = paths["html"].read_text(encoding="utf-8")
    assert "Road Inspector" in html_text
    assert "grass_wedge" in html_text
    assert "Reset map" in html_text
    assert "Copy teleport" in html_text
    assert "player setPos [" in html_text
    assert "roadById" in html_text
