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


def test_detects_visible_straight_miter_edge_discontinuity(tmp_path: Path) -> None:
    wrp = tmp_path / "miter.wrp"
    seam = (0.0, 6.25)
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = _straight_from_begin(2, seam, 6.0)
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    issue = next(issue for issue in result.issues if issue.category == "straight_miter")
    assert issue.object_ids == (1, 2)
    assert issue.metrics["center_gap_metres"] < 0.001
    assert issue.metrics["tangent_error_degrees"] > 5.9
    assert issue.metrics["edge_gap_max_metres"] > 0.45
    assert "connector-locked" in issue.candidate_fix


def test_tangent_continuous_straights_do_not_raise_seam_issue(tmp_path: Path) -> None:
    wrp = tmp_path / "straight.wrp"
    first = WorldObject(1, r"o\road\sil6.p3d", 0.0, 0.035, 3.125, 0.0)
    second = WorldObject(2, r"o\road\sil6.p3d", 0.0, 0.035, 9.375, 0.0)
    _write_world(wrp, (first, second))

    result = inspect_road_geometry(wrp)

    assert not [issue for issue in result.issues if issue.category in {"straight_miter", "connector_gap", "curve_transition"}]


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
    assert any(issue.category == "straight_miter" for issue in result.issues)


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
    assert "straight_miter" in html_text
    assert "Reset map" in html_text
