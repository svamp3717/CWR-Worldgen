from __future__ import annotations

import csv
from types import SimpleNamespace

from cwr_worldgen import road_inspector_coordinate_log as coordinate_log


def test_coordinate_log_contains_findings_and_every_road(tmp_path) -> None:
    issue = SimpleNamespace(
        issue_id="RI-0001",
        severity="high",
        category="curve_transition",
        x=123.5,
        z=456.25,
        object_ids=(7,),
        metrics={
            "source_road_ids": "road-42",
            "source_highways": "primary",
            "source_surfaces": "asphalt",
        },
    )
    roads = (
        SimpleNamespace(
            object_id=7,
            family="asf",
            kind="curve",
            model_path=r"o\road\asf6k10.p3d",
            logical_center=(124.0, 455.5),
        ),
        SimpleNamespace(
            object_id=8,
            family="sil",
            kind="straight",
            model_path=r"o\road\sil6.p3d",
            logical_center=(130.0, 460.0),
        ),
    )
    result = SimpleNamespace(issues=(issue,), road_objects=roads)
    path = tmp_path / "ingame-coordinates.csv"

    coordinate_log._write_coordinate_log(result, path)

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["record_type"] for row in rows] == ["finding", "road", "road"]
    assert rows[0]["issue_id"] == "RI-0001"
    assert rows[0]["source_surfaces"] == "asphalt"
    assert rows[0]["teleport_command"] == "player setPos [123.5, 456.25, 0]"
    assert rows[1]["object_id"] == "7"
    assert rows[1]["related_issue_ids"] == "RI-0001"
    assert rows[2]["object_id"] == "8"
    assert rows[2]["related_issue_ids"] == ""
    assert rows[2]["teleport_command"] == "player setPos [130, 460, 0]"
