from __future__ import annotations

import json
from pathlib import Path

# Import the public entry first so the inspector runtime is composed exactly as
# the installed cwr-road-inspector command sees it.
import cwr_worldgen.road_inspector_entry  # noqa: F401
from cwr_worldgen import road_inspector_source_context as _source
from cwr_worldgen.road_inspector import RoadIssue


def test_wgs84_source_segments_are_reported_in_world_metres(tmp_path: Path) -> None:
    roads = tmp_path / "roads.geojson"
    roads.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "bbox": [10.0, 20.0, 12.0, 22.0],
                "cwr_world": {
                    "coordinate_reference": "WGS84 longitude/latitude",
                    "world_size_metres": 1000.0,
                },
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "road_id": "road-000280",
                            "highway": "tertiary",
                            "surface": "asphalt",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[10.8, 21.0], [11.2, 21.0]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    segments = _source._segments(roads)
    assert len(segments) == 1
    assert segments[0].start == (400.00000000000034, 500.0)
    assert segments[0].end == (599.9999999999997, 500.0)

    issue = RoadIssue(
        issue_id="RI-00001",
        severity="high",
        score=60.0,
        category="straight_miter",
        x=500.0,
        z=500.0,
        object_ids=(1, 2),
        models=(r"o\road\sil6.p3d", r"o\road\sil6.p3d"),
        message="test",
        candidate_fix="test",
        metrics={},
    )
    metrics = _source._context_metrics(issue, segments)

    assert metrics["nearest_source_distance_metres"] == 0.0
    assert metrics["source_road_ids"] == "road-000280"
    assert metrics["source_highways"] == "tertiary"
    assert metrics["source_surfaces"] == "asphalt"
    assert metrics["source_segment_axes_degrees"] == "90.000"
