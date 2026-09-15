from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen import bridge_underlay_cleanup_policy as cleanup
from cwr_worldgen import bridge_underlay_spatial_policy as spatial
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import RoadFitReport


def _report(*objects: WorldObject) -> RoadFitReport:
    return RoadFitReport(
        objects=tuple(objects),
        chain_count=1,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def test_spatial_cleanup_matches_exact_full_span_scan() -> None:
    spans = (
        cleanup._BridgeSpan(
            points=((0.0, 0.0), (250.0, 0.0)),
            road_width=7.0,
        ),
        cleanup._BridgeSpan(
            points=((1000.0, 1000.0), (1250.0, 1000.0)),
            road_width=7.0,
        ),
    )
    objects = (
        WorldObject(1, r"o\road\sil25.p3d", 25.0, 0.0, 0.0, 90.0),
        WorldObject(2, r"o\road\sil25.p3d", 125.0, 0.0, 0.0, 90.0),
        WorldObject(3, r"o\road\sil25.p3d", 125.0, 0.0, 8.0, 90.0),
        WorldObject(4, r"o\road\sil25.p3d", 1125.0, 0.0, 1000.0, 90.0),
        WorldObject(5, r"o\road\sil25.p3d", 1125.0, 0.0, 1000.0, 0.0),
        WorldObject(6, r"ca\plants\smrk.p3d", 125.0, 0.0, 0.0, 0.0),
    )
    report = _report(*objects)

    expected_kept = tuple(
        obj
        for obj in report.objects
        if not cleanup._road_object_under_bridge(obj, spans)
    )
    actual, removed = spatial._remove_bridge_underlays(report, spans)

    assert actual.objects == expected_kept
    assert removed == len(report.objects) - len(expected_kept)


def test_span_index_prunes_distant_bridge_spans_without_losing_candidates() -> None:
    spans = tuple(
        cleanup._BridgeSpan(
            points=((index * 1000.0, 0.0), (index * 1000.0 + 200.0, 0.0)),
            road_width=7.0,
        )
        for index in range(40)
    )
    index = spatial._BridgeSpanIndex.create(spans)

    candidates = index.candidates(20_100.0, 0.0)

    assert candidates == (spans[20],)


def test_terminal_underlay_point_index_matches_existing_exact_check() -> None:
    matching = WorldObject(
        900,
        r"o\road\sil25.p3d",
        112.5,
        0.0,
        100.0,
        90.0,
    )
    far = tuple(
        WorldObject(
            index + 1,
            r"o\road\sil25.p3d",
            2000.0 + index * 10.0,
            0.0,
            2000.0,
            90.0,
        )
        for index in range(300)
    )
    objects = [*far, matching]
    start = (100.0, 100.0)
    end = (125.0, 100.0)
    index = spatial._ObjectPointIndex.create(objects)

    expected = cleanup._has_matching_underlay(objects, start, end, matching.model_path)
    actual = spatial._has_matching_underlay(index, start, end, matching.model_path)

    assert actual is expected is True


def test_spatial_terminal_generation_matches_existing_geometry() -> None:
    span = cleanup._BridgeSpan(
        points=((0.0, 100.0), (250.0, 100.0)),
        road_width=7.0,
        road_model_path=r"o\road\sil25.p3d",
    )
    spec = SimpleNamespace(cells=16, cell_size=50.0)
    elevations = (6.30,) * (spec.cells * spec.cells)

    filled, added = spatial._add_terminal_underlays(
        _report(),
        (span,),
        elevations,
        spec,
    )

    assert added == 4
    assert len(filled.objects) == 4
    assert [obj.x for obj in filled.objects] == pytest.approx(
        [12.5, 37.5, 212.5, 237.5]
    )
    assert all(obj.z == pytest.approx(100.0) for obj in filled.objects)
    assert all(obj.y == pytest.approx(6.335) for obj in filled.objects)
    assert all(obj.heading_degrees == pytest.approx(90.0) for obj in filled.objects)
