from __future__ import annotations

from cwr_worldgen import progress
from cwr_worldgen.building_prepare_progress_policy import (
    _dataset_building_candidate_total,
)
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmPointFeature
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary


def _empty_dataset(**overrides):
    values = dict(
        source_generator="building-prepare-progress",
        element_count=0,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
    )
    values.update(overrides)
    return OsmDataset(**values)


def test_prepare_reports_style_resolution_and_reuse_counters(monkeypatch) -> None:
    bbox = (59.0, 18.0, 59.02, 18.03)
    projection = BboxProjection.create(bbox, 1600.0)
    points = tuple(
        OsmPointFeature(
            f"node/prepare-progress-{index}",
            {
                "building": "house",
                "building:levels": str(index + 1),
                "addr:country": "SE",
            },
            projection.to_latlon((100.0 + index * 120.0, 180.0)),
        )
        for index in range(8)
    )
    dataset = _empty_dataset(
        element_count=len(points),
        building_points=points,
    )
    assert _dataset_building_candidate_total(dataset) == 8

    messages: list[tuple[int, str]] = []
    monkeypatch.setattr(
        progress,
        "report_progress",
        lambda percent, stage: messages.append((percent, stage)),
    )

    library = ProceduralBuildingLibrary(
        world_name="prepare_progress",
        maximum_variants=2,
        texture_variants=1,
    )
    library.prepare(dataset, projection, 8.0)

    assert messages
    assert all(percent == 23 for percent, _stage in messages)
    assert any(
        "resolving building styles" in stage
        and "(0/8, 0%;" in stage
        for _percent, stage in messages
    )
    assert any(
        "resolving building styles" in stage
        and "(8/8, 100%;" in stage
        for _percent, stage in messages
    )

    unique_total = len(library._request_counts)
    match_total = max(0, unique_total - 2)
    assert unique_total > 2
    assert any(
        "matching reusable variants" in stage
        and f"({match_total}/{match_total}, 100%;" in stage
        for _percent, stage in messages
    )
    selected_total = len(set(library._mapping.values()))
    assert selected_total == 2
    assert messages[-1] == (
        23,
        "Preparing procedural building variants complete "
        f"(8/8 buildings; {unique_total} unique; 2 selected)",
    )
