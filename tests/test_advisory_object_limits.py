from pathlib import Path

from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmPointFeature,
    OsmRaster,
    _advisory_object_limit,
    _object_threshold_warning,
    plan_building_placements,
)
from cwr_worldgen.semantic_features import generate_semantic_objects


def _empty_dataset(**overrides):
    values = dict(
        source_generator="advisory-limits",
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


def _empty_raster(cells: int) -> OsmRaster:
    empty = (False,) * (cells * cells)
    return OsmRaster(cells, empty, empty, empty, empty, empty, empty, cells, 0)


def test_advisory_limit_keeps_zero_as_disable_and_positive_as_unbounded() -> None:
    assert _advisory_object_limit(0, enabled=True) == 0
    assert _advisory_object_limit(3, enabled=False) == 3
    assert _advisory_object_limit(3, enabled=True) > 1_000_000_000
    assert _object_threshold_warning("test object", 4, 3) is not None
    assert _object_threshold_warning("test object", 3, 3) is None


def test_advisory_building_threshold_warns_but_keeps_all_mapped_buildings() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    buildings = tuple(
        OsmPointFeature(
            f"node/building-{index}",
            {"building": "house"},
            projection.to_latlon((50.0 + index * 75.0, 100.0)),
        )
        for index in range(2)
    )
    dataset = _empty_dataset(element_count=2, building_points=buildings)
    spec = _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=cells,
        cell_size=25.0,
        max_buildings=1,
        max_forest_objects=0,
        max_road_objects=0,
        advisory_object_limits=True,
        residential_infill_enabled=False,
        strict_assets=False,
    )
    progress: list[str] = []
    plans, truncated = plan_building_placements(
        dataset,
        projection,
        _empty_raster(cells),
        spec,
        progress_callback=lambda _percent, message: progress.append(message),
    )
    assert len(plans) == 2
    assert not truncated
    assert any("building footprint warning threshold exceeded" in message for message in progress)


def test_advisory_landmark_threshold_does_not_truncate_bus_stops() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    landmarks = tuple(
        OsmPointFeature(
            f"node/bus-{index}",
            {"landmark": "bus_stop"},
            projection.to_latlon((50.0 + index * 75.0, 100.0)),
        )
        for index in range(2)
    )
    dataset = _empty_dataset(element_count=2, landmarks=landmarks)
    spec = _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=cells,
        cell_size=25.0,
        max_buildings=0,
        max_forest_objects=0,
        max_road_objects=0,
        advisory_object_limits=True,
        maximum_landmark_objects=1,
        cemeteries_enabled=False,
        strict_assets=False,
    )
    result = generate_semantic_objects(
        dataset,
        projection,
        [0.0] * (cells * cells),
        spec,
        None,  # no site features are present, so no generated site library is needed
        starting_object_id=1,
        raster=_empty_raster(cells),
    )
    assert result.bus_stop_objects == 2


def test_all_user_facing_object_count_caps_route_through_advisory_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    osm_source = (root / "src" / "cwr_worldgen" / "osm.py").read_text(encoding="utf-8")
    semantic_source = (root / "src" / "cwr_worldgen" / "semantic_features.py").read_text(encoding="utf-8")
    cli_source = (root / "src" / "cwr_worldgen" / "cli.py").read_text(encoding="utf-8")

    effective_limits = (
        "building_limit = _advisory_object_limit",
        "infill_limit = _advisory_object_limit",
        "road_limit = _advisory_object_limit",
        "maximum_sidewalk_objects = _advisory_object_limit",
        "maximum_street_furniture_objects = _advisory_object_limit",
        "forest_limit = _advisory_object_limit",
        "extra_single_limit = _advisory_object_limit",
        "rocky_limit = _advisory_object_limit",
        "undergrowth_base_limit = _advisory_object_limit",
        "bush_limit = _advisory_object_limit",
        "border_limit = _advisory_object_limit",
        "ditch_limit = _advisory_object_limit",
        "barrier_limit = _advisory_object_limit",
        "bridge_limit = _advisory_object_limit",
        "meadow_limit = _advisory_object_limit",
        "haybale_limit = _advisory_object_limit",
        "wetland_limit = _advisory_object_limit",
        "mapped_tree_limit = _advisory_object_limit",
        "utility_limit = _advisory_object_limit",
    )
    for snippet in effective_limits:
        assert snippet in osm_source
    assert "max_landmarks = _advisory_object_limit" in semantic_source
    assert "grave_limit = _advisory_object_limit" in semantic_source
    assert '"advisory_object_limits": True' in cli_source
    assert cli_source.count("advisory_object_limits=True") >= 5
