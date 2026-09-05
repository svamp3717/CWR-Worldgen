from __future__ import annotations

from pathlib import Path

from cwr_worldgen import osm as osm_module
from cwr_worldgen import osm_house_modeler_runtime as runtime
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmPointFeature,
    OsmPolygonFeature,
    OsmRaster,
    plan_building_placements,
)
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary


def _empty_dataset(**overrides):
    values = dict(
        source_generator="building-resolve-performance",
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


def _rectangle_feature(projection, index: int, x: float, z: float) -> OsmPolygonFeature:
    width = 8.0 + (index % 4) * 2.0
    length = 10.0 + (index % 5) * 2.0
    half_w = width * 0.5
    half_l = length * 0.5
    world = (
        (x - half_w, z - half_l),
        (x + half_w, z - half_l),
        (x + half_w, z + half_l),
        (x - half_w, z + half_l),
        (x - half_w, z - half_l),
    )
    outer = tuple(projection.to_latlon(point) for point in world)
    return OsmPolygonFeature(
        f"way/resolve-{index}",
        {"building": "house", "addr:country": "SE"},
        (GeoPolygon(outer),),
    )


def test_prepared_modeler_style_keys_are_reused_during_rectangular_placement(monkeypatch) -> None:
    bbox = (59.0, 18.0, 59.03, 18.05)
    projection = BboxProjection.create(bbox, 2400.0)
    buildings = tuple(
        _rectangle_feature(
            projection,
            index,
            120.0 + (index % 6) * 260.0,
            120.0 + (index // 6) * 330.0,
        )
        for index in range(24)
    )
    dataset = _empty_dataset(element_count=len(buildings), building_polygons=buildings)

    calls = 0
    original = runtime.resolve_style

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "resolve_style", counted)
    library = ProceduralBuildingLibrary(
        world_name="resolve_perf",
        maximum_variants=128,
        texture_variants=1,
    )
    library.prepare(dataset, projection, 8.0)
    prepare_calls = calls
    assert prepare_calls >= len(buildings)

    projected_requests = []
    for feature in buildings:
        polygon = feature.polygons[0]
        points = tuple(projection.to_world(point) for point in polygon.outer[:-1])
        projected_requests.append((feature, points))
        library.plan_polygon(feature.tags, points)

    # These exact rectangular footprints were fully resolved during prepare(), so
    # placement must not repeat country/material/window/door style resolution.
    assert calls == prepare_calls

    # The cache is an optimization only. Removing one prepared entry must produce
    # the exact same requested/selected model while proving the fallback still
    # performs a real style resolution when no prepared request is available.
    feature, points = projected_requests[0]
    cached = library.plan_polygon(feature.tags, points)
    library._modeler_prepared_polygon_keys.clear()
    before_fallback = calls
    fallback = library.plan_polygon(feature.tags, points)
    assert calls == before_fallback + 1
    assert fallback.requested == cached.requested
    assert fallback.selected == cached.selected
    assert fallback.heading_degrees == cached.heading_degrees


def test_building_priority_is_classified_once_per_feature(monkeypatch) -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    points = tuple(
        OsmPointFeature(
            f"node/priority-{index}",
            {"building": ("house", "shed", "school", "apartments", "barn")[index % 5]},
            projection.to_latlon((20.0 + (index % 5) * 32.0, 20.0 + (index // 5) * 32.0)),
        )
        for index in range(25)
    )
    dataset = _empty_dataset(element_count=len(points), building_points=points)
    spec = _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=cells,
        cell_size=25.0,
        max_buildings=100,
        max_forest_objects=0,
        max_road_objects=0,
        residential_infill_enabled=False,
        strict_assets=False,
    )

    calls = 0
    original = osm_module._building_placement_priority

    def counted(tags):
        nonlocal calls
        calls += 1
        return original(tags)

    monkeypatch.setattr(osm_module, "_building_placement_priority", counted)
    plans, truncated = plan_building_placements(
        dataset, projection, _empty_raster(cells), spec
    )
    assert not truncated
    assert len(plans) == len(points)
    assert calls == len(points)
