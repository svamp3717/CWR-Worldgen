from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import residential_infill_performance_policy as policy


class _Projection:
    def __init__(self) -> None:
        self.calls = 0

    def to_world(self, point):
        self.calls += 1
        return point


def _polygon(*points, holes=()):
    closed = (*points, points[0])
    closed_holes = tuple((*hole, hole[0]) for hole in holes)
    return SimpleNamespace(outer=closed, holes=closed_holes)


def _dataset(*, points=(), polygons=(), urban=()):
    return SimpleNamespace(
        building_points=tuple(SimpleNamespace(point=point, tags={}) for point in points),
        building_polygons=(SimpleNamespace(polygons=tuple(polygons), tags={}),),
        urban=tuple(urban),
    )


def test_index_preserves_point_polygon_and_hole_occupancy_rules() -> None:
    projection = _Projection()
    dataset = _dataset(
        points=((10.0, 10.0), (50.0, 50.0), (5000.0, 5000.0)),
        polygons=(
            _polygon((150.0, 150.0), (170.0, 150.0), (170.0, 170.0), (150.0, 170.0)),
            _polygon((7000.0, 7000.0), (7020.0, 7000.0), (7020.0, 7020.0), (7000.0, 7020.0)),
        ),
    )
    index = policy._MappedBuildingOccupancyIndex.create(dataset, projection)
    outer = ((0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0))
    hole = ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0))

    assert index.contains_mapped_building(outer, (hole,)) is True

    only_hole_dataset = _dataset(points=((50.0, 50.0),))
    only_hole = policy._MappedBuildingOccupancyIndex.create(only_hole_dataset, _Projection())
    assert only_hole.contains_mapped_building(outer, (hole,)) is False


def test_index_accepts_polygon_when_historical_centroid_or_vertex_test_would() -> None:
    projection = _Projection()
    crossing = _polygon(
        (90.0, 90.0),
        (130.0, 90.0),
        (130.0, 130.0),
        (90.0, 130.0),
    )
    dataset = _dataset(polygons=(crossing,))
    index = policy._MappedBuildingOccupancyIndex.create(dataset, projection)
    outer = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))

    assert index.contains_mapped_building(outer, ()) is True


def test_near_point_lookup_preserves_overture_filtering() -> None:
    policy._clear_index_cache()
    projection = _Projection()
    dataset = SimpleNamespace(
        building_points=(
            SimpleNamespace(point=(100.0, 100.0), tags={}),
            SimpleNamespace(point=(500.0, 500.0), tags={"source": "overturemaps"}),
        ),
        building_polygons=(),
        urban=(),
    )

    assert policy._indexed_mapped_building_near_world_point(
        dataset, projection, 105.0, 100.0, 10.0
    ) is True
    assert policy._indexed_mapped_building_near_world_point(
        dataset, projection, 500.0, 500.0, 10.0
    ) is False
    assert policy._indexed_mapped_building_near_world_point(
        dataset, projection, 500.0, 500.0, 10.0, include_overture=True
    ) is True


def test_residential_area_index_preserves_holes_and_landuse_filter() -> None:
    policy._clear_index_cache()
    projection = _Projection()
    residential = SimpleNamespace(
        tags={"landuse": "residential"},
        polygons=(
            _polygon(
                (0.0, 0.0),
                (200.0, 0.0),
                (200.0, 200.0),
                (0.0, 200.0),
                holes=(((80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)),),
            ),
        ),
    )
    industrial = SimpleNamespace(
        tags={"landuse": "industrial"},
        polygons=(_polygon((300.0, 0.0), (500.0, 0.0), (500.0, 200.0), (300.0, 200.0)),),
    )
    dataset = _dataset(urban=(residential, industrial))

    inside = SimpleNamespace(point=(50.0, 50.0))
    hole = SimpleNamespace(point=(100.0, 100.0))
    industrial_only = SimpleNamespace(point=(400.0, 100.0))

    assert policy._indexed_place_inside_residential_area(inside, dataset, projection) is True
    assert policy._indexed_place_inside_residential_area(hole, dataset, projection) is False
    assert policy._indexed_place_inside_residential_area(industrial_only, dataset, projection) is False


def test_cached_context_projects_large_source_sets_only_once() -> None:
    policy._clear_index_cache()
    projection = _Projection()
    dataset = _dataset(
        points=tuple((float(index * 20), 5000.0) for index in range(1000)),
        polygons=tuple(
            _polygon(
                (float(index * 20), 10000.0),
                (float(index * 20 + 8), 10000.0),
                (float(index * 20 + 8), 10008.0),
                (float(index * 20), 10008.0),
            )
            for index in range(1000)
        ),
    )
    first = ((0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0))
    second = ((300.0, 0.0), (500.0, 0.0), (500.0, 200.0), (300.0, 200.0))

    policy._indexed_residential_area_has_mapped_building(dataset, projection, first, ())
    calls_after_first = projection.calls
    policy._indexed_residential_area_has_mapped_building(dataset, projection, second, ())
    policy._indexed_mapped_building_near_world_point(
        dataset, projection, 100.0, 5000.0, 30.0
    )

    assert calls_after_first == 5000
    assert projection.calls == calls_after_first


def test_plan_wrapper_reports_infill_subprogress(monkeypatch) -> None:
    policy._clear_index_cache()
    projection = _Projection()
    residential_polygon = _polygon(
        (0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)
    )
    residential = SimpleNamespace(
        tags={"landuse": "residential"}, polygons=(residential_polygon,)
    )
    place = SimpleNamespace(point=(50.0, 50.0))
    dataset = SimpleNamespace(
        building_points=(),
        building_polygons=(),
        urban=(residential,),
        places=(place,),
    )
    progress_messages: list[str] = []

    def fake_plan(
        dataset,
        projection,
        raster,
        spec,
        building_asset_library=None,
        progress_callback=None,
    ):
        assert progress_callback is not None
        progress_callback(90, "Planning residential infill")
        policy._indexed_place_inside_residential_area(place, dataset, projection)
        outer = tuple(residential_polygon.outer[:-1])
        policy._indexed_residential_area_has_mapped_building(
            dataset, projection, outer, ()
        )
        return (), False

    monkeypatch.setattr(policy, "_ORIGINAL_PLAN_BUILDINGS", fake_plan)
    plans, truncated = policy._plan_buildings_with_infill_progress(
        dataset,
        projection,
        None,
        None,
        progress_callback=lambda _percent, stage: progress_messages.append(stage),
    )

    assert plans == ()
    assert truncated is False
    assert "Planning residential infill settlement sources 1/1" in progress_messages
    assert "Planning residential infill area occupancy 1 checked" in progress_messages
