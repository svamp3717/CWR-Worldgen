from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import residential_infill_performance_policy as policy


class _Projection:
    def __init__(self) -> None:
        self.calls = 0

    def to_world(self, point):
        self.calls += 1
        return point


def _polygon(*points):
    closed = (*points, points[0])
    return SimpleNamespace(outer=closed, holes=())


def _dataset(*, points=(), polygons=()):
    return SimpleNamespace(
        building_points=tuple(SimpleNamespace(point=point) for point in points),
        building_polygons=(SimpleNamespace(polygons=tuple(polygons)),),
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


def test_cached_index_projects_large_mapped_building_set_only_once() -> None:
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

    assert calls_after_first == 5000
    assert projection.calls == calls_after_first
