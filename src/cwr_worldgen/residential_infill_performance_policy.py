# SPDX-License-Identifier: GPL-3.0-or-later
"""Accelerate residential-infill occupancy checks with a mapped-building index.

The stock infill path historically rescanned every mapped building for every
residential polygon. Large worlds turn that into residential areas multiplied
by tens of thousands of buildings. This policy projects/indexes mapped
buildings once per dataset/projection pair and keeps the original exact
point/centroid/vertex containment predicate for candidate buildings.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from . import osm as _osm

_INSTALLED = False
_ORIGINAL_HAS_MAPPED_BUILDING: Any = None
_DEFAULT_BUCKET_METRES = 256.0
_MAX_CACHED_INDEXES = 4
_INDEX_CACHE: dict[tuple[int, int], tuple[Any, Any, "_MappedBuildingOccupancyIndex"]] = {}

PointXZ = tuple[float, float]


def _bucket_range(minimum: float, maximum: float, bucket_size: float) -> range:
    return range(
        math.floor(float(minimum) / bucket_size),
        math.floor(float(maximum) / bucket_size) + 1,
    )


def _bounds(points: Sequence[PointXZ]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = tuple(float(point[0]) for point in points)
    zs = tuple(float(point[1]) for point in points)
    return min(xs), min(zs), max(xs), max(zs)


@dataclass(frozen=True, slots=True)
class _PolygonOccupancyRecord:
    samples: tuple[PointXZ, ...]
    bounds: tuple[float, float, float, float]


@dataclass(slots=True)
class _MappedBuildingOccupancyIndex:
    points: tuple[PointXZ, ...]
    polygons: tuple[_PolygonOccupancyRecord, ...]
    bucket_size: float
    point_buckets: dict[tuple[int, int], tuple[int, ...]]
    polygon_buckets: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def create(
        cls,
        dataset: Any,
        projection: Any,
        *,
        bucket_size: float = _DEFAULT_BUCKET_METRES,
    ) -> "_MappedBuildingOccupancyIndex":
        bucket_size = max(32.0, float(bucket_size))
        points: list[PointXZ] = []
        point_buckets: dict[tuple[int, int], list[int]] = {}
        for feature in getattr(dataset, "building_points", ()):
            point = projection.to_world(feature.point)
            world_point = (float(point[0]), float(point[1]))
            index = len(points)
            points.append(world_point)
            key = (
                math.floor(world_point[0] / bucket_size),
                math.floor(world_point[1] / bucket_size),
            )
            point_buckets.setdefault(key, []).append(index)

        polygons: list[_PolygonOccupancyRecord] = []
        polygon_buckets: dict[tuple[int, int], list[int]] = {}
        for feature in getattr(dataset, "building_polygons", ()):
            for polygon in getattr(feature, "polygons", ()):
                projected = tuple(
                    (float(point[0]), float(point[1]))
                    for point in (
                        projection.to_world(source_point)
                        for source_point in polygon.outer[:-1]
                    )
                )
                if len(projected) < 3:
                    continue
                _area, cx, cz = _osm._polygon_area_centroid(projected)
                record_bounds = _bounds(projected)
                if record_bounds is None:
                    continue
                record = _PolygonOccupancyRecord(
                    samples=((float(cx), float(cz)), *projected),
                    bounds=record_bounds,
                )
                index = len(polygons)
                polygons.append(record)
                minimum_x, minimum_z, maximum_x, maximum_z = record_bounds
                for bz in _bucket_range(minimum_z, maximum_z, bucket_size):
                    for bx in _bucket_range(minimum_x, maximum_x, bucket_size):
                        polygon_buckets.setdefault((bx, bz), []).append(index)

        return cls(
            points=tuple(points),
            polygons=tuple(polygons),
            bucket_size=bucket_size,
            point_buckets={key: tuple(values) for key, values in point_buckets.items()},
            polygon_buckets={key: tuple(values) for key, values in polygon_buckets.items()},
        )

    def contains_mapped_building(
        self,
        outer: Sequence[PointXZ],
        holes: Sequence[Sequence[PointXZ]],
    ) -> bool:
        area_bounds = _bounds(outer)
        if area_bounds is None:
            return False
        minimum_x, minimum_z, maximum_x, maximum_z = area_bounds
        point_indices: set[int] = set()
        polygon_indices: set[int] = set()
        for bz in _bucket_range(minimum_z, maximum_z, self.bucket_size):
            for bx in _bucket_range(minimum_x, maximum_x, self.bucket_size):
                key = (bx, bz)
                point_indices.update(self.point_buckets.get(key, ()))
                polygon_indices.update(self.polygon_buckets.get(key, ()))

        # Preserve the historical test order: mapped points first, then polygon
        # centroids/vertices. The bucket layer is only a conservative prefilter.
        for index in sorted(point_indices):
            if _osm._polygon_contains_with_holes(self.points[index], outer, holes):
                return True
        for index in sorted(polygon_indices):
            record = self.polygons[index]
            if any(
                _osm._polygon_contains_with_holes(sample, outer, holes)
                for sample in record.samples
            ):
                return True
        return False


def _index_for(dataset: Any, projection: Any) -> _MappedBuildingOccupancyIndex:
    key = (id(dataset), id(projection))
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] is dataset and cached[1] is projection:
        return cached[2]

    index = _MappedBuildingOccupancyIndex.create(dataset, projection)
    _INDEX_CACHE[key] = (dataset, projection, index)
    while len(_INDEX_CACHE) > _MAX_CACHED_INDEXES:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    return index


def _indexed_residential_area_has_mapped_building(
    dataset: Any,
    projection: Any,
    outer: Sequence[PointXZ],
    holes: Sequence[Sequence[PointXZ]],
) -> bool:
    return _index_for(dataset, projection).contains_mapped_building(outer, holes)


def _clear_index_cache() -> None:
    _INDEX_CACHE.clear()


def install_residential_infill_performance_policy() -> None:
    global _INSTALLED, _ORIGINAL_HAS_MAPPED_BUILDING
    if _INSTALLED:
        return
    _ORIGINAL_HAS_MAPPED_BUILDING = _osm._residential_area_has_mapped_building
    _osm._residential_area_has_mapped_building = _indexed_residential_area_has_mapped_building
    _INSTALLED = True
