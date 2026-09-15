# SPDX-License-Identifier: GPL-3.0-or-later
"""Accelerate residential-infill source lookups with reusable spatial indexes.

Large worlds can contain tens of thousands of mapped buildings and thousands of
place/residential features.  The stock infill path historically rescanned those
collections for every residential polygon and every place label.  This policy
projects/indexes both mapped buildings and residential landuse polygons once per
dataset/projection pair while preserving the original exact point/centroid/
vertex and polygon-with-holes predicates after spatial candidate pruning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from . import osm as _osm

_INSTALLED = False
_ORIGINAL_HAS_MAPPED_BUILDING: Any = None
_ORIGINAL_MAPPED_BUILDING_NEAR_POINT: Any = None
_ORIGINAL_PLACE_INSIDE_RESIDENTIAL_AREA: Any = None
_DEFAULT_BUCKET_METRES = 256.0
_RESIDENTIAL_BUCKET_METRES = 512.0
_MAX_CACHED_INDEXES = 4
_CONTEXT_CACHE: dict[tuple[int, int], tuple[Any, Any, "_InfillSpatialContext"]] = {}

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


def _is_overture(feature: Any) -> bool:
    tags = getattr(feature, "tags", {}) or {}
    return str(tags.get("source", "")).casefold() == "overturemaps"


@dataclass(frozen=True, slots=True)
class _PolygonOccupancyRecord:
    samples: tuple[PointXZ, ...]
    bounds: tuple[float, float, float, float]
    overture: bool


@dataclass(slots=True)
class _MappedBuildingOccupancyIndex:
    points: tuple[PointXZ, ...]
    point_overture: tuple[bool, ...]
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
        point_overture: list[bool] = []
        point_buckets: dict[tuple[int, int], list[int]] = {}
        for feature in getattr(dataset, "building_points", ()):
            point = projection.to_world(feature.point)
            world_point = (float(point[0]), float(point[1]))
            index = len(points)
            points.append(world_point)
            point_overture.append(_is_overture(feature))
            key = (
                math.floor(world_point[0] / bucket_size),
                math.floor(world_point[1] / bucket_size),
            )
            point_buckets.setdefault(key, []).append(index)

        polygons: list[_PolygonOccupancyRecord] = []
        polygon_buckets: dict[tuple[int, int], list[int]] = {}
        for feature in getattr(dataset, "building_polygons", ()):
            overture = _is_overture(feature)
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
                    overture=overture,
                )
                index = len(polygons)
                polygons.append(record)
                minimum_x, minimum_z, maximum_x, maximum_z = record_bounds
                for bz in _bucket_range(minimum_z, maximum_z, bucket_size):
                    for bx in _bucket_range(minimum_x, maximum_x, bucket_size):
                        polygon_buckets.setdefault((bx, bz), []).append(index)

        return cls(
            points=tuple(points),
            point_overture=tuple(point_overture),
            polygons=tuple(polygons),
            bucket_size=bucket_size,
            point_buckets={key: tuple(values) for key, values in point_buckets.items()},
            polygon_buckets={key: tuple(values) for key, values in polygon_buckets.items()},
        )

    def _candidate_indices(
        self,
        minimum_x: float,
        minimum_z: float,
        maximum_x: float,
        maximum_z: float,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        point_indices: set[int] = set()
        polygon_indices: set[int] = set()
        for bz in _bucket_range(minimum_z, maximum_z, self.bucket_size):
            for bx in _bucket_range(minimum_x, maximum_x, self.bucket_size):
                key = (bx, bz)
                point_indices.update(self.point_buckets.get(key, ()))
                polygon_indices.update(self.polygon_buckets.get(key, ()))
        return tuple(sorted(point_indices)), tuple(sorted(polygon_indices))

    def contains_mapped_building(
        self,
        outer: Sequence[PointXZ],
        holes: Sequence[Sequence[PointXZ]],
    ) -> bool:
        area_bounds = _bounds(outer)
        if area_bounds is None:
            return False
        point_indices, polygon_indices = self._candidate_indices(*area_bounds)

        # Preserve the historical test order: mapped points first, then polygon
        # centroids/vertices. The bucket layer is only a conservative prefilter.
        for index in point_indices:
            if _osm._polygon_contains_with_holes(self.points[index], outer, holes):
                return True
        for index in polygon_indices:
            record = self.polygons[index]
            if any(
                _osm._polygon_contains_with_holes(sample, outer, holes)
                for sample in record.samples
            ):
                return True
        return False

    def near_world_point(
        self,
        x: float,
        z: float,
        radius: float,
        *,
        include_overture: bool,
    ) -> bool:
        radius = max(0.0, float(radius))
        radius_squared = radius * radius
        point_indices, polygon_indices = self._candidate_indices(
            float(x) - radius,
            float(z) - radius,
            float(x) + radius,
            float(z) + radius,
        )
        for index in point_indices:
            if not include_overture and self.point_overture[index]:
                continue
            px, pz = self.points[index]
            if (px - x) * (px - x) + (pz - z) * (pz - z) <= radius_squared:
                return True
        for index in polygon_indices:
            record = self.polygons[index]
            if not include_overture and record.overture:
                continue
            for px, pz in record.samples:
                if (px - x) * (px - x) + (pz - z) * (pz - z) <= radius_squared:
                    return True
        return False


@dataclass(frozen=True, slots=True)
class _ResidentialAreaRecord:
    outer: tuple[PointXZ, ...]
    holes: tuple[tuple[PointXZ, ...], ...]


@dataclass(slots=True)
class _ResidentialAreaIndex:
    polygons: tuple[_ResidentialAreaRecord, ...]
    bucket_size: float
    buckets: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def create(
        cls,
        dataset: Any,
        projection: Any,
        *,
        bucket_size: float = _RESIDENTIAL_BUCKET_METRES,
    ) -> "_ResidentialAreaIndex":
        bucket_size = max(64.0, float(bucket_size))
        polygons: list[_ResidentialAreaRecord] = []
        mutable: dict[tuple[int, int], list[int]] = {}
        for feature in getattr(dataset, "urban", ()):
            tags = getattr(feature, "tags", {}) or {}
            if str(tags.get("landuse", "")).casefold() != "residential":
                continue
            for polygon in getattr(feature, "polygons", ()):
                outer = tuple(
                    (float(point[0]), float(point[1]))
                    for point in (
                        projection.to_world(source_point)
                        for source_point in polygon.outer[:-1]
                    )
                )
                if len(outer) < 3:
                    continue
                holes = tuple(
                    tuple(
                        (float(point[0]), float(point[1]))
                        for point in (
                            projection.to_world(source_point)
                            for source_point in hole[:-1]
                        )
                    )
                    for hole in getattr(polygon, "holes", ())
                )
                record_bounds = _bounds(outer)
                if record_bounds is None:
                    continue
                index = len(polygons)
                polygons.append(_ResidentialAreaRecord(outer=outer, holes=holes))
                minimum_x, minimum_z, maximum_x, maximum_z = record_bounds
                for bz in _bucket_range(minimum_z, maximum_z, bucket_size):
                    for bx in _bucket_range(minimum_x, maximum_x, bucket_size):
                        mutable.setdefault((bx, bz), []).append(index)
        return cls(
            polygons=tuple(polygons),
            bucket_size=bucket_size,
            buckets={key: tuple(values) for key, values in mutable.items()},
        )

    def contains_point(self, point: PointXZ) -> bool:
        x, z = float(point[0]), float(point[1])
        key = (
            math.floor(x / self.bucket_size),
            math.floor(z / self.bucket_size),
        )
        for index in self.buckets.get(key, ()):
            record = self.polygons[index]
            if _osm._polygon_contains_with_holes(point, record.outer, record.holes):
                return True
        return False


@dataclass(frozen=True, slots=True)
class _InfillSpatialContext:
    buildings: _MappedBuildingOccupancyIndex
    residential: _ResidentialAreaIndex


def _context_for(dataset: Any, projection: Any) -> _InfillSpatialContext:
    key = (id(dataset), id(projection))
    cached = _CONTEXT_CACHE.get(key)
    if cached is not None and cached[0] is dataset and cached[1] is projection:
        return cached[2]

    context = _InfillSpatialContext(
        buildings=_MappedBuildingOccupancyIndex.create(dataset, projection),
        residential=_ResidentialAreaIndex.create(dataset, projection),
    )
    _CONTEXT_CACHE[key] = (dataset, projection, context)
    while len(_CONTEXT_CACHE) > _MAX_CACHED_INDEXES:
        _CONTEXT_CACHE.pop(next(iter(_CONTEXT_CACHE)))
    return context


def _index_for(dataset: Any, projection: Any) -> _MappedBuildingOccupancyIndex:
    return _context_for(dataset, projection).buildings


def _indexed_residential_area_has_mapped_building(
    dataset: Any,
    projection: Any,
    outer: Sequence[PointXZ],
    holes: Sequence[Sequence[PointXZ]],
) -> bool:
    return _context_for(dataset, projection).buildings.contains_mapped_building(
        outer, holes
    )


def _indexed_mapped_building_near_world_point(
    dataset: Any,
    projection: Any,
    x: float,
    z: float,
    radius: float,
    *,
    include_overture: bool = False,
) -> bool:
    return _context_for(dataset, projection).buildings.near_world_point(
        x, z, radius, include_overture=include_overture
    )


def _indexed_place_inside_residential_area(
    place: Any,
    dataset: Any,
    projection: Any,
) -> bool:
    point = projection.to_world(place.point)
    return _context_for(dataset, projection).residential.contains_point(
        (float(point[0]), float(point[1]))
    )


def _clear_index_cache() -> None:
    _CONTEXT_CACHE.clear()


def install_residential_infill_performance_policy() -> None:
    global _INSTALLED
    global _ORIGINAL_HAS_MAPPED_BUILDING
    global _ORIGINAL_MAPPED_BUILDING_NEAR_POINT
    global _ORIGINAL_PLACE_INSIDE_RESIDENTIAL_AREA
    if _INSTALLED:
        return
    _ORIGINAL_HAS_MAPPED_BUILDING = _osm._residential_area_has_mapped_building
    _ORIGINAL_MAPPED_BUILDING_NEAR_POINT = _osm._mapped_building_near_world_point
    _ORIGINAL_PLACE_INSIDE_RESIDENTIAL_AREA = _osm._place_inside_residential_area
    _osm._residential_area_has_mapped_building = _indexed_residential_area_has_mapped_building
    _osm._mapped_building_near_world_point = _indexed_mapped_building_near_world_point
    _osm._place_inside_residential_area = _indexed_place_inside_residential_area
    _INSTALLED = True
