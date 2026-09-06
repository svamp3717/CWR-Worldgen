# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep the mature legacy church renderer while using real OSM road footprints.

Christian churches already have a dedicated rectangular CWR renderer with an
integrated end-mounted tower/spire and roof cut-out.  The final road-clearance
work exposed one unrelated problem: for an irregular mapped church the fitted
rectangle can be substantially larger than the source footprint, so road safety
may reject the church because the *proxy rectangle* touches a nearby street.

Do not replace the church renderer to solve that.  Keep ProceduralBuildingLibrary
completely untouched and substitute only the support polygon used by terrain,
building-overlap and final road-clearance policies.  The support follows the real
mapped OSM outer ring and is translated with any placement nudge, while the model
path, dimensions, heading, worship style, tower and spire remain the established
legacy implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from . import final_building_road_clearance_policy as _clearance
from . import generator as _generator
from . import osm as _osm

PointXZ = tuple[float, float]

_CACHE_REVISION = "final-road-building-clearance-v6-legacy-church-source-road-footprints"
_INSTALLED = False
_ORIGINAL_OSM_PLAN_BUILDINGS = None
_ORIGINAL_GENERATOR_PLAN_BUILDINGS = None


def _open_ring(points) -> tuple:
    values = tuple(points or ())
    if len(values) >= 2 and values[0] == values[-1]:
        return values[:-1]
    return values


def _polygon_centroid(points: Sequence[PointXZ]) -> PointXZ:
    """Return an area centroid with a stable average fallback."""
    if not points:
        return 0.0, 0.0
    try:
        _area, cx, cz = _osm._polygon_area_centroid(points)
        if math.isfinite(float(cx)) and math.isfinite(float(cz)):
            return float(cx), float(cz)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    return (
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    )


def _mapped_church_support_polygon(plan, dataset, projection) -> tuple[PointXZ, ...] | None:
    """Return the source church outline translated to the plan's current origin."""
    osm_key = str(getattr(plan, "osm_key", "") or "")
    if not osm_key or getattr(plan, "synthetic_infill", False):
        return None
    if str(getattr(plan, "building_family", "") or "").casefold() != "church":
        return None

    candidates: list[tuple[float, tuple[PointXZ, ...], PointXZ]] = []
    for feature in getattr(dataset, "building_polygons", ()):
        if str(getattr(feature, "osm_key", "") or "") != osm_key:
            continue
        if not _osm.is_actual_church(getattr(feature, "tags", {}) or {}):
            continue
        for polygon in getattr(feature, "polygons", ()):
            ring = _open_ring(getattr(polygon, "outer", ()))
            if len(ring) < 3:
                continue
            projected = tuple(projection.to_world(point) for point in ring)
            if len(projected) < 3:
                continue
            centre = _polygon_centroid(projected)
            distance_sq = (
                (float(getattr(plan, "x", 0.0)) - centre[0]) ** 2
                + (float(getattr(plan, "z", 0.0)) - centre[1]) ** 2
            )
            candidates.append((distance_sq, projected, centre))

    if not candidates:
        return None
    _distance, polygon, centre = min(candidates, key=lambda item: item[0])
    # Large-building preprocessing may already have translated the legacy model.
    # Apply the same rigid translation to the authoritative source footprint so
    # road checks and the rendered model still describe the same placement.
    dx = float(plan.x) - centre[0]
    dz = float(plan.z) - centre[1]
    return tuple((x + dx, z + dz) for x, z in polygon)


def _apply_legacy_church_supports(plans, dataset, projection):
    updated = []
    for plan in plans or ():
        source_polygon = _mapped_church_support_polygon(plan, dataset, projection)
        if source_polygon is None:
            updated.append(plan)
        else:
            updated.append(replace(plan, support_polygon=source_polygon))
    return tuple(updated)


def _plan_buildings_with_source_church_supports(*args, **kwargs):
    plans, truncated = _ORIGINAL_OSM_PLAN_BUILDINGS(*args, **kwargs)
    dataset = args[0] if args else kwargs.get("dataset")
    projection = args[1] if len(args) > 1 else kwargs.get("projection")
    if dataset is None or projection is None:
        return plans, truncated
    return _apply_legacy_church_supports(plans, dataset, projection), truncated


def install_church_native_polygon_policy() -> None:
    """Compatibility installer: retain legacy churches, replace collision footprint only."""
    global _INSTALLED, _ORIGINAL_OSM_PLAN_BUILDINGS, _ORIGINAL_GENERATOR_PLAN_BUILDINGS
    if _INSTALLED:
        return

    _ORIGINAL_OSM_PLAN_BUILDINGS = _osm.plan_building_placements
    _ORIGINAL_GENERATOR_PLAN_BUILDINGS = _generator.plan_building_placements

    # generator imported the function directly, so patch both references. They
    # normally point at the same callable; the generator wrapper deliberately
    # delegates to the OSM original to avoid double-applying this policy.
    _osm.plan_building_placements = _plan_buildings_with_source_church_supports
    _generator.plan_building_placements = _plan_buildings_with_source_church_supports

    # Old cached placement tuples may contain the oversized rectangle. Force one
    # fresh non-road planning pass, but leave the mature church P3D cache alone.
    _clearance._CACHE_REVISION = _CACHE_REVISION
    _INSTALLED = True
