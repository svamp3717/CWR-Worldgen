# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove tree/bush objects from runway and sports-pitch clear zones.

The filter is intentionally applied at the final WRP write boundary. Placement
caches may predate this policy, but cached vegetation must never be allowed to
reappear on a runway or mapped playing field. Spatial membership is vectorized
with Shapely so dense worlds do not pay one geometry allocation per tree.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import json
import math
from typing import Iterable, Sequence

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union


_INSTALLED = False
RUNWAY_CLEARANCE_METRES = 2.0
SPORTS_CLEARANCE_METRES = 0.75
# Generated forest/undergrowth clusters can place proxies roughly 12 m from the
# WRP object's origin. Account for that footprint instead of testing only the
# cluster centre against the semantic surface polygon.
CLUSTER_EXTRA_CLEARANCE_METRES = 14.0


def _canonical(value: object) -> str:
    return str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()


def _iter_model_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            if isinstance(item, str) and item:
                yield item


def _configured_vegetation_models(spec) -> set[str]:
    from . import osm

    values: set[str] = set()
    for name in (
        "OSM_INDIVIDUAL_TREE_MODELS",
        "NOGOVA_LEAF_INDIVIDUAL_TREE_MODELS",
        "NOGOVA_PINE_INDIVIDUAL_TREE_MODELS",
        "KOLGUJEV_INDIVIDUAL_TREE_MODELS",
        "KOLGUJEV_BROADLEAF_INDIVIDUAL_TREE_MODELS",
        "KOLGUJEV_CONIFER_INDIVIDUAL_TREE_MODELS",
        "ROADSIDE_TREE_MODELS",
        "STOCK_HEDGE_MODELS",
        "STOCK_STREET_TREE_MODELS",
        "STOCK_SETTLEMENT_FRUIT_TREE_MODELS",
    ):
        for model in getattr(osm, name, ()):
            values.add(_canonical(model))

    # Pick up user/preset overrides without hard-coding every generation-era
    # field name. Only model fields whose names explicitly describe woody
    # vegetation are considered; grass/reed/rock models remain untouched.
    names: list[str] = []
    if is_dataclass(spec):
        names.extend(field.name for field in fields(spec))
    else:
        names.extend(name for name in dir(spec) if not name.startswith("_"))
    for name in names:
        lowered = name.casefold()
        if "model" not in lowered:
            continue
        if not any(token in lowered for token in ("tree", "bush", "undergrowth", "forest_border", "hedge")):
            continue
        try:
            raw = getattr(spec, name)
        except (AttributeError, TypeError):
            continue
        for model in _iter_model_values(raw):
            values.add(_canonical(model))
    return values


def _is_tree_or_bush(model_path: str, spec, configured: set[str]) -> bool:
    from .procedural_forests import is_generated_cluster_model

    canonical = _canonical(model_path)
    if canonical in configured:
        return True
    if is_generated_cluster_model(str(getattr(spec, "name", "")), model_path):
        return True
    if canonical.startswith("o\\tree\\"):
        return True
    filename = canonical.rsplit("\\", 1)[-1]
    if filename.startswith(("str ", "ker ", "krovi")):
        return True
    return filename in {
        "jablon.p3d", "hrusen.p3d", "str_jablon.p3d",
        "dub.p3d", "briza.p3d", "javor.p3d", "lipa.p3d", "vrba.p3d",
        "smrk.p3d", "borovice.p3d", "jedle.p3d",
    }



def _configured_barrier_models(spec) -> dict[str, str]:
    """Return canonical barrier model -> family for final-road clearance."""

    from . import osm

    result: dict[str, str] = {}
    families = (
        ("hedge", tuple(getattr(spec, "stock_hedge_models", osm.STOCK_HEDGE_MODELS))),
        ("wall", tuple(getattr(spec, "stock_wall_models", osm.STOCK_WALL_MODELS))),
        ("fence", tuple(getattr(spec, "stock_metal_fence_models", osm.STOCK_METAL_FENCE_MODELS))),
        ("fence", tuple(getattr(osm, "STOCK_FARMLAND_FENCE_MODELS", ()))),
    )
    for family, models in families:
        for model in models:
            result[_canonical(model)] = family
    return result


def _oriented_box(
    x: float,
    z: float,
    heading_degrees: float,
    half_length: float,
    half_width: float,
) -> tuple[tuple[float, float], ...]:
    angle = math.radians(float(heading_degrees))
    tx, tz = math.sin(angle), math.cos(angle)
    nx, nz = -tz, tx
    return (
        (x - tx * half_length - nx * half_width, z - tz * half_length - nz * half_width),
        (x + tx * half_length - nx * half_width, z + tz * half_length - nz * half_width),
        (x + tx * half_length + nx * half_width, z + tz * half_length + nz * half_width),
        (x - tx * half_length + nx * half_width, z - tz * half_length + nz * half_width),
    )


def _final_road_conflict_indices(
    objects: Sequence[object],
    candidate_indices: Sequence[int],
    cluster_flags: Sequence[bool],
    barrier_families: Sequence[str | None],
    *,
    elevations: Sequence[float] | None,
    road_objects: Sequence[object] | None,
    spec,
) -> set[int]:
    """Find tree/barrier objects intersecting the actual fitted road surfaces.

    Source OSM corridors are insufficient once stock junction replacement and
    generated paved fallbacks move/widen the final carriageway. Work from the
    exact road objects that will be serialized instead, so a cached tree or
    fence cannot reappear on a generated junction.
    """

    if elevations is None:
        return set()
    from . import final_building_road_clearance_policy as final_roads
    from . import osm

    roads = tuple(road_objects if road_objects is not None else objects)
    primitives = []
    for obj in roads:
        for primitive in final_roads._road_object_primitives(obj, spec):
            midpoint = (
                (primitive.start[0] + primitive.end[0]) * 0.5,
                (primitive.start[1] + primitive.end[1]) * 0.5,
            )
            terrain = osm._sample_elevation(
                elevations,
                spec.cells,
                spec.cell_size,
                midpoint[0],
                midpoint[1],
            )
            if (
                abs(float(primitive.elevation) - float(terrain))
                <= final_roads._MAXIMUM_VERTICAL_TERRAIN_GAP_METRES
            ):
                primitives.append(primitive)
    if not primitives:
        return set()

    index = final_roads._RoadPrimitiveIndex(tuple(primitives))
    barrier_length = max(2.0, float(getattr(spec, "barrier_segment_length", 6.0)))
    removed: set[int] = set()
    for object_index, cluster, barrier_family in zip(
        candidate_indices, cluster_flags, barrier_families
    ):
        obj = objects[int(object_index)]
        x = float(getattr(obj, "x", 0.0))
        z = float(getattr(obj, "z", 0.0))
        if barrier_family:
            if barrier_family == "wall":
                half_length = float(getattr(osm, "STOCK_WALL_EFFECTIVE_LENGTH_METRES", 2.45)) * 0.5
                half_width = 0.30
            elif barrier_family == "hedge":
                half_length = barrier_length * 0.5
                half_width = max(
                    0.45,
                    float(getattr(osm, "HEDGE_FOOTPRINT_HALF_WIDTH_METRES", 1.25)),
                )
            else:
                half_length = barrier_length * 0.5
                half_width = 0.35
            # Stock barrier P3Ds are authored along local X; placement rotates
            # them by +90 degrees relative to their mapped line heading.
            polygon = _oriented_box(
                x,
                z,
                float(getattr(obj, "heading_degrees", 0.0)) - 90.0,
                half_length,
                half_width,
            )
        else:
            half = (
                float(CLUSTER_EXTRA_CLEARANCE_METRES)
                if cluster
                else 0.45
            )
            polygon = (
                (x - half, z - half),
                (x + half, z - half),
                (x + half, z + half),
                (x - half, z + half),
            )
        conflicts, _checked = final_roads._conflicts(polygon, index)
        if conflicts:
            removed.add(int(object_index))
    return removed


def _runway_clear_shapes(dataset, projection, spec):
    from . import runway_surface_policy as runway

    result = []
    profile = getattr(spec, "ground_texture_profile", "generated")
    for geometry in runway._runway_geometries(dataset, projection, profile):
        sx, sz = geometry.start_x, geometry.start_z
        ex, ez = geometry.end_x, geometry.end_z
        px = geometry.px * geometry.half_width
        pz = geometry.pz * geometry.half_width
        shape = Polygon((
            (sx + px, sz + pz),
            (ex + px, ez + pz),
            (ex - px, ez - pz),
            (sx - px, sz - pz),
        ))
        if not shape.is_empty:
            result.append(shape.buffer(RUNWAY_CLEARANCE_METRES, join_style=2))
    return tuple(result)


def _sports_clear_shapes(dataset, projection):
    result = []
    for feature in getattr(dataset, "sites", ()):
        tags = getattr(feature, "tags", {}) or {}
        if str(tags.get("site", "")).casefold() != "sports_pitch":
            continue
        for polygon in getattr(feature, "polygons", ()):
            outer = [projection.to_world(point) for point in polygon.outer[:-1]]
            if len(outer) < 3:
                continue
            holes = [
                [projection.to_world(point) for point in ring[:-1]]
                for ring in getattr(polygon, "holes", ())
                if len(ring) >= 4
            ]
            shape = Polygon(outer, holes)
            if not shape.is_empty and shape.is_valid:
                result.append(shape.buffer(SPORTS_CLEARANCE_METRES, join_style=2))
    return tuple(result)


def _contains(geometry, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    if geometry is None or geometry.is_empty or xs.size == 0:
        return np.zeros(xs.shape, dtype=bool)
    return np.asarray(contains_xy(geometry, xs, zs), dtype=bool)


def filter_vegetation_objects(
    objects: Sequence[object],
    dataset,
    projection,
    spec,
    *,
    elevations: Sequence[float] | None = None,
    road_objects: Sequence[object] | None = None,
):
    from .procedural_forests import is_generated_cluster_model

    runway_shapes = _runway_clear_shapes(dataset, projection, spec)
    sports_shapes = _sports_clear_shapes(dataset, projection)

    runway_union = unary_union(runway_shapes) if runway_shapes else None
    sports_union = unary_union(sports_shapes) if sports_shapes else None
    combined_parts = [
        shape for shape in (runway_union, sports_union)
        if shape is not None and not shape.is_empty
    ]
    combined = unary_union(combined_parts) if combined_parts else None
    runway_cluster_union = (
        runway_union.buffer(CLUSTER_EXTRA_CLEARANCE_METRES, join_style=2)
        if runway_union is not None and not runway_union.is_empty else None
    )
    sports_cluster_union = (
        sports_union.buffer(CLUSTER_EXTRA_CLEARANCE_METRES, join_style=2)
        if sports_union is not None and not sports_union.is_empty else None
    )
    cluster_parts = [
        shape for shape in (runway_cluster_union, sports_cluster_union)
        if shape is not None and not shape.is_empty
    ]
    cluster_combined = unary_union(cluster_parts) if cluster_parts else None
    configured = _configured_vegetation_models(spec)
    barrier_models = _configured_barrier_models(spec)

    candidate_indices: list[int] = []
    xs: list[float] = []
    zs: list[float] = []
    cluster_flags: list[bool] = []
    barrier_families: list[str | None] = []
    vegetation_flags: list[bool] = []
    model_classification: dict[str, tuple[bool, bool, str | None]] = {}
    world_name = str(getattr(spec, "name", ""))
    for index, obj in enumerate(objects):
        model_path = str(getattr(obj, "model_path", ""))
        canonical = _canonical(model_path)
        classification = model_classification.get(canonical)
        if classification is None:
            cluster = is_generated_cluster_model(world_name, model_path)
            classification = (
                _is_tree_or_bush(model_path, spec, configured),
                cluster,
                barrier_models.get(canonical),
            )
            model_classification[canonical] = classification
        vegetation, cluster, barrier_family = classification
        if not vegetation and barrier_family is None:
            continue
        candidate_indices.append(index)
        xs.append(float(getattr(obj, "x", 0.0)))
        zs.append(float(getattr(obj, "z", 0.0)))
        cluster_flags.append(cluster)
        barrier_families.append(barrier_family)
        vegetation_flags.append(vegetation)

    if not candidate_indices:
        return tuple(objects), {
            "removed": 0, "runway": 0, "sports_pitch": 0,
            "final_road": 0, "final_road_vegetation": 0,
            "final_road_barrier": 0,
        }

    x_array = np.asarray(xs, dtype=np.float64)
    z_array = np.asarray(zs, dtype=np.float64)
    clusters = np.asarray(cluster_flags, dtype=bool)
    vegetation = np.asarray(vegetation_flags, dtype=bool)
    normal_hits = vegetation & _contains(combined, x_array, z_array)
    cluster_hits = vegetation & clusters & _contains(
        cluster_combined, x_array, z_array
    )
    remove_mask = normal_hits | cluster_hits

    road_remove_indices = _final_road_conflict_indices(
        tuple(objects),
        candidate_indices,
        cluster_flags,
        barrier_families,
        elevations=elevations,
        road_objects=road_objects,
        spec=spec,
    )
    for offset, object_index in enumerate(candidate_indices):
        if object_index in road_remove_indices:
            remove_mask[offset] = True

    if not bool(np.any(remove_mask)):
        return tuple(objects), {
            "removed": 0, "runway": 0, "sports_pitch": 0,
            "final_road": 0, "final_road_vegetation": 0,
            "final_road_barrier": 0,
        }

    runway_mask = (
        vegetation
        & (
            _contains(runway_union, x_array, z_array)
            | (clusters & _contains(runway_cluster_union, x_array, z_array))
        )
        & remove_mask
    )
    sports_mask = (
        vegetation
        & (
            _contains(sports_union, x_array, z_array)
            | (clusters & _contains(sports_cluster_union, x_array, z_array))
        )
        & remove_mask
    )
    remove_indices = {
        candidate_indices[offset]
        for offset in np.flatnonzero(remove_mask)
    }
    filtered = tuple(obj for index, obj in enumerate(objects) if index not in remove_indices)
    return filtered, {
        "removed": len(remove_indices),
        "runway": int(np.count_nonzero(runway_mask)),
        "sports_pitch": int(np.count_nonzero(sports_mask & ~runway_mask)),
        "final_road": len(road_remove_indices),
        "final_road_vegetation": sum(
            1
            for offset, object_index in enumerate(candidate_indices)
            if object_index in road_remove_indices and vegetation_flags[offset]
        ),
        "final_road_barrier": sum(
            1
            for offset, object_index in enumerate(candidate_indices)
            if object_index in road_remove_indices and barrier_families[offset] is not None
        ),
        "candidate_vegetation": int(np.count_nonzero(vegetation)),
        "candidate_barriers": sum(1 for value in barrier_families if value is not None),
        "cluster_candidates": int(np.count_nonzero(clusters)),
        "original_objects": len(objects),
        "final_objects": len(filtered),
    }


def install_vegetation_clearance_policy() -> None:
    """Install a cache-proof final vegetation clearance filter."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import runway_surface_policy as runway
    from .progress import report_progress

    original_writer = runway._ORIGINAL_WRITE_RVW4
    if not callable(original_writer):
        return

    def write_rvw4_without_surface_vegetation(
        path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
    ):
        context = runway._RUNWAY_CONTEXT.get()
        if (
            context is None
            or context.world_name.casefold() != Path(path).stem.casefold()
            or context.cells != int(width)
            or context.cells != int(height)
        ):
            return original_writer(
                path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
            )

        filtered, report = filter_vegetation_objects(
            tuple(objects),
            context.dataset,
            context.projection,
            context.spec,
            elevations=elevations,
            road_objects=tuple(objects),
        )
        report_path = Path(path).parent / "vegetation-exclusions.json"
        report_path.write_text(json.dumps({
            "schema": 1,
            "policy": "no-trees-bushes-or-barriers-on-final-road-surfaces",
            "runway_clearance_metres": RUNWAY_CLEARANCE_METRES,
            "sports_pitch_clearance_metres": SPORTS_CLEARANCE_METRES,
            "cluster_extra_clearance_metres": CLUSTER_EXTRA_CLEARANCE_METRES,
            **report,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        removed = int(report.get("removed", 0))
        if removed:
            report_progress(
                86,
                "Cleared vegetation/barriers from protected surfaces "
                f"({removed:,} removed; final roads {int(report.get('final_road', 0)):,}; "
                f"runway {int(report.get('runway', 0)):,}; "
                f"sports {int(report.get('sports_pitch', 0)):,})",
            )
        return original_writer(
            path, width, height, elevations, texture_indices, texture_paths, filtered, **kwargs
        )

    runway._ORIGINAL_WRITE_RVW4 = write_rvw4_without_surface_vegetation

    # Milestone validation compares the serialized WRP object count with
    # generated.objects. Present the same filtered view to validation so the
    # final write-time safety policy does not create a false count mismatch.
    original_validate = runway._ORIGINAL_VALIDATE_MILESTONE4

    def validate_without_surface_vegetation(*args, **kwargs):
        if len(args) >= 10:
            values = list(args)
            spec = values[1]
            dataset = values[3]
            projection = values[4]
            generated = values[9]
            road_fit = values[10] if len(values) > 10 else None
            elevations = values[6] if len(values) > 6 else None
            filtered, _report = filter_vegetation_objects(
                tuple(getattr(generated, "objects", ())),
                dataset,
                projection,
                spec,
                elevations=elevations,
                road_objects=tuple(getattr(road_fit, "objects", ())),
            )
            if len(filtered) != len(getattr(generated, "objects", ())):
                values[9] = replace(generated, objects=filtered)
            args = tuple(values)
        return original_validate(*args, **kwargs)

    runway._ORIGINAL_VALIDATE_MILESTONE4 = validate_without_surface_vegetation
    _INSTALLED = True
