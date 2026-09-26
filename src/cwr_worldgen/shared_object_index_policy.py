# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared immutable-object indexes for final world post-processing.

Several late policies independently walked the complete WRP object tuple to
case-fold model paths, classify object families, collect coordinates and rebuild
model counts. Large worlds can exceed 150k objects, so the repeated O(N) setup is
noticeable even when each actual policy only cares about a few hundred objects.

The index is identity-cached for immutable tuples and provides model groups,
coordinate arrays and a small spatial hash. Consumers still preserve their old
filter/grounding semantics; they simply start from the relevant model groups.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Sequence

import numpy as np

from . import osm as _osm
from . import stock_utility_policy as _stock_utility
from . import vegetation_clearance_policy as _vegetation

_INSTALLED = False
_ORIGINAL_VEGETATION_FILTER: Any = None
_ORIGINAL_UTILITY_REWRITE: Any = None
_ORIGINAL_VEGETATION_AUDIT: Any = None
_MAX_INDEXES = 5


def _canonical(value: object) -> str:
    return str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()


@dataclass(slots=True)
class _ObjectIndex:
    owner: tuple[Any, ...]
    canonical_models: tuple[str, ...]
    model_groups: dict[str, tuple[int, ...]]
    original_model_paths: dict[str, str]
    xs: np.ndarray
    zs: np.ndarray
    bucket_size: float
    buckets: dict[tuple[int, int], tuple[int, ...]]

    def nearby_indices(self, x: float, z: float, radius: float) -> tuple[int, ...]:
        radius = max(0.0, float(radius))
        bx0 = math.floor((x - radius) / self.bucket_size)
        bx1 = math.floor((x + radius) / self.bucket_size)
        bz0 = math.floor((z - radius) / self.bucket_size)
        bz1 = math.floor((z + radius) / self.bucket_size)
        result: list[int] = []
        for bz in range(bz0, bz1 + 1):
            for bx in range(bx0, bx1 + 1):
                result.extend(self.buckets.get((bx, bz), ()))
        return tuple(result)


_INDEXES: "OrderedDict[int, _ObjectIndex]" = OrderedDict()


def object_index(objects: Sequence[Any]) -> _ObjectIndex:
    owner = objects if isinstance(objects, tuple) else tuple(objects)
    identity = id(owner)
    cached = _INDEXES.get(identity)
    if cached is not None and cached.owner is owner:
        _INDEXES.move_to_end(identity)
        return cached

    canonical_models = tuple(
        _canonical(getattr(obj, "model_path", "")) for obj in owner
    )
    mutable_groups: dict[str, list[int]] = {}
    original_paths: dict[str, str] = {}
    xs = np.fromiter(
        (float(getattr(obj, "x", 0.0)) for obj in owner),
        dtype=np.float64,
        count=len(owner),
    )
    zs = np.fromiter(
        (float(getattr(obj, "z", 0.0)) for obj in owner),
        dtype=np.float64,
        count=len(owner),
    )
    for index, (obj, canonical) in enumerate(zip(owner, canonical_models)):
        mutable_groups.setdefault(canonical, []).append(index)
        original_paths.setdefault(canonical, str(getattr(obj, "model_path", "")))

    bucket_size = 64.0
    mutable_buckets: dict[tuple[int, int], list[int]] = {}
    for index in range(len(owner)):
        key = (
            math.floor(float(xs[index]) / bucket_size),
            math.floor(float(zs[index]) / bucket_size),
        )
        mutable_buckets.setdefault(key, []).append(index)

    value = _ObjectIndex(
        owner=owner,
        canonical_models=canonical_models,
        model_groups={key: tuple(values) for key, values in mutable_groups.items()},
        original_model_paths=original_paths,
        xs=xs,
        zs=zs,
        bucket_size=bucket_size,
        buckets={key: tuple(values) for key, values in mutable_buckets.items()},
    )
    _INDEXES[identity] = value
    _INDEXES.move_to_end(identity)
    while len(_INDEXES) > _MAX_INDEXES:
        _INDEXES.popitem(last=False)
    return value


def _indexed_filter_vegetation_objects(
    objects,
    dataset,
    projection,
    spec,
    *,
    elevations=None,
    road_objects=None,
):
    from shapely.ops import unary_union
    from .procedural_forests import is_generated_cluster_model

    owner = objects if isinstance(objects, tuple) else tuple(objects)
    # Final fitted-road clearance needs the exact road-object primitives and
    # barrier footprints implemented by the authoritative policy. Keep the
    # vectorized runway/sports fast path for ordinary calls, but delegate when
    # the WRP writer supplies final road geometry.
    if elevations is not None and _ORIGINAL_VEGETATION_FILTER is not None:
        return _ORIGINAL_VEGETATION_FILTER(
            owner,
            dataset,
            projection,
            spec,
            elevations=elevations,
            road_objects=road_objects,
        )
    runway_shapes = _vegetation._runway_clear_shapes(dataset, projection, spec)
    sports_shapes = _vegetation._sports_clear_shapes(dataset, projection)
    if not runway_shapes and not sports_shapes:
        return owner, {"removed": 0, "runway": 0, "sports_pitch": 0}

    runway_union = unary_union(runway_shapes) if runway_shapes else None
    sports_union = unary_union(sports_shapes) if sports_shapes else None
    combined_parts = [
        shape for shape in (runway_union, sports_union)
        if shape is not None and not shape.is_empty
    ]
    combined = unary_union(combined_parts) if combined_parts else None
    runway_cluster_union = (
        runway_union.buffer(_vegetation.CLUSTER_EXTRA_CLEARANCE_METRES, join_style=2)
        if runway_union is not None and not runway_union.is_empty else None
    )
    sports_cluster_union = (
        sports_union.buffer(_vegetation.CLUSTER_EXTRA_CLEARANCE_METRES, join_style=2)
        if sports_union is not None and not sports_union.is_empty else None
    )
    cluster_parts = [
        shape for shape in (runway_cluster_union, sports_cluster_union)
        if shape is not None and not shape.is_empty
    ]
    cluster_combined = unary_union(cluster_parts) if cluster_parts else None
    configured = _vegetation._configured_vegetation_models(spec)
    world_name = str(getattr(spec, "name", ""))
    index = object_index(owner)

    candidate_indices: list[int] = []
    cluster_flags: list[bool] = []
    for canonical, indices in index.model_groups.items():
        model_path = index.original_model_paths[canonical]
        cluster = is_generated_cluster_model(world_name, model_path)
        if not _vegetation._is_tree_or_bush(model_path, spec, configured):
            continue
        candidate_indices.extend(indices)
        cluster_flags.extend([cluster] * len(indices))

    if not candidate_indices:
        return owner, {"removed": 0, "runway": 0, "sports_pitch": 0}

    # Restore original object order after collecting whole model groups.
    order = np.argsort(np.asarray(candidate_indices, dtype=np.int64), kind="stable")
    candidate_array = np.asarray(candidate_indices, dtype=np.int64)[order]
    clusters = np.asarray(cluster_flags, dtype=np.bool_)[order]
    x_array = index.xs[candidate_array]
    z_array = index.zs[candidate_array]
    normal_hits = _vegetation._contains(combined, x_array, z_array)
    cluster_hits = clusters & _vegetation._contains(
        cluster_combined, x_array, z_array
    )
    remove_mask = normal_hits | cluster_hits
    if not bool(np.any(remove_mask)):
        return owner, {"removed": 0, "runway": 0, "sports_pitch": 0}

    runway_mask = (
        _vegetation._contains(runway_union, x_array, z_array)
        | (clusters & _vegetation._contains(
            runway_cluster_union, x_array, z_array
        ))
    ) & remove_mask
    sports_mask = (
        _vegetation._contains(sports_union, x_array, z_array)
        | (clusters & _vegetation._contains(
            sports_cluster_union, x_array, z_array
        ))
    ) & remove_mask
    remove_indices = set(candidate_array[np.flatnonzero(remove_mask)].tolist())
    filtered = tuple(
        obj for object_index_value, obj in enumerate(owner)
        if object_index_value not in remove_indices
    )
    return filtered, {
        "removed": len(remove_indices),
        "runway": int(np.count_nonzero(runway_mask)),
        "sports_pitch": int(np.count_nonzero(sports_mask & ~runway_mask)),
        "candidate_vegetation": len(candidate_array),
        "cluster_candidates": int(np.count_nonzero(clusters)),
        "original_objects": len(owner),
        "final_objects": len(filtered),
    }


def _indexed_rewrite_stock_utilities(
    result,
    dataset,
    projection,
    raster,
    elevations,
    spec,
    progress: Callable[[int, str], None] | None = None,
):
    owner = tuple(getattr(result, "objects", ()))
    if not owner:
        return result
    index = object_index(owner)

    utility_indices: list[int] = []
    utility_kinds: dict[int, str] = {}
    for canonical, indices in index.model_groups.items():
        model_path = index.original_model_paths[canonical]
        kind = _stock_utility._utility_kind(model_path)
        if not kind:
            continue
        for object_index_value in indices:
            utility_indices.append(object_index_value)
            utility_kinds[object_index_value] = kind
    if not utility_indices:
        return result

    corridors = _osm._project_vehicle_road_corridors(dataset, projection)
    mapped_tower_positions = _stock_utility._mapped_power_tower_positions(
        dataset, projection
    )
    rewritten = list(owner)
    changed = 0
    unresolved = 0

    existing_usage = getattr(result, "model_usage", ())
    if existing_usage:
        model_usage = Counter(dict(existing_usage))
    else:
        model_usage = Counter(obj.model_path for obj in owner)

    for object_index_value in sorted(utility_indices):
        obj = owner[object_index_value]
        kind = utility_kinds[object_index_value]
        if kind == "stock_power_tower":
            kind = (
                "mapped_power_tower"
                if _stock_utility._matches_mapped_tower(
                    obj.x, obj.z, mapped_tower_positions
                )
                else "settlement_power_tower"
            )

        if kind == "mapped_power_tower":
            model = _stock_utility.STOCK_POWER_TOWER_MODELS[0]
            footprint = _stock_utility._TOWER_FOOTPRINT_METRES
            target_distance = (
                _stock_utility._TOWER_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES
            )
        else:
            model = _stock_utility._stock_pole_model(obj.object_id)
            footprint = _stock_utility._POLE_FOOTPRINT_METRES
            target_distance = (
                _stock_utility._POLE_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES
            )

        safe = _stock_utility._road_safe_position(
            dataset,
            projection,
            raster,
            elevations,
            spec,
            corridors,
            obj.x,
            obj.z,
            footprint=footprint,
            target_distance=target_distance,
        )
        if safe is None:
            unresolved += 1
            new_obj = replace(obj, model_path=model)
        else:
            sx, sz, sy = safe
            new_obj = replace(obj, model_path=model, x=sx, z=sz, y=sy)

        if new_obj != obj:
            changed += 1
            model_usage[obj.model_path] -= 1
            if model_usage[obj.model_path] <= 0:
                model_usage.pop(obj.model_path, None)
            model_usage[new_obj.model_path] += 1
            rewritten[object_index_value] = new_obj

    if not changed:
        return result
    if unresolved and progress is not None:
        progress(
            66,
            f"WARNING: {unresolved:,} stock utility pole/tower placements could not be moved fully clear of roads/buildings; kept at their source coordinates.",
        )
    return replace(
        result,
        objects=tuple(rewritten),
        model_usage=tuple(sorted(
            model_usage.items(), key=lambda item: item[0].casefold()
        )),
    )


def _indexed_vegetation_grounding_audit(objects, elevations, spec):
    owner = objects if isinstance(objects, tuple) else tuple(objects)
    cells = int(getattr(spec, "cells"))
    cell_size = float(getattr(spec, "cell_size"))
    world_name = str(getattr(spec, "name", "cwr_world"))
    tree_limit = max(
        0.0, float(getattr(spec, "forest_single_tree_maximum_float", 0.15))
    )
    cluster_tree_limit = max(
        0.0, float(getattr(spec, "forest_cluster_tree_maximum_float", 0.20))
    )
    cluster_bush_limit = max(
        0.0, float(getattr(spec, "forest_cluster_bush_maximum_float", 0.60))
    )
    individual_models = {
        value.casefold() for value in _osm.OSM_INDIVIDUAL_TREE_MODELS
    }
    individual_models.update(
        str(getattr(spec, field, "")).casefold()
        for field in (
            "forest_single_tree_model",
            "forest_hillside_tree_model",
            "forest_roadside_tree_model",
        )
        if getattr(spec, field, "")
    )
    individual_models.update(
        str(value).casefold()
        for value in getattr(spec, "forest_roadside_tree_models", _osm.ROADSIDE_TREE_MODELS)
    )

    tree_objects = cluster_trees = cluster_bushes = violations = 0
    maximum_tree_float = maximum_bush_float = 0.0
    index = object_index(owner)

    for canonical, indices in index.model_groups.items():
        model_path = index.original_model_paths[canonical]
        parsed = _osm._parse_generated_cluster_model(model_path, world_name)
        if parsed is not None:
            variant, grade = parsed
            for object_index_value in indices:
                obj = owner[object_index_value]
                tree_float, bush_float, trees, bushes = _osm._cluster_proxy_floats(
                    variant,
                    grade=grade,
                    heading=obj.heading_degrees,
                    x=obj.x,
                    y=obj.y,
                    z=obj.z,
                    elevations=elevations,
                    cells=cells,
                    cell_size=cell_size,
                )
                cluster_trees += trees
                cluster_bushes += bushes
                maximum_tree_float = max(maximum_tree_float, tree_float)
                maximum_bush_float = max(maximum_bush_float, bush_float)
                violations += int(
                    trees > 0 and tree_float > cluster_tree_limit + 1.0e-6
                )
                violations += int(
                    bushes > 0 and bush_float > cluster_bush_limit + 1.0e-6
                )
            continue

        folded = model_path.casefold()
        if not (folded.startswith("data3d\\str") or folded in individual_models):
            continue
        tree_objects += len(indices)
        for object_index_value in indices:
            obj = owner[object_index_value]
            minimum_ground, _maximum_ground = _osm._triangle_elevation_bounds(
                elevations, cells, cell_size, obj.x, obj.z
            )
            floating = max(0.0, obj.y - minimum_ground)
            maximum_tree_float = max(maximum_tree_float, floating)
            violations += int(floating > tree_limit + 1.0e-6)

    return (
        tree_objects,
        cluster_trees,
        cluster_bushes,
        violations,
        maximum_tree_float,
        maximum_bush_float,
    )


def install_shared_object_index_policy() -> None:
    global _INSTALLED
    global _ORIGINAL_VEGETATION_FILTER, _ORIGINAL_UTILITY_REWRITE
    global _ORIGINAL_VEGETATION_AUDIT
    if _INSTALLED:
        return
    _ORIGINAL_VEGETATION_FILTER = _vegetation.filter_vegetation_objects
    _ORIGINAL_UTILITY_REWRITE = _stock_utility._rewrite_stock_utilities
    _ORIGINAL_VEGETATION_AUDIT = _osm._audit_vegetation_grounding
    _vegetation.filter_vegetation_objects = _indexed_filter_vegetation_objects
    _stock_utility._rewrite_stock_utilities = _indexed_rewrite_stock_utilities
    _osm._audit_vegetation_grounding = _indexed_vegetation_grounding_audit
    _INSTALLED = True
