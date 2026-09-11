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
from typing import Iterable, Sequence

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union


_INSTALLED = False
RUNWAY_CLEARANCE_METRES = 2.0
SPORTS_CLEARANCE_METRES = 0.75


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
        # Generated forest clusters contain tree/bush proxies. Clearing the
        # whole cluster is the only way to guarantee none of those proxies sits
        # on a runway or pitch.
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


def filter_vegetation_objects(objects: Sequence[object], dataset, projection, spec):
    runway_shapes = _runway_clear_shapes(dataset, projection, spec)
    sports_shapes = _sports_clear_shapes(dataset, projection)
    if not runway_shapes and not sports_shapes:
        return tuple(objects), {"removed": 0, "runway": 0, "sports_pitch": 0}

    runway_union = unary_union(runway_shapes) if runway_shapes else None
    sports_union = unary_union(sports_shapes) if sports_shapes else None
    combined_parts = [
        shape for shape in (runway_union, sports_union)
        if shape is not None and not shape.is_empty
    ]
    combined = unary_union(combined_parts) if combined_parts else None
    configured = _configured_vegetation_models(spec)

    candidate_indices: list[int] = []
    xs: list[float] = []
    zs: list[float] = []
    for index, obj in enumerate(objects):
        if not _is_tree_or_bush(str(getattr(obj, "model_path", "")), spec, configured):
            continue
        candidate_indices.append(index)
        xs.append(float(getattr(obj, "x", 0.0)))
        zs.append(float(getattr(obj, "z", 0.0)))

    if not candidate_indices:
        return tuple(objects), {"removed": 0, "runway": 0, "sports_pitch": 0}

    x_array = np.asarray(xs, dtype=np.float64)
    z_array = np.asarray(zs, dtype=np.float64)
    remove_mask = _contains(combined, x_array, z_array)
    if not bool(np.any(remove_mask)):
        return tuple(objects), {"removed": 0, "runway": 0, "sports_pitch": 0}

    runway_mask = _contains(runway_union, x_array, z_array) & remove_mask
    sports_mask = _contains(sports_union, x_array, z_array) & remove_mask
    remove_indices = {
        candidate_indices[offset]
        for offset in np.flatnonzero(remove_mask)
    }
    filtered = tuple(obj for index, obj in enumerate(objects) if index not in remove_indices)
    return filtered, {
        "removed": len(remove_indices),
        "runway": int(np.count_nonzero(runway_mask)),
        "sports_pitch": int(np.count_nonzero(sports_mask & ~runway_mask)),
        "candidate_vegetation": len(candidate_indices),
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
            tuple(objects), context.dataset, context.projection, context.spec
        )
        report_path = Path(path).parent / "vegetation-exclusions.json"
        report_path.write_text(json.dumps({
            "schema": 1,
            "policy": "no-trees-or-bushes-on-runways-or-sports-pitches",
            "runway_clearance_metres": RUNWAY_CLEARANCE_METRES,
            "sports_pitch_clearance_metres": SPORTS_CLEARANCE_METRES,
            **report,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        removed = int(report.get("removed", 0))
        if removed:
            report_progress(
                86,
                "Cleared tree/bush objects from runways and sports pitches "
                f"({removed:,} removed; runway {int(report.get('runway', 0)):,}; "
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
            filtered, _report = filter_vegetation_objects(
                tuple(getattr(generated, "objects", ())), dataset, projection, spec
            )
            if len(filtered) != len(getattr(generated, "objects", ())):
                values[9] = replace(generated, objects=filtered)
            args = tuple(values)
        return original_validate(*args, **kwargs)

    runway._ORIGINAL_VALIDATE_MILESTONE4 = validate_without_surface_vegetation
    _INSTALLED = True
