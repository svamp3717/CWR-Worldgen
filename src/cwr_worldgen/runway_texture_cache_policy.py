# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache and reuse three generated terrain textures for every OSM runway.

The first generated-runway implementation rasterized every touched WRP cell and
only deduplicated after compression. That preserves arbitrary bearings perfectly,
but makes a long runway pay the full paint/compress cost dozens of times.

This policy changes the unit of reuse to one runway. Each runway gets a start,
middle and end texture. Touched WRP cells are classified along the runway axis and
all cells in a role share the same WRP texture slot. The three resulting PAA files
are persisted under a dedicated cache directory and restored on later unchanged
builds without repainting or recompressing them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Sequence

from .cache import cache_key, resolve_cache_dir, restore_or_create_file
from . import runway_surface_policy as _runway


RUNWAY_GROUND_CACHE_DIRNAME = "runway-ground-textures"
RUNWAY_GROUND_CACHE_SCHEMA = 1
_ROLE_CODES = (("start", "z"), ("middle", "d"), ("end", "k"))
_INSTALLED = False


def _cell_centre(cell_index: int, spec) -> tuple[float, float]:
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    row, column = divmod(int(cell_index), cells)
    return (column + 0.5) * cell_size, (row + 0.5) * cell_size


def _project_to_runway(
    cell_index: int, geometry: _runway._RunwayGeometry, spec
) -> tuple[float, float]:
    x, z = _cell_centre(cell_index, spec)
    rel_x, rel_z = x - geometry.start_x, z - geometry.start_z
    along = rel_x * geometry.ux + rel_z * geometry.uz
    lateral = rel_x * geometry.px + rel_z * geometry.pz
    return along, lateral


def _distance_to_runway_segment(
    cell_index: int, geometry: _runway._RunwayGeometry, spec
) -> float:
    along, lateral = _project_to_runway(cell_index, geometry, spec)
    clamped = min(max(along, 0.0), geometry.length)
    outside = along - clamped
    return math.hypot(outside, lateral)


def _assign_cells_to_runways(
    cell_indices: Sequence[int],
    geometries: Sequence[_runway._RunwayGeometry],
    spec,
) -> dict[int, list[int]]:
    assigned: dict[int, list[int]] = defaultdict(list)
    for cell_index in cell_indices:
        nearest = min(
            range(len(geometries)),
            key=lambda index: (
                _distance_to_runway_segment(cell_index, geometries[index], spec),
                index,
            ),
        )
        assigned[nearest].append(int(cell_index))
    return assigned


def _role_for_cell(
    cell_index: int, geometry: _runway._RunwayGeometry, spec
) -> str:
    along, _lateral = _project_to_runway(cell_index, geometry, spec)
    endpoint_band = min(
        geometry.length / 3.0,
        max(1.0, float(spec.cell_size) * 1.10),
    )
    if along <= endpoint_band:
        return "start"
    if along >= geometry.length - endpoint_band:
        return "end"
    return "middle"


def _cells_by_role(
    cell_indices: Sequence[int], geometry: _runway._RunwayGeometry, spec
) -> dict[str, list[int]]:
    groups = {role: [] for role, _code in _ROLE_CODES}
    for cell_index in cell_indices:
        groups[_role_for_cell(cell_index, geometry, spec)].append(int(cell_index))
    return groups


def _representative_cell(
    candidates: Sequence[int],
    all_cells: Sequence[int],
    geometry: _runway._RunwayGeometry,
    spec,
    role: str,
) -> int:
    pool = tuple(candidates) or tuple(all_cells)
    target = {
        "start": 0.0,
        "middle": geometry.length * 0.5,
        "end": geometry.length,
    }[role]
    return min(
        pool,
        key=lambda cell_index: (
            abs(_project_to_runway(cell_index, geometry, spec)[0] - target),
            abs(_project_to_runway(cell_index, geometry, spec)[1]),
            int(cell_index),
        ),
    )


def _cache_root(source_dir: Path, spec) -> Path:
    requested = getattr(spec, "cache_dir", None)
    cache_dir = resolve_cache_dir(
        Path(source_dir),
        Path(requested).expanduser() if requested is not None else None,
    )
    return cache_dir / RUNWAY_GROUND_CACHE_DIRNAME


def _exact_background_fingerprint(ground_path: str) -> str | None:
    try:
        from . import runway_exact_background_policy as exact_policy
    except ImportError:
        return None
    state = exact_policy._EXACT_STATE.get()
    if state is None:
        return None
    exact = state.textures.get(exact_policy._canonical(ground_path))
    if exact is None:
        return None
    return sha256(exact.data).hexdigest()


def _texture_cache_key(
    *,
    spec,
    geometry: _runway._RunwayGeometry,
    role: str,
    representative_cell: int,
    original_wrp_texture_index: int,
) -> str:
    material_index = int(original_wrp_texture_index) - 1
    ground_path = _runway._ground_path_for_material(spec, material_index)
    payload = {
        "schema": RUNWAY_GROUND_CACHE_SCHEMA,
        "renderer": "three-role-runway-v1",
        "role": role,
        "texture_size": int(_runway.RUNWAY_TEXTURE_SIZE),
        "profile": str(getattr(spec, "ground_texture_profile", "generated")),
        "seed": str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
        "world": str(getattr(spec, "name", "")),
        "cells": int(spec.cells),
        "cell_size": float(spec.cell_size),
        "representative_cell": int(representative_cell),
        "original_wrp_texture_index": int(original_wrp_texture_index),
        "ground_path": _runway._canonical_texture_path(ground_path),
        "exact_background_sha256": _exact_background_fingerprint(ground_path),
        "geometry": {
            "osm_key": geometry.osm_key,
            "start": (geometry.start_x, geometry.start_z),
            "end": (geometry.end_x, geometry.end_z),
            "length": geometry.length,
            "half_width": geometry.half_width,
        },
    }
    return cache_key("runway-ground-texture-v1", payload)


def _sync_exact_assignment_counts(
    cell_indices: Sequence[int], texture_indices: Sequence[int], spec
) -> None:
    """Keep exact-background diagnostics meaningful even when all files hit cache."""
    try:
        from . import runway_exact_background_policy as exact_policy
    except ImportError:
        return
    state = exact_policy._EXACT_STATE.get()
    if state is None:
        return
    exact_count = 0
    fallback_count = 0
    for cell_index in cell_indices:
        material_index = int(texture_indices[cell_index]) - 1
        ground_path = _runway._ground_path_for_material(spec, material_index)
        if exact_policy._canonical(ground_path) in state.textures:
            exact_count += 1
        else:
            fallback_count += 1
    state.exact_cell_assignments = exact_count
    state.fallback_cell_assignments = fallback_count


def apply_cached_runway_texture_triplets(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Render/cache one start-middle-end texture set for every active runway."""
    from . import generator

    cell_indices = _runway.runway_texture_cell_indices(dataset, projection, spec)
    if not cell_indices:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    if len(texture_indices) != int(spec.cells) * int(spec.cells):
        raise ValueError("runway texture generation received a mismatched WRP grid")

    geometries = _runway._runway_geometries(
        dataset, projection, getattr(spec, "ground_texture_profile", "generated")
    )
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    assignments = _assign_cells_to_runways(cell_indices, geometries, spec)
    active = tuple(index for index in range(len(geometries)) if assignments.get(index))
    required_slots = len(active) * len(_ROLE_CODES)
    if len(texture_paths) + required_slots > _runway.RVW4_TEXTURE_LIMIT:
        raise ValueError("cached runway texture triplets exceed the RVW4 512-entry texture table")

    source_dir = Path(source_dir)
    for stale in source_dir.glob(f"{_runway.RUNWAY_TEXTURE_PREFIX}*.paa"):
        stale.unlink()
    report_path = source_dir / "runway-textures.json"
    if report_path.exists():
        report_path.unlink()

    materials = tuple(generator._material_definitions(spec))
    revised_indices = [int(value) for value in texture_indices]
    revised_paths = [str(value) for value in texture_paths]
    generated_paths: list[str] = []
    background_cache: dict[tuple[int, str], object] = {}
    cache_root = _cache_root(source_dir, spec)
    cache_enabled = bool(getattr(spec, "cache_enabled", True))
    cache_refresh = bool(getattr(spec, "cache_refresh", False))
    cache_hits = 0
    cache_misses = 0
    texture_sets: list[dict[str, object]] = []

    for serial, geometry_index in enumerate(active):
        geometry = geometries[geometry_index]
        runway_cells = tuple(assignments[geometry_index])
        groups = _cells_by_role(runway_cells, geometry, spec)
        role_slots: dict[str, int] = {}
        role_paths: dict[str, str] = {}
        role_counts: dict[str, int] = {}

        for role, code in _ROLE_CODES:
            representative = _representative_cell(
                groups[role], runway_cells, geometry, spec, role
            )
            original_index = int(texture_indices[representative])
            filename = f"{_runway.RUNWAY_TEXTURE_PREFIX}{serial:03x}{code}.paa"
            wire_path = rf"{spec.name}\{filename}"
            destination = source_dir / filename
            key = _texture_cache_key(
                spec=spec,
                geometry=geometry,
                role=role,
                representative_cell=representative,
                original_wrp_texture_index=original_index,
            )
            cache_path = cache_root / f"{key}.paa"

            def produce(path: Path, *, representative=representative, original_index=original_index) -> None:
                image = _runway._render_runway_cell(
                    cell_index=representative,
                    original_wrp_texture_index=original_index,
                    geometries=(geometry,),
                    materials=materials,
                    spec=spec,
                    size=_runway.RUNWAY_TEXTURE_SIZE,
                    background_cache=background_cache,
                )
                _runway.write_rgb_dxt1_paa(path, image)

            hit = restore_or_create_file(
                cache_path=cache_path,
                destination=destination,
                producer=produce,
                enabled=cache_enabled,
                refresh=cache_refresh,
            )
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1

            slot = len(revised_paths)
            revised_paths.append(wire_path)
            generated_paths.append(wire_path)
            role_slots[role] = slot
            role_paths[role] = wire_path
            role_counts[role] = len(groups[role])

        for cell_index in runway_cells:
            revised_indices[cell_index] = role_slots[
                _role_for_cell(cell_index, geometry, spec)
            ]

        texture_sets.append({
            "osm_key": geometry.osm_key,
            "runway_index": serial,
            "cell_count": len(runway_cells),
            "role_cell_counts": role_counts,
            "role_texture_paths": role_paths,
        })

    _sync_exact_assignment_counts(cell_indices, texture_indices, spec)
    reused_cells = max(0, len(cell_indices) - len(generated_paths))
    report_path.write_text(json.dumps({
        "schema": 3,
        "mode": "generated-terrain-textures",
        "strategy": "three-textures-per-runway",
        "texture_limit": _runway.RVW4_TEXTURE_LIMIT,
        "texture_size": _runway.RUNWAY_TEXTURE_SIZE,
        "base_texture_entries": len(texture_paths),
        "runway_count": len(active),
        "runway_cells": len(cell_indices),
        "generated_runway_textures": len(generated_paths),
        "reused_runway_cell_assignments": reused_cells,
        "reuse_ratio": (reused_cells / len(cell_indices)) if cell_indices else 0.0,
        "final_texture_entries": len(revised_paths),
        "runway_cell_indices": list(cell_indices),
        "texture_paths": generated_paths,
        "texture_sets": texture_sets,
        "cache_directory": str(cache_root),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "stock_reference_family": {
            "grass": list(_runway.runway_texture_triplet("grass")),
            "desert": list(_runway.runway_texture_triplet("desert")),
        },
        "nogova_background_mode": "selected-stock-path-colour-match",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def _install_slot_budget_wrapper() -> None:
    """Budget three texture-table slots per runway instead of one per touched cell."""
    from . import generator
    from . import playability

    current = generator.fit_road_objects
    if getattr(current, "_cwr_three_runway_texture_budget", False):
        return

    def fit_road_objects_with_triplet_budget(
        dataset, projection, elevations, spec, *, starting_id: int = 1,
        progress_callback=None,
    ):
        report = _runway._ORIGINAL_FIT_ROAD_OBJECTS(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        cell_indices = _runway.runway_texture_cell_indices(dataset, projection, spec)
        geometries = _runway._runway_geometries(
            dataset, projection, getattr(spec, "ground_texture_profile", "generated")
        )
        active_count = 0
        if cell_indices and geometries:
            assignments = _assign_cells_to_runways(cell_indices, geometries, spec)
            active_count = sum(bool(assignments.get(index)) for index in range(len(geometries)))
        _fits, base_entries, _old_final = _runway._runway_texture_budget(spec, 0)
        final_entries = base_entries + active_count * len(_ROLE_CODES)
        mode = (
            "textures" if cell_indices and active_count and final_entries <= _runway.RVW4_TEXTURE_LIMIT else
            "p3d-fallback" if cell_indices else "none"
        )
        _runway._RUNWAY_CONTEXT.set(_runway._RunwayBuildContext(
            world_name=str(spec.name),
            cells=int(spec.cells),
            dataset=dataset,
            projection=projection,
            spec=spec,
            cell_indices=cell_indices,
            mode=mode,
            base_texture_entries=base_entries,
        ))
        _runway._GENERATED_RUNWAY_PATHS.set(())
        if mode != "p3d-fallback":
            return report
        next_id = max(
            (int(obj.object_id) for obj in report.objects),
            default=int(starting_id) - 1,
        ) + 1
        runways = _runway.runway_overlay_objects(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=next_id,
        )
        return report if not runways else replace(
            report, objects=tuple((*report.objects, *runways))
        )

    fit_road_objects_with_triplet_budget._cwr_three_runway_texture_budget = True
    generator.fit_road_objects = fit_road_objects_with_triplet_budget
    playability.fit_road_objects = fit_road_objects_with_triplet_budget


def install_runway_texture_cache_policy() -> None:
    """Install cached start/middle/end runway textures exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return

    # The exact-background policy is the outer wrapper. Replace the callable it
    # delegates to so exact stock PAA compositing remains active while the inner
    # renderer changes from per-cell generation to cached three-role generation.
    try:
        from . import runway_exact_background_policy as exact_policy
    except ImportError:
        exact_policy = None
    if exact_policy is not None and getattr(exact_policy, "_INSTALLED", False):
        exact_policy._ORIGINAL_APPLY_RUNWAY_TEXTURES = apply_cached_runway_texture_triplets
    else:
        _runway.apply_generated_runway_texture_table = apply_cached_runway_texture_triplets

    _install_slot_budget_wrapper()
    _INSTALLED = True
