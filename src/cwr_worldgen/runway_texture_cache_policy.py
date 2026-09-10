# SPDX-License-Identifier: GPL-3.0-or-later
"""Cache three generated ground textures per runway instead of painting every cell.

The runway surface policy originally rendered every touched RVW4 cell and only
deduplicated the finished RGB images afterwards. That preserves arbitrary
sub-cell alignment, but it makes long airfields pay the full paint/DXT1 cost on
every build. This policy deliberately chooses the cheaper representation the GUI
workflow needs: one start, one middle, and one end texture per OSM runway. Those
three PAAs are persisted in a dedicated cache outside disposable build-stage
caches and restored on later builds.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
from typing import Sequence

from .cache import cache_key, resolve_cache_dir, restore_or_create_file
from . import runway_surface_policy as _runway


RUNWAY_GROUND_CACHE_DIRNAME = ".cwr-worldgen-runway-cache"
RUNWAY_TRIPLET_CACHE_SCHEMA = 1
_ROLE_SPECS = (
    ("start", "z", 0.0),
    ("middle", "d", 0.5),
    ("end", "k", 1.0),
)
_INSTALLED = False


def _cell_metrics(cell_index: int, geometry, spec) -> tuple[float, float, float]:
    """Return along, lateral, and squared segment distance for one cell centre."""
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    cz, cx = divmod(int(cell_index), cells)
    x = (cx + 0.5) * cell_size
    z = (cz + 0.5) * cell_size
    rel_x, rel_z = x - geometry.start_x, z - geometry.start_z
    along = rel_x * geometry.ux + rel_z * geometry.uz
    lateral = rel_x * geometry.px + rel_z * geometry.pz
    if along < 0.0:
        longitudinal = -along
    elif along > geometry.length:
        longitudinal = along - geometry.length
    else:
        longitudinal = 0.0
    return along, lateral, lateral * lateral + longitudinal * longitudinal


def _assign_cells_to_runways(
    cell_indices: Sequence[int], geometries: Sequence[object], spec
) -> tuple[tuple[tuple[int, float, float], ...], ...]:
    """Assign every touched cell to its nearest runway centre line."""
    assigned: list[list[tuple[int, float, float]]] = [
        [] for _ in geometries
    ]
    for raw_index in cell_indices:
        cell_index = int(raw_index)
        choices = []
        for geometry_index, geometry in enumerate(geometries):
            along, lateral, distance_sq = _cell_metrics(cell_index, geometry, spec)
            choices.append((distance_sq, abs(lateral), geometry_index, along, lateral))
        if not choices:
            continue
        _distance, _lateral_abs, geometry_index, along, lateral = min(choices)
        assigned[geometry_index].append((cell_index, along, lateral))
    return tuple(
        tuple(sorted(values, key=lambda item: (item[1], abs(item[2]), item[0])))
        for values in assigned
    )


def _role_for_cell(along: float, geometry, spec) -> str:
    """Use end-cap textures only for roughly one terrain-cell length."""
    length = float(geometry.length)
    if length <= 0.0:
        return "middle"
    band = min(length / 3.0, max(1.0, float(spec.cell_size) * 1.10))
    if along <= band:
        return "start"
    if along >= length - band:
        return "end"
    return "middle"


def _representative_cell(
    values: Sequence[tuple[int, float, float]], geometry, role: str
) -> int:
    """Pick the cell whose local slice best represents one reusable role."""
    target = {
        "start": 0.0,
        "middle": float(geometry.length) * 0.5,
        "end": float(geometry.length),
    }[role]
    cell_index, _along, _lateral = min(
        values,
        key=lambda item: (abs(item[1] - target), abs(item[2]), item[0]),
    )
    return int(cell_index)


def _requested_cache_dir(spec) -> Path | None:
    value = getattr(spec, "cache_dir", None)
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser()


def _runway_cache_dir(source_dir: Path, spec) -> Path:
    """Return a persistent cache that survives normal GUI post-build cleanup."""
    requested = _requested_cache_dir(spec)
    if requested is not None:
        requested = requested.resolve()
        try:
            from .build_cache_policy import BUILD_CACHE_DIRNAME
        except ImportError:
            BUILD_CACHE_DIRNAME = ".cwr-worldgen-build-cache"
        if requested.parent.name == BUILD_CACHE_DIRNAME:
            return requested.parent.parent / RUNWAY_GROUND_CACHE_DIRNAME
        return requested / RUNWAY_GROUND_CACHE_DIRNAME
    return resolve_cache_dir(Path(source_dir), None) / RUNWAY_GROUND_CACHE_DIRNAME


def _exact_background_fingerprint(spec, original_texture_index: int) -> dict[str, object]:
    material_index = int(original_texture_index) - 1
    ground_path = _runway._ground_path_for_material(spec, material_index)
    result: dict[str, object] = {
        "material_index": material_index,
        "ground_path": _runway._canonical_texture_path(ground_path),
    }
    try:
        from . import runway_exact_background_policy as exact_policy

        state = exact_policy._EXACT_STATE.get()
        if state is not None:
            exact = state.textures.get(exact_policy._canonical(ground_path))
            if exact is not None:
                result["exact_sha256"] = sha256(exact.data).hexdigest()
    except (AttributeError, ImportError):
        pass
    return result


def _cache_path_for_role(
    cache_dir: Path,
    *,
    geometry,
    role: str,
    representative_cell: int,
    original_texture_index: int,
    spec,
) -> Path:
    payload = {
        "schema": RUNWAY_TRIPLET_CACHE_SCHEMA,
        "renderer": "runway-triplet-v1",
        "role": role,
        "texture_size": int(_runway.RUNWAY_TEXTURE_SIZE),
        "profile": str(getattr(spec, "ground_texture_profile", "generated")),
        "seed": str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
        "world": str(getattr(spec, "name", "")),
        "cells": int(getattr(spec, "cells", 0)),
        "cell_size": float(getattr(spec, "cell_size", 0.0)),
        "representative_cell": int(representative_cell),
        "geometry": {
            "osm_key": str(getattr(geometry, "osm_key", "")),
            "start": [round(float(geometry.start_x), 6), round(float(geometry.start_z), 6)],
            "end": [round(float(geometry.end_x), 6), round(float(geometry.end_z), 6)],
            "length": round(float(geometry.length), 6),
            "half_width": round(float(geometry.half_width), 6),
        },
        "background": _exact_background_fingerprint(spec, original_texture_index),
    }
    key = cache_key("runway-ground-texture-triplet-v1", payload)
    return cache_dir / f"{key}.paa"


def _sync_exact_assignment_counts(
    touched: Sequence[int], texture_indices: Sequence[int], spec
) -> None:
    """Keep exact-background diagnostics meaningful when rendered PAAs hit cache."""
    try:
        from . import runway_exact_background_policy as exact_policy

        state = exact_policy._EXACT_STATE.get()
    except (AttributeError, ImportError):
        return
    if state is None:
        return
    exact_count = 0
    fallback_count = 0
    for cell_index in touched:
        material_index = int(texture_indices[int(cell_index)]) - 1
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
    """Create at most three reusable ground textures for each OSM runway."""
    from . import generator

    touched = _runway.runway_texture_cell_indices(dataset, projection, spec)
    if not touched:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    if len(texture_indices) != int(spec.cells) * int(spec.cells):
        raise ValueError("runway texture generation received a mismatched WRP grid")

    geometries = _runway._runway_geometries(
        dataset, projection, getattr(spec, "ground_texture_profile", "generated")
    )
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    assignments = _assign_cells_to_runways(touched, geometries, spec)
    active = tuple(
        (geometry_index, geometry, values)
        for geometry_index, (geometry, values) in enumerate(zip(geometries, assignments))
        if values
    )
    required_slots = len(active) * len(_ROLE_SPECS)
    if len(texture_paths) + required_slots > _runway.RVW4_TEXTURE_LIMIT:
        raise ValueError("runway texture triplets exceed the RVW4 512-entry texture table")

    source_dir = Path(source_dir)
    for pattern in (
        f"{_runway.RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa",
        f"{_runway.RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f][zdk].paa",
    ):
        for stale in source_dir.glob(pattern):
            stale.unlink()
    report_path = source_dir / "runway-textures.json"
    if report_path.exists():
        report_path.unlink()

    runway_cache_dir = _runway_cache_dir(source_dir, spec)
    cache_enabled = bool(getattr(spec, "cache_enabled", True))
    cache_refresh = bool(getattr(spec, "cache_refresh", False))

    materials = tuple(generator._material_definitions(spec))
    revised_indices = [int(value) for value in texture_indices]
    revised_paths = [str(value) for value in texture_paths]
    generated_paths: list[str] = []
    texture_sets: list[dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0
    background_cache: dict[tuple[int, str], object] = {}

    for serial, (geometry_index, geometry, values) in enumerate(active):
        role_slots: dict[str, int] = {}
        role_paths: dict[str, str] = {}
        role_cells = {"start": 0, "middle": 0, "end": 0}

        for role, role_code, _fraction in _ROLE_SPECS:
            representative = _representative_cell(values, geometry, role)
            original_index = int(texture_indices[representative])
            filename = f"{_runway.RUNWAY_TEXTURE_PREFIX}{serial:03x}{role_code}.paa"
            wire_path = rf"{spec.name}\{filename}"
            slot = len(revised_paths)
            if slot >= _runway.RVW4_TEXTURE_LIMIT:
                raise ValueError("runway texture triplets exceed the RVW4 texture table")

            cache_path = _cache_path_for_role(
                runway_cache_dir,
                geometry=geometry,
                role=role,
                representative_cell=representative,
                original_texture_index=original_index,
                spec=spec,
            )

            def paint(destination: Path, *, cell=representative, wrp_index=original_index, geom=geometry):
                image = _runway._render_runway_cell(
                    cell_index=cell,
                    original_wrp_texture_index=wrp_index,
                    geometries=(geom,),
                    materials=materials,
                    spec=spec,
                    size=_runway.RUNWAY_TEXTURE_SIZE,
                    background_cache=background_cache,
                )
                _runway.write_rgb_dxt1_paa(destination, image)

            hit = restore_or_create_file(
                cache_path=cache_path,
                destination=source_dir / filename,
                producer=paint,
                enabled=cache_enabled,
                refresh=cache_refresh,
            )
            cache_hits += int(hit)
            cache_misses += int(not hit)
            revised_paths.append(wire_path)
            generated_paths.append(wire_path)
            role_slots[role] = slot
            role_paths[role] = wire_path

        for cell_index, along, _lateral in values:
            role = _role_for_cell(along, geometry, spec)
            revised_indices[cell_index] = role_slots[role]
            role_cells[role] += 1

        texture_sets.append({
            "runway_index": int(geometry_index),
            "osm_key": str(getattr(geometry, "osm_key", "")),
            "cells": len(values),
            "role_cell_counts": role_cells,
            "texture_paths": role_paths,
        })

    _sync_exact_assignment_counts(touched, texture_indices, spec)
    reused_cells = max(0, len(touched) - len(generated_paths))
    report_path.write_text(json.dumps({
        "schema": 3,
        "mode": "generated-terrain-textures",
        "strategy": "three-textures-per-runway",
        "texture_limit": _runway.RVW4_TEXTURE_LIMIT,
        "texture_size": _runway.RUNWAY_TEXTURE_SIZE,
        "base_texture_entries": len(texture_paths),
        "runway_cells": len(touched),
        "runway_count": len(active),
        "generated_runway_textures": len(generated_paths),
        "reused_runway_cell_assignments": reused_cells,
        "reuse_ratio": (reused_cells / len(touched)) if touched else 0.0,
        "final_texture_entries": len(revised_paths),
        "runway_cell_indices": list(map(int, touched)),
        "texture_paths": generated_paths,
        "texture_sets": texture_sets,
        "cache_directory": str(runway_cache_dir),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "stock_reference_family": {
            "grass": list(_runway.runway_texture_triplet("grass")),
            "desert": list(_runway.runway_texture_triplet("desert")),
        },
        "nogova_background_mode": "selected-stock-path-colour-match",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def _fit_road_objects_with_triplet_budget(
    dataset,
    projection,
    elevations: Sequence[float],
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    """Mirror the runway wrapper but budget three slots per runway, not per cell."""
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
        active_count = sum(bool(values) for values in assignments)
    _unused_fits, base_entries, _unused_final = _runway._runway_texture_budget(spec, 0)
    final_entries = base_entries + active_count * len(_ROLE_SPECS)
    fits = final_entries <= _runway.RVW4_TEXTURE_LIMIT
    mode = (
        "textures" if cell_indices and active_count and fits else
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


def install_runway_texture_cache_policy() -> None:
    """Install reusable runway triplets after exact-background support is active."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator
    from . import playability
    from . import runway_exact_background_policy as exact_policy

    if getattr(exact_policy, "_INSTALLED", False):
        # Preserve the exact-background wrapper. It will call this optimized
        # implementation inside its build-scoped exact-texture context.
        exact_policy._ORIGINAL_APPLY_RUNWAY_TEXTURES = apply_cached_runway_texture_triplets
    else:
        _runway.apply_generated_runway_texture_table = apply_cached_runway_texture_triplets

    generator.fit_road_objects = _fit_road_objects_with_triplet_budget
    playability.fit_road_objects = _fit_road_objects_with_triplet_budget
    _INSTALLED = True
