# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist correctly aligned generated runway-cell textures between builds.

Each RVW4 terrain cell has its own fixed UV space. Reusing one representative
start/middle/end image across different cells therefore moves the runway inside
those cells and can create fake parallel strips. Keep the surface policy's
world-coordinate renderer instead: render every touched cell at its real world
position, cache that result persistently, and deduplicate byte-identical PAAs in
the final WRP texture table.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
from typing import Sequence

from .cache import cache_key, resolve_cache_dir, restore_or_create_file
from . import runway_surface_policy as _runway


RUNWAY_GROUND_CACHE_DIRNAME = ".cwr-worldgen-runway-cache"
RUNWAY_CELL_CACHE_SCHEMA = 2
_INSTALLED = False


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


def _geometry_fingerprint(geometry) -> dict[str, object]:
    return {
        "osm_key": str(getattr(geometry, "osm_key", "")),
        "start": [round(float(geometry.start_x), 6), round(float(geometry.start_z), 6)],
        "end": [round(float(geometry.end_x), 6), round(float(geometry.end_z), 6)],
        "length": round(float(geometry.length), 6),
        "half_width": round(float(geometry.half_width), 6),
    }


def _cache_path_for_cell(
    cache_dir: Path,
    *,
    geometries: Sequence[object],
    cell_index: int,
    original_texture_index: int,
    spec,
) -> Path:
    """Return a key that changes whenever this cell's rendered pixels can change."""
    payload = {
        "schema": RUNWAY_CELL_CACHE_SCHEMA,
        "renderer": "runway-world-aligned-cell-v2",
        "texture_size": int(_runway.RUNWAY_TEXTURE_SIZE),
        "profile": str(getattr(spec, "ground_texture_profile", "generated")),
        "seed": str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
        "world": str(getattr(spec, "name", "")),
        "cells": int(getattr(spec, "cells", 0)),
        "cell_size": float(getattr(spec, "cell_size", 0.0)),
        "cell_index": int(cell_index),
        "original_texture_index": int(original_texture_index),
        "geometries": [_geometry_fingerprint(value) for value in geometries],
        "background": _exact_background_fingerprint(spec, original_texture_index),
    }
    key = cache_key("runway-ground-texture-cell-v2", payload)
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


def apply_cached_runway_cell_textures(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Restore/render correctly aligned runway cells and deduplicate final PAAs."""
    from . import generator

    touched = _runway.runway_texture_cell_indices(dataset, projection, spec)
    if not touched:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    if len(texture_indices) != int(spec.cells) * int(spec.cells):
        raise ValueError("runway texture generation received a mismatched WRP grid")
    # The base runway policy deliberately uses this conservative bound to choose
    # P3D fallback before writing the WRP. Preserve that safety contract here.
    if len(texture_paths) + len(touched) > _runway.RVW4_TEXTURE_LIMIT:
        raise ValueError("generated runway textures exceed the RVW4 512-entry texture table")

    geometries = _runway._runway_geometries(
        dataset, projection, getattr(spec, "ground_texture_profile", "generated")
    )
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    source_dir = Path(source_dir)
    for pattern in (
        f"{_runway.RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa",
        f"{_runway.RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f][zdk].paa",
        f".{_runway.RUNWAY_TEXTURE_PREFIX}cell*.paa",
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
    background_cache: dict[tuple[int, str], object] = {}
    # PAA digest -> (full bytes, WRP slot). Compare bytes too so a theoretical
    # hash collision cannot turn the runway into abstract art.
    reusable: dict[bytes, tuple[bytes, int]] = {}
    cache_hits = 0
    cache_misses = 0
    reused_cells = 0

    for cell_index in map(int, touched):
        original_index = int(texture_indices[cell_index])
        cache_path = _cache_path_for_cell(
            runway_cache_dir,
            geometries=geometries,
            cell_index=cell_index,
            original_texture_index=original_index,
            spec=spec,
        )
        candidate = source_dir / f".{_runway.RUNWAY_TEXTURE_PREFIX}cell{cell_index:06x}.paa"

        def paint(destination: Path, *, cell=cell_index, wrp_index=original_index):
            image = _runway._render_runway_cell(
                cell_index=cell,
                original_wrp_texture_index=wrp_index,
                geometries=geometries,
                materials=materials,
                spec=spec,
                size=_runway.RUNWAY_TEXTURE_SIZE,
                background_cache=background_cache,
            )
            _runway.write_rgb_dxt1_paa(destination, image)

        hit = restore_or_create_file(
            cache_path=cache_path,
            destination=candidate,
            producer=paint,
            enabled=cache_enabled,
            refresh=cache_refresh,
        )
        cache_hits += int(hit)
        cache_misses += int(not hit)
        data = candidate.read_bytes()
        digest = sha256(data).digest()
        prior = reusable.get(digest)
        if prior is not None and prior[0] == data:
            revised_indices[cell_index] = prior[1]
            reused_cells += 1
            candidate.unlink(missing_ok=True)
            continue

        slot = len(revised_paths)
        if slot >= _runway.RVW4_TEXTURE_LIMIT:
            candidate.unlink(missing_ok=True)
            raise ValueError("deduplicated runway textures exceed the RVW4 texture table")
        filename = f"{_runway.RUNWAY_TEXTURE_PREFIX}{len(generated_paths):03x}.paa"
        final_path = source_dir / filename
        candidate.replace(final_path)
        wire_path = rf"{spec.name}\{filename}"
        revised_paths.append(wire_path)
        revised_indices[cell_index] = slot
        generated_paths.append(wire_path)
        reusable[digest] = (data, slot)

    _sync_exact_assignment_counts(touched, texture_indices, spec)
    report_path.write_text(json.dumps({
        "schema": 4,
        "mode": "generated-terrain-textures",
        "strategy": "per-cell-content-addressed-cache",
        "texture_limit": _runway.RVW4_TEXTURE_LIMIT,
        "texture_size": _runway.RUNWAY_TEXTURE_SIZE,
        "base_texture_entries": len(texture_paths),
        "runway_cells": len(touched),
        "runway_count": len(geometries),
        "generated_runway_textures": len(generated_paths),
        "reused_runway_cell_assignments": reused_cells,
        "reuse_ratio": (reused_cells / len(touched)) if touched else 0.0,
        "final_texture_entries": len(revised_paths),
        "runway_cell_indices": list(map(int, touched)),
        "texture_paths": generated_paths,
        "cache_directory": str(runway_cache_dir),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_entries_considered": len(touched),
        "stock_reference_family": {
            "grass": list(_runway.runway_texture_triplet("grass")),
            "desert": list(_runway.runway_texture_triplet("desert")),
        },
        "nogova_background_mode": "selected-stock-path-colour-match",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


# Compatibility name retained for callers/tests from the short-lived triplet
# implementation. Its behavior is now the safe per-cell cache above.
apply_cached_runway_texture_triplets = apply_cached_runway_cell_textures


def install_runway_texture_cache_policy() -> None:
    """Install persistent per-cell runway caching after exact backgrounds."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import runway_exact_background_policy as exact_policy

    if getattr(exact_policy, "_INSTALLED", False):
        # Preserve the exact-background wrapper. It calls this implementation
        # while its build-scoped source-texture context is active.
        exact_policy._ORIGINAL_APPLY_RUNWAY_TEXTURES = apply_cached_runway_cell_textures
    else:
        _runway.apply_generated_runway_texture_table = apply_cached_runway_cell_textures

    # Do not replace fit_road_objects here. The base runway surface policy budgets
    # the worst-case one texture per touched cell and selects P3D fallback before
    # the WRP is written. That conservative budget is what keeps this correct.
    _INSTALLED = True
