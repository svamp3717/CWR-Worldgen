# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound and parallelize expensive modeler-backed building asset generation.

The OSM House Modeler integration carries much richer architectural identity than
the legacy CWR palette system.  That fidelity is useful, but the old placement
pipeline could fan a nominal 128 reusable variants back out into hundreds of
final P3Ds through cosmetic texture variation, foundation buckets and placement
state.  Texture cache misses were also rendered serially before the P3D worker
pool started.

This policy keeps country/material/opening/detail identity intact while removing
that accidental multiplication:

* one cosmetic texture/weather variant is used by default; country/material/style
  choices already provide substantial visual variation,
* exact polygon-native buildings retain a separate bounded budget,
* the normal variant budget is enforced again on final registered rectangular
  P3Ds using the existing family/physical-fit reuse machinery,
* missing modeler PAA cache entries are rendered through the existing bounded
  process pool before the serial writer performs cheap cache restores, and
* progress reports expose texture/P3D cache misses instead of one opaque label.

Power users can raise the cosmetic/final/polygon budgets with environment
variables documented below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_STANDARD_VARIANTS = 128
_DEFAULT_POLYGON_DIVISOR = 2
_DEFAULT_POLYGON_FLOOR = 16
_DEFAULT_POLYGON_CEILING = 96
_DEFAULT_BUILDING_PARALLEL_MINIMUM = 16
_DEFAULT_MODELER_TEXTURE_VARIANTS = 1
_TEXTURE_PARALLEL_MINIMUM = 4
_ENV_POLYGON_VARIANTS = "CWR_WORLDGEN_POLYGON_BUILDING_VARIANTS"
_ENV_FINAL_VARIANTS = "CWR_WORLDGEN_FINAL_BUILDING_VARIANTS"
_ENV_TEXTURE_VARIANTS = "CWR_WORLDGEN_BUILDING_TEXTURE_VARIANTS"
_INSTALLED = False


def _integer_override(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(minimum, min(maximum, int(default)))
    try:
        value = int(raw)
    except ValueError:
        value = int(default)
    return max(minimum, min(maximum, value))


def _polygon_variant_budget(maximum_variants: int) -> int:
    """Return the default exact-polygon budget for one building library."""
    override = os.environ.get(_ENV_POLYGON_VARIANTS, "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = -1
        if requested >= 0:
            return requested

    standard = max(1, int(maximum_variants))
    return max(
        _DEFAULT_POLYGON_FLOOR,
        min(_DEFAULT_POLYGON_CEILING, standard // _DEFAULT_POLYGON_DIVISOR),
    )


def _final_variant_budget(maximum_variants: int) -> int:
    return _integer_override(
        _ENV_FINAL_VARIANTS,
        max(1, int(maximum_variants)),
        minimum=1,
        maximum=4096,
    )


def _texture_variant_budget(requested: int) -> int:
    del requested
    return _integer_override(
        _ENV_TEXTURE_VARIANTS,
        _DEFAULT_MODELER_TEXTURE_VARIANTS,
        minimum=1,
        maximum=10,
    )


@dataclass(frozen=True, slots=True)
class _TextureCacheTask:
    cache_path: Path
    kind: str
    size: int
    family: str = ""
    style_token: str = "default"
    texture_variant: int = 0
    outbuilding_kind: str = ""
    roof_token: str = ""
    cache_enabled: bool = True
    cache_refresh: bool = False


def _write_modeler_texture_cache_task(task: _TextureCacheTask) -> str:
    """Render one modeler PAA directly into its persistent cache entry."""
    if task.cache_path.is_file() and not task.cache_refresh:
        return str(task.cache_path)

    from .osm_house_modeler_texture_bridge import (
        modeler_door_texture_image,
        modeler_foundation_texture_image,
        modeler_front_texture_image,
        modeler_interior_wall_texture_image,
        modeler_open_wall_texture_image,
        modeler_roof_texture_image,
        modeler_wall_texture_image,
        modeler_window_frame_texture_image,
    )
    from .paa import write_rgb_dxt1_paa

    if task.kind == "wall":
        image = modeler_wall_texture_image(
            task.family, size=task.size,
            regional_style=task.style_token,
            texture_variant=task.texture_variant,
        )
    elif task.kind == "open_wall":
        image = modeler_open_wall_texture_image(
            task.family, size=task.size,
            regional_style=task.style_token,
            texture_variant=task.texture_variant,
        )
    elif task.kind == "interior_wall":
        image = modeler_interior_wall_texture_image(
            task.family, size=task.size,
            regional_style=task.style_token,
            texture_variant=task.texture_variant,
        )
    elif task.kind == "front":
        image = modeler_front_texture_image(
            task.family, size=task.size,
            regional_style=task.style_token,
            texture_variant=task.texture_variant,
            outbuilding_kind=task.outbuilding_kind,
        )
    elif task.kind == "door":
        image = modeler_door_texture_image(
            task.size,
            family=task.family,
            regional_style=task.style_token,
            texture_variant=task.texture_variant,
            outbuilding_kind=task.outbuilding_kind,
        )
    elif task.kind == "roof":
        image = modeler_roof_texture_image(
            task.roof_token,
            size=task.size,
            texture_variant=task.texture_variant,
        )
    elif task.kind == "foundation":
        image = modeler_foundation_texture_image(task.size)
    elif task.kind == "window_frame":
        image = modeler_window_frame_texture_image(task.size)
    else:  # pragma: no cover - programmer error guarded by task construction
        raise ValueError(f"unknown modeler texture cache task {task.kind!r}")

    task.cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.cache_path.with_name(
        f"{task.cache_path.stem}.{os.getpid()}.tmp{task.cache_path.suffix}"
    )
    try:
        write_rgb_dxt1_paa(temporary, image)
        os.replace(temporary, task.cache_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return str(task.cache_path)


def _texture_cache_tasks(library, buildings) -> tuple[int, list[_TextureCacheTask]]:
    """Return total referenced modeler cache entries and the current misses."""
    if not library.cache_enabled or library.cache_dir is None or library.cache_refresh:
        return 0, []

    selected = sorted(library._usage)
    size = int(library.texture_size)
    entries: dict[Path, _TextureCacheTask] = {}

    def add(namespace: str, payload: Any, kind: str, **kwargs: Any) -> None:
        key = buildings.cache_key(namespace, payload)
        path = library.cache_dir / "procedural-assets" / f"{key}.paa"
        entries.setdefault(
            path,
            _TextureCacheTask(
                cache_path=path,
                kind=kind,
                size=size,
                **kwargs,
            ),
        )

    if any(key.foundation_depth_m > 0.0 for key in selected):
        add(
            "procedural-building-foundation-modeler-v1-cwa78",
            {"texture": "stone", "texture_size": size},
            "foundation",
        )

    uses_light_window_trim = any(
        key.interiors
        and key.family not in buildings.UTILITY_INTERIOR_FAMILIES
        and buildings._uses_light_window_trim(key)
        for key in selected
    )
    if uses_light_window_trim:
        add(
            "procedural-building-window-frame-modeler-v1-cwa84",
            {"material": "weathered-white-painted-wood", "texture_size": size},
            "window_frame",
        )

    used_palette_variants = {
        (key.family, key.texture_style_token or key.regional_style, key.texture_variant)
        for key in selected
    }
    for family, token, variant in used_palette_variants:
        add(
            "procedural-building-wall-modeler-v1-cwa78",
            {
                "family": family,
                "regional_style": token,
                "texture_variant": variant,
                "texture_size": size,
            },
            "wall",
            family=family,
            style_token=token,
            texture_variant=variant,
        )

    used_open_variants = {
        (key.family, key.texture_style_token or key.regional_style, key.texture_variant)
        for key in selected
        if key.interiors
        or key.family == "church"
        or key.family in buildings._GROUND_FLOOR_ONLY_FACADE_FAMILIES
        or key.family in buildings._PAINTED_WINDOW_FAMILIES
    }
    for family, token, variant in used_open_variants:
        add(
            "procedural-building-open-wall-modeler-v1-cwa78",
            {
                "family": family,
                "regional_style": token,
                "texture_variant": variant,
                "texture_size": size,
            },
            "open_wall",
            family=family,
            style_token=token,
            texture_variant=variant,
        )

    used_interior_variants = {
        (key.family, key.texture_style_token or key.regional_style, key.texture_variant)
        for key in selected if key.interiors
    }
    for family, token, variant in used_interior_variants:
        add(
            "procedural-building-interior-wall-modeler-v1-cwa58",
            {
                "family": family,
                "regional_style": token,
                "texture_variant": variant,
                "texture_size": size,
            },
            "interior_wall",
            family=family,
            style_token=token,
            texture_variant=variant,
        )

    used_front_variants = {
        (
            key.family,
            key.texture_style_token or key.regional_style,
            key.texture_variant,
            key.outbuilding_kind,
        )
        for key in selected
    }
    for family, token, variant, outbuilding_kind in used_front_variants:
        add(
            "procedural-building-front-modeler-v1-cwa78",
            {
                "family": family,
                "regional_style": token,
                "texture_variant": variant,
                "outbuilding_kind": outbuilding_kind,
                "texture_size": size,
            },
            "front",
            family=family,
            style_token=token,
            texture_variant=variant,
            outbuilding_kind=outbuilding_kind,
        )

    used_doors: set[tuple[str, str, int, str]] = set()
    for key in selected:
        if not key.interiors and not key.footprint_vertices:
            continue
        used_doors.add((
            key.family,
            key.texture_style_token or key.regional_style,
            key.texture_variant,
            key.outbuilding_kind,
        ))
    for family, token, variant, outbuilding_kind in used_doors:
        add(
            "procedural-building-door-modeler-v1-cwa84",
            {
                "family": family,
                "regional_style": token,
                "texture_variant": variant,
                "outbuilding_kind": outbuilding_kind,
                "texture_size": size,
            },
            "door",
            family=family,
            style_token=token,
            texture_variant=variant,
            outbuilding_kind=outbuilding_kind,
        )

    used_roofs = {
        (
            f"{key.roof_style}|{key.roof_material}|{','.join(key.colour_palette[:4])}",
            key.texture_variant,
        )
        for key in selected
    }
    for roof, variant in used_roofs:
        add(
            "procedural-building-roof-modeler-v1-cwa78",
            {"roof": roof, "texture_variant": variant, "texture_size": size},
            "roof",
            roof_token=roof,
            texture_variant=variant,
        )

    misses = [task for path, task in entries.items() if not path.is_file()]
    return len(entries), misses


def _model_cache_counts(library, buildings) -> tuple[int, int]:
    hits = misses = 0
    for key in library._usage:
        asset_key = buildings.cache_key(
            "procedural-building-model-v49-robust-polygon-roof-triangulation",
            {
                "world_name": library.world_name,
                "variant": asdict(key),
                "roof_pitch_degrees": (key.roof_pitch_degrees or library.roof_pitch_degrees),
                "foundation_depth": key.foundation_depth_m,
                "church_plinth_height": library.church_plinth_height,
            },
        )
        cached = (
            library.cache_dir / "procedural-assets" / f"{asset_key}.p3d"
            if library.cache_dir is not None else None
        )
        is_hit = (
            library.cache_enabled
            and not library.cache_refresh
            and cached is not None
            and cached.is_file()
        )
        hits += int(is_hit)
        misses += int(not is_hit)
    return hits, misses


def _safe_final_reuse(library, requested, candidates: Iterable[Any]):
    """Return an existing final P3D key only when engine semantics remain safe."""
    pool = [candidate for candidate in candidates if not candidate.footprint_vertices]
    same_mode = [candidate for candidate in pool if candidate.interiors == requested.interiors]
    if not same_mode:
        return None
    pool = same_mode

    compatible = set(library._compatible_families(requested.family))
    compatible_pool = [candidate for candidate in pool if candidate.family in compatible]
    if not compatible_pool:
        return None
    pool = compatible_pool

    if requested.family in {"church", "school", "shop"}:
        same_family = [candidate for candidate in pool if candidate.family == requested.family]
        if not same_family:
            return None
        pool = same_family
    elif requested.family == "outbuilding" and requested.outbuilding_kind:
        same_kind = [
            candidate for candidate in pool
            if candidate.family == "outbuilding"
            and candidate.outbuilding_kind == requested.outbuilding_kind
        ]
        if not same_kind:
            return None
        pool = same_kind

    if requested.interiors:
        same_storey = [
            candidate for candidate in pool
            if candidate.second_storey == requested.second_storey
        ]
        if not same_storey:
            return None
        pool = same_storey
        same_foundation = [
            candidate for candidate in pool
            if abs(candidate.foundation_depth_m - requested.foundation_depth_m) <= 1.0e-6
        ]
        if not same_foundation:
            return None
        pool = same_foundation
        # Window/door dimensions are actual holes in enterable geometry. Do not
        # save time by replacing them with a different architectural opening set.
        same_openings = [
            candidate for candidate in pool
            if candidate.texture_style_token == requested.texture_style_token
        ]
        if not same_openings:
            return None
        pool = same_openings
    else:
        # A deeper exterior skirt is safe; a shallower one can expose terrain gaps.
        safe_foundation = [
            candidate for candidate in pool
            if candidate.foundation_depth_m + 1.0e-6 >= requested.foundation_depth_m
        ]
        if not safe_foundation:
            return None
        pool = safe_foundation
        same_style = [
            candidate for candidate in pool
            if candidate.texture_style_token == requested.texture_style_token
        ]
        if same_style:
            pool = same_style

    candidates = library._reuse_candidates(requested, pool)
    return library._best_variant(requested, candidates) if candidates else None


def install_building_asset_budget_policy() -> None:
    """Install final budgeting, texture prewarming and earlier building parallelism."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import parallel_assets
    from . import procedural_buildings as buildings

    original_init = buildings.ProceduralBuildingLibrary.__init__
    original_register = buildings.ProceduralBuildingLibrary.register_placement
    original_write_assets = buildings.ProceduralBuildingLibrary.write_assets

    def budgeted_init(self, *args, **kwargs):
        if "maximum_polygon_variants" not in kwargs:
            kwargs["maximum_polygon_variants"] = _polygon_variant_budget(
                int(kwargs.get("maximum_variants", _DEFAULT_STANDARD_VARIANTS))
            )
        kwargs["texture_variants"] = _texture_variant_budget(
            int(kwargs.get("texture_variants", buildings.DEFAULT_BUILDING_TEXTURE_VARIANTS))
        )
        result = original_init(self, *args, **kwargs)
        self._cwr_final_standard_variant_budget = _final_variant_budget(self.maximum_variants)
        return result

    def register_with_final_budget(self, placement, *, foundation_depth_m=None):
        registered = original_register(
            self, placement, foundation_depth_m=foundation_depth_m
        )
        selected = registered.selected
        if selected.footprint_vertices:
            return registered
        if self._usage[selected] > 1:
            return registered

        budget = int(getattr(
            self, "_cwr_final_standard_variant_budget", self.maximum_variants
        ))
        standard_keys = [key for key in self._usage if not key.footprint_vertices]
        if len(standard_keys) <= budget:
            return registered

        # The original registration has already counted this one new key. Undo
        # it while looking for a safe existing final model, then either reuse or
        # restore the new key if semantic/opening/foundation constraints require it.
        self._usage[selected] -= 1
        if self._usage[selected] <= 0:
            del self._usage[selected]
        replacement = _safe_final_reuse(self, selected, tuple(self._usage))
        if replacement is None:
            self._usage[selected] += 1
            return registered

        self._usage[replacement] += 1
        return buildings.BuildingPlacement(
            self.model_path(replacement),
            registered.heading_degrees,
            registered.requested,
            replacement,
        )

    def write_assets_with_budget_progress(self, source_dir, catalogue_path):
        total = len(getattr(self, "_usage", {}))
        polygons = sum(
            bool(getattr(key, "footprint_vertices", ()))
            for key in getattr(self, "_usage", {})
        )
        interiors = sum(
            bool(getattr(key, "interiors", False))
            for key in getattr(self, "_usage", {})
        )
        texture_total, texture_misses = _texture_cache_tasks(self, buildings)
        model_hits, model_misses = _model_cache_counts(self, buildings)

        try:
            from .progress import report_progress
            report_progress(
                73,
                "Generating procedural building assets "
                f"({total} P3Ds: {model_misses} misses/{model_hits} hits; "
                f"{interiors} interiors, {polygons} exact; "
                f"{len(texture_misses)} texture misses/{texture_total} refs)",
            )
        except Exception:
            pass

        if texture_misses:
            parallel_assets.process_asset_tasks(
                _write_modeler_texture_cache_task,
                texture_misses,
                minimum_parallel_tasks=_TEXTURE_PARALLEL_MINIMUM,
            )
            try:
                from .progress import report_progress
                report_progress(
                    74,
                    f"Generating procedural building P3Ds ({model_misses} cache misses, {model_hits} hits)",
                )
            except Exception:
                pass

        return original_write_assets(self, source_dir, catalogue_path)

    buildings.ProceduralBuildingLibrary.__init__ = budgeted_init
    buildings.ProceduralBuildingLibrary.register_placement = register_with_final_budget
    buildings.ProceduralBuildingLibrary.write_assets = write_assets_with_budget_progress

    parallel_assets._BUILDING_PARALLEL_MINIMUM = min(
        int(getattr(parallel_assets, "_BUILDING_PARALLEL_MINIMUM", 64)),
        _DEFAULT_BUILDING_PARALLEL_MINIMUM,
    )

    _INSTALLED = True
