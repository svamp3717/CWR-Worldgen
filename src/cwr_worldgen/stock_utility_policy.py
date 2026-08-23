# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from typing import Callable

from . import asset_mapping as _asset_mapping
from . import generator as _generator
from . import osm as _osm
from . import procedural_infrastructure as _procedural_infrastructure

STOCK_POWER_POLE_MODELS: tuple[str, ...] = (
    r"data3d\sloupyelA.p3d",
    r"data3d\sloupyell.p3d",
)
STOCK_POWER_TOWER_MODELS: tuple[str, ...] = (
    r"O\Hous\stozarvn_1.p3d",
)

_GENERATED_POWER_POLE_SUFFIX = r"\i\util_power_pole.p3d"
_GENERATED_POWER_TOWER_SUFFIX = r"\i\util_power_tower.p3d"
_POLE_FOOTPRINT_METRES = 3.0
_TOWER_FOOTPRINT_METRES = 9.0
_POLE_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES = 7.0
_TOWER_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES = 10.0
_MAPPED_TOWER_MATCH_DISTANCE_METRES = 1.0

_INSTALLED = False


def _stock_pole_model(object_id: int) -> str:
    # Stable and cheap. The WRP object id is deterministic for a given placement plan.
    return STOCK_POWER_POLE_MODELS[int(object_id) % len(STOCK_POWER_POLE_MODELS)]


def _stock_utility_model(original):
    """Return original-game pole/tower models before procedural usage is registered."""

    def wrapped(self, subtype: str) -> str:
        kind = str(subtype).casefold()
        if kind == "power_pole":
            return STOCK_POWER_POLE_MODELS[0]
        if kind == "power_tower":
            return STOCK_POWER_TOWER_MODELS[0]
        return original(self, subtype)

    wrapped._cwr_stock_utility_policy = True  # type: ignore[attr-defined]
    return wrapped


def _utility_kind(model_path: str) -> str:
    folded = model_path.casefold()
    if folded.endswith(_GENERATED_POWER_POLE_SUFFIX):
        return "mapped_power_pole"
    if folded.endswith(_GENERATED_POWER_TOWER_SUFFIX):
        return "mapped_power_tower"
    if folded in {path.casefold() for path in STOCK_POWER_POLE_MODELS}:
        return "stock_power_pole"
    if folded in {path.casefold() for path in STOCK_POWER_TOWER_MODELS}:
        # This stock mast may be an explicit mapped tower or historical settlement
        # clutter. The dataset-aware rewrite below disambiguates those cases.
        return "stock_power_tower"
    return ""


def _candidate_is_free(
    raster,
    corridors,
    spec,
    x: float,
    z: float,
    footprint: float,
) -> bool:
    if not (0.0 <= x < spec.world_size and 0.0 <= z < spec.world_size):
        return False
    if _osm._mask_at(raster.water, spec.cells, spec.world_size, x, z):
        return False
    if _osm._mask_at(raster.buildings, spec.cells, spec.world_size, x, z):
        return False
    return not _osm.forest_block_intersects_road_corridors(
        corridors, x, z, block_size=footprint
    )


def _road_safe_position(
    dataset,
    projection,
    raster,
    elevations,
    spec,
    corridors,
    x: float,
    z: float,
    *,
    footprint: float,
    target_distance: float,
) -> tuple[float, float, float] | None:
    if not _osm.forest_block_intersects_road_corridors(
        corridors, x, z, block_size=footprint
    ):
        _minimum, maximum = _osm._square_elevation_extrema(
            elevations, spec.cells, spec.cell_size, x, z, footprint
        )
        return x, z, maximum + max(0.0, float(getattr(spec, "utility_ground_clearance", 0.05)))

    road_point = _osm.nearest_road_point(dataset, projection, x, z)
    if road_point is None:
        return None
    rx, rz = road_point
    dx, dz = x - rx, z - rz
    length = math.hypot(dx, dz)
    if length > 1.0e-6:
        ux, uz = dx / length, dz / length
    else:
        heading = _osm.nearest_road_heading(dataset, projection, x, z)
        ux, uz = _osm._heading_right_vector(heading)

    # Try both sides of the road and progressively larger setbacks. The first
    # valid position wins, keeping poles roadside without letting their model
    # footprint intrude into the carriageway.
    distances = (
        target_distance,
        target_distance + 2.0,
        target_distance + 4.0,
        target_distance + 7.0,
        target_distance + 11.0,
        target_distance + 16.0,
    )
    for distance in distances:
        for sign in (1.0, -1.0):
            cx = rx + ux * distance * sign
            cz = rz + uz * distance * sign
            if not _candidate_is_free(raster, corridors, spec, cx, cz, footprint):
                continue
            _minimum, maximum = _osm._square_elevation_extrema(
                elevations, spec.cells, spec.cell_size, cx, cz, footprint
            )
            y = maximum + max(0.0, float(getattr(spec, "utility_ground_clearance", 0.05)))
            return cx, cz, y
    return None


def _mapped_power_tower_positions(dataset, projection) -> tuple[tuple[float, float], ...]:
    return tuple(
        projection.to_world(feature.point)
        for feature in getattr(dataset, "utility_points", ())
        if feature.tags.get("utility", "").casefold() == "power_tower"
    )


def _matches_mapped_tower(x: float, z: float, positions: tuple[tuple[float, float], ...]) -> bool:
    limit2 = _MAPPED_TOWER_MATCH_DISTANCE_METRES ** 2
    return any((x - tx) ** 2 + (z - tz) ** 2 <= limit2 for tx, tz in positions)


def _rewrite_stock_utilities(result, dataset, projection, raster, elevations, spec, progress: Callable[[int, str], None] | None = None):
    if not getattr(result, "objects", ()):
        return result

    corridors = _osm._project_vehicle_road_corridors(dataset, projection)
    mapped_tower_positions = _mapped_power_tower_positions(dataset, projection)
    rewritten = []
    changed = 0
    unresolved = 0

    for obj in result.objects:
        kind = _utility_kind(obj.model_path)
        if not kind:
            rewritten.append(obj)
            continue

        if kind == "stock_power_tower":
            kind = (
                "mapped_power_tower"
                if _matches_mapped_tower(obj.x, obj.z, mapped_tower_positions)
                else "settlement_power_tower"
            )

        if kind == "mapped_power_tower":
            model = STOCK_POWER_TOWER_MODELS[0]
            footprint = _TOWER_FOOTPRINT_METRES
            target_distance = _TOWER_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES
        else:
            # Legacy generated power poles, ordinary stock poles, and the oversized
            # mast accidentally used as settlement clutter all normalize to a normal
            # original-game pole family.
            model = _stock_pole_model(obj.object_id)
            footprint = _POLE_FOOTPRINT_METRES
            target_distance = _POLE_TARGET_CLEARANCE_FROM_ROAD_CENTRE_METRES

        safe = _road_safe_position(
            dataset, projection, raster, elevations, spec, corridors,
            obj.x, obj.z,
            footprint=footprint,
            target_distance=target_distance,
        )
        if safe is None:
            # Keep the source placement rather than silently deleting an OSM
            # utility object, but surface the rare impossible placement loudly.
            unresolved += 1
            new_obj = replace(obj, model_path=model)
        else:
            sx, sz, sy = safe
            new_obj = replace(obj, model_path=model, x=sx, z=sz, y=sy)

        changed += int(new_obj != obj)
        rewritten.append(new_obj)

    if not changed:
        return result

    model_usage = Counter(obj.model_path for obj in rewritten)
    if unresolved and progress is not None:
        progress(
            66,
            f"WARNING: {unresolved:,} stock utility pole/tower placements could not be moved fully clear of roads/buildings; kept at their source coordinates.",
        )
    return replace(
        result,
        objects=tuple(rewritten),
        model_usage=tuple(sorted(model_usage.items(), key=lambda item: item[0].casefold())),
    )


def _stock_default_osm_asset_mapping(original):
    def wrapped(spec, milestone_number: int, *, global_textures=()):
        mapping = original(spec, milestone_number, global_textures=global_textures)
        replacement_models = (*STOCK_POWER_POLE_MODELS, *STOCK_POWER_TOWER_MODELS)
        rules = tuple(
            replace(
                rule,
                models=replacement_models,
                description="Stock OFP/CWA power pole and high-voltage tower models",
            )
            if rule.rule_id == "osm-power-utilities"
            else rule
            for rule in mapping.rules
        )
        return replace(mapping, rules=rules)

    wrapped._cwr_stock_utility_policy = True  # type: ignore[attr-defined]
    return wrapped


def install_stock_utility_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_utility_model = _procedural_infrastructure.ProceduralInfrastructureLibrary.utility_model
    if not getattr(original_utility_model, "_cwr_stock_utility_policy", False):
        _procedural_infrastructure.ProceduralInfrastructureLibrary.utility_model = _stock_utility_model(
            original_utility_model
        )

    original_loader = _generator._load_nonroad_objects

    def load_nonroad_objects(*args, **kwargs):
        value = original_loader(*args, **kwargs)
        result, cached_library, hit, key, path = value
        bound = {
            "dataset": args[0] if len(args) > 0 else kwargs["dataset"],
            "projection": args[1] if len(args) > 1 else kwargs["projection"],
            "raster": args[2] if len(args) > 2 else kwargs["raster"],
            "elevations": args[3] if len(args) > 3 else kwargs["elevations"],
            "spec": args[4] if len(args) > 4 else kwargs["spec"],
        }
        progress = kwargs.get("progress_callback")
        result = _rewrite_stock_utilities(result, progress=progress, **bound)
        return result, cached_library, hit, key, path

    load_nonroad_objects._cwr_stock_utility_policy = True  # type: ignore[attr-defined]
    _generator._load_nonroad_objects = load_nonroad_objects

    original_mapping = _asset_mapping.default_osm_asset_mapping
    wrapped_mapping = _stock_default_osm_asset_mapping(original_mapping)
    _asset_mapping.default_osm_asset_mapping = wrapped_mapping
    _generator.default_osm_asset_mapping = wrapped_mapping

    _INSTALLED = True
