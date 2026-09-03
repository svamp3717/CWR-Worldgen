# SPDX-License-Identifier: GPL-3.0-or-later
"""Performance policy for modeler-backed enterable procedural buildings.

This module is intentionally output-preserving.  It memoizes deterministic
layout helpers within each generator process, starts the existing worker pool for
smaller batches of expensive interior P3Ds, and keeps the 20 m distance LOD free
of close-up architectural attachments that are already present on the detail LOD.
"""
from __future__ import annotations

from functools import lru_cache
import threading
from typing import Sequence

_INTERIOR_PARALLEL_MINIMUM = 8
_INSTALLED = False
_WRITE_STATE = threading.local()


def _worker_identity(worker) -> tuple[str, str]:
    return (
        str(getattr(worker, "__module__", "")),
        str(getattr(worker, "__name__", "")),
    )


def install_interior_performance_policy() -> None:
    """Install deterministic memoization and interior-specific scheduling."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import osm_house_modeler_upgrade as upgrade
    from . import procedural_buildings as buildings

    # The existing scheduler deliberately keeps cache hits in the parent process.
    # Only lower the subprocess threshold when at least one actual building miss
    # can be an enterable model; forests/roads/sites retain their old policy.
    original_process_asset_tasks = buildings.process_asset_tasks

    def process_interior_assets_earlier(
        worker,
        tasks,
        *,
        minimum_parallel_tasks: int = 768,
    ):
        task_list = list(tasks)
        identity = _worker_identity(worker)
        is_building_worker = identity == (
            "cwr_worldgen.procedural_buildings",
            "_write_building_asset_task",
        )
        has_interior = is_building_worker and any(
            bool(getattr(getattr(task, "key", None), "interiors", False))
            for task in task_list
        )
        threshold = (
            min(int(minimum_parallel_tasks), _INTERIOR_PARALLEL_MINIMUM)
            if has_interior
            else int(minimum_parallel_tasks)
        )
        return original_process_asset_tasks(
            worker,
            task_list,
            minimum_parallel_tasks=threshold,
        )

    buildings.process_asset_tasks = process_interior_assets_earlier

    # These helpers are pure functions of the immutable BuildingVariantKey and
    # their scalar arguments. One enterable P3D asks for several of them from
    # Visual, Geometry, Roadway, Memory and Paths LOD construction. Cache them
    # per worker instead of decoding/recomputing the same architecture each time.
    buildings._door_dimensions = lru_cache(maxsize=4096)(buildings._door_dimensions)
    buildings._interior_wall_thickness = lru_cache(maxsize=4096)(
        buildings._interior_wall_thickness
    )
    buildings._second_storey_layout = lru_cache(maxsize=4096)(
        buildings._second_storey_layout
    )
    buildings._interior_storey_count = lru_cache(maxsize=8192)(
        buildings._interior_storey_count
    )
    buildings._visible_window_storey_count = lru_cache(maxsize=8192)(
        buildings._visible_window_storey_count
    )
    buildings._interior_stair_profile = lru_cache(maxsize=4096)(
        buildings._interior_stair_profile
    )
    buildings._interior_vehicle_ramp_profile = lru_cache(maxsize=4096)(
        buildings._interior_vehicle_ramp_profile
    )

    original_window_openings = buildings._interior_window_openings

    @lru_cache(maxsize=16384)
    def cached_window_openings(
        key,
        horizontal_min: float,
        horizontal_max: float,
        wall_top: float,
        exclusions: tuple[tuple[float, float], ...],
    ):
        return tuple(original_window_openings(
            key,
            horizontal_min,
            horizontal_max,
            wall_top,
            ground_exclusions=exclusions,
        ))

    def memoized_window_openings(
        key,
        horizontal_min: float,
        horizontal_max: float,
        wall_top: float,
        *,
        ground_exclusions: Sequence[tuple[float, float]] = (),
    ):
        exclusions = tuple(
            (float(start), float(end)) for start, end in ground_exclusions
        )
        return cached_window_openings(
            key,
            float(horizontal_min),
            float(horizontal_max),
            float(wall_top),
            exclusions,
        )

    buildings._interior_window_openings = memoized_window_openings

    # Shapely construction/validation is expensive and polygon-native interiors
    # query the same footprint from visual, collision, roadway and path builders.
    buildings._polygon_native_shape = lru_cache(maxsize=1024)(
        buildings._polygon_native_shape
    )
    buildings._polygon_native_roof_mesh = lru_cache(maxsize=1024)(
        buildings._polygon_native_roof_mesh
    )

    # write_building_mlod builds a close detail shell and then a second closed
    # shell for the 20 m distance LOD. The runtime modeler adapter normally adds
    # porches/chimneys/balconies/rainwater details to every visual call. Suppress
    # that append only on the second top-level visual call; the inexpensive base
    # facade/roof (including exact door/window scaling) remains intact.
    original_write_building_mlod = buildings.write_building_mlod
    original_visual_lod = buildings._visual_lod
    original_polygon_visual_lod = buildings._polygon_native_visual_lod

    def write_building_mlod_with_state(*args, **kwargs):
        previous_active = bool(getattr(_WRITE_STATE, "active", False))
        previous_visual_calls = int(getattr(_WRITE_STATE, "visual_calls", 0))
        previous_polygon_calls = int(getattr(_WRITE_STATE, "polygon_calls", 0))
        _WRITE_STATE.active = True
        _WRITE_STATE.visual_calls = 0
        _WRITE_STATE.polygon_calls = 0
        try:
            return original_write_building_mlod(*args, **kwargs)
        finally:
            _WRITE_STATE.active = previous_active
            _WRITE_STATE.visual_calls = previous_visual_calls
            _WRITE_STATE.polygon_calls = previous_polygon_calls

    def compact_distance_visual(*args, **kwargs):
        depth = int(getattr(upgrade._CALL_STATE, "depth", 0))
        if bool(getattr(_WRITE_STATE, "active", False)) and depth == 0:
            _WRITE_STATE.visual_calls = int(
                getattr(_WRITE_STATE, "visual_calls", 0)
            ) + 1
            key = args[0] if args else kwargs.get("key")
            if (
                _WRITE_STATE.visual_calls == 2
                and key is not None
                and not bool(getattr(key, "interiors", False))
            ):
                previous = depth
                upgrade._CALL_STATE.depth = previous + 1
                try:
                    return original_visual_lod(*args, **kwargs)
                finally:
                    upgrade._CALL_STATE.depth = previous
        return original_visual_lod(*args, **kwargs)

    def compact_polygon_distance_visual(*args, **kwargs):
        polygon_depth = int(getattr(upgrade._CALL_STATE, "polygon_depth", 0))
        if bool(getattr(_WRITE_STATE, "active", False)) and polygon_depth == 0:
            _WRITE_STATE.polygon_calls = int(
                getattr(_WRITE_STATE, "polygon_calls", 0)
            ) + 1
            key = args[0] if args else kwargs.get("key")
            if (
                _WRITE_STATE.polygon_calls == 2
                and key is not None
                and not bool(getattr(key, "interiors", False))
            ):
                previous = polygon_depth
                upgrade._CALL_STATE.polygon_depth = previous + 1
                try:
                    return original_polygon_visual_lod(*args, **kwargs)
                finally:
                    upgrade._CALL_STATE.polygon_depth = previous
        return original_polygon_visual_lod(*args, **kwargs)

    buildings.write_building_mlod = write_building_mlod_with_state
    buildings._visual_lod = compact_distance_visual
    buildings._polygon_native_visual_lod = compact_polygon_distance_visual
    _INSTALLED = True
