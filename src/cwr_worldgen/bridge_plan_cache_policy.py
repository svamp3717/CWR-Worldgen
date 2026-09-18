# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist pre-fill stock-bridge plans beside cached terrain solutions.

Bridge abutment grading intentionally changes the terrain at each bridge end. The
final stock-bridge planner must therefore reuse the plan captured immediately
before that fill; replanning against the filled terrain can shorten a bridge by a
module and leave its real endpoint over the reopened water channel.

The abutment policy historically kept those plans only in process memory. A
terrain-solution cache hit restores the raised terrain but cannot restore that
memory, so the final planner sees half of an old build and half of a new one.
Keep the small plan dictionary in a sidecar keyed by the terrain cache itself.
Existing terrain caches without the sidecar are refreshed once, after which both
terrain and bridge-plan state can be reused normally.
"""
from __future__ import annotations

from dataclasses import is_dataclass, replace
from functools import wraps
from pathlib import Path
import pickle
from typing import Any

from . import bridge_abutment_terrain_policy as _abutment
from .cache import atomic_write_bytes

_SCHEMA = 1
_SIDECAR_SUFFIX = ".bridge-plans-v1.pickle"
_INSTALLED = False
_ORIGINAL_LOAD_TERRAIN_SOLUTION: Any = None


def _has_explicit_bridge(dataset: Any) -> bool:
    """Return whether source roads contain a real bridge-like tag."""
    for feature in getattr(dataset, "roads", ()):
        tags = getattr(feature, "tags", {}) or {}
        bridge = str(tags.get("bridge", "")).strip().casefold()
        if (
            bridge not in {"", "no", "false", "0", "none"}
            or str(tags.get("man_made", "")).strip().casefold() == "bridge"
            or str(tags.get("special", "")).strip().casefold() == "bridge"
        ):
            return True
    return False


def _sidecar_path(cache_path: str | Path | None) -> Path | None:
    if cache_path is None:
        return None
    path = Path(cache_path)
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def _valid_plan(plan: Any) -> bool:
    try:
        points = tuple(plan.points)
        module_count = int(plan.module_count)
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        len(points) == 2
        and module_count > 0
        and all(len(tuple(point)) == 2 for point in points)
    )


def _write_plan_sidecar(cache_path: str | Path | None, spec: Any) -> bool:
    """Best-effort persistence; cache metadata must never fail a valid build."""
    path = _sidecar_path(cache_path)
    if path is None:
        return False
    payload = {
        "schema": _SCHEMA,
        "world_key": _abutment._plan_world_key(spec),
        "plans": dict(_abutment._PLAN_CACHE),
    }
    try:
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write_bytes(path, encoded)
    except (OSError, pickle.PickleError, AttributeError, TypeError, ValueError):
        return False
    return True


def _restore_plan_sidecar(cache_path: str | Path | None, spec: Any) -> bool:
    path = _sidecar_path(cache_path)
    if path is None or not path.is_file():
        return False
    try:
        payload = pickle.loads(path.read_bytes())
        schema = int(payload.get("schema", 0)) if isinstance(payload, dict) else 0
    except (
        OSError,
        EOFError,
        pickle.PickleError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return False
    if not isinstance(payload, dict) or schema != _SCHEMA:
        return False
    world_key = _abutment._plan_world_key(spec)
    if tuple(payload.get("world_key", ())) != tuple(world_key):
        return False
    plans = payload.get("plans")
    if not isinstance(plans, dict):
        return False
    for key, plan in plans.items():
        if not isinstance(key, tuple) or tuple(key[:3]) != tuple(world_key):
            return False
        if not _valid_plan(plan):
            return False
    _abutment._PLAN_CACHE.clear()
    _abutment._PLAN_CACHE.update(plans)
    return True


def _refresh_spec(spec: Any) -> Any | None:
    """Return an equivalent spec that forces one cache regeneration."""
    if not hasattr(spec, "cache_refresh") or not is_dataclass(spec):
        return None
    try:
        return replace(spec, cache_refresh=True)
    except (TypeError, ValueError):
        return None


def install_bridge_plan_cache_policy() -> None:
    """Keep bridge planning state synchronized with the terrain cache."""
    global _INSTALLED, _ORIGINAL_LOAD_TERRAIN_SOLUTION
    if _INSTALLED:
        return

    from . import generator

    _ORIGINAL_LOAD_TERRAIN_SOLUTION = generator._load_terrain_solution

    @wraps(_ORIGINAL_LOAD_TERRAIN_SOLUTION)
    def load_terrain_solution(
        loaded,
        dataset,
        projection,
        raster,
        spec,
        dem_key,
        raster_key,
        dataset_identity,
        *,
        building_placement_plans=(),
        progress_callback=None,
    ):
        # Never let bridge plans from another build survive a cache lookup.
        _abutment._PLAN_CACHE.clear()
        value = _ORIGINAL_LOAD_TERRAIN_SOLUTION(
            loaded,
            dataset,
            projection,
            raster,
            spec,
            dem_key,
            raster_key,
            dataset_identity,
            building_placement_plans=building_placement_plans,
            progress_callback=progress_callback,
        )
        _grading, _slopes, hit, _key, cache_path = value

        if not _has_explicit_bridge(dataset):
            return value

        if hit:
            if _restore_plan_sidecar(cache_path, spec):
                return value

            # Old caches contain the post-fill terrain but not the pre-fill plan.
            # Rebuild exactly once so the authoritative plan is captured from the
            # same solver pass that produced the terrain. Future hits restore the
            # tiny sidecar and retain the normal terrain-cache speedup.
            refresh_spec = _refresh_spec(spec)
            if refresh_spec is None:
                return value
            _abutment._PLAN_CACHE.clear()
            value = _ORIGINAL_LOAD_TERRAIN_SOLUTION(
                loaded,
                dataset,
                projection,
                raster,
                refresh_spec,
                dem_key,
                raster_key,
                dataset_identity,
                building_placement_plans=building_placement_plans,
                progress_callback=progress_callback,
            )
            _grading, _slopes, _hit, _key, cache_path = value
            _write_plan_sidecar(cache_path, refresh_spec)
            return value

        _write_plan_sidecar(cache_path, spec)
        return value

    load_terrain_solution._cwr_bridge_plan_cache_policy = True  # type: ignore[attr-defined]
    generator._load_terrain_solution = load_terrain_solution
    _INSTALLED = True
