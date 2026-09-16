# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge advisory object-limit policy from Milestone9Spec to the runtime spec."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace

from . import milestone9 as _milestone9
from .road_chain_parallel_policy import install_road_chain_parallel_policy
from .road_quality_parallel_compat_policy import install_road_quality_parallel_compat_policy
from .bridge_runtime_policy import install_bridge_runtime_policy
from .bridge_source_water_performance_policy import install_bridge_source_water_performance_policy
from .bridge_postsolve_water_performance_policy import install_bridge_postsolve_water_performance_policy
from .bridge_ditch_spatial_policy import install_bridge_ditch_spatial_policy
from .bridge_underlay_spatial_policy import install_bridge_underlay_spatial_policy
from .bridge_plan_cache_policy import install_bridge_plan_cache_policy
from .road_constraint_performance_policy import install_road_constraint_performance_policy
from .road_profile_performance_policy import install_road_profile_performance_policy
from .building_pad_performance_policy import install_building_pad_performance_policy
from .road_constraint_pathology_policy import install_road_constraint_pathology_policy
from .terrain_postsolve_watchdog_policy import install_terrain_postsolve_watchdog_policy
from .residential_infill_performance_policy import install_residential_infill_performance_policy
from .terrain_postprocess_performance_policy import install_terrain_postprocess_performance_policy
from .terrain_grid_performance_policy import install_terrain_grid_performance_policy
from .object_placement_performance_policy import install_object_placement_performance_policy
from .object_sampling_cache_policy import install_object_sampling_cache_policy
from .forest_vector_performance_policy import install_forest_vector_performance_policy
from .forest_array_cache_policy import install_forest_array_cache_policy
from .forest_generation_binding_policy import install_forest_generation_binding_policy
from .object_stage_parallel_policy import install_object_stage_parallel_policy
from .forest_primary_parallel_policy import install_forest_primary_parallel_policy
from .road_finish_parallel_policy import install_road_finish_parallel_policy
from .shared_object_index_policy import install_shared_object_index_policy
from .road_fit_cache_policy import install_road_fit_cache_policy

_ADVISORY_OBJECT_LIMITS: ContextVar[bool | None] = ContextVar(
    "cwr_milestone9_advisory_object_limits", default=None
)
_INSTALLED = False
_ORIGINAL_BUILD_MILESTONE9 = _milestone9.build_milestone9
_ORIGINAL_BUILD_MILESTONE4 = _milestone9.build_milestone4


def install_milestone9_advisory_policy() -> None:
    """Preserve the external Milestone 9 limit policy through spec conversion.

    Milestone9Spec is part of the source-bundle-facing API while the actual world
    builder consumes _Milestone9PlayabilitySpec. Keep the conversion explicit
    without changing direct low-level playability callers, whose historical
    default remains hard/truncating limits unless they opt in.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    # The stock-road process pool and its road-quality ContextVar bridge must
    # become the base implementation before source-water bridge policy captures
    # the fitter. Workers therefore use the same terrain/junction scoring as the
    # serial road-quality path, while object IDs remain assigned in the parent.
    install_road_chain_parallel_policy()
    install_road_quality_parallel_compat_policy()
    install_bridge_runtime_policy()
    # Source-water bridge detection is exact but its historical 0.5 m sampling
    # performed Python point-in-ring work for every sample. Index and batch that
    # lookup before road terrain constraints capture the live water predicate.
    install_bridge_source_water_performance_policy()
    # Bridge-or-causeway terrain repair duplicated that same scalar sampler after
    # the core terrain solver returned. Batch its separate wet-run detection too,
    # preserving islands/dry gaps while avoiding silent post-solve stalls.
    install_bridge_postsolve_water_performance_policy()
    # Explicit bridge classification historically compared every bridge segment
    # against every mapped watercourse segment. Bound that exact predicate with a
    # cached spatial watercourse index before terrain road processing begins.
    install_bridge_ditch_spatial_policy()
    # Bridge runtime installs exact underlay cleanup. Add only a conservative
    # spatial candidate layer around it so distant road objects never reach the
    # existing distance/heading predicates.
    install_bridge_underlay_spatial_policy()
    # Bridge abutment grading stores the authoritative pre-fill span in memory.
    # Keep that tiny state beside cached terrain so a cache hit cannot replan a
    # shorter bridge against the already-raised endpoint cells.
    install_bridge_plan_cache_policy()
    install_road_constraint_performance_policy()
    # The corridor/cell half above is vectorized already. Long roads still paid
    # for scalar line interpolation and six terrain samples per cross-slope
    # station, so batch those profile samples as well before building pads run.
    install_road_profile_performance_policy()
    install_building_pad_performance_policy()
    # Complex roads can still make vectorized GEOS distance/project operations
    # scale with both corridor-cell count and source vertex count. Resolve those
    # values while walking local segments and arm a no-progress traceback guard.
    install_road_constraint_pathology_policy()
    # The core terrain solver reports 100% before bridge/causeway/abutment wrappers
    # have returned. Keep a traceback guard armed across that hidden post-solve gap.
    install_terrain_postsolve_watchdog_policy()
    # Residential infill must not rescan every mapped building for every
    # residential polygon. Index source building occupancy once per world while
    # preserving the existing exact point/centroid/vertex containment rules.
    install_residential_infill_performance_policy()
    install_terrain_postprocess_performance_policy()
    install_terrain_grid_performance_policy()
    install_object_placement_performance_policy()
    install_object_sampling_cache_policy()
    install_forest_vector_performance_policy()
    install_forest_array_cache_policy()
    install_forest_generation_binding_policy()
    # Multicore object-stage precomputation is deliberately late: it must wrap
    # the final bridge-aware stock fitter and the vector/cached forest helpers.
    install_object_stage_parallel_policy()
    # Replace the bounded primary-forest tuple cache with compact worker plans
    # for every regular eligible block. The parent remains authoritative for
    # row-major acceptance, object limits, counters and object ids.
    install_forest_primary_parallel_policy()
    # Supersede only the road half with NumPy endpoint sampling plus parallel
    # whole-chain transform/model/axis finalization. The non-road caches above
    # remain active and the parent still owns object ids and junction validation.
    install_road_finish_parallel_policy()
    install_shared_object_index_policy()
    # Cache the deterministic expensive inner road fit after the parallel finish
    # layer is final. Later package-startup wrappers still perform final dedupe,
    # bridge underlay cleanup and road/building ContextVar recording on every run.
    install_road_fit_cache_policy()

    def build_milestone9(output_dir, spec, *, clean: bool = True):
        token = _ADVISORY_OBJECT_LIMITS.set(
            bool(getattr(spec, "advisory_object_limits", False))
        )
        try:
            return _ORIGINAL_BUILD_MILESTONE9(output_dir, spec, clean=clean)
        finally:
            _ADVISORY_OBJECT_LIMITS.reset(token)

    def build_milestone4(output_dir, spec, *args, **kwargs):
        enabled = _ADVISORY_OBJECT_LIMITS.get()
        if enabled is not None and hasattr(spec, "advisory_object_limits"):
            spec = replace(spec, advisory_object_limits=enabled)
        return _ORIGINAL_BUILD_MILESTONE4(output_dir, spec, *args, **kwargs)

    build_milestone9._cwr_milestone9_advisory_policy = True  # type: ignore[attr-defined]
    build_milestone4._cwr_milestone9_advisory_policy = True  # type: ignore[attr-defined]
    _milestone9.build_milestone9 = build_milestone9
    _milestone9.build_milestone4 = build_milestone4
    _INSTALLED = True