# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge advisory object-limit policy from Milestone9Spec to the runtime spec."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace

from . import milestone9 as _milestone9
from .road_chain_parallel_policy import install_road_chain_parallel_policy
from .road_quality_parallel_compat_policy import install_road_quality_parallel_compat_policy
from .bridge_runtime_policy import install_bridge_runtime_policy
from .road_constraint_performance_policy import install_road_constraint_performance_policy
from .building_pad_performance_policy import install_building_pad_performance_policy
from .terrain_postprocess_performance_policy import install_terrain_postprocess_performance_policy
from .terrain_grid_performance_policy import install_terrain_grid_performance_policy
from .object_placement_performance_policy import install_object_placement_performance_policy
from .object_sampling_cache_policy import install_object_sampling_cache_policy
from .forest_vector_performance_policy import install_forest_vector_performance_policy
from .forest_array_cache_policy import install_forest_array_cache_policy
from .shared_object_index_policy import install_shared_object_index_policy

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
    install_road_constraint_performance_policy()
    install_building_pad_performance_policy()
    install_terrain_postprocess_performance_policy()
    install_terrain_grid_performance_policy()
    install_object_placement_performance_policy()
    install_object_sampling_cache_policy()
    install_forest_vector_performance_policy()
    install_forest_array_cache_policy()
    install_shared_object_index_policy()

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
