# SPDX-License-Identifier: GPL-3.0-or-later
"""Single, explicit installation order for world-generation road policies.

Historically the road fitter was patched from ``cwr_worldgen.__init__``,
``stock_road_late_policy_stack`` and ``raceway_policy``.  That made the effective
runtime fitter difficult to inspect because installation order was spread across
multiple modules and some installers re-installed earlier stages.

This module is now the one production authority for road-policy activation.
Individual policy modules still contain their algorithms, but they no longer
choose when they become active.

Set ``CWR_WORLDGEN_ROAD_PIPELINE_TRACE=1`` to print each installed stage during
package startup.  This is intentionally opt-in so normal CLI/GUI output stays
unchanged.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable

_INSTALLED = False
_TRACE_ENV = "CWR_WORLDGEN_ROAD_PIPELINE_TRACE"


# Ordered stage names are public on purpose.  A debugger or regression test can
# inspect the exact production composition without reverse-engineering monkey
# patches from several import sites.
ROAD_PIPELINE_STAGES: tuple[str, ...] = (
    # Baseline stock-road fitting.
    "road_quality",
    "stock_curve_base",
    "stock_geometry",
    "stock_transform",
    "stock_3d_connector",
    # Gravel and mixed-surface integration.
    "gravel_junction",
    "gravel_gap",
    "gravel_family",
    "stock_junction",
    "stock_measured_junction",
    "stock_skew",
    "gravel_asphalt_transition",
    "stock_connector",
    "stock_surface_overlap",
    # Source conditioning / obstacle-safe fitting.
    "stock_relaxation",
    "stock_obstacles",
    "stock_local_fit",
    "stock_relaxation_transaction",
    "stock_path_conditioning",
    "stock_curve_preservation",
    # Native curve and bend selection.
    "wrptool_catalogue",
    "curve_regularization",
    "sharp_turn",
    "sharp_exact",
    "s_bend",
    "micro_bend",
    "s_bend_exact",
    "long_s_bend",
    "single_vertex_bend",
    "curve_usage",
    "junction_endpoint",
    # Final continuity / physical emitted geometry.
    "visual_finish",
    "final_continuity",
    "skew_orientation",
    "turning_t_fallback",
    "straight_seam",
    "curve_seam_fallback",
    "intersection_edge",
    "emitted_seam",
    "paved_wedge_geometry",
    "emitted_seam_refinement",
    "stock_paved_only",
    "inspector_candidates",
    "candidate_enforcement",
    "paved_junction_completion",
    "fit_first",
    "native_junction_ownership",
    "candidate_final_enforcement",
    "reference_wrp",
    "kodiak_reference",
    "stock_assets_only",
    # OSM classification belongs last so it cannot secretly compose fitter
    # wrappers.  It only broadens supported paved source classes/assets.
    "raceway_classification",
)


def _trace(stage: str) -> None:
    value = os.environ.get(_TRACE_ENV, "").strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        print(f"[cwr-road-pipeline] installed {stage}", file=sys.stderr)


def _run(stage: str, installer: Callable[[], None]) -> None:
    installer()
    _trace(stage)


def install_road_pipeline() -> None:
    """Install the complete production road fitter exactly once.

    The ordering below intentionally mirrors the previous effective package
    startup order.  The refactor changes ownership and observability, not road
    geometry, so it is suitable as a first step before deleting redundant
    policy implementations.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .road_quality_policy import install_road_quality_policy
    from .stock_road_curve_policy import install_stock_road_curve_policy
    from .stock_road_geometry_policy import install_stock_road_geometry_policy
    from .stock_road_transform_policy import install_stock_road_transform_policy
    from .stock_road_3d_connector_policy import install_stock_road_3d_connector_policy
    from .gravel_junction_policy import install_gravel_junction_policy
    from .gravel_gap_policy import install_gravel_gap_policy
    from .gravel_family_policy import install_gravel_family_policy
    from .stock_road_junction_policy import install_stock_road_junction_policy
    from .stock_road_measured_junction_policy import install_stock_road_measured_junction_policy
    from .stock_road_skew_policy import install_stock_road_skew_policy
    from .gravel_asphalt_transition_policy import install_gravel_asphalt_transition_policy
    from .stock_road_connector_policy import install_stock_road_connector_policy
    from .stock_road_surface_overlap_policy import install_stock_road_surface_overlap_policy
    from .stock_road_relaxation_policy import install_stock_road_relaxation_policy
    from .stock_road_obstacle_policy import install_stock_road_obstacle_policy
    from .stock_road_local_fit_policy import install_stock_road_local_fit_policy
    from .stock_road_relaxation_transaction_policy import install_stock_road_relaxation_transaction_policy
    from .stock_road_path_conditioning_policy import install_stock_road_path_conditioning_policy
    from .stock_road_curve_preservation_policy import install_stock_road_curve_preservation_policy

    _run("road_quality", install_road_quality_policy)
    _run("stock_curve_base", install_stock_road_curve_policy)
    _run("stock_geometry", install_stock_road_geometry_policy)
    _run("stock_transform", install_stock_road_transform_policy)
    _run("stock_3d_connector", install_stock_road_3d_connector_policy)
    _run("gravel_junction", install_gravel_junction_policy)
    _run("gravel_gap", install_gravel_gap_policy)
    _run("gravel_family", install_gravel_family_policy)
    _run("stock_junction", install_stock_road_junction_policy)
    _run("stock_measured_junction", install_stock_road_measured_junction_policy)
    _run("stock_skew", install_stock_road_skew_policy)
    _run("gravel_asphalt_transition", install_gravel_asphalt_transition_policy)
    _run("stock_connector", install_stock_road_connector_policy)
    _run("stock_surface_overlap", install_stock_road_surface_overlap_policy)
    _run("stock_relaxation", install_stock_road_relaxation_policy)
    _run("stock_obstacles", install_stock_road_obstacle_policy)
    _run("stock_local_fit", install_stock_road_local_fit_policy)
    _run("stock_relaxation_transaction", install_stock_road_relaxation_transaction_policy)
    _run("stock_path_conditioning", install_stock_road_path_conditioning_policy)
    _run("stock_curve_preservation", install_stock_road_curve_preservation_policy)

    # Former stock_road_late_policy_stack.py, kept here in the same order.
    from .stock_road_wrptool_catalogue_policy import install_stock_road_wrptool_catalogue_policy
    from .stock_road_curve_regularization_policy import install_stock_road_curve_regularization_policy
    from .stock_road_sharp_turn_policy import install_stock_road_sharp_turn_policy
    from .stock_road_sharp_exact_policy import install_stock_road_sharp_exact_policy
    from .stock_road_s_bend_policy import install_stock_road_s_bend_policy
    from .stock_road_micro_bend_policy import install_stock_road_micro_bend_policy
    from .stock_road_s_bend_exact_policy import install_stock_road_s_bend_exact_policy
    from .stock_road_long_s_bend_policy import install_stock_road_long_s_bend_policy
    from .stock_road_single_vertex_bend_policy import install_stock_road_single_vertex_bend_policy
    from .stock_road_curve_usage_policy import install_stock_road_curve_usage_policy
    from .stock_road_junction_endpoint_policy import install_stock_road_junction_endpoint_policy
    from .stock_road_visual_finish_policy import install_stock_road_visual_finish_policy
    from .stock_road_final_continuity_policy import install_stock_road_final_continuity_policy
    from .stock_road_skew_orientation_policy import install_stock_road_skew_orientation_policy
    from .stock_road_turning_t_fallback_policy import install_stock_road_turning_t_fallback_policy
    from .stock_road_straight_seam_policy import install_stock_road_straight_seam_policy
    from .stock_road_curve_seam_fallback_policy import install_stock_road_curve_seam_fallback_policy
    from .stock_road_intersection_edge_policy import install_stock_road_intersection_edge_policy
    from .stock_road_emitted_seam_policy import install_stock_road_emitted_seam_policy
    from .stock_road_paved_wedge_policy import install_stock_road_paved_wedge_policy
    from .stock_road_emitted_seam_refinement_policy import install_stock_road_emitted_seam_refinement_policy
    from .stock_road_stock_paved_only_policy import install_stock_road_stock_paved_only_policy
    from .stock_road_inspector_candidate_policy import install_stock_road_inspector_candidate_policy
    from .stock_road_inspector_candidate_enforcement_policy import (
        install_stock_road_inspector_candidate_final_policy,
        install_stock_road_inspector_candidate_selector_policy,
    )
    from .stock_road_paved_junction_completion_policy import install_stock_road_paved_junction_completion_policy
    from .stock_road_fit_first_policy import install_stock_road_fit_first_policy
    from .stock_road_native_junction_ownership_policy import install_stock_road_native_junction_ownership_policy
    from .stock_road_reference_wrp_policy import install_stock_road_reference_wrp_policy
    from .stock_road_kodiak_reference_policy import install_stock_road_kodiak_reference_policy
    from .stock_road_stock_assets_only_policy import install_stock_road_stock_assets_only_policy

    _run("wrptool_catalogue", install_stock_road_wrptool_catalogue_policy)
    _run("curve_regularization", install_stock_road_curve_regularization_policy)
    _run("sharp_turn", install_stock_road_sharp_turn_policy)
    _run("sharp_exact", install_stock_road_sharp_exact_policy)
    _run("s_bend", install_stock_road_s_bend_policy)
    _run("micro_bend", install_stock_road_micro_bend_policy)
    _run("s_bend_exact", install_stock_road_s_bend_exact_policy)
    _run("long_s_bend", install_stock_road_long_s_bend_policy)
    _run("single_vertex_bend", install_stock_road_single_vertex_bend_policy)
    _run("curve_usage", install_stock_road_curve_usage_policy)
    _run("junction_endpoint", install_stock_road_junction_endpoint_policy)
    _run("visual_finish", install_stock_road_visual_finish_policy)
    _run("final_continuity", install_stock_road_final_continuity_policy)
    _run("skew_orientation", install_stock_road_skew_orientation_policy)
    _run("turning_t_fallback", install_stock_road_turning_t_fallback_policy)
    _run("straight_seam", install_stock_road_straight_seam_policy)
    _run("curve_seam_fallback", install_stock_road_curve_seam_fallback_policy)
    _run("intersection_edge", install_stock_road_intersection_edge_policy)
    _run("emitted_seam", install_stock_road_emitted_seam_policy)
    _run("paved_wedge_geometry", install_stock_road_paved_wedge_policy)
    _run("emitted_seam_refinement", install_stock_road_emitted_seam_refinement_policy)
    _run("stock_paved_only", install_stock_road_stock_paved_only_policy)
    _run("inspector_candidates", install_stock_road_inspector_candidate_policy)
    _run("candidate_enforcement", install_stock_road_inspector_candidate_selector_policy)
    _run("paved_junction_completion", install_stock_road_paved_junction_completion_policy)
    _run("fit_first", install_stock_road_fit_first_policy)
    _run("native_junction_ownership", install_stock_road_native_junction_ownership_policy)
    _run("candidate_final_enforcement", install_stock_road_inspector_candidate_final_policy)
    _run("reference_wrp", install_stock_road_reference_wrp_policy)
    _run("kodiak_reference", install_stock_road_kodiak_reference_policy)
    _run("stock_assets_only", install_stock_road_stock_assets_only_policy)

    from .raceway_policy import install_raceway_policy

    _run("raceway_classification", install_raceway_policy)
    _INSTALLED = True
