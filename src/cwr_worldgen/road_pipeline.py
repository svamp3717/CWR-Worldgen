# SPDX-License-Identifier: GPL-3.0-or-later
"""Single, explicit installation order for world-generation road policies.

The historical road stack relied on import-time capture of mutable fitter
functions. That means import order is part of the behaviour, not merely the
installer call order. Keep that behaviour explicit here:

* base policies are imported and installed one at a time, matching the old
  package startup sequence;
* late policies are preloaded as a group and then installed in declared order,
  matching the old late-stack module; and
* non-fitting source classification is applied once at the end.

Set ``CWR_WORLDGEN_ROAD_PIPELINE_TRACE=1`` to print each installed stage during
package startup.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import os
import sys
from types import ModuleType

_INSTALLED = False
_TRACE_ENV = "CWR_WORLDGEN_ROAD_PIPELINE_TRACE"
_PACKAGE = __package__ or "cwr_worldgen"
_MAXIMUM_PAVED_SEAM_TANGENT_ERROR_DEGREES = 8.0
_MAXIMUM_EXACT_S_BEND_RUN_METRES = 1200.0

# The base stack historically used import -> install -> import -> install.
# Preserve that exact timing because several policy modules bind the current
# fitter or helper implementation when the module itself is imported.
_BASE_STAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("road_quality", "road_quality_policy", "install_road_quality_policy"),
    ("stock_curve_base", "stock_road_curve_policy", "install_stock_road_curve_policy"),
    ("stock_geometry", "stock_road_geometry_policy", "install_stock_road_geometry_policy"),
    ("stock_transform", "stock_road_transform_policy", "install_stock_road_transform_policy"),
    ("stock_3d_connector", "stock_road_3d_connector_policy", "install_stock_road_3d_connector_policy"),
    ("gravel_junction", "gravel_junction_policy", "install_gravel_junction_policy"),
    ("gravel_gap", "gravel_gap_policy", "install_gravel_gap_policy"),
    ("gravel_family", "gravel_family_policy", "install_gravel_family_policy"),
    ("stock_junction", "stock_road_junction_policy", "install_stock_road_junction_policy"),
    ("stock_measured_junction", "stock_road_measured_junction_policy", "install_stock_road_measured_junction_policy"),
    ("stock_skew", "stock_road_skew_policy", "install_stock_road_skew_policy"),
    ("gravel_asphalt_transition", "gravel_asphalt_transition_policy", "install_gravel_asphalt_transition_policy"),
    ("stock_connector", "stock_road_connector_policy", "install_stock_road_connector_policy"),
    ("stock_surface_overlap", "stock_road_surface_overlap_policy", "install_stock_road_surface_overlap_policy"),
    ("stock_relaxation", "stock_road_relaxation_policy", "install_stock_road_relaxation_policy"),
    ("stock_obstacles", "stock_road_obstacle_policy", "install_stock_road_obstacle_policy"),
    ("stock_local_fit", "stock_road_local_fit_policy", "install_stock_road_local_fit_policy"),
    ("stock_relaxation_transaction", "stock_road_relaxation_transaction_policy", "install_stock_road_relaxation_transaction_policy"),
    ("stock_path_conditioning", "stock_road_path_conditioning_policy", "install_stock_road_path_conditioning_policy"),
    ("stock_curve_preservation", "stock_road_curve_preservation_policy", "install_stock_road_curve_preservation_policy"),
)

# The old stock_road_late_policy_stack imported this entire family first and
# installed it second. Keep that semantic while deleting the orchestration
# module itself. Cancelled overlap layers and one-assignment compatibility
# modules are intentionally absent.
_LATE_STAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("wrptool_catalogue", "stock_road_wrptool_catalogue_policy", "install_stock_road_wrptool_catalogue_policy"),
    ("curve_regularization", "stock_road_curve_regularization_policy", "install_stock_road_curve_regularization_policy"),
    ("sharp_turn", "stock_road_sharp_turn_policy", "install_stock_road_sharp_turn_policy"),
    ("sharp_exact", "stock_road_sharp_exact_policy", "install_stock_road_sharp_exact_policy"),
    ("s_bend", "stock_road_s_bend_policy", "install_stock_road_s_bend_policy"),
    ("micro_bend", "stock_road_micro_bend_policy", "install_stock_road_micro_bend_policy"),
    ("s_bend_exact", "stock_road_s_bend_exact_policy", "install_stock_road_s_bend_exact_policy"),
    ("single_vertex_bend", "stock_road_single_vertex_bend_policy", "install_stock_road_single_vertex_bend_policy"),
    ("curve_usage", "stock_road_curve_usage_policy", "install_stock_road_curve_usage_policy"),
    ("junction_endpoint", "stock_road_junction_endpoint_policy", "install_stock_road_junction_endpoint_policy"),
    ("visual_finish", "stock_road_visual_finish_policy", "install_stock_road_visual_finish_policy"),
    ("final_continuity", "stock_road_final_continuity_policy", "install_stock_road_final_continuity_policy"),
    ("skew_orientation", "stock_road_skew_orientation_policy", "install_stock_road_skew_orientation_policy"),
    ("turning_t_fallback", "stock_road_turning_t_fallback_policy", "install_stock_road_turning_t_fallback_policy"),
    ("emitted_seam", "stock_road_emitted_seam_policy", "install_stock_road_emitted_seam_policy"),
    ("paved_wedge_geometry", "stock_road_paved_wedge_policy", "install_stock_road_paved_wedge_policy"),
    ("emitted_seam_refinement", "stock_road_emitted_seam_refinement_policy", "install_stock_road_emitted_seam_refinement_policy"),
    ("stock_paved_only", "stock_road_stock_paved_only_policy", "install_stock_road_stock_paved_only_policy"),
    ("inspector_candidates", "stock_road_inspector_candidate_policy", "install_stock_road_inspector_candidate_policy"),
    ("candidate_enforcement", "stock_road_inspector_candidate_enforcement_policy", "install_stock_road_inspector_candidate_selector_policy"),
    ("paved_junction_completion", "stock_road_paved_junction_completion_policy", "install_stock_road_paved_junction_completion_policy"),
    ("native_junction_ownership", "stock_road_native_junction_ownership_policy", "install_stock_road_native_junction_ownership_policy"),
    ("candidate_final_enforcement", "stock_road_inspector_candidate_enforcement_policy", "install_stock_road_inspector_candidate_final_policy"),
    ("reference_wrp", "stock_road_reference_wrp_policy", "install_stock_road_reference_wrp_policy"),
    ("kodiak_reference", "stock_road_kodiak_reference_policy", "install_stock_road_kodiak_reference_policy"),
    ("stock_assets_only", "stock_road_stock_assets_only_policy", "install_stock_road_stock_assets_only_policy"),
)

ROAD_PIPELINE_STAGES: tuple[str, ...] = (
    *(stage for stage, _module, _installer in _BASE_STAGE_SPECS),
    *(stage for stage, _module, _installer in _LATE_STAGE_SPECS),
    "raceway_classification",
)


def _trace(stage: str) -> None:
    value = os.environ.get(_TRACE_ENV, "").strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        print(f"[cwr-road-pipeline] installed {stage}", file=sys.stderr)


def _load(module_name: str) -> ModuleType:
    return import_module(f".{module_name}", _PACKAGE)


def _invoke(stage: str, module: ModuleType, installer_name: str) -> None:
    getattr(module, installer_name)()
    _trace(stage)


def _install_base_stages() -> None:
    for stage, module_name, installer_name in _BASE_STAGE_SPECS:
        module = _load(module_name)
        _invoke(stage, module, installer_name)


def _install_late_stages() -> None:
    # Import every late module before installing any late module. This mirrors
    # the historical late-stack module and preserves import-time helper capture.
    modules = {
        module_name: _load(module_name)
        for _stage, module_name, _installer_name in _LATE_STAGE_SPECS
    }
    for stage, module_name, installer_name in _LATE_STAGE_SPECS:
        _invoke(stage, modules[module_name], installer_name)
        if stage == "s_bend_exact":
            # A retired compatibility policy existed only to raise this exact
            # covered-run search cap from 360 m to 1200 m.
            modules["stock_road_s_bend_exact_policy"].MAXIMUM_EXACT_S_BEND_RUN_METRES = (
                _MAXIMUM_EXACT_S_BEND_RUN_METRES
            )
        elif stage == "final_continuity":
            # The retired curve-seam fallback used to leave this 8-degree
            # geometry bound behind even though fit-first later disabled its
            # visual object-appending hook. The final physical emitted-seam
            # planner still consumes the bound, so preserve the value directly.
            modules[
                "stock_road_visual_finish_policy"
            ].MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES = (
                _MAXIMUM_PAVED_SEAM_TANGENT_ERROR_DEGREES
            )


def _install_raceway_classification() -> None:
    """Treat OSM motor raceways as ordinary supported paved vehicle roads."""

    asset_mapping = _load("asset_mapping")
    generator = _load("generator")
    normalization = _load("normalization")
    osm = _load("osm")

    osm._MAJOR_HIGHWAYS.add("raceway")
    normalization._MAJOR_HIGHWAYS.add("raceway")

    original_mapping = asset_mapping.default_osm_asset_mapping
    if getattr(original_mapping, "_cwr_raceway_classification", False):
        return

    def wrapped(spec, milestone_number: int, *, global_textures=()):
        mapping = original_mapping(
            spec,
            milestone_number,
            global_textures=global_textures,
        )
        rules = []
        for rule in mapping.rules:
            if rule.rule_id != "road-paved":
                rules.append(rule)
                continue
            match = []
            for key, values in rule.match:
                if key == "highway" and "raceway" not in values:
                    values = (*values, "raceway")
                match.append((key, values))
            rules.append(replace(rule, match=tuple(match)))
        return replace(mapping, rules=tuple(rules))

    wrapped._cwr_raceway_classification = True  # type: ignore[attr-defined]
    asset_mapping.default_osm_asset_mapping = wrapped
    # generator.py imports the mapping function directly.
    generator.default_osm_asset_mapping = wrapped


def _synchronise_public_fitter() -> None:
    """Expose the same final fitter through playability and generator."""

    generator = _load("generator")
    playability = _load("playability")
    playability.fit_road_objects = generator.fit_road_objects


def install_road_pipeline() -> None:
    """Install the production road fitter exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    _install_base_stages()
    _install_late_stages()
    _synchronise_public_fitter()
    _install_raceway_classification()
    _trace("raceway_classification")
    _INSTALLED = True
