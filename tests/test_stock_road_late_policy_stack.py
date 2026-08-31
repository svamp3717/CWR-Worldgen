# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_curve_regularization_policy as _regularization
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp_turn
from cwr_worldgen import stock_road_sharp_exact_policy as _sharp_exact
from cwr_worldgen import stock_road_s_bend_policy as _s_bend
from cwr_worldgen import stock_road_micro_bend_policy as _micro_bend
from cwr_worldgen import stock_road_long_s_bend_policy as _long_s_bend
from cwr_worldgen import stock_road_single_vertex_bend_policy as _single_vertex_bend
from cwr_worldgen import stock_road_curve_usage_policy as _curve_usage
from cwr_worldgen import stock_road_junction_endpoint_policy as _junction_endpoint
from cwr_worldgen import stock_road_visual_finish_policy as _visual_finish
from cwr_worldgen import stock_road_final_continuity_policy as _final_continuity
from cwr_worldgen import stock_road_skew_orientation_policy as _skew_orientation
from cwr_worldgen import stock_road_turning_t_fallback_policy as _turning_t_fallback
from cwr_worldgen import stock_road_straight_seam_policy as _straight_seam
from cwr_worldgen import stock_road_curve_seam_fallback_policy as _curve_seam_fallback
from cwr_worldgen import stock_road_intersection_edge_policy as _intersection_edge
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted_seam
from cwr_worldgen import stock_road_emitted_seam_refinement_policy as _emitted_refinement
from cwr_worldgen import stock_road_fit_first_policy as _fit_first
from cwr_worldgen import stock_road_inspector_candidate_completion_policy as _completion
from cwr_worldgen import stock_road_inspector_candidate_enforcement_policy as _enforcement
from cwr_worldgen import stock_road_inspector_candidate_policy as _candidate
from cwr_worldgen import stock_road_reference_wrp_policy as _reference
from cwr_worldgen import stock_road_kodiak_reference_policy as _kodiak
from cwr_worldgen import stock_road_stock_assets_only_policy as _stock_only
from cwr_worldgen import stock_road_late_policy_stack as _stack


def test_late_stock_road_policies_are_active_on_package_import() -> None:
    policies = (
        _regularization,
        _sharp_turn,
        _sharp_exact,
        _s_bend,
        _micro_bend,
        _long_s_bend,
        _single_vertex_bend,
        _curve_usage,
        _junction_endpoint,
        _visual_finish,
        _final_continuity,
        _skew_orientation,
        _turning_t_fallback,
        _straight_seam,
        _curve_seam_fallback,
        _intersection_edge,
        _emitted_seam,
        _emitted_refinement,
        _fit_first,
        _reference,
        _kodiak,
        _stock_only,
    )

    assert _stack._INSTALLED
    assert all(policy._INSTALLED for policy in policies)
    assert not _completion._INSTALLED
    assert _enforcement._SELECTOR_INSTALLED
    assert _enforcement._FINAL_INSTALLED
    assert _sharp_turn._sharp_turn_spans is _single_vertex_bend._single_vertex_sharp_turn_spans
    assert _p._stock_piece_chain is _junction_endpoint._junction_endpoint_chain


def test_final_wrappers_are_not_left_disconnected() -> None:
    # Kodiak remains the outer playability wrapper. The final stock-only guard is
    # attached to generator serialization rather than disturbing this road-policy
    # composition.
    assert _p.fit_road_objects is _kodiak._fit
    assert _kodiak._ORIGINAL_FIT is _reference._fit
    assert _reference._ORIGINAL_FIT is _enforcement._fit
    assert _enforcement._ORIGINAL_FINAL_FIT is _emitted_seam._fit
    assert _emitted_seam._ORIGINAL_FIT is _intersection_edge._fit
    assert (
        _emitted_seam._emitted_seam_cover_plans
        is _emitted_refinement._refined_emitted_seam_cover_plans
    )
    assert not _completion._INSTALLED
    assert _emitted_seam._apply_emitted_seam_covers is _candidate._apply_wedge_candidates
    assert _fit_first._ORIGINAL_EMITTED_SEAM_APPLY is _candidate._apply_wedge_candidates

    # Keep the older seam planners wired for regression analysis, but the final
    # production visual hook must preserve the fitted objects rather than append
    # their low straight/curve underlays.
    assert (
        _fit_first._ORIGINAL_VISUAL_SEAM_APPLY
        is _curve_seam_fallback._apply_paved_curve_seam_fallback
    )
    assert (
        _visual_finish._apply_curve_seam_covers
        is _fit_first._preserve_fitted_visual_seam
    )
    assert _curve_seam_fallback._ORIGINAL_FINISH is _straight_seam._apply_straight_seam_covers

    # The same rule applies after native junction fitting: the historical edge
    # policy remains in the wrapper chain, but it may no longer append overlapping
    # tongue pieces when no rigid stock junction fits cleanly.
    assert (
        _intersection_edge._seal_legacy_paved_intersections
        is _fit_first._preserve_fitted_intersection
    )
    assert _fit_first._ORIGINAL_INTERSECTION_EDGE_APPLY is not None

    # Native T placement uses the later measured skew/orientation chooser rather
    # than the earlier centre-only fallback, with the Lundby turning-T bound
    # applied after that chooser is installed.
    assert _final_continuity._same_family_paved_skew_t is _skew_orientation._same_family_paved_skew_t
    assert (
        _skew_orientation.MAXIMUM_TURNING_T_MAIN_BEND_DEGREES
        == _turning_t_fallback.MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES
    )
