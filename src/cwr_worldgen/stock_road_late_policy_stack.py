# SPDX-License-Identifier: GPL-3.0-or-later
"""Install the late stock-road continuity policies in dependency order.

The road fitter is intentionally layered: early policies establish measured stock
geometry and obstacle-safe source conditioning, while these later policies repair
engine-visible paved-road failures found in generated Lundby WRPs. Keeping the
ordering in one place prevents a policy from existing in source and tests without
ever participating in a normal world build.
"""
from __future__ import annotations

from . import stock_road_wrptool_catalogue_policy as _wrptool_catalogue
from . import stock_road_curve_regularization_policy as _curve_regularization
from . import stock_road_sharp_turn_policy as _sharp_turn
from . import stock_road_sharp_exact_policy as _sharp_exact
from . import stock_road_s_bend_policy as _s_bend
from . import stock_road_micro_bend_policy as _micro_bend
from . import stock_road_s_bend_exact_policy as _s_bend_exact
from . import stock_road_long_s_bend_policy as _long_s_bend
from . import stock_road_single_vertex_bend_policy as _single_vertex_bend
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_junction_endpoint_policy as _junction_endpoint
from . import stock_road_visual_finish_policy as _visual_finish
from . import stock_road_final_continuity_policy as _final_continuity
from . import stock_road_skew_orientation_policy as _skew_orientation
from . import stock_road_turning_t_fallback_policy as _turning_t_fallback
from . import stock_road_straight_seam_policy as _straight_seam
from . import stock_road_curve_seam_fallback_policy as _curve_seam_fallback
from . import stock_road_intersection_edge_policy as _intersection_edge
from . import stock_road_emitted_seam_policy as _emitted_seam
from . import stock_road_paved_wedge_policy as _paved_wedge
from . import stock_road_emitted_seam_refinement_policy as _emitted_seam_refinement
from . import stock_road_stock_paved_only_policy as _stock_paved_only
from . import stock_road_paved_junction_completion_policy as _paved_junctions
from . import stock_road_fit_first_policy as _fit_first
from . import stock_road_native_junction_ownership_policy as _native_junction_ownership


_INSTALLED = False


def install_stock_road_late_policy_stack() -> None:
    """Activate paved-road fixes after the base stock-road stack is installed."""

    global _INSTALLED
    if _INSTALLED:
        return

    # WrpTool's Resistance road inventory is the catalogue humans actually use
    # when assembling WRPs by hand. Install that exact T/X set before any late
    # junction policy asks the base fitter which native model exists.
    _wrptool_catalogue.install_stock_road_wrptool_catalogue_policy()

    # Preserve and regularize coherent source curvature before exact-pose bend
    # promotion. These passes are paved-only where they alter stock piece use.
    _curve_regularization.install_stock_road_curve_regularization_policy()
    _sharp_turn.install_stock_road_sharp_turn_policy()
    _sharp_exact.install_stock_road_sharp_exact_policy()
    _s_bend.install_stock_road_s_bend_policy()
    _micro_bend.install_stock_road_micro_bend_policy()
    # The S-bend beam used to hand its sampled centreline back to the greedy
    # fitter. Preserve its real stock actions after the micro-bend wrapper so a
    # successful alternating-curve fit cannot be re-faceted into short slabs.
    _s_bend_exact.install_stock_road_s_bend_exact_policy()
    # Lundby32 contains a covered paved S-bend a little over one kilometre long.
    # Keep the exact fitter's existing acceptance gates but let that already-safe
    # junction-to-junction case reach the beam instead of failing the old 360 m cap.
    _long_s_bend.install_stock_road_long_s_bend_policy()
    # Lundby28 exposes isolated source corners that the sustained-turn detector
    # never handed to the curve beam. Give those corners the same exact connector
    # search before broader curve promotion runs.
    _single_vertex_bend.install_stock_road_single_vertex_bend_policy()
    _curve_usage.install_stock_road_curve_usage_policy()
    # Exact policies above can return their recovered actions directly. Keep the
    # original junction cover information for acceptance, but reject a late fit
    # that accidentally restores the old trimmed endpoint instead of reaching
    # the logical node underneath its cap.
    _junction_endpoint.install_stock_road_junction_endpoint_policy()

    # Final visual/physical passes. The dependency comments in these modules
    # are deliberate: visual finish must precede final continuity; skew
    # orientation must precede the turning-T acceptance clamp; straight/curve
    # seam and intersection-edge policies retain their planners as regression
    # evidence; the emitted pass still measures final pitched WRP geometry.
    _visual_finish.install_stock_road_visual_finish_policy()
    _final_continuity.install_stock_road_final_continuity_policy()
    _skew_orientation.install_stock_road_skew_orientation_policy()
    _turning_t_fallback.install_stock_road_turning_t_fallback_policy()
    _straight_seam.install_stock_road_straight_seam_policy()
    _curve_seam_fallback.install_stock_road_curve_seam_fallback_policy()
    _intersection_edge.install_stock_road_intersection_edge_policy()
    _emitted_seam.install_stock_road_emitted_seam_policy()
    # Keep the physical endpoint reconstruction from the paved-wedge work. It is
    # still useful for measuring the real WRP seams, but production must no
    # longer serialize any world-local paved helper P3Ds.
    _paved_wedge.install_stock_road_paved_wedge_policy()
    _emitted_seam_refinement.install_stock_road_emitted_seam_refinement_policy()
    _stock_paved_only.install_stock_road_stock_paved_only_policy()
    # Native paved T/X meshes are visually trustworthy only while every measured
    # connector is essentially exact. Otherwise keep source-aligned approaches
    # visible over one low stock-family central cap. Dirt/gravel stays untouched.
    _paved_junctions.install_stock_road_paved_junction_completion_policy()

    # Production builds keep fitting first. Disable the older intermediate seam
    # and junction overlap helpers, but leave the final pitch-aware emitted-seam
    # audit active so unambiguous residual paved wedges are not serialized open.
    _fit_first.install_stock_road_fit_first_policy()

    # Once a purpose-built native T/X is selected, it owns the entire visible
    # intersection centre. Restore the measured connector trim after every older
    # fallback layer has been composed, and re-enable the transaction-checked
    # connector alignment for skewed all-paved T junctions.
    _native_junction_ownership.install_stock_road_native_junction_ownership_policy()

    _INSTALLED = True
