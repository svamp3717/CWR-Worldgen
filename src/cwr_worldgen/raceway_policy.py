# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import replace

from . import asset_mapping as _asset_mapping
from . import generator as _generator
from . import osm as _osm

_INSTALLED = False


def _raceway_default_osm_asset_mapping(original):
    """Add ``highway=raceway`` to the built-in paved-road asset rule."""

    def wrapped(spec, milestone_number: int, *, global_textures=()):
        mapping = original(spec, milestone_number, global_textures=global_textures)
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

    wrapped._cwr_raceway_policy = True  # type: ignore[attr-defined]
    return wrapped


def install_raceway_policy() -> None:
    """Treat OSM motor raceways as supported paved vehicle roads.

    ``highway=raceway`` is a normal linear OSM highway class, but the importer
    historically omitted it from the supported-road sets. Raceway ways now
    default to the ordinary asphalt road family. Explicit unpaved ``surface=*``
    tags continue to win, so dirt and motocross circuits are not incorrectly
    paved.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    # The direct Overpass importer and normalized source-bundle path keep their
    # own supported-highway sets. Update both so GUI/source builds and direct
    # Python builds agree about raceways.
    _osm._MAJOR_HIGHWAYS.add("raceway")
    from . import normalization as _normalization

    _normalization._MAJOR_HIGHWAYS.add("raceway")

    # The actual model chooser already treats an otherwise-unspecified raceway
    # as paved because raceway is not a dirt-default highway. Extend the asset
    # dependency rule as well, otherwise a raceway-only world could select the
    # asphalt P3D at placement time without copying that model into the build.
    original_mapping = _asset_mapping.default_osm_asset_mapping
    if not getattr(original_mapping, "_cwr_raceway_policy", False):
        wrapped_mapping = _raceway_default_osm_asset_mapping(original_mapping)
        _asset_mapping.default_osm_asset_mapping = wrapped_mapping
        # generator.py imports the function directly, so patch its bound module
        # reference in parallel with the source module.
        _generator.default_osm_asset_mapping = wrapped_mapping

    # Raceway policy is the final installer imported by cwr_worldgen.__init__.
    # Use that stable end-of-stack point to attach the stock-road visual and
    # continuity safeguards after every geometry/junction wrapper has composed
    # itself.
    from .stock_road_visual_finish_policy import install_stock_road_visual_finish_policy

    install_stock_road_visual_finish_policy()
    from .stock_road_curve_regularization_policy import (
        install_stock_road_curve_regularization_policy,
    )

    install_stock_road_curve_regularization_policy()
    from .stock_road_final_continuity_policy import (
        install_stock_road_final_continuity_policy,
    )

    install_stock_road_final_continuity_policy()
    # Some real paved bends cannot be represented by one fixed-radius arc but
    # are still well approximated by a connector-locked sequence of stock
    # curves and straights. Fit those only after the ordinary coherent-curve
    # path has had first refusal, so already-correct curves remain untouched.
    from .stock_road_sharp_turn_policy import install_stock_road_sharp_turn_policy

    install_stock_road_sharp_turn_policy()
    # On short junction-to-junction paved bends the generic sharp policy can
    # find the right native curves and then lose their tangent continuity when
    # it sends the sampled line back through the greedy fitter. Accept the
    # beam's exact stock sequence directly in that narrow case so the road edges
    # meet instead of leaving a triangular grass wedge at each heading change.
    from .stock_road_sharp_exact_policy import install_stock_road_sharp_exact_policy

    install_stock_road_sharp_exact_policy()
    # Promote a broader set of coherent paved bends to the same exact-pose
    # stock-curve representation. This is still road-only and keeps every
    # accepted connector inside the existing sharp-turn source corridor.
    from .stock_road_curve_usage_policy import install_stock_road_curve_usage_policy

    install_stock_road_curve_usage_policy()
    # Gentle 8-15 degree bends are a separate residual seam class: they can be
    # spread over several short straight slabs but need only one stock 10-degree
    # curve. Let the same connector-locked beam accept that one-curve solution
    # before visual seam fallbacks are considered.
    from .stock_road_micro_bend_policy import install_stock_road_micro_bend_policy

    install_stock_road_micro_bend_policy()
    # Consecutive bend spans can reverse direction at one shared vertex. The
    # one-sign sharp fitter cannot repair that local S-bend as a unit, so allow
    # one road-only beam to use both handednesses while keeping the same source
    # corridor and exact stock connector geometry.
    from .stock_road_s_bend_policy import install_stock_road_s_bend_policy

    install_stock_road_s_bend_policy()
    # The final-continuity skew chooser is later than the measured-junction
    # installer, so restate the verified local -X branch side here at the true
    # end of the policy stack. Near-orthogonal T nodes may still use the native
    # model; strongly skewed ones keep the much smaller through-axis fallback.
    from .stock_road_skew_orientation_policy import (
        install_stock_road_skew_orientation_policy,
    )

    install_stock_road_skew_orientation_policy()
    # When a bend cannot use native curves, short rectangular paved pieces can
    # still share a centreline connector while their outer edges open into a
    # triangular grass wedge. Install the low same-family seam underlay late so
    # no continuity wrapper can disable this strictly visual fallback.
    from .stock_road_straight_seam_policy import (
        install_stock_road_straight_seam_policy,
    )

    install_stock_road_straight_seam_policy()
    # The supplied Lundby18 screenshots show that the same failure can survive
    # at a curve-to-straight connector. Final continuity used to disable curve
    # seam underlays on the assumption that selection had made every curve seam
    # tangent-continuous. Re-enable that fallback for paved roads only, after
    # straight coverage, while leaving dirt/gravel and junctions untouched.
    from .stock_road_curve_seam_fallback_policy import (
        install_stock_road_curve_seam_fallback_policy,
    )

    install_stock_road_curve_seam_fallback_policy()
    # A legacy sil/asf/kos six-metre cap is a poor visible surface for skewed
    # or turning intersections. Keep it below the real approach pieces and add
    # low incident-aligned tongues only where its axis does not cover an arm.
    # This final road-only pass lets the approach geometry own the visible road
    # edges instead of one rectangular fallback slab.
    from .stock_road_intersection_edge_policy import (
        install_stock_road_intersection_edge_policy,
    )

    install_stock_road_intersection_edge_policy()
    _INSTALLED = True
