# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep late skew-T replacement aligned with measured Memory-LOD geometry.

The measured junction policy establishes that Resistance T junctions use local
0/180 degrees for the through road and local -X (270 degrees) for the branch.
The later final-continuity skew fallback originally reintroduced the older +X
assumption.  That still selected the correct P3D, but rotated it 180 degrees so
its visible branch tongue appeared on the opposite side of the logical node.

Install this at the very end of the stock-road policy stack.  It changes only
the strongly-skewed same-family T chooser; placement continues through the
measured junction object transform, including the verified asymmetric model
origin offset.
"""
from __future__ import annotations

import math

from . import stock_road_final_continuity_policy as _final
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry

_INSTALLED = False


def _same_family_paved_skew_t(incidents, family: str):
    if len(incidents) != 3 or family not in {"sil", "asf", "kos"}:
        return None
    if any(incident.family != family for incident in incidents):
        return None
    model = _junction._T_JUNCTION_MODELS.get((family, family))
    if model is None:
        return None

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    branch_heading = _junction._heading(incidents[branch].direction)

    candidates = []
    for zero, opposite in ((first, second), (second, first)):
        rotation, main_error = _junction._best_rotation(
            (
                (0.0, _junction._heading(incidents[zero].direction)),
                (180.0, _junction._heading(incidents[opposite].direction)),
            )
        )
        # Memory LOD measurements put the T branch on local -X, not +X.
        # Preserve the dominant through-road exactly, then choose the main-axis
        # orientation whose 270-degree connector lies nearest the source branch.
        branch_error = _junction._angular_distance(
            (rotation + 270.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > _final.MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
        return None

    half_width = float(_model_geometry.STOCK_HALF_WIDTHS_METRES[family])
    lateral = (
        _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
        * math.sin(math.radians(branch_error))
    )
    if lateral > half_width - _final.SKEW_T_CONNECTOR_EDGE_MARGIN_METRES:
        return None

    return _junction._NativeJunction(
        model_path=model,
        heading_degrees=rotation % 360.0,
        maximum_heading_error_degrees=max(main_error, branch_error),
        cap_family=family,
    )


def install_stock_road_skew_orientation_policy() -> None:
    """Patch the final skew chooser after all earlier junction policies."""

    global _INSTALLED
    if _INSTALLED:
        return
    _final._same_family_paved_skew_t = _same_family_paved_skew_t
    _INSTALLED = True
