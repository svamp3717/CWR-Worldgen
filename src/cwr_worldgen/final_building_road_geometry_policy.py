# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry corrections for final road/building clearance.

Generated gravel curves are quadratic bowed ribbons, not circular arcs. Keep the
post-road building-clearance primitive model on the exact same centreline recipe
as the infrastructure mesh generator.
"""
from __future__ import annotations

import math

from . import final_building_road_clearance_policy as _clearance

_INSTALLED = False


def gravel_curve_centreline_points(
    length: float,
    side: str,
    degrees: float,
) -> tuple[tuple[float, float], ...]:
    """Return the centreline used by generated gravel ribbon geometry."""
    length = max(0.0, float(length))
    amount = abs(float(degrees))
    if amount <= 1.0e-9 or length <= 1.0e-9:
        return ((0.0, -length * 0.5), (0.0, length * 0.5))

    signed = amount if str(side).casefold() == "r" else -amount
    theta = math.radians(amount)
    radius = length / max(1.0e-9, 2.0 * math.sin(theta * 0.5))
    sagitta = math.copysign(
        radius * (1.0 - math.cos(theta * 0.5)),
        signed,
    )
    control_x = sagitta * 2.0
    half_length = length * 0.5
    section_count = max(2, int(math.ceil(amount / 10.0)))

    points: list[tuple[float, float]] = []
    for index in range(section_count + 1):
        t = index / section_count
        one_minus = 1.0 - t
        points.append((
            2.0 * one_minus * t * control_x,
            -half_length * one_minus * one_minus + half_length * t * t,
        ))
    return tuple(points)


def install_final_building_road_geometry_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _clearance._gravel_curve_points = gravel_curve_centreline_points
    _INSTALLED = True

    # Step 3 depends on the exact final-road primitive geometry above. Install
    # right-of-way priority last so unresolved buildings may suppress only a
    # tightly bounded set of low-priority dirt/gravel road pieces.
    from .road_building_priority_policy import install_road_building_priority_policy

    install_road_building_priority_policy()

    # Step 4 is the final invariant. It sees the exact road set that survives
    # Step 3 suppression and refuses to serialize any procedural building that
    # still intersects a ground-level road surface.
    from .final_road_building_audit_policy import (
        install_final_road_building_audit_policy,
    )

    install_final_road_building_audit_policy()
