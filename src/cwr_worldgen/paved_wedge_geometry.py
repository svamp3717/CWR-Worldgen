# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

# The generated wedge starts slightly inside the stock road so there cannot be a
# floating-point hairline between meshes.  Because the mesh is triangular, that
# longitudinal inset also narrows the triangle at the *actual* road edge.  The
# old mesh added 20 mm to its base width but then lost considerably more than
# that to taper, especially on shallow turns.  Preserve the requested overlap at
# the real road edge instead, and leave a small lateral safety allowance for the
# sub-decimetre connector offsets seen in fitted/pitched WRP geometry.
PAVED_WEDGE_BASE_INSET_METRES = 0.020
PAVED_WEDGE_APEX_OVERLAP_METRES = 0.020
PAVED_WEDGE_LATERAL_OVERLAP_METRES = 0.100


def paved_wedge_local_points(
    turn_degrees: float,
    *,
    radius_metres: float = 4.55,
    maximum_turn_degrees: float = 35.0,
) -> tuple[tuple[float, float, float], ...]:
    """Return a terrain-covering outside-miter triangle in model X/Z space.

    The base is inset by ``PAVED_WEDGE_BASE_INSET_METRES`` so it overlaps the
    stock road.  Its model-space half-width is deliberately expanded so that,
    *after the triangle has tapered across that inset*, the cross-section at the
    true road edge still reaches the geometric miter base plus 10 cm.  This is
    the part the previous formula got backwards.
    """

    turn = max(0.0, min(float(turn_degrees), float(maximum_turn_degrees)))
    radius = float(radius_metres)
    half_angle = math.radians(turn * 0.5)
    cosine = max(1.0e-9, math.cos(half_angle))

    base_distance = (
        radius * cosine - PAVED_WEDGE_BASE_INSET_METRES
    )
    apex_distance = (
        radius + PAVED_WEDGE_APEX_OVERLAP_METRES
    ) / cosine
    depth = max(1.0e-6, apex_distance - base_distance)

    # Width required at the real stock-road edge, which lies BASE_INSET metres
    # forward of this triangle's model-space base.
    required_edge_half_width = (
        radius * math.sin(half_angle) + PAVED_WEDGE_LATERAL_OVERLAP_METRES
    )
    remaining_fraction = max(
        1.0e-4,
        1.0 - PAVED_WEDGE_BASE_INSET_METRES / depth,
    )
    model_base_half_width = required_edge_half_width / remaining_fraction

    return (
        (0.0, 0.0, depth),
        (model_base_half_width, 0.0, 0.0),
        (-model_base_half_width, 0.0, 0.0),
    )


def triangle_samples(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Return vertices, edge samples and interior samples for visibility tests."""

    points: list[tuple[float, float]] = [first, second, third]
    for start, end in ((first, second), (second, third), (third, first)):
        for fraction in (0.25, 0.50, 0.75):
            points.append((
                float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
                float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
            ))
    points.extend((
        (
            (float(first[0]) + float(second[0]) + float(third[0])) / 3.0,
            (float(first[1]) + float(second[1]) + float(third[1])) / 3.0,
        ),
        (
            (float(first[0]) * 0.50 + float(second[0]) * 0.25 + float(third[0]) * 0.25),
            (float(first[1]) * 0.50 + float(second[1]) * 0.25 + float(third[1]) * 0.25),
        ),
        (
            (float(first[0]) * 0.25 + float(second[0]) * 0.50 + float(third[0]) * 0.25),
            (float(first[1]) * 0.25 + float(second[1]) * 0.50 + float(third[1]) * 0.25),
        ),
        (
            (float(first[0]) * 0.25 + float(second[0]) * 0.25 + float(third[0]) * 0.50),
            (float(first[1]) * 0.25 + float(second[1]) * 0.25 + float(third[1]) * 0.50),
        ),
    ))
    return tuple(points)
