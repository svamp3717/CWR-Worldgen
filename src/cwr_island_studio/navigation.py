# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport navigation maths shared by pointer, keyboard and button zoom."""
from __future__ import annotations

import math


def zoom_around(
    left: float, top: float, zoom: float,
    pointer_x: float, pointer_y: float, factor: float,
    minimum: float, maximum: float,
) -> tuple[float, float, float]:
    """Keep the world coordinate under the pointer fixed while zooming."""
    values = (left, top, zoom, pointer_x, pointer_y, factor, minimum, maximum)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("zoom inputs must be finite")
    if zoom <= 0 or factor <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid zoom range")
    target = min(max(zoom * factor, minimum), maximum)
    world_x = left + pointer_x / zoom
    world_z = top + pointer_y / zoom
    return world_x - pointer_x / target, world_z - pointer_y / target, target


def pan_with_arrow(
    left: float, top: float, zoom: float, direction: str,
    *, fast: bool = False, step_pixels: float = 80.0,
) -> tuple[float, float]:
    """Pan the viewport by a screen-space step in the arrow's direction.

    The camera moves in the indicated direction, so pressing Right reveals
    map terrain farther to the right. Screen-pixel steps feel consistent at
    any zoom; Shift triples the distance. Like mouse-drag panning, this does
    not clamp the viewport to the map edges.
    """
    if not all(math.isfinite(value) for value in (left, top, zoom, step_pixels)):
        raise ValueError("pan inputs must be finite")
    if zoom <= 0 or step_pixels <= 0:
        raise ValueError("zoom and pan step must be positive")
    vectors = {
        "Left": (-1, 0),
        "Right": (1, 0),
        "Up": (0, -1),
        "Down": (0, 1),
    }
    if direction not in vectors:
        raise ValueError(f"unsupported pan direction: {direction!r}")
    dx, dz = vectors[direction]
    distance = step_pixels * (3 if fast else 1) / zoom
    return left + dx * distance, top + dz * distance
