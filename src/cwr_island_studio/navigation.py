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
