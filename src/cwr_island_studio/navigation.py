# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport navigation maths shared by pointer, keyboard and button zoom."""
from __future__ import annotations

import math
import re


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


def parse_game_position(raw: str) -> tuple[float, float, float | None]:
    """Parse OFP/CWA console positions in [X, Y, altitude] order.

    The in-game Y axis is the map's north/south direction. Internally the
    RVW4 editor displays that direction as Z, with Y reserved for elevation.
    Accept the exact getPos array or two horizontal numbers; altitude is
    parsed for validation but does not affect a top-down camera.
    """
    text = raw.strip()
    text = re.sub(r"^(?:position|pos|getpos\s+[a-z_]\w*)\s*[:=]\s*",
                  "", text, flags=re.IGNORECASE)
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        text = text[1:-1].strip()
    elif any(character in text for character in "[]{}"):
        raise ValueError("Use [X, Y, altitude] or X, Y")
    components = (
        [part.strip() for part in text.split(",")]
        if "," in text else text.split()
    )
    if len(components) not in (2, 3) or any(not part for part in components):
        raise ValueError("Enter [X, Y, altitude] from the game console, or X, Y")
    try:
        values = tuple(float(part) for part in components)
    except ValueError as exc:
        raise ValueError("Coordinates must be numeric metres") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Coordinates must be finite numbers")
    return values[0], values[1], values[2] if len(values) == 3 else None


def center_on_position(
    game_x: float, game_y: float, canvas_width: float,
    canvas_height: float, zoom: float,
) -> tuple[float, float]:
    """Center the viewport on in-game X/Y without changing the zoom.

    World X is east/west and in-game Y is north/south (the editor's Z).
    The third console coordinate, if provided, is height and is ignored.
    """
    if not all(math.isfinite(value) for value in (
        game_x, game_y, canvas_width, canvas_height, zoom,
    )):
        raise ValueError("Position and viewport parameters must be finite")
    if canvas_width <= 0 or canvas_height <= 0 or zoom <= 0:
        raise ValueError("Viewport dimensions and zoom must be positive")
    return (
        game_x - canvas_width / (2 * zoom),
        game_y - canvas_height / (2 * zoom),
    )


def game_console_teleport_command(x: float, z: float) -> str:
    """Return an OFP/CWA console teleport command for Island Studio's X/Z.

    Game getPos/setPos uses [east X, north Y, height], whereas WRP stores
    north as Z. Zero height places the player at ground level.
    """
    if not (math.isfinite(x) and math.isfinite(z)):
        raise ValueError("Map coordinates must be finite")
    return f"player setPos [{x:.3f}, {z:.3f}, 0]"
