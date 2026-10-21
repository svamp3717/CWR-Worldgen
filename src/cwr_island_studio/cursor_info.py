# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast, read-only cursor inspection of terrain and texture map cells."""
from __future__ import annotations

import math

from .core import IslandDocument


def cursor_readout(
    document: IslandDocument, x: float, z: float, layer: str,
) -> str:
    """Report exactly the cell used by the elevation/texture viewport.

    The underlying terrain_at and texture_at APIs include unsaved brush edits.
    No image, object or asset parsing is needed when the pointer moves.
    """
    if not (math.isfinite(x) and math.isfinite(z)):
        return "Outside map"
    if not (0.0 <= x < document.world_width and
            0.0 <= z < document.world_height):
        return "Outside map"
    column = min(document.width - 1, int(x / document.cell_size))
    row = min(document.height - 1, int(z / document.cell_size))
    position = f"X {x:,.1f} m  Z {z:,.1f} m"
    if layer == "Elevation":
        height = document.terrain_at(column, row)
        return f"{position}  |  Height {height:,.2f} m"
    if layer in ("Texture slots", "Actual textures"):
        slot = document.texture_at(column, row)
        name = (
            document.texture_paths[slot]
            if 0 <= slot < len(document.texture_paths) else ""
        )
        return f"{position}  |  Texture {slot}: {name or '(unnamed)'}"
    return position
