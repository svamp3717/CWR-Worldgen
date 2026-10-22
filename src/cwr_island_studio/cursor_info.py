# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast, read-only cursor inspection of terrain and texture map cells."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .core import IslandDocument, MapObject
from .model_bounds import footprint_contains


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
        texture = texture_under_cursor(document, x, z)
        if texture is not None:
            slot, name = texture
            return f"{position}  |  Texture {slot}: {name or '(unnamed)'}"
    return position


def texture_under_cursor(
    document: IslandDocument, x: float, z: float,
) -> tuple[int, str] | None:
    """Return the texture slot and original WRP path at a world position.

    OPRW texture layers can be a different size from the elevation grid.
    Sample their own grid directly, as the actual-texture viewport does.
    """
    if not (math.isfinite(x) and math.isfinite(z)):
        return None
    if not (0 <= x < document.world_width and 0 <= z < document.world_height):
        return None
    if hasattr(document, "layer_width") and hasattr(document, "layer_height"):
        col = min(document.layer_width - 1,
                  int(x / document.world_width * document.layer_width))
        row = min(document.layer_height - 1,
                  int(z / document.world_height * document.layer_height))
        slot = int(document.textures[row, col])
    else:
        col = min(document.width - 1, int(x / document.cell_size))
        row = min(document.height - 1, int(z / document.cell_size))
        slot = document.texture_at(col, row)
    path = (
        document.texture_paths[slot]
        if 0 <= slot < len(document.texture_paths) else ""
    )
    return slot, path


@dataclass(frozen=True, slots=True)
class ObjectHoverTarget:
    """The screen-space geometry that was actually rendered for an object."""

    obj: MapObject
    sx: float
    sy: float
    kind: str = "marker"  # marker, segment, or outline
    points: tuple[tuple[float, float], ...] = ()
    bounds: tuple[float, float, float, float] | None = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.points:
            xs, ys = zip(*self.points)
            object.__setattr__(
                self, "bounds", (min(xs), min(ys), max(xs), max(ys)),
            )


def object_under_cursor(
    targets: Sequence[ObjectHoverTarget], sx: float, sy: float,
    *, marker_radius: float = 7.0, segment_radius: float = 5.0,
) -> MapObject | None:
    """Find the visible object under the pointer without scanning the WRP.

    The draw loop captures at most a few thousand visible model shapes.
    Check their screen geometry, preferring nearby visible markers or road
    strokes to large overlapping polygons. No P3D decoding happens on hover.
    """
    if not math.isfinite(sx) or not math.isfinite(sy):
        return None
    best: MapObject | None = None
    best_score = (math.inf, math.inf, math.inf)
    for target in targets:
        dx, dy = target.sx - sx, target.sy - sy
        origin_distance = dx * dx + dy * dy
        if target.kind == "marker":
            if origin_distance > marker_radius * marker_radius:
                continue
            score = (0, origin_distance, target.obj.index)
        else:
            box = target.bounds
            radius = segment_radius if target.kind == "segment" else 0.0
            if box is None or not (
                box[0] - radius <= sx <= box[2] + radius
                and box[1] - radius <= sy <= box[3] + radius
            ):
                continue
            if target.kind == "outline":
                if not footprint_contains(target.points, sx, sy):
                    continue
                score = (1, origin_distance, target.obj.index)
            elif target.kind == "segment" and len(target.points) == 2:
                (ax, ay), (bx, by) = target.points
                length_sq = (bx - ax) ** 2 + (by - ay) ** 2
                t = max(0.0, min(
                    1.0, ((sx - ax) * (bx - ax) + (sy - ay) * (by - ay)) / length_sq,
                )) if length_sq else 0.0
                distance_sq = (sx - ax - t * (bx - ax)) ** 2 + (
                    sy - ay - t * (by - ay)
                ) ** 2
                if distance_sq > segment_radius * segment_radius:
                    continue
                score = (0, distance_sq, target.obj.index)
            else:
                continue
        if score < best_score:
            best_score, best = score, target.obj
    return best


def object_cursor_readout(obj: MapObject | None) -> str:
    """Identify the model and its exact placement without changing selection."""
    if obj is None:
        return ""
    return (
        f"Object ID {obj.object_id:,} (record {obj.index:,}): {obj.model_path}  "
        f"|  Origin X {obj.x:,.1f}, Y {obj.y:,.1f}, Z {obj.z:,.1f} m  "
        f"|  Heading {obj.heading:.1f}°"
    )
