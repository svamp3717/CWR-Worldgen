# SPDX-License-Identifier: GPL-3.0-or-later
"""Road editing and fast 2D previews, independent of terrain texture slots."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_STOCK = re.compile(r"(?:^|\\)(sil|asf|kos|ces)(25|12|6)\.p3d$", re.I)
_GENERATED = re.compile(
    r"(?:^|\\)(?:road_(?:paved|gravel|dirt)|paved|gravel|dirt)_w\d+_l(\d+)(?:_[lr]\d+)?\.p3d$",
    re.I,
)
# Stock modules have nominal 25 / 12.5 / 6.25 metre lengths.
_STOCK_LENGTH = {"25": 25.0, "12": 12.5, "6": 6.25}


@dataclass(frozen=True, slots=True)
class RoadPlacement:
    x: float
    z: float
    heading: float


def is_road_model(path: str | bytes) -> bool:
    if isinstance(path, bytes):
        raw = path.split(b"\0", 1)[0].decode("ascii", "ignore")
    else:
        raw = path
    value = raw.replace("/", "\\").lower()
    if not value.endswith(".p3d"):
        return False
    return bool(_STOCK.search(value) or _GENERATED.search(value)
                or value.startswith("o\\road\\")
                or "\\road\\" in value
                or "\\gravel_" in value)


def straight_length(model_path: str) -> float | None:
    path = model_path.replace("/", "\\")
    stock = _STOCK.search(path)
    if stock:
        return _STOCK_LENGTH[stock.group(2)]
    generated = _GENERATED.search(path)
    if generated:
        return int(generated.group(1)) / 10.0
    return None


def road_segment(
    x: float, z: float, heading: float, model_path: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    length = straight_length(model_path)
    if length is None or length <= 0:
        return None
    theta = math.radians(heading)
    dx, dz = math.sin(theta) * length * 0.5, math.cos(theta) * length * 0.5
    return (x - dx, z - dz), (x + dx, z + dz)


def plan_straight_road(
    start: tuple[float, float], end: tuple[float, float],
    model_path: str, *, maximum_pieces: int = 10000,
) -> tuple[RoadPlacement, ...]:
    """Produce a deterministic line of complete, equal-length road objects.

    This does not create junctions or curves, and leaves any terminal gap
    shorter than one segment visible rather than stretching a stock model.
    """
    length = straight_length(model_path)
    if length is None or length <= 0:
        raise ValueError("Road drawing requires a supported straight road model")
    x0, z0 = start
    x1, z1 = end
    if not all(math.isfinite(v) for v in (x0, z0, x1, z1)):
        raise ValueError("road endpoints must be finite")
    distance = math.hypot(x1 - x0, z1 - z0)
    count = int(math.floor((distance + 1.0e-8) / length))
    if count > maximum_pieces:
        raise ValueError("road stroke is too long; draw a shorter section")
    if count == 0:
        return ()
    dx, dz = (x1 - x0) / distance, (z1 - z0) / distance
    angle = math.degrees(math.atan2(dx, dz)) % 360.0
    return tuple(
        RoadPlacement(x0 + (i + 0.5) * length * dx,
                      z0 + (i + 0.5) * length * dz, angle)
        for i in range(count)
    )
