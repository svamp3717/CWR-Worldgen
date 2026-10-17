# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect a straight OFP/CWA P3D fence donor's visible segment geometry.

An unmodified donor is placed directly in the world.  Its visual LOD tells
us the segment length and whether the long axis is local X or local Z.
No mesh conversion, rescaling or ODOL version guessing is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .legacy_proxy_models import (
    ProxyCloneError,
    _read_mlod_visual,
    _read_odol_visual,
)


@dataclass(frozen=True, slots=True)
class FenceDonorGeometry:
    source_format: str
    length_metres: float
    width_metres: float
    heading_offset_degrees: float
    long_axis: str
    minimum_y_metres: float = 0.0


def inspect_fence_donor_geometry(data: bytes, *, allow_ambiguous: bool = False) -> FenceDonorGeometry:
    """Infer horizontal length/axis from the first visible P3D LOD.

    The mapped line heading is the direction of travel. A +Z-aligned P3D
    needs 0 degrees, and a +X-aligned model needs 90 degrees, matching the
    existing OFP fence placement convention.
    """
    if data.startswith(b"MLOD"):
        lod, _, _ = _read_mlod_visual(data)
        format_name = "MLOD"
    elif data.startswith(b"ODOL"):
        lod, _, _ = _read_odol_visual(data)
        format_name = "ODOL"
    else:
        raise ProxyCloneError(
            f"unsupported fence P3D format {data[:4]!r}; expected MLOD or ODOL v6/v7"
        )
    points = tuple(lod.points)
    if not points:
        raise ProxyCloneError("fence visual LOD contains no points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    zs = [float(point[2]) for point in points]
    if not all(math.isfinite(v) for v in (*xs, *ys, *zs)):
        raise ProxyCloneError("fence visual LOD contains non-finite positions")
    x_length = max(xs) - min(xs)
    z_length = max(zs) - min(zs)
    length, width = max(x_length, z_length), min(x_length, z_length)
    if not 0.25 <= length <= 200.0:
        raise ProxyCloneError(
            f"implausible visual fence length {length:.2f} m; specify overrides instead"
        )
    if width >= length * 0.85 and not allow_ambiguous:
        raise ProxyCloneError(
            f"ambiguous fence axis (X={x_length:.2f} m, Z={z_length:.2f} m); "
            "supply explicit length and heading offset"
        )
    axis = "X" if x_length > z_length else "Z"
    return FenceDonorGeometry(
        source_format=format_name,
        length_metres=length,
        width_metres=width,
        heading_offset_degrees=90.0 if axis == "X" else 0.0,
        long_axis=axis,
        minimum_y_metres=min(ys),
    )
