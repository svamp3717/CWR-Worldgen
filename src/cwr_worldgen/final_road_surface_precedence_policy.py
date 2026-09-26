# SPDX-License-Identifier: GPL-3.0-or-later
"""Guarantee paved-over-dirt precedence in the final generated road objects.

OSM-node underlay handling catches the normal dirt-to-asphalt T join, but late
junction replacement and fitted-curve geometry can move the final asphalt
surface away from the source centreline. This pass operates on the actual final
road objects, not inspector output: if an endpoint of a stock dirt slab lands
inside a final paved surface, only that endpoint is lowered while the far end
keeps the normal dirt elevation.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re
from typing import Callable, Sequence

from . import generator as _generator
from . import playability as _p

_DIRT_STRAIGHT = re.compile(
    r"^(?P<family>ces|cesta)(?P<nominal>25|12|6)\.p3d$",
    re.IGNORECASE,
)
_UNPAVED = re.compile(
    r"^(?:ces|cesta)(?:25|12|6|10\s+\d+)\.p3d$|^gravel",
    re.IGNORECASE,
)
_RAW_PROGRESS_PERCENT = 99
_INSTALLED = False
_ORIGINAL_FIT = None


def _filename(path: str) -> str:
    return str(path).replace("/", "\\").rsplit("\\", 1)[-1].casefold()


def _endpoint_polygon(point: tuple[float, float], half: float = 0.05):
    x, z = point
    return (
        (x - half, z - half),
        (x + half, z - half),
        (x + half, z + half),
        (x - half, z + half),
    )


def enforce_final_paved_over_dirt(
    report,
    elevations: Sequence[float],
    spec,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
):
    """Lower dirt endpoints that actually touch final paved road surfaces."""

    if not getattr(report, "objects", ()):
        return report

    # Reuse the same road-surface primitives used by final building clearance.
    # They cover stock curves/T/X pieces and generated paved ribbons/junctions.
    from . import final_building_road_clearance_policy as road_geometry

    paved_primitives = []
    for obj in report.objects:
        filename = _filename(obj.model_path)
        if _UNPAVED.match(filename):
            continue
        for primitive in road_geometry._road_object_primitives(obj, spec):
            midpoint = (
                (primitive.start[0] + primitive.end[0]) * 0.5,
                (primitive.start[1] + primitive.end[1]) * 0.5,
            )
            terrain = _p._sample_elevation(
                elevations,
                spec.cells,
                spec.cell_size,
                midpoint[0],
                midpoint[1],
            )
            # Do not bury a ground dirt track merely because a bridge crosses
            # above it in X/Z.
            if (
                abs(float(primitive.elevation) - float(terrain))
                <= road_geometry._MAXIMUM_VERTICAL_TERRAIN_GAP_METRES
            ):
                paved_primitives.append(primitive)

    if not paved_primitives:
        return report

    paved_index = road_geometry._RoadPrimitiveIndex(tuple(paved_primitives))
    replacements = {}
    lowered_endpoints = 0

    for obj in report.objects:
        match = _DIRT_STRAIGHT.fullmatch(_filename(obj.model_path))
        if match is None:
            continue
        nominal = int(match.group("nominal"))
        length = _p.stock_road_piece_length_metres(
            obj.model_path,
            nominal,
            float(spec.road_segment_length),
        )
        start, end = _p._model_axis(obj, length)

        hits = []
        for endpoint in (start, end):
            conflicts, _checked = road_geometry._conflicts(
                _endpoint_polygon(endpoint),
                paved_index,
            )
            hits.append(bool(conflicts))
        if not any(hits):
            continue

        replacements[int(obj.object_id)] = _p._road_object_on_slope_endpoint_offsets(
            int(obj.object_id),
            obj.model_path,
            start,
            end,
            elevations,
            spec,
            start_vertical_offset=(
                _p._MIXED_DIRT_UNDERLAY_VERTICAL_OFFSET_METRES
                if hits[0]
                else _p._STOCK_DIRT_VERTICAL_OFFSET_METRES
            ),
            end_vertical_offset=(
                _p._MIXED_DIRT_UNDERLAY_VERTICAL_OFFSET_METRES
                if hits[1]
                else _p._STOCK_DIRT_VERTICAL_OFFSET_METRES
            ),
        )
        lowered_endpoints += int(hits[0]) + int(hits[1])

    if not replacements:
        return report

    if progress_callback is not None:
        progress_callback(
            _RAW_PROGRESS_PERCENT,
            "Enforcing final paved-over-dirt precedence "
            f"({lowered_endpoints:,} dirt endpoint(s) lowered)",
        )
    return replace(
        report,
        objects=tuple(
            replacements.get(int(obj.object_id), obj)
            for obj in report.objects
        ),
    )


def install_final_road_surface_precedence_policy() -> None:
    """Install after final paved-junction/seam generation."""
    global _INSTALLED, _ORIGINAL_FIT
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects

    def precedence_fit(
        dataset,
        projection,
        elevations,
        spec,
        *,
        starting_id: int = 1,
        progress_callback=None,
    ):
        report = _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        return enforce_final_paved_over_dirt(
            report,
            elevations,
            spec,
            progress_callback=progress_callback,
        )

    _p.fit_road_objects = precedence_fit
    _generator.fit_road_objects = precedence_fit
    _INSTALLED = True
