# SPDX-License-Identifier: GPL-3.0-or-later
"""Prevent CWA dirt/gravel road artwork from printing over paved road artwork.

CWA's stock and generated road materials cannot be relied on to depth-sort
overlapping visual faces. A lower road's nominal position or its pitch does not
make its road texture invisible beneath a higher road. The legacy final fitter
deliberately extends pitched lower-priority terminal pieces *under* paving.
Resolve those intersections by clipping away the lower VISUAL footprint. Keep
unaffected chain pieces intact and preserve the generated Roadway contact LOD.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re
from typing import Sequence

from .model import WorldObject
from .procedural_infrastructure import (
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
    GENERATED_GRAVEL_VISUAL_TOP_METRES,
    custom_road_model_path,
    custom_road_model_signature,
)

# Leave only a few millimetres of coordinate tolerance. Seam models now encode
# lengths to 1 cm, so we need not withdraw 35-85 mm at either boundary.
_EDGE_SAFETY_METRES = 0.003
_MIN_EDGE_MODEL_METRES = 0.5


def _visual_footprint(axis, *, overlap: float, lateral_padding: float = 0.0):
    from . import final_road_dedup_policy as road

    if overlap <= 0.0:
        return road._surface_polygon(axis, padding=lateral_padding)
    # Gravel/dirt ribbons have a visual lip at *both* ends. Axis-only
    # intersection tests miss these 0.9 m faces even when nominally clipped.
    x0, z0 = axis.start
    x1, z1 = axis.end
    ux, uz = axis.ux, axis.uz
    width = axis.half_width + lateral_padding
    nx, nz = -uz, ux
    return (
        (x0 - ux * overlap + nx * width, z0 - uz * overlap + nz * width),
        (x0 - ux * overlap - nx * width, z0 - uz * overlap - nz * width),
        (x1 + ux * overlap - nx * width, z1 + uz * overlap - nz * width),
        (x1 + ux * overlap + nx * width, z1 + uz * overlap + nz * width),
    )


def _flush_piece(obj, axis, begin: float, end: float, *, object_id: int,
                 spec, elevations: Sequence[float]):
    from . import final_road_dedup_policy as road

    # One-centimetre P3D lengths keep the artwork within about 8 mm of its
    # boundary. The previous 0.1m floor and 35mm inset were visibly detached.
    available = end - begin - 2.0 * _EDGE_SAFETY_METRES
    length = math.floor(available * 100.0 + 1.0e-8) / 100.0
    if length < _MIN_EDGE_MODEL_METRES:
        return None
    mid = (begin + end) * 0.5
    left = mid - length * 0.5
    right = mid + length * 0.5
    start = road._axis_point(axis, left)
    finish = road._axis_point(axis, right)
    surface = "gravel" if axis.family in {"asf", "gravel"} else "dirt"
    path = custom_road_model_path(
        str(getattr(spec, "name", "world")),
        surface,
        axis.half_width * 2.0,
        length,
        flush_ends=True,
    )
    cells = int(getattr(spec, "cells", 0))
    size = float(getattr(spec, "cell_size", 0.0))
    if cells > 1 and size > 0 and len(elevations) == cells * cells:
        # Old terminal underlays deliberately pitch into the asphalt. Their
        # truncated visible portion must be fitted afresh to final terrain.
        # The following cross-slope grounding pass can then bank the edges.
        return road._p._road_object_on_slope(
            object_id, path, start, finish, elevations, spec,
            vertical_offset=road._p._road_surface_vertical_offset(surface),
        )
    # Lightweight geometry-only callers without elevation fixtures still
    # preserve the source skin elevation, including generated model origin lift.
    point = road._axis_point(axis, mid)
    source_skin = road._object_plane_height(obj, point)
    if custom_road_model_signature(obj.model_path) is not None:
        source_skin += (
            GENERATED_GRAVEL_VISUAL_TOP_METRES
            * math.cos(math.radians(float(obj.pitch_degrees)))
        )
    return WorldObject(
        object_id, path, point[0],
        source_skin - GENERATED_GRAVEL_VISUAL_TOP_METRES,
        point[1], float(obj.heading_degrees), 0.0,
    )


def _painted_gravel_blockers(report, spec):
    """Expose complete gravel artwork, including lips, as blockers for dirt."""
    from . import final_road_dedup_policy as road

    blockers = []
    for index, obj in enumerate(report.objects):
        axis = road._road_axis(obj, index, spec)
        if axis is not None and axis.family in {"asf", "gravel"}:
            signature = custom_road_model_signature(obj.model_path)
            lip = (
                GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
                if signature is not None and signature[0] == "gravel"
                and not re.search(r"_edge[1-9]?\.p3d$", str(obj.model_path), re.I)
                else 0.0
            )
            padding = (
                road._generated_curve_padding(obj.model_path, axis)
                if axis.curved_model else 0.0
            )
            blockers.append(road._blocker_from_polygon(
                obj, _visual_footprint(
                    axis, overlap=lip, lateral_padding=padding,
                ),
            ))
        elif road._is_gravel_junction_cap_model(obj.model_path):
            blockers.append(road._junction_blocker(obj))
    return tuple(blockers)


def enforce_visible_road_priority(report, spec, elevations: Sequence[float]):
    """Make lower-priority road *visual geometry* stop before higher surfaces.

    Retain the old road fitter's network connection calculation, but never
    trust an under-asphalt overlap to be visually hidden by its WRP height.
    Near intersections, replace only the conflicting dirt/gravel objects by
    exact-length flush-end models. Their Roadway LOD remains present, and paved
    Roadway takes over inside the intersection.
    """
    from . import final_road_dedup_policy as road

    if not report.objects:
        return report

    current = report
    for surface in ("gravel", "dirt"):
        paved = road._paved_blockers(current, spec)
        blockers = (
            paved if surface == "gravel"
            else paved + _painted_gravel_blockers(current, spec)
        )
        if not blockers:
            continue
        buckets: dict[tuple[int, int], set[int]] = {}
        for index, blocker in enumerate(blockers):
            for bucket in road._polygon_buckets(blocker.polygon):
                buckets.setdefault(bucket, set()).add(index)

        changed = False
        next_id = max(int(o.object_id) for o in current.objects) + 1
        output = []
        for index, obj in enumerate(current.objects):
            axis = road._lower_surface_axis(obj, index, spec)
            if axis is None or (
                (axis.family in {"asf", "gravel"}) != (surface == "gravel")
            ):
                output.append(obj)
                continue
            signature = custom_road_model_signature(obj.model_path)
            is_generated = signature is not None and signature[0] == surface
            edge = re.search(r"_edge[1-9]?\.p3d$", str(obj.model_path), re.I) is not None
            lip = (
                GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
                if is_generated and not edge else 0.0
            )
            polygon = road._surface_polygon(axis)
            curved_padding = (
                road._generated_curve_padding(obj.model_path, axis)
                if axis.curved_model else 0.0
            )
            painted_polygon = _visual_footprint(
                axis, overlap=lip, lateral_padding=curved_padding,
            )
            possible = set()
            for bucket in road._polygon_buckets(painted_polygon):
                possible.update(buckets.get(bucket, ()))
            nominal_intervals = []
            hits_visual_lip = False
            for bidx in sorted(possible):
                blocker = blockers[bidx]
                if (
                    blocker.elevation - axis.elevation
                    > road._DIRT_PAVED_OVERPASS_CLEARANCE_METRES
                ):
                    continue
                if road._polygon_area(road._clip_convex_polygon(
                    painted_polygon, blocker.polygon,
                )) <= 1.0e-5:
                    continue
                hits_visual_lip = True
                interval = road._blocked_interval(axis, polygon, blocker)
                if interval is not None:
                    nominal_intervals.append(interval)
            if not hits_visual_lip:
                output.append(obj)
                continue
            changed = True
            if axis.curved_model:
                # A custom or stock curve's visual ribbon bows outside the
                # chord's rectangular axis. A straight cutout would create an
                # incorrect shape; do not leave painted road faces in paving.
                continue
            if not nominal_intervals:
                # Nominal model rectangle is safe, but the 0.9m translucent
                # painted lip reaches asphalt. Same P3D, without either lip.
                if is_generated and not edge:
                    flush = custom_road_model_path(
                        str(getattr(spec, "name", "world")), surface,
                        axis.half_width * 2.0, axis.length,
                        float(signature[3]), flush_ends=True,
                    )
                    output.append(replace(obj, model_path=flush))
                else:
                    output.append(obj)
                continue

            # Do not preserve the old pitched underlays: the renderer can draw
            # their dirt/gravel faces over higher asphalt despite lower WRP Y.
            blocked = road._merge_intervals(nominal_intervals)
            if axis.curved_model:
                # A curved visual surface cannot be retiled as one straight
                # cutout; the paved intersection owns the conflicting curve.
                continue
            visible = road._clear_intervals(axis.length, blocked)
            first_piece = True
            for begin, end in visible:
                oid = int(obj.object_id) if first_piece else next_id
                piece = _flush_piece(
                    obj, axis, begin, end, object_id=oid,
                    spec=spec, elevations=elevations,
                )
                if piece is None:
                    continue
                output.append(piece)
                if not first_piece:
                    next_id += 1
                first_piece = False

        if changed:
            current = replace(current, objects=tuple(output))
    return current
