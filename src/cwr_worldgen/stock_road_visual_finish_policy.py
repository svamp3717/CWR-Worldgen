# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual safeguards for stock-road junctions.

CWA renders the whole road strip, not just its centreline. An unsupported skew T
keeps the legacy six-metre straight cap, and the core fitter may orient that
symmetric cap along the side arm. In game that produces a conspicuous rectangular
road slab across the main carriageway even though the logical road graph is valid.

Align every legacy stock cap with the most nearly continuous incident pair. This
is deliberately post-fit and does not move source road geometry. Curve-seam
helpers remain here for regression/diagnostic work, but are not allowed to veto
native curve selection until their engine-visible edge model is fully calibrated.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_geometry_policy as _geometry
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local_fit
from . import stock_road_model_geometry as _model_geometry

MAXIMUM_VISUAL_SEAM_TANGENT_ERROR_DEGREES = 0.75
LEGACY_CAP_AXIS_TOLERANCE_DEGREES = 0.50

_ORIGINAL_FIT = None
_ORIGINAL_CHAIN_IS_SEAM_SAFE = None
_INSTALLED = False


def _axis_heading_difference(first: float, second: float) -> float:
    """Angular difference for a symmetric straight-road axis."""

    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _junction_incident_map(dataset, projection, spec):
    raw = {}
    positions = {}
    for feature, projected in zip(
        dataset.roads, _p.projected_road_polylines(dataset, projection)
    ):
        if not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads):
            continue
        points = tuple(_p._clean_road_points(projected))
        if len(points) < 2:
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        dirt = _p.road_is_dirt(feature.tags)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= 0.05:
                continue
            segment_key = f"{feature.osm_key}/{index:06d}"
            forward = _p._normalised_direction(start, end)
            reverse = (-forward[0], -forward[1])
            start_key = _p._road_node_key(start)
            end_key = _p._road_node_key(end)
            raw.setdefault(start_key, []).append(
                (forward, dirt, model, segment_key, feature.osm_key)
            )
            raw.setdefault(end_key, []).append(
                (reverse, dirt, model, segment_key, feature.osm_key)
            )
            positions.setdefault(start_key, start)
            positions.setdefault(end_key, end)

    result = {}
    for key, values in raw.items():
        unique = _p._unique_incidents(values)
        if len(unique) not in {3, 4}:
            continue
        incidents = tuple(
            _junction._Incident(value[0], _junction._family(value[2]), value[2])
            for value in unique
        )
        result[key] = (positions[key], incidents)
    return result


def _dominant_cap_heading(incidents, family: str) -> float | None:
    """Return the continuous same-family axis a legacy cap should follow."""

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    if incidents[first].family != family or incidents[second].family != family:
        return None
    return _junction._heading(incidents[first].direction)


def _realign_legacy_caps(report, dataset, projection, elevations, spec):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    incident_map = _junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        old = objects[index]
        match = _model_geometry.stock_straight_match(old.model_path)
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        key = _p._road_node_key((float(old.x), float(old.z)))
        junction = incident_map.get(key)
        if junction is None:
            continue
        node, incidents = junction
        if math.dist((float(old.x), float(old.z)), node) > 0.25:
            continue
        heading = _dominant_cap_heading(incidents, family)
        if heading is None:
            continue
        if (
            _axis_heading_difference(float(old.heading_degrees), heading)
            <= LEGACY_CAP_AXIS_TOLERANCE_DEGREES
        ):
            continue

        half = _model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6] * 0.5
        angle = math.radians(heading)
        direction = (math.sin(angle), math.cos(angle))
        start = (
            node[0] - direction[0] * half,
            node[1] - direction[1] * half,
        )
        end = (
            node[0] + direction[0] * half,
            node[1] + direction[1] * half,
        )
        fixed = _p._road_object_on_slope(
            int(old.object_id),
            old.model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + _local_fit.LEGACY_CAP_VERTICAL_BIAS_METRES
            ),
        )
        objects[index] = replace(
            fixed,
            x=float(node[0]),
            z=float(node[1]),
            heading_degrees=heading % 360.0,
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def _chord_heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _piece_tangents(measure, piece, start, end) -> tuple[float, float]:
    """Return rendered centreline tangent headings at a fitted piece's ends."""

    chord = _chord_heading(start, end)
    if _model_geometry.stock_curve_match(piece.model_path) is None:
        return chord, chord

    source_start = _p._nearest_polyline_heading(measure.points, start)
    source_end = _p._nearest_polyline_heading(measure.points, end)
    signed_turn = _p._signed_heading_delta(source_start, source_end)
    half_turn = _model_geometry.STOCK_CURVE_ANGLE_DEGREES * 0.5
    if signed_turn < 0.0:
        return (chord + half_turn) % 360.0, (chord - half_turn) % 360.0
    return (chord - half_turn) % 360.0, (chord + half_turn) % 360.0


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id=1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock road visual finish policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return report
    return _realign_legacy_caps(report, dataset, projection, elevations, spec)


def install_stock_road_visual_finish_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
