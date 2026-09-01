# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual safeguards for stock-road junction caps.

CWA renders the whole road strip, not just its centreline. An unsupported skew T
keeps the legacy six-metre straight cap, and the core fitter may orient that
symmetric cap along the side arm. In game that produces a conspicuous rectangular
road slab across the main carriageway even though the logical road graph is valid.

Align every legacy stock cap with the most nearly continuous incident pair. When
that through pair itself turns at the node, keep the fitted approaches as the
visible surface and sink the rigid straight cap into a low central-fill role.
Curve continuity is owned later by ``stock_road_final_continuity_policy`` and
physical final-WRP seam repair by ``stock_road_emitted_seam_policy``. The old
intermediate curve-underlay planner was disabled in production and is therefore
retained only as a no-op hook for composition compatibility.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_local_fit_policy as _local_fit
from . import stock_road_model_geometry as _model_geometry

LEGACY_CAP_AXIS_TOLERANCE_DEGREES = 0.50
MINIMUM_TURNING_LEGACY_CAP_TURN_DEGREES = 1.0
TURNING_LEGACY_CAP_VERTICAL_BIAS_METRES = -0.006
# Compatibility value still set by final continuity. Intermediate curve seam
# underlays themselves are retired; emitted-seam owns actual final WRP gaps.
MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES = 3.25

_ORIGINAL_FIT = None
_INSTALLED = False


def _axis_heading_difference(first: float, second: float) -> float:
    """Angular difference for a symmetric road-surface axis."""

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


def _dominant_through_geometry(incidents) -> tuple[float, float] | None:
    """Return the dominant through heading and its deviation from straight."""

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    first_heading = _junction._heading(incidents[first].direction)
    second_heading = _junction._heading(incidents[second].direction)
    separation = _junction._angular_distance(first_heading, second_heading)
    return first_heading, abs(180.0 - separation)


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

        through = _dominant_through_geometry(incidents)
        if through is None:
            continue
        through_heading, through_turn = through
        turning_through = (
            through_turn >= MINIMUM_TURNING_LEGACY_CAP_TURN_DEGREES
        )

        heading = _dominant_cap_heading(incidents, family)
        if heading is None:
            if not turning_through:
                continue
            # A mixed-surface turning node can still use its legacy cap as low
            # central fill. Once sunk, following the dominant through direction
            # gives that fill the most useful coverage without making it the
            # visible surface of either incident family.
            heading = through_heading

        axis_aligned = (
            _axis_heading_difference(float(old.heading_degrees), heading)
            <= LEGACY_CAP_AXIS_TOLERANCE_DEGREES
        )
        if axis_aligned and not turning_through:
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
        vertical_bias = (
            TURNING_LEGACY_CAP_VERTICAL_BIAS_METRES
            if turning_through
            else _local_fit.LEGACY_CAP_VERTICAL_BIAS_METRES
        )
        fixed = _p._road_object_on_slope(
            int(old.object_id),
            old.model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + vertical_bias
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


def _apply_curve_seam_covers(report, elevations, spec):
    """Compatibility no-op for the retired intermediate curve-underlay stage."""

    return report


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
    report = _realign_legacy_caps(report, dataset, projection, elevations, spec)
    return _apply_curve_seam_covers(report, elevations, spec)


def install_stock_road_visual_finish_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
