# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep legacy paved intersection caps below the actual road approaches.

A stock six-metre straight is a poor visible intersection surface when the
through road bends at the node or the side arm is skewed.  The approaches are
already fitted all the way to the logical node, so letting that straight cap sit
above them replaces good road edges with one rectangular slab and creates the
mismatched kerbs/painted borders visible in game.

For same-family paved legacy T/X fallbacks, keep the existing cap only as a low
central underlay and add low same-family six-metre tongues along any incident
axis the cap itself does not cover.  The real road approaches remain on top and
therefore own the visible road edges.  The underlays merely fill the triangular
holes between those approaches.  Native measured junction P3Ds are left alone,
as are dirt, gravel and mixed-surface intersections.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_visual_finish_policy as _finish

_PAVED_FAMILIES = {"sil", "asf", "kos"}

# A legacy cap is useful as a central fill, but it must not win the z-buffer
# over the actual approach pieces.  Tongues sit one millimetre higher than the
# central fill so the directionally correct arm wins where both underlays meet.
INTERSECTION_CAP_UNDERLAY_BIAS_METRES = -0.004
INTERSECTION_TONGUE_UNDERLAY_BIAS_METRES = -0.003

# A 6.25 m stock short piece is shifted slightly out from the node.  It still
# reaches 1.75 m behind the logical intersection while overlapping 4.50 m of
# the incoming approach, which is enough to hide the wedge without throwing a
# large rectangle across the opposite carriageway.
INTERSECTION_TONGUE_BACKTRACK_METRES = 1.75
INTERSECTION_AXIS_DUPLICATE_TOLERANCE_DEGREES = 1.0
INTERSECTION_NODE_MATCH_TOLERANCE_METRES = 0.35

_ORIGINAL_FIT = None
_INSTALLED = False


def _legacy_paved_cap_family(obj) -> str | None:
    match = _geometry.stock_straight_match(str(obj.model_path))
    if match is None or int(match.group("length")) != 6:
        return None
    family = match.group("family").casefold()
    return family if family in _PAVED_FAMILIES else None


def _axis_heading_difference(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _heading_unit(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _matching_junction(incident_map, point: tuple[float, float]):
    key = _p._road_node_key(point)
    direct = incident_map.get(key)
    if direct is not None and math.dist(point, direct[0]) <= INTERSECTION_NODE_MATCH_TOLERANCE_METRES:
        return direct

    nearest = min(
        incident_map.values(),
        key=lambda value: math.dist(point, value[0]),
        default=None,
    )
    if nearest is None:
        return None
    if math.dist(point, nearest[0]) > INTERSECTION_NODE_MATCH_TOLERANCE_METRES:
        return None
    return nearest


def _same_family_paved_incidents(incidents, family: str) -> bool:
    return (
        len(incidents) in {3, 4}
        and family in _PAVED_FAMILIES
        and all(incident.family == family for incident in incidents)
    )


def _uncovered_incident_headings(incidents, cap_heading: float) -> tuple[float, ...]:
    """Return unique outward incident headings not covered by the cap axis."""

    result: list[float] = []
    for incident in incidents:
        heading = _junction._heading(incident.direction)
        if (
            _axis_heading_difference(heading, cap_heading)
            <= INTERSECTION_AXIS_DUPLICATE_TOLERANCE_DEGREES
        ):
            continue
        if any(
            _axis_heading_difference(heading, existing)
            <= INTERSECTION_AXIS_DUPLICATE_TOLERANCE_DEGREES
            for existing in result
        ):
            continue
        result.append(heading)
    return tuple(result)


def _road_object_for_span(
    object_id: int,
    model_path: str,
    start: tuple[float, float],
    end: tuple[float, float],
    elevations,
    spec,
    *,
    vertical_bias: float,
):
    return _p._road_object_on_slope(
        object_id,
        model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + float(vertical_bias)
        ),
    )


def _lower_legacy_cap(cap, node, elevations, spec):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    direction = _heading_unit(float(cap.heading_degrees))
    start = (
        float(node[0]) - direction[0] * half,
        float(node[1]) - direction[1] * half,
    )
    end = (
        float(node[0]) + direction[0] * half,
        float(node[1]) + direction[1] * half,
    )
    fixed = _road_object_for_span(
        int(cap.object_id),
        str(cap.model_path),
        start,
        end,
        elevations,
        spec,
        vertical_bias=INTERSECTION_CAP_UNDERLAY_BIAS_METRES,
    )
    # Keep the logical cap center exactly on the junction node.  A straight road
    # is symmetric, so this only guards against tiny slope-solver roundoff.
    return replace(
        fixed,
        x=float(node[0]),
        z=float(node[1]),
        heading_degrees=float(cap.heading_degrees) % 360.0,
    )


def _tongue_object(
    object_id: int,
    family: str,
    node: tuple[float, float],
    outward_heading: float,
    elevations,
    spec,
):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    back = min(
        max(0.0, float(INTERSECTION_TONGUE_BACKTRACK_METRES)),
        length - 0.25,
    )
    forward = length - back
    direction = _heading_unit(outward_heading)
    start = (
        float(node[0]) - direction[0] * back,
        float(node[1]) - direction[1] * back,
    )
    end = (
        float(node[0]) + direction[0] * forward,
        float(node[1]) + direction[1] * forward,
    )
    return _road_object_for_span(
        object_id,
        rf"o\road\{family}6.p3d",
        start,
        end,
        elevations,
        spec,
        vertical_bias=INTERSECTION_TONGUE_UNDERLAY_BIAS_METRES,
    )


def _seal_legacy_paved_intersections(report, dataset, projection, elevations, spec):
    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)),
        len(report.objects),
    )
    if cap_count <= 0:
        return report

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0
    changed = False

    for index in range(cap_count):
        cap = objects[index]
        family = _legacy_paved_cap_family(cap)
        if family is None:
            continue

        junction = _matching_junction(
            incident_map,
            (float(cap.x), float(cap.z)),
        )
        if junction is None:
            continue
        node, incidents = junction
        if not _same_family_paved_incidents(incidents, family):
            continue

        # This is exactly the fallback case where the rigid native junction was
        # rejected.  Keep its small central rectangle as fill only, not as the
        # visible road surface.
        objects[index] = _lower_legacy_cap(
            cap,
            node,
            elevations,
            spec,
        )
        changed = True

        for heading in _uncovered_incident_headings(
            incidents,
            float(cap.heading_degrees),
        ):
            objects.append(
                _tongue_object(
                    next_id,
                    family,
                    node,
                    heading,
                    elevations,
                    spec,
                )
            )
            next_id += 1
            added += 1

    if not changed and added == 0:
        return report

    required = len(objects)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after paved-intersection edge "
            f"coverage: requires {required:,} objects, "
            f"limit is {int(spec.max_road_objects):,}"
        )

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + added
        ),
    )


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
        raise RuntimeError("stock road intersection-edge policy is not installed")
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
    return _seal_legacy_paved_intersections(
        report,
        dataset,
        projection,
        elevations,
        spec,
    )


def install_stock_road_intersection_edge_policy() -> None:
    """Install the final paved-intersection edge underlay pass."""

    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _INSTALLED = True
