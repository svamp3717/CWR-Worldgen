# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep Road Inspector on the same Resistance road catalogue as WrpTool.

The core parser understands the measured T/X geometry, but catalogue knowledge
used to be implicit in filename regexes. This diagnostic layer makes WrpTool's
actual stock inventory authoritative and reports a concrete model recommendation
when a source intersection is rendered with a generic or wrong central cap even
though a purpose-built Resistance junction exists.

``kr_new_kos.p3d`` is also retained as a known WrpTool road object instead of
being silently discarded. Its connector geometry is intentionally left empty
until the model's Memory LOD is measured, so Inspector never invents endpoints.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import road_inspector as _core
from . import stock_road_model_geometry as _geometry
from . import stock_road_wrp_catalogue as _catalogue


MAXIMUM_CATALOGUE_APPROACH_DISTANCE_METRES = 7.25
MAXIMUM_CATALOGUE_APPROACH_HEADING_ERROR_DEGREES = 35.0
_INSTALLED = False
_ORIGINAL_ROAD_OBJECT_FROM_RECORD = None
_ORIGINAL_SOURCE_INTERSECTION_ISSUES = None


def _heading(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _decode_model(values) -> str | None:
    raw = values[13].split(b"\0", 1)[0]
    if not raw:
        return None
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return None


def _road_object_from_record(values):
    if _ORIGINAL_ROAD_OBJECT_FROM_RECORD is None:
        raise RuntimeError("WrpTool Road Inspector catalogue is not installed")
    road = _ORIGINAL_ROAD_OBJECT_FROM_RECORD(values)
    if road is not None:
        asset = _catalogue.native_junction_asset(str(road.model_path))
        if asset is None:
            return road
        # The catalogue is authoritative for family/kind metadata. Physical
        # connector points remain those reconstructed by the measured core.
        return replace(
            road,
            family=asset.main_family,
            kind="junction_t" if asset.kind == "t" else "junction_x",
        )

    model_path = _decode_model(values)
    if model_path is None:
        return None
    normalized = _catalogue.normalise_stock_road_path(model_path)
    special = {
        _catalogue.normalise_stock_road_path(path)
        for path in _catalogue.WRPTOOL_SPECIAL_ROAD_MODELS
    }
    if normalized not in special:
        return None

    object_id = int(values[12])
    x, y, z = float(values[9]), float(values[10]), float(values[11])
    heading = math.degrees(math.atan2(-float(values[2]), float(values[0]))) % 360.0
    pitch_sine = max(-1.0, min(1.0, float(values[7])))
    pitch = math.degrees(math.asin(pitch_sine))
    return _core.RoadObject(
        object_id,
        model_path,
        x,
        y,
        z,
        heading,
        pitch,
        "kos",
        "stock_special",
        0.0,
        (x, z),
        (),
    )


def _dominant_pair(headings: tuple[float, ...]) -> tuple[int, int] | None:
    if len(headings) < 2:
        return None
    best = None
    for first in range(len(headings)):
        for second in range(first + 1, len(headings)):
            separation = _core._angular_distance(headings[first], headings[second])
            candidate = (abs(180.0 - separation), first, second)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else (best[1], best[2])


def _approach_family_for_heading(roads, node, heading, used_ids):
    candidates = []
    for road in roads:
        if road.kind in {"junction_t", "junction_x", "stock_special"}:
            continue
        if road.family not in _geometry.STOCK_HALF_WIDTHS_METRES:
            continue
        for endpoint in road.endpoints:
            if endpoint.object_id in used_ids:
                continue
            distance = math.dist(node, endpoint.point)
            if distance > MAXIMUM_CATALOGUE_APPROACH_DISTANCE_METRES:
                continue
            node_heading = _heading(node, endpoint.point)
            error = _core._angular_distance(node_heading, heading)
            if error > MAXIMUM_CATALOGUE_APPROACH_HEADING_ERROR_DEGREES:
                continue
            candidates.append(
                (error, distance, endpoint.object_id, endpoint.family, endpoint)
            )
    if not candidates:
        return None
    _error, _distance, object_id, family, endpoint = min(candidates)
    used_ids.add(object_id)
    return family, endpoint


def _catalogue_expected_model(roads, source):
    headings = tuple(source.headings_degrees)
    if len(headings) not in {3, 4}:
        return None

    used_ids: set[int] = set()
    matched = []
    for heading in headings:
        value = _approach_family_for_heading(roads, source.point, heading, used_ids)
        if value is None:
            return None
        matched.append(value)
    families = tuple(value[0] for value in matched)

    if len(headings) == 3:
        pair = _dominant_pair(headings)
        if pair is None:
            return None
        first, second = pair
        if families[first] != families[second]:
            return None
        branch = next(index for index in range(3) if index not in pair)
        model = _catalogue.WRPTOOL_T_JUNCTION_MODELS.get(
            (families[first], families[branch])
        )
        return None if model is None else (model, tuple(value[1] for value in matched))

    if len(set(families)) != 1:
        return None
    model = _catalogue.WRPTOOL_X_JUNCTION_MODELS.get(families[0])
    return None if model is None else (model, tuple(value[1] for value in matched))


def _central_cap(roads, node, tolerance):
    candidates = []
    for road in roads:
        if road.kind in {"junction_t", "junction_x"}:
            pass
        elif road.kind == "straight" and road.nominal_length_metres <= 6.26:
            pass
        else:
            continue
        distance = math.dist(road.logical_center, node)
        if distance <= tolerance:
            candidates.append((distance, road.object_id, road))
    return None if not candidates else min(candidates)[2]


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    if _ORIGINAL_SOURCE_INTERSECTION_ISSUES is None:
        raise RuntimeError("WrpTool Road Inspector catalogue is not installed")
    issues = list(
        _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
            roads,
            junctions,
            match_tolerance=match_tolerance,
        )
    )

    existing = {
        (
            issue.category,
            round(float(issue.x), 2),
            round(float(issue.z), 2),
        )
        for issue in issues
    }
    for source in junctions:
        expected = _catalogue_expected_model(roads, source)
        if expected is None:
            continue
        expected_model, approaches = expected
        cap = _central_cap(roads, source.point, max(0.90, float(match_tolerance)))
        if cap is not None and (
            _catalogue.normalise_stock_road_path(cap.model_path)
            == _catalogue.normalise_stock_road_path(expected_model)
        ):
            continue

        key = (
            "intersection_stock_asset_mismatch",
            round(float(source.point[0]), 2),
            round(float(source.point[1]), 2),
        )
        if key in existing:
            continue
        existing.add(key)

        actual = "no central road object" if cap is None else str(cap.model_path)
        object_ids = {endpoint.object_id for endpoint in approaches}
        models = {endpoint.model_path for endpoint in approaches}
        if cap is not None:
            object_ids.add(cap.object_id)
            models.add(cap.model_path)
        score = 72.0 if cap is not None else 82.0
        issues.append(
            _core.RoadIssue(
                "",
                _core._severity(score),
                score,
                "intersection_stock_asset_mismatch",
                float(source.point[0]),
                float(source.point[1]),
                tuple(sorted(object_ids)),
                tuple(sorted(models)),
                (
                    f"WrpTool's Resistance road catalogue has {expected_model} for "
                    f"this {len(source.headings_degrees)}-arm stock-road family combination, "
                    f"but the WRP uses {actual} at the intersection centre."
                ),
                (
                    f"Use one {expected_model} at the logical intersection and fit the "
                    "approaches to its measured Memory-LOD connectors instead of layering short caps."
                ),
                {
                    "expected_stock_model": expected_model,
                    "actual_central_model": actual,
                    "source_arm_count": len(source.headings_degrees),
                },
            )
        )
    return issues


def install() -> None:
    """Install catalogue-aware parsing and source-intersection diagnostics."""

    global _ORIGINAL_ROAD_OBJECT_FROM_RECORD, _ORIGINAL_SOURCE_INTERSECTION_ISSUES
    global _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_ROAD_OBJECT_FROM_RECORD = _core._road_object_from_record
    _ORIGINAL_SOURCE_INTERSECTION_ISSUES = _core._source_intersection_issues
    _core._road_object_from_record = _road_object_from_record
    _core._source_intersection_issues = _source_intersection_issues
    _INSTALLED = True
