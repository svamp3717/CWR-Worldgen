# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove duplicate road beneath bridge interiors and add terminal road underlays."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from . import bridge_render_policy as _bridge
from . import generator as _generator
from . import osm as _osm
from . import paved_junction_policy as _paved
from . import playability as _p

_ALIGNMENT_TOLERANCE_DEGREES = 30.0
_ENDPOINT_KEEP_METRES = 0.10
_TERMINAL_UNDERLAY_METRES = _bridge._STOCK_MODULE_SPACING_METRES
_TERMINAL_PIECE_LENGTH_METRES = 25.0
_ORIGINAL_FIT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _BridgeSpan:
    points: tuple[tuple[float, float], ...]
    road_width: float
    source_points: tuple[tuple[float, float], ...] = ()
    source_start_measure: float = 0.0
    source_end_measure: float = 0.0
    road_model_path: str = r"o\road\sil25.p3d"


def _undirected_heading_difference(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, 180.0 - difference)


def _nearest_polyline_measure(points, point) -> tuple[float, float, float, float]:
    cumulative = 0.0
    total = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
    best = None
    px, pz = point
    for index, (start, end) in enumerate(zip(points, points[1:])):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length2 = dx * dx + dz * dz
        length = math.sqrt(length2)
        if length <= 1.0e-9:
            continue
        fraction = ((px - start[0]) * dx + (pz - start[1]) * dz) / length2
        fraction = max(0.0, min(1.0, fraction))
        nearest = (start[0] + dx * fraction, start[1] + dz * fraction)
        distance = math.dist(point, nearest)
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        along = cumulative + length * fraction
        candidate = (distance, heading, along, index)
        if best is None or (candidate[0], candidate[3]) < (best[0], best[3]):
            best = candidate
        cumulative += length
    if best is None:
        return math.inf, 0.0, 0.0, total
    return best[0], best[1], best[2], total


def _feature_needs_bridge(feature, points, elevations, spec) -> bool:
    tags = feature.tags
    bridge = str(tags.get("bridge", "")).strip().casefold()
    explicit = (
        bridge not in {"", "no", "false", "0", "none"}
        or str(tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(tags.get("special", "")).strip().casefold() == "bridge"
    )
    elevated = _osm._numeric_tag(tags, "layer", 0.0) > 0.0
    if not explicit and not elevated:
        return False
    width = max(6.0, _osm.road_width_metres(tags))
    return _osm.road_span_has_in_game_water(
        points,
        elevations,
        cells=spec.cells,
        cell_size=spec.cell_size,
        sea_level=spec.sea_level,
        width=width,
    )


def _bridge_spans(dataset, projection, elevations, spec) -> tuple[_BridgeSpan, ...]:
    if not bool(getattr(spec, "bridges_enabled", True)):
        return ()
    warning_threshold = max(0, int(getattr(spec, "maximum_bridge_objects", 1000)))
    if warning_threshold <= 0:
        return ()
    bridge_limit = _osm._advisory_object_limit(
        warning_threshold,
        enabled=bool(getattr(spec, "advisory_object_limits", False)),
    )
    stock_spec = _bridge._stock_bridge_spec(spec)
    spans = []
    used_objects = 0
    projected = _osm.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        if _osm.road_bridge_crosses_ditch_only(feature, dataset, projection):
            continue
        source_chunks = _osm._bridge_module_chunks(points, _bridge._STOCK_MODULE_SPACING_METRES)
        if not source_chunks or not _feature_needs_bridge(feature, points, elevations, stock_spec):
            continue
        plan = _bridge.stock_bridge_span_plan(points, elevations, stock_spec)
        span_points = tuple(plan.points) if plan is not None else points
        required = plan.module_count if plan is not None else len(source_chunks)
        if required > bridge_limit - used_objects:
            continue
        used_objects += required
        source_start = 0.0
        source_end = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
        if plan is not None:
            first = _nearest_polyline_measure(points, span_points[0])[2]
            second = _nearest_polyline_measure(points, span_points[-1])[2]
            source_start, source_end = sorted((first, second))
        spans.append(
            _BridgeSpan(
                points=span_points,
                road_width=max(6.0, _osm.road_width_metres(feature.tags)),
                source_points=points,
                source_start_measure=source_start,
                source_end_measure=source_end,
                road_model_path=_p.road_model_for_tags(spec, feature.tags),
            )
        )
    return tuple(spans)


def _road_matches_interval(obj, points, road_width, start_measure, end_measure) -> bool:
    if len(points) < 2:
        return False
    distance, heading, along, total = _nearest_polyline_measure(points, (float(obj.x), float(obj.z)))
    lower = max(0.0, float(start_measure))
    upper = total if end_measure is None else min(total, float(end_measure))
    if upper <= lower + 1.0e-9:
        return False
    endpoint_keep = min(_ENDPOINT_KEEP_METRES, (upper - lower) * 0.02)
    if along <= lower + endpoint_keep or along >= upper - endpoint_keep:
        return False
    corridor = max(1.25, float(road_width) * 0.40)
    if distance > corridor:
        return False
    return _undirected_heading_difference(obj.heading_degrees, heading) <= _ALIGNMENT_TOLERANCE_DEGREES


def _road_matches_terminal_underlay(obj, points, road_width, start_measure=0.0, end_measure=None) -> bool:
    if len(points) < 2:
        return False
    total = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
    lower = max(0.0, float(start_measure))
    upper = total if end_measure is None else min(total, float(end_measure))
    length = upper - lower
    if length <= 1.0e-9:
        return False
    terminal = min(float(_TERMINAL_UNDERLAY_METRES), length * 0.5)
    return (
        _road_matches_interval(obj, points, road_width, lower, lower + terminal)
        or _road_matches_interval(obj, points, road_width, upper - terminal, upper)
    )


def _road_object_under_bridge(obj, spans) -> bool:
    if _paved._family(obj.model_path) is None:
        return False
    for span in spans:
        if _road_matches_terminal_underlay(obj, span.points, span.road_width):
            return False
        if (
            span.source_points
            and span.source_end_measure > span.source_start_measure
            and _road_matches_terminal_underlay(
                obj,
                span.source_points,
                span.road_width,
                span.source_start_measure,
                span.source_end_measure,
            )
        ):
            return False
        if _road_matches_interval(obj, span.points, span.road_width, 0.0, None):
            return True
        if (
            span.source_points
            and span.source_end_measure > span.source_start_measure
            and _road_matches_interval(
                obj,
                span.source_points,
                span.road_width,
                span.source_start_measure,
                span.source_end_measure,
            )
        ):
            return True
    return False


def _remove_bridge_underlays(report, spans):
    if not spans or not getattr(report, "objects", ()):
        return report, 0
    kept = tuple(obj for obj in report.objects if not _road_object_under_bridge(obj, spans))
    removed = len(report.objects) - len(kept)
    return (replace(report, objects=kept), removed) if removed else (report, 0)


def _terminal_model(path: str) -> str | None:
    family = _paved._family(path)
    mapping = {
        "sil": r"o\road\sil25.p3d",
        "asf": r"o\road\asf25.p3d",
        "kos": r"o\road\kos25.p3d",
        "ces": r"o\road\ces25.p3d",
        "silnice": r"data3d\silnice25.p3d",
        "asfaltka": r"data3d\asfaltka25.p3d",
        "cesta": r"data3d\cesta25.p3d",
    }
    return mapping.get(family)


def _point_on_chord(span: _BridgeSpan, distance: float) -> tuple[float, float]:
    start, end = span.points[0], span.points[-1]
    total = max(1.0e-9, math.dist(start, end))
    fraction = max(0.0, min(1.0, float(distance) / total))
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def _terminal_segments(span: _BridgeSpan):
    """Return 25 m ground-road segments covering the first bridge module at each end."""
    total = math.dist(span.points[0], span.points[-1])
    terminal = min(float(_TERMINAL_UNDERLAY_METRES), total * 0.5)
    if terminal <= 0.5:
        return ()
    segments = []
    seen = set()
    for base in (0.0, total - terminal):
        cursor = base
        limit = base + terminal
        while cursor + 0.5 < limit:
            length = min(_TERMINAL_PIECE_LENGTH_METRES, limit - cursor)
            if length < 5.0:
                break
            start = _point_on_chord(span, cursor)
            end = _point_on_chord(span, cursor + length)
            key = tuple(round(value, 3) for point in (start, end) for value in point)
            reverse = tuple(round(value, 3) for point in (end, start) for value in point)
            canonical = min(key, reverse)
            if canonical not in seen:
                seen.add(canonical)
                segments.append((start, end))
            cursor += length
    return tuple(segments)


def _has_matching_underlay(objects, start, end, model_path) -> bool:
    midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    heading = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
    target_family = _paved._family(model_path)
    for obj in objects:
        if _paved._family(obj.model_path) != target_family:
            continue
        if math.dist((float(obj.x), float(obj.z)), midpoint) > 4.0:
            continue
        if _undirected_heading_difference(obj.heading_degrees, heading) <= 12.0:
            return True
    return False


def _add_terminal_underlays(report, spans, elevations, spec):
    """Synthesize road pieces where the fitter stops before the terminal bridge modules."""
    if not spans:
        return report, 0
    objects = list(getattr(report, "objects", ()))
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0
    for span in spans:
        model = _terminal_model(span.road_model_path)
        if model is None:
            continue
        for start, end in _terminal_segments(span):
            if _has_matching_underlay(objects, start, end, model):
                continue
            obj = _p._road_object_on_slope(
                next_id,
                model,
                start,
                end,
                elevations,
                spec,
                vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
            )
            objects.append(obj)
            next_id += 1
            added += 1
    if not added:
        return report, 0
    return replace(report, objects=tuple(objects)), added


def _fit(dataset, projection, elevations, spec, *, starting_id: int = 1, progress_callback=None):
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    spans = _bridge_spans(dataset, projection, elevations, spec)
    cleaned, removed = _remove_bridge_underlays(report, spans)
    finished, added = _add_terminal_underlays(cleaned, spans, elevations, spec)
    if progress_callback is not None and (removed or added):
        progress_callback(
            99,
            f"Bridge road underlay: removed {removed:,} interior piece(s), added {added:,} terminal mask piece(s)",
        )
    return finished


def install_bridge_underlay_cleanup_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
