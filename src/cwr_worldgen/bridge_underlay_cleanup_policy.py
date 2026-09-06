# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove ordinary road pieces only beneath the stock bridge actually emitted."""
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
_ORIGINAL_FIT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _BridgeSpan:
    points: tuple[tuple[float, float], ...]
    road_width: float


def _undirected_heading_difference(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, 180.0 - difference)


def _nearest_polyline_measure(
    points: tuple[tuple[float, float], ...],
    point: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return distance, segment heading, along-distance and total polyline length."""

    cumulative = 0.0
    total = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
    best: tuple[float, float, float, int] | None = None
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


def _feature_needs_bridge(
    feature,
    points,
    elevations,
    spec,
) -> bool:
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
    """Reproduce the exact wet-only stock bridge corridor used by non-road output."""

    if not bool(getattr(spec, "bridges_enabled", True)):
        return ()

    warning_threshold = max(
        0, int(getattr(spec, "maximum_bridge_objects", 1000))
    )
    if warning_threshold <= 0:
        return ()
    bridge_limit = _osm._advisory_object_limit(
        warning_threshold,
        enabled=bool(getattr(spec, "advisory_object_limits", False)),
    )
    stock_spec = _bridge._stock_bridge_spec(spec)

    spans: list[_BridgeSpan] = []
    used_objects = 0
    projected = _osm.projected_road_polylines(dataset, projection)
    for feature, raw_points in zip(dataset.roads, projected):
        points = tuple(_p._clean_road_points(raw_points))
        if len(points) < 2:
            continue
        if _osm.road_bridge_crosses_ditch_only(feature, dataset, projection):
            continue
        source_chunks = _osm._bridge_module_chunks(
            points, _bridge._STOCK_MODULE_SPACING_METRES
        )
        if not source_chunks or not _feature_needs_bridge(
            feature, points, elevations, stock_spec
        ):
            continue

        plan = _bridge.stock_bridge_span_plan(
            points, elevations, stock_spec
        )
        # If the narrow centreline sampler cannot resolve a wet interval but the
        # core width-aware water test can, retain the historical full candidate
        # corridor rather than remove no underlay at all.
        span_points = plan.points if plan is not None else points
        required = plan.module_count if plan is not None else len(source_chunks)
        if required > bridge_limit - used_objects:
            continue
        used_objects += required
        spans.append(
            _BridgeSpan(
                points=tuple(span_points),
                road_width=max(6.0, _osm.road_width_metres(feature.tags)),
            )
        )
    return tuple(spans)


def _road_object_under_bridge(obj, spans: tuple[_BridgeSpan, ...]) -> bool:
    # Reuse the road-family catalogue so this covers Data3D roads, Resistance
    # roads and generated gravel, while excluding bridge P3Ds.
    if _paved._family(obj.model_path) is None:
        return False

    for span in spans:
        distance, heading, along, total = _nearest_polyline_measure(
            span.points, (float(obj.x), float(obj.z))
        )
        endpoint_keep = min(_ENDPOINT_KEEP_METRES, total * 0.02)
        if along <= endpoint_keep or along >= total - endpoint_keep:
            continue
        # Only the actual emitted bridge corridor is suppressed.  The dry
        # prefix/suffix of a long bridge-tagged OSM way therefore remains normal
        # fitted road all the way toward the stock bridge abutment.
        corridor = max(1.25, span.road_width * 0.40)
        if distance > corridor:
            continue
        if (
            _undirected_heading_difference(obj.heading_degrees, heading)
            > _ALIGNMENT_TOLERANCE_DEGREES
        ):
            continue
        return True
    return False


def _remove_bridge_underlays(report, spans: tuple[_BridgeSpan, ...]):
    if not spans or not getattr(report, "objects", ()):
        return report, 0
    kept = tuple(
        obj for obj in report.objects
        if not _road_object_under_bridge(obj, spans)
    )
    removed = len(report.objects) - len(kept)
    if not removed:
        return report, 0
    return replace(report, objects=kept), removed


def _fit(
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
    spans = _bridge_spans(dataset, projection, elevations, spec)
    cleaned, removed = _remove_bridge_underlays(report, spans)
    if removed and progress_callback is not None:
        progress_callback(
            99,
            f"Removed {removed:,} ordinary road piece(s) underneath emitted bridge footprints",
        )
    return cleaned


def install_bridge_underlay_cleanup_policy() -> None:
    """Install after final road deduplication and before building-road clearance."""

    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
