# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove duplicate road beneath bridge interiors, but keep terminal road underlay."""
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
# Keep ordinary fitted road underneath the first stock bridge module at each
# end.  The bridge remains the drivable/visible upper surface; this lower road
# only masks coarse-grid grass/terrain that can otherwise show through around
# the abutment.  Interior bridge modules still have their duplicate road removed.
_TERMINAL_UNDERLAY_METRES = _bridge._STOCK_MODULE_SPACING_METRES
_ORIGINAL_FIT = None
_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _BridgeSpan:
    # Straight emitted bridge chord.
    points: tuple[tuple[float, float], ...]
    road_width: float
    # Original OSM/fitted-road polyline and the along-range replaced by the
    # emitted bridge. The stock bridge is straight, while the source road can
    # bow several metres away from that chord.
    source_points: tuple[tuple[float, float], ...] = ()
    source_start_measure: float = 0.0
    source_end_measure: float = 0.0


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
    """Reproduce the emitted bridge and source-road interval it replaces."""

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
        span_points = tuple(plan.points) if plan is not None else points
        required = plan.module_count if plan is not None else len(source_chunks)
        if required > bridge_limit - used_objects:
            continue
        used_objects += required

        source_start = 0.0
        source_end = sum(
            math.dist(start, end) for start, end in zip(points, points[1:])
        )
        if plan is not None:
            # The emitted stock chain is a straight chord. Project both chord
            # ends back onto the source road so cleanup can remove the curved
            # duplicate road even when it bows several metres away from the
            # bridge centreline.
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
            )
        )
    return tuple(spans)


def _road_matches_interval(
    obj,
    points: tuple[tuple[float, float], ...],
    road_width: float,
    start_measure: float,
    end_measure: float | None,
) -> bool:
    if len(points) < 2:
        return False
    distance, heading, along, total = _nearest_polyline_measure(
        points, (float(obj.x), float(obj.z))
    )
    lower = max(0.0, float(start_measure))
    upper = total if end_measure is None else min(total, float(end_measure))
    if upper <= lower + 1.0e-9:
        return False

    endpoint_keep = min(_ENDPOINT_KEEP_METRES, (upper - lower) * 0.02)
    if along <= lower + endpoint_keep or along >= upper - endpoint_keep:
        return False

    # Keep this narrow because the second check below follows the original
    # bridge-tagged source road exactly. Nearby parallel roads should survive.
    corridor = max(1.25, float(road_width) * 0.40)
    if distance > corridor:
        return False
    if (
        _undirected_heading_difference(obj.heading_degrees, heading)
        > _ALIGNMENT_TOLERANCE_DEGREES
    ):
        return False
    return True


def _road_matches_terminal_underlay(
    obj,
    points: tuple[tuple[float, float], ...],
    road_width: float,
    start_measure: float = 0.0,
    end_measure: float | None = None,
) -> bool:
    """Return True for aligned road beneath the first bridge module at either end."""
    if len(points) < 2:
        return False
    total = sum(math.dist(start, end) for start, end in zip(points, points[1:]))
    lower = max(0.0, float(start_measure))
    upper = total if end_measure is None else min(total, float(end_measure))
    length = upper - lower
    if length <= 1.0e-9:
        return False

    # If a bridge is only one or two stock modules long, its two terminal modules
    # legitimately cover the whole span. In that case keeping all fitted road
    # beneath it is intentional: the road is a lower visual mask, not the deck.
    terminal = min(float(_TERMINAL_UNDERLAY_METRES), length * 0.5)
    return (
        _road_matches_interval(
            obj, points, road_width, lower, lower + terminal
        )
        or _road_matches_interval(
            obj, points, road_width, upper - terminal, upper
        )
    )


def _road_object_under_bridge(obj, spans: tuple[_BridgeSpan, ...]) -> bool:
    # Reuse the road-family catalogue so this covers Data3D roads, Resistance
    # roads and generated gravel, while excluding bridge P3Ds.
    if _paved._family(obj.model_path) is None:
        return False

    for span in spans:
        # Preserve fitted road under the first stock module at both abutments.
        # It sits on the graded terrain below the bridge and masks any grass that
        # the coarse 50 m height grid would otherwise expose around the joint.
        if _road_matches_terminal_underlay(
            obj,
            span.points,
            span.road_width,
        ):
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

        # Remove anything beneath the interior of the emitted straight stock
        # bridge. This catches the common straight-road case while the terminal
        # module underlays above are deliberately retained.
        if _road_matches_interval(
            obj,
            span.points,
            span.road_width,
            0.0,
            None,
        ):
            return True

        # The stock bridge is a straight chord but the fitted OSM road can curve
        # several metres away from it. Remove the matching source-road pieces only
        # over the along-range actually replaced by the emitted bridge. Terminal
        # underlays remain as the grass-hiding abutment mask.
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
            f"Removed {removed:,} ordinary road piece(s) underneath bridge interiors; kept terminal underlays",
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
