# SPDX-License-Identifier: GPL-3.0-or-later
"""Restore ordinary road geometry when a stock paved junction cannot be fitted."""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import paved_junction_policy as _paved
from . import playability as _p
from . import road_quality_policy as _rq

_SUCCESS_DISTANCE_METRES = 0.75
_MAX_STABILIZATION_REFITS = 2
_ORIGINAL_FIT = None
_INSTALLED = False


def _junction_geometry(dataset, projection, spec):
    """Reserve stock-junction space only for plans still eligible for stock fitting.

    ``paved_junction_policy`` normally exposes only stock-paved junction geometry.
    During a fallback refit, ``_PLANS`` contains the subset that actually fitted in
    the previous pass. Failed stock-paved plans are restored to the ordinary road
    quality geometry so their arms are no longer trimmed back by the 32 m stock
    approach reserve. Non-stock junctions remain excluded exactly as before.
    """

    base = dict(_paved._ORIGINAL_GEOMETRY(dataset, projection, spec))
    all_plans = _paved._plans(dataset, projection, spec)
    active_plans = _paved._PLANS.get()
    if active_plans is None:
        active_plans = all_plans
        fallback_keys: set[tuple[int, int]] = set()
    else:
        fallback_keys = set(all_plans).difference(active_plans)

    result = {
        key: base[key]
        for key in fallback_keys
        if key in base
    }
    for key, plan in active_plans.items():
        if key not in base:
            continue
        result[key] = replace(
            base[key],
            axis=plan.axis,
            half_length=_paved._APPROACH_RESERVE,
            half_width=_paved._APPROACH_RESERVE,
            directions=tuple(connector.direction for connector in plan.connectors),
        )
    return result


def _successful_plan_keys(report, plans) -> frozenset[tuple[int, int]]:
    """Identify plans that actually emitted their stock T/X junction model."""

    wanted_models = {
        plan.model_path.replace("/", "\\").casefold()
        for plan in plans.values()
    }
    emitted_by_model: dict[str, list[tuple[float, float]]] = {}
    for obj in getattr(report, "objects", ()):
        model = obj.model_path.replace("/", "\\").casefold()
        if model not in wanted_models:
            continue
        emitted_by_model.setdefault(model, []).append((float(obj.x), float(obj.z)))

    successful = []
    for key, plan in plans.items():
        model = plan.model_path.replace("/", "\\").casefold()
        if any(
            math.dist(position, plan.point) <= _SUCCESS_DISTANCE_METRES
            for position in emitted_by_model.get(model, ())
        ):
            successful.append(key)
    return frozenset(successful)


def _base_refit(
    dataset,
    projection,
    elevations,
    spec,
    active_plans,
    *,
    starting_id: int,
    progress_callback,
):
    token = _paved._PLANS.set(dict(active_plans))
    try:
        return _paved._ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
    finally:
        _paved._PLANS.reset(token)


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    plans = _paved._plans(dataset, projection, spec)
    if not plans:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    # First preserve the normal paved-junction path. Most maps never need the
    # fallback and therefore pay only a cheap scan for emitted junction models.
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    successful_keys = _successful_plan_keys(report, plans)
    if len(successful_keys) == len(plans):
        return report

    active = {
        key: plans[key]
        for key in successful_keys
    }
    if progress_callback is not None:
        progress_callback(
            99,
            "Refitting paved junction fallbacks: "
            f"{len(plans) - len(active):,} stock junction(s) use ordinary connected roads",
        )

    # Refit from the road-quality layer with failed plans restored to ordinary
    # junction geometry. Re-apply only the stock junctions proven viable above.
    # A neighbouring fallback can very occasionally invalidate a formerly viable
    # stock approach, so allow two shrinking stabilization passes. If it still
    # changes after that, prefer a fully ordinary connected result over gaps.
    for _attempt in range(_MAX_STABILIZATION_REFITS):
        base_report = _base_refit(
            dataset,
            projection,
            elevations,
            spec,
            active,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        if not active:
            return base_report

        fitted = _paved._apply_plans(base_report, active, elevations, spec)
        stable_keys = _successful_plan_keys(fitted, active)
        if len(stable_keys) == len(active):
            return fitted
        active = {
            key: active[key]
            for key in stable_keys
        }

    # Geometry remained coupled after the bounded stabilization attempts. An
    # ordinary junction is visually less fancy but, unlike a 30 m grass gap,
    # remains a road.
    return _base_refit(
        dataset,
        projection,
        elevations,
        spec,
        {},
        starting_id=starting_id,
        progress_callback=progress_callback,
    )


def install_paved_junction_fallback_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_FIT = _p.fit_road_objects

    # Replace the geometry hook before gravel-junction policy captures it as its
    # underlying implementation. This preserves the existing wrapper order.
    _paved._junction_geometry = _junction_geometry
    _rq._junction_geometry = _junction_geometry
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
