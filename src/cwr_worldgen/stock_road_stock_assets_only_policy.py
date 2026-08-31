# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep paved and dirt roads stock-only while allowing procedural gravel.

Paved production roads must use the stock ``sil/asf/kos`` families and measured
Resistance junctions. Dirt roads must use the configured stock ``ces`` family.
Generated gravel ribbons, curves and junctions are a separate supported road
family and remain enabled when ``procedural_gravel_roads`` is requested.

This outer policy also keeps two reference-WRP lessons that were previously only
partly implemented. During an obstacle-checked junction transaction, a source T
may be locally regularised by as much as 15 degrees onto a measured native T;
the final modified geometry still has to pass the Inspector's strict connector
matcher. Native paved curve fitting also gets a wider, still road-width-bounded
source corridor so real 10-degree curve chains are not rejected merely because
OSM encoded the bend as a hard vertex.
"""
from __future__ import annotations

from . import generator as _generator
from . import playability as _p
from . import stock_road_inspector_candidate_enforcement_policy as _enforcement
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_relaxation_transaction_policy as _transaction
from . import stock_road_sharp_turn_policy as _sharp
from .procedural_infrastructure import (
    paved_miter_angle_degrees,
    paved_wedge_angle_degrees,
)


MAXIMUM_NATIVE_T_PLANNING_THROUGH_TURN_DEGREES = 15.0
STOCK_CURVE_SOURCE_CORRIDOR_METRES = 1.25

_ORIGINAL_NATIVE_T = None
_ORIGINAL_FIT = None
_INSTALLED = False


def _normalised_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _generated_dirt_model(model_path: str) -> bool:
    """Reject any future world-local dirt P3D without confusing gravel with dirt."""

    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return "\\i\\" in path and filename.startswith("dirt") and filename.endswith(".p3d")


def _stock_native_t_dispatch(incidents):
    """Let the transaction regularise a skew T, then enforce strict final fit."""

    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("stock paved/dirt policy is not installed")

    # Generated gravel keeps its existing junction path. The stock T planning
    # changes below are for stock paved and stock ces incidents only.
    if _enforcement._contains_generated_gravel(incidents):
        return _ORIGINAL_NATIVE_T(incidents)

    if _transaction._PLANNING_RELAXED_JUNCTION.get():
        if _enforcement._through_turn_degrees(incidents) <= (
            MAXIMUM_NATIVE_T_PLANNING_THROUGH_TURN_DEGREES + 1.0e-9
        ):
            limit = _enforcement._planning_tolerance_degrees(incidents)
            if limit is not None:
                return _enforcement._measured_native_t_with_limit(incidents, limit)
        return None

    # Final placement uses the strict measured connector candidate against the
    # geometry that actually survived the obstacle-checked transaction.
    return _candidate._measured_native_t_junction(incidents)


def _generated_road_model(model_path: str) -> bool:
    """Return True only for generated paved/dirt road models forbidden in output."""

    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return (
        filename == "paved_fill.p3d"
        or paved_miter_angle_degrees(filename) is not None
        or paved_wedge_angle_degrees(filename) is not None
        or _generated_dirt_model(path)
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("stock paved/dirt final guard is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    forbidden = tuple(
        obj for obj in report.objects if _generated_road_model(str(obj.model_path))
    )
    if forbidden:
        sample = ", ".join(sorted({str(obj.model_path) for obj in forbidden})[:6])
        raise ValueError(
            "stock paved/dirt policy violation: generated paved or dirt road P3Ds "
            f"survived final fitting ({len(forbidden)} objects; {sample})"
        )
    return report


def install_stock_road_stock_assets_only_policy() -> None:
    """Install final stock-only paved/dirt rules while preserving generated gravel."""

    global _ORIGINAL_NATIVE_T, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _enforcement._FINAL_INSTALLED:
        raise RuntimeError("Inspector candidate final policy must install first")

    _ORIGINAL_NATIVE_T = _enforcement._junction._native_t_junction
    _ORIGINAL_FIT = _p.fit_road_objects

    # Give a skew Lundby-style stock T a chance to become a native T by moving
    # connector-local approach points inside the all-or-nothing obstacle-checked
    # transaction. Generated gravel incidents retain their own selector.
    _enforcement._junction._native_t_junction = _stock_native_t_dispatch

    # 1.25 m is still far inside the 4.55 m paved surface half-width. The beam's
    # exact connector/tangent tests and obstacle-conditioned source geometry stay
    # authoritative, but native curves are no longer rejected by the old 0.60 m
    # centreline leash inherited from the faceted-straight era.
    _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES = STOCK_CURVE_SOURCE_CORRIDOR_METRES

    # Last word before WRP serialization: generated paved or dirt helpers are a
    # build error. Generated gravel is intentionally allowed.
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
