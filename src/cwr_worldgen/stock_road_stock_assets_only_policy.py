# SPDX-License-Identifier: GPL-3.0-or-later
"""Make stock OFP/CWA P3Ds the only road models emitted by production builds.

The road fitter accumulated several world-local helper families while debugging
Lundby: procedural gravel ribbons/junctions and paved fill/miter/wedge meshes.
They are useful experiments, but they also make the final WRP diverge from the
way hand-authored OFP/CWA worlds are assembled. Production road geometry now
uses stock ``sil/asf/kos/ces`` pieces and measured Resistance junction models
only.

This outer policy also fixes two reference-WRP lessons that were previously only
partly implemented. During an obstacle-checked junction transaction, a source T
may be locally regularised by as much as 15 degrees onto a measured native T;
the final modified geometry still has to pass the Inspector's strict connector
matcher. Native paved curve fitting also gets a wider, still road-width-bounded
source corridor so real 10-degree curve chains are not rejected merely because
OSM encoded the bend as a hard vertex.
"""
from __future__ import annotations

from . import asset_mapping as _asset_mapping
from . import generator as _generator
from . import osm as _osm
from . import playability as _p
from . import stock_road_inspector_candidate_enforcement_policy as _enforcement
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_relaxation_transaction_policy as _transaction
from . import stock_road_sharp_turn_policy as _sharp
from .procedural_infrastructure import (
    is_generated_gravel_road_model,
    paved_miter_angle_degrees,
    paved_wedge_angle_degrees,
)


MAXIMUM_NATIVE_T_PLANNING_THROUGH_TURN_DEGREES = 15.0
STOCK_CURVE_SOURCE_CORRIDOR_METRES = 1.25

_ORIGINAL_ROAD_MODEL_FOR_TAGS = None
_ORIGINAL_ASSET_MAPPING = None
_ORIGINAL_NATIVE_T = None
_ORIGINAL_FIT = None
_INSTALLED = False


class _StockRoadSpecProxy:
    """Expose the normal world spec while disabling procedural road assets."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    @property
    def procedural_gravel_roads(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _normalised_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _generated_gravel_model(model_path: str) -> bool:
    """Recognise every historical world-local gravel road variant."""

    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    # The public helper recognises the ordinary generated ribbon family. Curves
    # and junctions use sibling world-local names, so catch those by the common
    # gravel basename as well. Stock OFP/CWA roads are sil/asf/kos/ces and can
    # never collide with this test.
    return is_generated_gravel_road_model(model_path) or (
        "\\i\\" in path and filename.startswith("gravel")
    )


def _stock_road_model_for_tags(spec, tags):
    if _ORIGINAL_ROAD_MODEL_FOR_TAGS is None:
        raise RuntimeError("stock-road-only model policy is not installed")
    model = _ORIGINAL_ROAD_MODEL_FOR_TAGS(spec, tags)
    if _generated_gravel_model(str(model)):
        return str(getattr(spec, "dirt_road_model", r"o\road\ces25.p3d"))
    return model


def _stock_asset_mapping(spec, milestone_number: int, *, global_textures=()):
    if _ORIGINAL_ASSET_MAPPING is None:
        raise RuntimeError("stock-road-only asset mapping is not installed")
    return _ORIGINAL_ASSET_MAPPING(
        _StockRoadSpecProxy(spec),
        milestone_number,
        global_textures=global_textures,
    )


def _stock_native_t_dispatch(incidents):
    """Let the transaction regularise a skew T, then enforce strict final fit."""

    if _ORIGINAL_NATIVE_T is None:
        raise RuntimeError("stock-road-only native-T policy is not installed")

    # Backward compatibility for explicit synthetic generated-gravel incidents.
    # Normal production road selection above never creates them.
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

    # Do not reject the final geometry merely because the original source T
    # turned by more than 1.25 degrees. If the transaction succeeded, the
    # inserted connector-aligned approach points are what matter here. The
    # Inspector matcher still allows only 0.90 degree connector error.
    return _candidate._measured_native_t_junction(incidents)


def _generated_road_model(model_path: str) -> bool:
    path = _normalised_path(model_path)
    filename = path.rsplit("\\", 1)[-1]
    return (
        _generated_gravel_model(path)
        or filename == "paved_fill.p3d"
        or paved_miter_angle_degrees(filename) is not None
        or paved_wedge_angle_degrees(filename) is not None
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
        raise RuntimeError("stock-road-only final guard is not installed")
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
            "stock-road-only policy violation: generated road P3Ds survived "
            f"final fitting ({len(forbidden)} objects; {sample})"
        )
    return report


def install_stock_road_stock_assets_only_policy() -> None:
    """Install the final stock-only road selection, fitting and audit rules."""

    global _ORIGINAL_ROAD_MODEL_FOR_TAGS, _ORIGINAL_ASSET_MAPPING
    global _ORIGINAL_NATIVE_T, _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    if not _enforcement._FINAL_INSTALLED:
        raise RuntimeError("Inspector candidate final policy must install first")

    _ORIGINAL_ROAD_MODEL_FOR_TAGS = _osm.road_model_for_tags
    _ORIGINAL_ASSET_MAPPING = _asset_mapping.default_osm_asset_mapping
    _ORIGINAL_NATIVE_T = _enforcement._junction._native_t_junction
    _ORIGINAL_FIT = _p.fit_road_objects

    # Service/track/gravel source roads now resolve to the configured stock dirt
    # family even when an old milestone/config still says procedural_gravel_roads.
    _osm.road_model_for_tags = _stock_road_model_for_tags
    _p.road_model_for_tags = _stock_road_model_for_tags

    # The asset requirement pass must agree with placement, otherwise an unused
    # world-local gravel family would still be packed into the PBO.
    _asset_mapping.default_osm_asset_mapping = _stock_asset_mapping
    _generator.default_osm_asset_mapping = _stock_asset_mapping

    # Give an 11-degree Lundby-style T a chance to become a stock native T by
    # moving only its connector-local approach points inside the existing
    # all-or-nothing obstacle-checked transaction.
    _enforcement._junction._native_t_junction = _stock_native_t_dispatch

    # 1.25 m is still far inside the 4.55 m paved surface half-width. The beam's
    # exact connector/tangent tests and obstacle-conditioned source geometry stay
    # authoritative, but native curves are no longer rejected by a 0.60 m
    # centreline leash inherited from the faceted-straight era.
    _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES = STOCK_CURVE_SOURCE_CORRIDOR_METRES

    # Last word before WRP serialization: generated road models are a build
    # error, not a quiet fallback.
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
