# SPDX-License-Identifier: GPL-3.0-or-later
"""Adopt stock paved-road habits measured in a hand-authored WrpTool WRP.

The reference ``CEEB_rezina.wrp`` supplied for road-placement comparison has a
very different failure profile from generated Lundby worlds: stock road and
junction P3Ds are yaw-only, ordinary connectors meet in plan, and native ten-
degree curve models are used as normal turn primitives instead of faceting most
bends with short rotated straights.

Apply those lessons conservatively to paved Resistance families only:

* ``sil``, ``asf`` and ``kos`` stock pieces are fitted in planar connector space,
  so their X/Z connector span remains the exact model-space length instead of
  shrinking by ``cos(pitch)``;
* final paved stock road/junction objects are emitted with zero pitch while
  retaining the terrain-derived centre Y, matching the reference WRP's stepped
  horizontal-piece placement style; and
* coherent paved bends are allowed to promote to a native ten-degree curve with
  fewer prerequisite short facets than the old emergency-only curve policy.

The later Kodiak reference stage owns final native-junction stub cleanup with a
broader tolerance, so this stage no longer adds a redundant fitter wrapper.
Stock ``ces`` and generated gravel retain their terrain-following 3D connector
policy until they are studied separately.
"""
from __future__ import annotations

from dataclasses import replace
import re

from . import playability as _p
from . import stock_road_3d_connector_policy as _three_d
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_model_geometry as _geometry


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_NATIVE_T = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?:sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_NATIVE_X = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)

REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS = 2
REFERENCE_MINIMUM_TOTAL_TURN_DEGREES = 8.0
REFERENCE_MINIMUM_PROMOTED_CURVES = 1
REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES = 3
REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = 3.0

_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_ORIGINAL_USES_MEASURED_RIGID_CONNECTORS = None
_INSTALLED = False


def _stock_family(model_path: str) -> str | None:
    straight = _geometry.stock_straight_match(str(model_path))
    if straight is not None:
        return straight.group("family").casefold()
    curve = _geometry.stock_curve_match(str(model_path))
    if curve is not None:
        return curve.group("family").casefold()
    return None


def _is_paved_stock_surface(model_path: str) -> bool:
    """Return True for stock paved straights, curves and native paved junctions."""

    family = _stock_family(model_path)
    if family is not None:
        return family in _PAVED_FAMILIES
    normalised = str(model_path).replace("/", "\\")
    return (
        _NATIVE_T.fullmatch(normalised) is not None
        or _NATIVE_X.fullmatch(normalised) is not None
    )


def _uses_measured_rigid_connectors(pieces) -> bool:
    """Keep terrain-length fitting for dirt/gravel, but not stock paved roads."""

    if _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS is None:
        raise RuntimeError("reference WRP road policy is not installed")

    families = set()
    for piece in pieces:
        model_path = str(piece.model_path)
        if _p.is_generated_gravel_road_model(model_path):
            return _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS(pieces)
        family = _stock_family(model_path)
        if family is not None:
            families.add(family)

    if families and families <= _PAVED_FAMILIES:
        return False
    return _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS(pieces)


def _road_object_on_slope(*args, **kwargs):
    """Emit paved stock road P3Ds horizontal while preserving their fitted Y."""

    if _ORIGINAL_ROAD_OBJECT_ON_SLOPE is None:
        raise RuntimeError("reference WRP road policy is not installed")
    obj = _ORIGINAL_ROAD_OBJECT_ON_SLOPE(*args, **kwargs)
    model_path = str(args[1] if len(args) > 1 else kwargs.get("model_path", ""))
    if not _is_paved_stock_surface(model_path):
        return obj
    if abs(float(obj.pitch_degrees)) <= 1.0e-12:
        return obj
    return replace(obj, pitch_degrees=0.0)


def install_stock_road_reference_wrp_policy() -> None:
    """Install the paved-only placement rules learned from the reference WRP."""

    global _ORIGINAL_ROAD_OBJECT_ON_SLOPE
    global _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS
    global _INSTALLED
    if _INSTALLED:
        return

    if not _three_d._INSTALLED:
        raise RuntimeError("3D stock-road connector policy must install first")
    if not _curve_usage._INSTALLED:
        raise RuntimeError("stock road curve-usage policy must install first")
    if not _candidate._INSTALLED:
        raise RuntimeError("Inspector candidate policy must install first")

    _ORIGINAL_ROAD_OBJECT_ON_SLOPE = _p._road_object_on_slope
    _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS = (
        _three_d._uses_measured_rigid_connectors
    )

    _three_d._uses_measured_rigid_connectors = _uses_measured_rigid_connectors
    _p._road_object_on_slope = _road_object_on_slope

    _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS = (
        REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS
    )
    _curve_usage._MINIMUM_TOTAL_TURN_DEGREES = REFERENCE_MINIMUM_TOTAL_TURN_DEGREES
    _curve_usage._MINIMUM_PROMOTED_CURVES = REFERENCE_MINIMUM_PROMOTED_CURVES
    _curve_usage._MAXIMUM_EXTRA_PIECES = REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES
    _candidate.INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = (
        REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
    )
    _candidate.INSPECTOR_CURVE_MAXIMUM_EXTRA_PIECES = (
        REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES
    )

    _INSTALLED = True
