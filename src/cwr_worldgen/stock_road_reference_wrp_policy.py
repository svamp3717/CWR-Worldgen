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
  horizontal-piece placement style;
* coherent paved bends are allowed to promote to a native ten-degree curve with
  fewer prerequisite short facets than the old emergency-only curve policy; and
* a native T/X owns the road centre all the way to its measured 6.25 m connector,
  so a late ordinary paved stub from the logical node to that connector is
  removed before serialization.

This deliberately does not alter stock ``ces`` or generated gravel. Those road
families retain the existing terrain-following 3D connector policy until they are
studied separately.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import generator as _generator
from . import playability as _p
from . import stock_road_3d_connector_policy as _three_d
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_inspector_candidate_policy as _candidate
from . import stock_road_model_geometry as _geometry
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_visual_finish_policy as _finish


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_NATIVE_T = re.compile(
    r"^(?:.*[\\/])kr_new_(?P<main>sil|asf|kos)_(?:sil|ces|asf|kos)_t\.p3d$",
    re.IGNORECASE,
)
_NATIVE_X = re.compile(
    r"^(?:.*[\\/])kr_new_silxsil\.p3d$",
    re.IGNORECASE,
)

# The reference WRP uses native curves as ordinary construction pieces. Keep the
# exact connector/corridor acceptance gates, but stop requiring a bend to have
# already degraded into three short straight slabs before curve promotion gets a
# chance to repair it.
REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS = 2
REFERENCE_MINIMUM_TOTAL_TURN_DEGREES = 8.0
REFERENCE_MINIMUM_PROMOTED_CURVES = 1
REFERENCE_MAXIMUM_EXTRA_CURVE_PIECES = 3
REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = 3.0

REFERENCE_NATIVE_NODE_TOLERANCE_METRES = 0.15
REFERENCE_NATIVE_CONNECTOR_MARGIN_METRES = 0.35

_ORIGINAL_ROAD_OBJECT_ON_SLOPE = None
_ORIGINAL_USES_MEASURED_RIGID_CONNECTORS = None
_ORIGINAL_FIT = None
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
    """Keep terrain-length fitting for dirt/gravel, but not stock paved roads.

    The 3D policy asks this helper at call time. Returning False for a pure paved
    stock variant set makes the chain solver use the model's full planar chord.
    That is required before the resulting P3D can safely be emitted at zero
    pitch; otherwise the fitter would still shorten X/Z by the old cosine term
    and flattening the object afterward would create a gap of its own.
    """

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


def _source_node_for_native_cap(cap, incident_map):
    logical = _paved._logical_center(cap)
    if logical is None:
        return None
    if incident_map:
        matched = _paved._matching_junction(incident_map, logical)
        if matched is not None:
            return tuple(matched[0])
    return tuple(logical)


def _drop_native_node_to_connector_stubs(report, dataset, projection, spec):
    """Remove a paved straight that exists only inside a native T/X footprint.

    The reference WRP's ordinary approaches terminate at the native connector;
    they do not continue from that connector to the logical intersection node.
    Some older composed policies can still leave exactly that 6.25 m centre stub
    after native selection. Use the source junction node as the final authority
    so the cleanup remains correct even if an asymmetric T's P3D origin is not at
    the logical road crossing.
    """

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    native_nodes = []
    for cap in report.objects[:cap_count]:
        if _paved._native_signature(str(cap.model_path)) is None:
            continue
        node = _source_node_for_native_cap(cap, incident_map)
        if node is not None:
            native_nodes.append(node)
    if not native_nodes:
        return report

    radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    connector_limit = radius + REFERENCE_NATIVE_CONNECTOR_MARGIN_METRES
    remove_ids: set[int] = set()
    for obj in report.objects[cap_count:]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None:
            continue
        family = match.group("family").casefold()
        if family not in _PAVED_FAMILIES:
            continue
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))]
        )
        axis = _p._model_axis(obj, length)
        for node in native_nodes:
            distances = tuple(math.dist(node, endpoint) for endpoint in axis)
            # The characteristic stale object has one endpoint at the logical
            # node and the other at, or just inside, the native 6.25 m connector.
            # Longer connector-to-outside approaches never satisfy this gate.
            if (
                min(distances) <= REFERENCE_NATIVE_NODE_TOLERANCE_METRES
                and max(distances) <= connector_limit
            ):
                remove_ids.add(int(obj.object_id))
                break

    if not remove_ids:
        return report
    return replace(
        report,
        objects=tuple(
            obj for obj in report.objects if int(obj.object_id) not in remove_ids
        ),
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
        raise RuntimeError("reference WRP road policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    # A report containing a native stock T/X has already gone through stock-road
    # fitting, even for lightweight/internal specs that do not expose the public
    # ``stock_road_piece_fitting`` flag. Run this harmless ownership cleanup on
    # the report itself rather than silently skipping it because a config field is
    # absent. The old guard did exactly that in the junction regression.
    return _drop_native_node_to_connector_stubs(
        report, dataset, projection, spec
    )


def install_stock_road_reference_wrp_policy() -> None:
    """Install the paved-only placement rules learned from the reference WRP."""

    global _ORIGINAL_ROAD_OBJECT_ON_SLOPE
    global _ORIGINAL_USES_MEASURED_RIGID_CONNECTORS
    global _ORIGINAL_FIT
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
    _ORIGINAL_FIT = _p.fit_road_objects

    # ``_three_d._stock_piece_chain`` resolves this helper dynamically even
    # though later curve wrappers sit outside it, so the paved chain reverts to
    # exact planar model lengths without disturbing generated gravel or ces.
    _three_d._uses_measured_rigid_connectors = _uses_measured_rigid_connectors
    _p._road_object_on_slope = _road_object_on_slope

    # Promote curves as a normal paved-road primitive rather than only after a
    # run has already become a conspicuous collection of short facets. The beam,
    # corridor and tangent checks remain unchanged and therefore authoritative.
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

    # Keep this outermost within the road stack. It does not invent geometry; it
    # only drops a redundant node-to-connector paved stub after every older
    # junction wrapper has had its chance to run.
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit

    _INSTALLED = True
