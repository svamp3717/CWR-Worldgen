# SPDX-License-Identifier: GPL-3.0-or-later
"""Make WrpTool's Resistance junction inventory authoritative for generation.

The old junction policy already knew most of the Resistance names, but its
post-fit replacement could still reject a valid native junction when the legacy
six-metre cap happened to use a different family from the measured through road.
That made some WrpTool-listed ``asf``/``kos`` combinations effectively dead
assets even though the incident roads selected them correctly.

Install the exact WrpTool catalogue and let the measured incident families own
native-junction selection. Generated gravel is unaffected because it has no
WrpTool native junction asset and therefore keeps its existing fallback path.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_wrp_catalogue as _catalogue


_INSTALLED = False


def _replace_stock_junction_caps(report, dataset, projection, elevations, spec):
    count = int(getattr(report, "junction_cap_objects", 0))
    if count <= 0 or not report.objects:
        return report

    junctions = _junction._junction_incidents(dataset, projection, spec)
    objects = list(report.objects)
    changed = False

    for index in range(min(count, len(objects))):
        old = objects[index]
        cap_match = _junction._STOCK_CAP_MODEL.fullmatch(
            str(old.model_path).replace("/", "\\")
        )
        if cap_match is None:
            continue

        key = _p._road_node_key((float(old.x), float(old.z)))
        match = junctions.get(key)
        if match is not None:
            point, native = match
            asset = _catalogue.native_junction_asset(str(native.model_path))
            if (
                asset is not None
                and math.dist((float(old.x), float(old.z)), point) <= 0.20
            ):
                # The incident roads, not the generic legacy cap, determine the
                # surface combination. This is what makes every WrpTool-listed
                # T/X combination reachable by the production generator.
                objects[index] = _junction._native_junction_object(
                    old, native, elevations, spec
                )
                changed = True
                continue

        lowered = _junction._lower_legacy_stock_cap(old, elevations, spec)
        if lowered != old:
            objects[index] = lowered
            changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def install_stock_road_wrptool_catalogue_policy() -> None:
    """Install WrpTool's stock junction catalogue into the live fitter."""

    global _INSTALLED
    if _INSTALLED:
        return

    _junction._T_JUNCTION_MODELS = dict(_catalogue.WRPTOOL_T_JUNCTION_MODELS)
    _junction._X_JUNCTION_MODELS = dict(_catalogue.WRPTOOL_X_JUNCTION_MODELS)
    _junction._ALL_NATIVE_JUNCTION_MODELS = tuple(
        _catalogue.WRPTOOL_NATIVE_JUNCTION_MODELS
    )
    _junction._replace_stock_junction_caps = _replace_stock_junction_caps
    _INSTALLED = True
