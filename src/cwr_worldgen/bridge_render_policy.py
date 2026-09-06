# SPDX-License-Identifier: GPL-3.0-or-later
"""Render generated bridge plans with stock CWA bridge modules.

The OSM bridge planner already solves the useful part of the problem: which road
span is a bridge, where its centreline lies, and what absolute deck Y/pitch it
needs. Earlier revisions represented that solution with world-local custom P3Ds.
Long P3Ds exceeded legacy Roadway limits; splitting them exposed more terrain-
object quirks around custom model classes and grounding.

For CWA output the conservative answer is to stop reimplementing a bridge model.
Translate each generated ``br_single`` placement into a contiguous chain of the
stock Resistance/Nogova 30 m bridge module. The WRP transform remains the one
solved by the procedural planner, while model behavior comes from a known-good
engine asset.

Cached non-road placement results are rewritten too, so users do not need to
clear their placement cache after upgrading this policy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import re

from . import generator as _generator
from . import osm as _osm

_STOCK_BRIDGE_MODEL = _osm.NOGOVA_BRIDGE_MODEL
_STOCK_BRIDGE_LENGTH_METRES = float(_osm.NOGOVA_BRIDGE_MODULE_LENGTH_METRES)
_BRIDGE_SINGLE = re.compile(
    r"^br_single_w(?P<width>\d+)_l(?P<length>\d+)\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None
_INSTALLED = False


def _target_spacing_metres(spec) -> float:
    """Return requested centre spacing, bounded by the fixed stock module length."""

    try:
        configured = float(getattr(spec, "bridge_module_length", 30.0))
    except (TypeError, ValueError):
        configured = _STOCK_BRIDGE_LENGTH_METRES
    if not math.isfinite(configured):
        configured = _STOCK_BRIDGE_LENGTH_METRES
    return min(_STOCK_BRIDGE_LENGTH_METRES, max(3.0, configured))


def _module_offsets(total_metres: float, target_spacing: float) -> tuple[float, ...]:
    """Return evenly spaced stock-module centres covering the complete span.

    Stock bridge models are fixed at 30 m. Centre spacing is therefore no greater
    than 30 m, giving either exact joins or a small deterministic overlap when a
    bridge is not an exact multiple. The first/last modules may extend slightly
    onto dry land, which is preferable to a seam over water and mirrors the
    existing stock-bridge path in ``osm.py``.
    """

    total = max(3.0, float(total_metres))
    spacing = min(_STOCK_BRIDGE_LENGTH_METRES, max(3.0, float(target_spacing)))
    tolerance = max(1.0e-6, spacing * 1.0e-9)
    count = max(1, int(math.ceil((total - tolerance) / spacing)))
    covered_step = total / count
    first = -total * 0.5 + covered_step * 0.5
    return tuple(first + index * covered_step for index in range(count))


def _split_bridge_object(obj, spec, *, next_object_id: int):
    """Replace one generated bridge P3D with stock CWA bridge modules."""

    path = str(getattr(obj, "model_path", "")).replace("/", "\\")
    if "\\" not in path:
        return (obj,), next_object_id
    _prefix, filename = path.rsplit("\\", 1)
    match = _BRIDGE_SINGLE.fullmatch(filename)
    if match is None:
        return (obj,), next_object_id

    total_metres = max(3.0, int(match.group("length")) / 10.0)
    offsets = _module_offsets(total_metres, _target_spacing_metres(spec))
    heading_degrees = float(getattr(obj, "heading_degrees", 0.0))
    pitch_degrees = float(getattr(obj, "pitch_degrees", 0.0))
    heading = math.radians(heading_degrees)
    sin_heading = math.sin(heading)
    cos_heading = math.cos(heading)
    vertical_per_horizontal_metre = math.tan(math.radians(pitch_degrees))

    pieces = []
    for index, offset in enumerate(offsets):
        object_id = int(getattr(obj, "object_id", 0)) if index == 0 else next_object_id
        if index != 0:
            next_object_id += 1
        pieces.append(
            replace(
                obj,
                object_id=object_id,
                model_path=_STOCK_BRIDGE_MODEL,
                x=float(obj.x) + sin_heading * offset,
                y=float(obj.y) + vertical_per_horizontal_metre * offset,
                z=float(obj.z) + cos_heading * offset,
            )
        )
    return tuple(pieces), next_object_id


def _modularize_result(result, spec):
    """Rewrite generated bridge objects and keep placement accounting exact."""

    if result is None or not bool(getattr(spec, "procedural_bridges", True)):
        return result
    objects = tuple(getattr(result, "objects", ()) or ())
    if not objects:
        return result

    next_object_id = max(
        (int(getattr(obj, "object_id", 0)) for obj in objects), default=0
    ) + 1
    rewritten = []
    replacements: list[tuple[object, tuple[object, ...]]] = []
    for obj in objects:
        pieces, next_object_id = _split_bridge_object(
            obj, spec, next_object_id=next_object_id
        )
        rewritten.extend(pieces)
        # Even a short one-piece bridge is a replacement when its custom model
        # becomes the stock CWA bridge model.
        if len(pieces) != 1 or pieces[0] is not obj:
            replacements.append((obj, pieces))

    if not replacements:
        return result

    added = sum(len(pieces) - 1 for _old, pieces in replacements)
    updates = {
        "objects": tuple(rewritten),
        "bridge_objects": int(getattr(result, "bridge_objects", 0)) + added,
    }

    original_usage = tuple(getattr(result, "model_usage", ()) or ())
    if original_usage:
        usage = Counter(dict(original_usage))
        for old, pieces in replacements:
            old_path = str(old.model_path)
            usage[old_path] -= 1
            if usage[old_path] <= 0:
                del usage[old_path]
            usage.update(piece.model_path for piece in pieces)
        updates["model_usage"] = tuple(
            sorted(usage.items(), key=lambda item: item[0].casefold())
        )
    elif hasattr(result, "model_usage"):
        usage = Counter(obj.model_path for obj in rewritten)
        updates["model_usage"] = tuple(
            sorted(usage.items(), key=lambda item: item[0].casefold())
        )

    return replace(result, **updates)


def _generate_world_objects(
    dataset,
    projection,
    raster,
    elevations,
    spec,
    *args,
    **kwargs,
):
    result = _ORIGINAL_GENERATE_WORLD_OBJECTS(
        dataset, projection, raster, elevations, spec, *args, **kwargs
    )
    return _modularize_result(result, spec)


def _load_nonroad_objects(*args, **kwargs):
    """Rewrite both fresh and cached non-road placement results."""

    value = _ORIGINAL_LOAD_NONROAD_OBJECTS(*args, **kwargs)
    spec = kwargs.get("spec")
    if spec is None and len(args) >= 5:
        spec = args[4]
    if spec is None or not isinstance(value, tuple) or not value:
        return value
    rewritten = _modularize_result(value[0], spec)
    if rewritten is value[0]:
        return value
    return (rewritten, *value[1:])


def install_bridge_render_policy() -> None:
    """Install before final building/road clearance captures generation hooks."""

    global _ORIGINAL_GENERATE_WORLD_OBJECTS, _ORIGINAL_LOAD_NONROAD_OBJECTS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_GENERATE_WORLD_OBJECTS = _osm.generate_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    _osm.generate_world_objects = _generate_world_objects
    _generator.generate_world_objects = _generate_world_objects
    _generator._load_nonroad_objects = _load_nonroad_objects
    _INSTALLED = True
