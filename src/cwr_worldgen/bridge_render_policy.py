# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep procedural bridge P3Ds within OFP/CWA model-size limits.

The core bridge planner currently emits one ``br_single`` object for the complete
extended bridge span. That is attractive on paper, but OFP/CWA has practical
Geometry/Roadway size limits. Long real-world bridges can therefore exist in the
WRP and PBO while rendering as nothing in game.

This policy preserves the planner's exact straight bridge axis, elevation and
heading, but replaces oversized single-span bridge objects with contiguous
start/middle/end modules no longer than 30 metres. It also post-processes cached
non-road placement results so an old cached 300 m bridge cannot bypass the fix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import re

from . import generator as _generator
from . import osm as _osm

# Stock OFP/CWA bridge/road infrastructure is conventionally modular at about
# 25-30 m. Keep generated Geometry and Roadway LODs in that same conservative
# range instead of relying on one terrain object hundreds of metres long.
_MAX_CWA_BRIDGE_MODULE_METRES = 30.0
_BRIDGE_SINGLE = re.compile(
    r"^br_single_w(?P<width>\d+)_l(?P<length>\d+)\.p3d$",
    re.IGNORECASE,
)

_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None
_INSTALLED = False


def _target_module_dm(spec) -> int:
    try:
        configured = float(getattr(spec, "bridge_module_length", 30.0))
    except (TypeError, ValueError):
        configured = 30.0
    if not math.isfinite(configured):
        configured = 30.0
    metres = min(_MAX_CWA_BRIDGE_MODULE_METRES, max(3.0, configured))
    return max(30, int(round(metres * 10.0)))


def _module_lengths_dm(total_dm: int, target_dm: int) -> tuple[int, ...]:
    """Partition one bridge length exactly into near-equal safe modules."""

    total_dm = max(30, int(total_dm))
    target_dm = max(30, int(target_dm))
    if total_dm <= target_dm:
        return (total_dm,)
    count = max(2, int(math.ceil(total_dm / target_dm)))
    base, remainder = divmod(total_dm, count)
    # ``count`` comes from a >=3 m target, so real bridge spans cannot create a
    # sub-3 m module here. Keep the guard for malformed/cached model names.
    if base < 30:
        return (total_dm,)
    return tuple(base + (1 if index < remainder else 0) for index in range(count))


def _split_bridge_object(obj, spec, *, next_object_id: int):
    """Return CWA-sized replacements for one generated ``br_single`` object."""

    path = str(getattr(obj, "model_path", "")).replace("/", "\\")
    if "\\" not in path:
        return (obj,), next_object_id
    prefix, filename = path.rsplit("\\", 1)
    match = _BRIDGE_SINGLE.fullmatch(filename)
    if match is None:
        return (obj,), next_object_id

    width_dm = int(match.group("width"))
    total_dm = int(match.group("length"))
    lengths_dm = _module_lengths_dm(total_dm, _target_module_dm(spec))
    if len(lengths_dm) == 1:
        return (obj,), next_object_id

    total_m = total_dm / 10.0
    heading_degrees = float(getattr(obj, "heading_degrees", 0.0))
    pitch_degrees = float(getattr(obj, "pitch_degrees", 0.0))
    heading = math.radians(heading_degrees)
    sin_heading = math.sin(heading)
    cos_heading = math.cos(heading)
    vertical_per_metre = math.tan(math.radians(pitch_degrees))

    cursor = -total_m * 0.5
    pieces = []
    last = len(lengths_dm) - 1
    for index, length_dm in enumerate(lengths_dm):
        length_m = length_dm / 10.0
        offset = cursor + length_m * 0.5
        if index == 0:
            subtype = "start"
        elif index == last:
            subtype = "end"
        else:
            subtype = "middle"
        model_path = (
            f"{prefix}\\br_{subtype}_w{width_dm:03d}_l{length_dm:03d}.p3d"
        )
        object_id = int(getattr(obj, "object_id", 0)) if index == 0 else next_object_id
        if index != 0:
            next_object_id += 1
        pieces.append(
            replace(
                obj,
                object_id=object_id,
                model_path=model_path,
                x=float(obj.x) + sin_heading * offset,
                y=float(obj.y) + vertical_per_metre * offset,
                z=float(obj.z) + cos_heading * offset,
            )
        )
        cursor += length_m
    return tuple(pieces), next_object_id


def _modularize_result(result, spec):
    """Replace oversized generated bridge objects and keep result accounting exact."""

    if result is None or not bool(getattr(spec, "procedural_bridges", True)):
        return result
    objects = tuple(getattr(result, "objects", ()) or ())
    if not objects:
        return result

    next_object_id = max((int(getattr(obj, "object_id", 0)) for obj in objects), default=0) + 1
    rewritten = []
    replacements: list[tuple[object, tuple[object, ...]]] = []
    for obj in objects:
        pieces, next_object_id = _split_bridge_object(
            obj, spec, next_object_id=next_object_id
        )
        rewritten.extend(pieces)
        if len(pieces) > 1:
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
        # Compatibility for hand-built/direct API results that predate the
        # emission-time aggregate. Asset generation still needs the new module
        # paths, so build the aggregate once from the already-rewritten objects.
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
    """Modularize both fresh and cached non-road placement results."""

    value = _ORIGINAL_LOAD_NONROAD_OBJECTS(*args, **kwargs)
    spec = kwargs.get("spec")
    if spec is None and len(args) >= 5:
        spec = args[4]
    if spec is None or not isinstance(value, tuple) or not value:
        return value
    modularized = _modularize_result(value[0], spec)
    if modularized is value[0]:
        return value
    return (modularized, *value[1:])


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

    # Bridge modules already carry absolute WRP deck transforms. Do not let a
    # LandContact LOD independently conform each module to bank/seabed terrain.
    from .bridge_grounding_policy import install_bridge_grounding_policy

    install_bridge_grounding_policy()
    _INSTALLED = True
