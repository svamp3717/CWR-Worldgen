# SPDX-License-Identifier: GPL-3.0-or-later
"""Route CWA bridge generation through the existing stock bridge planner.

The core OSM generator already has two bridge implementations:

* ``procedural_bridges=True`` authors one world-local generated P3D spanning the
  complete extended bridge; and
* ``procedural_bridges=False`` places the stock Resistance/Nogova 30 m bridge
  modules, with stock-specific deck offsets, footprints and object budgeting.

The generated-P3D route has proven unreliable in OFP/CWA: a single long model
exceeds Roadway limits, while split custom models still interact badly with the
legacy terrain-object rules. Do not translate those placements after the fact,
because the stock bridge path intentionally uses different deck-height math.
Instead, override only the bridge-mode field before the existing generator and
placement cache see the spec. Everything else remains the user's original spec.

Passing the overridden spec into ``_load_nonroad_objects`` is deliberate: the
placement-cache key includes ``procedural_bridges``, so old cached custom bridge
placements cannot survive this policy.
"""
from __future__ import annotations

from dataclasses import is_dataclass, replace

from . import generator as _generator
from . import osm as _osm

_ORIGINAL_GENERATE_WORLD_OBJECTS = None
_ORIGINAL_LOAD_NONROAD_OBJECTS = None
_INSTALLED = False


class _StockBridgeSpecProxy:
    """Read-through spec view for non-dataclass compatibility callers/tests."""

    __slots__ = ("_base", "procedural_bridges")

    def __init__(self, base) -> None:
        self._base = base
        self.procedural_bridges = False

    def __getattr__(self, name):
        return getattr(self._base, name)


def _stock_bridge_spec(spec):
    """Return an equivalent spec whose bridge implementation is stock CWA."""

    if not bool(getattr(spec, "procedural_bridges", True)):
        return spec
    if is_dataclass(spec):
        return replace(spec, procedural_bridges=False)
    return _StockBridgeSpecProxy(spec)


def _generate_world_objects(
    dataset,
    projection,
    raster,
    elevations,
    spec,
    *args,
    **kwargs,
):
    """Use the core generator unchanged except for its bridge implementation."""

    return _ORIGINAL_GENERATE_WORLD_OBJECTS(
        dataset,
        projection,
        raster,
        elevations,
        _stock_bridge_spec(spec),
        *args,
        **kwargs,
    )


def _load_nonroad_objects(*args, **kwargs):
    """Make the stock bridge mode part of the non-road placement cache key."""

    positional = list(args)
    named = dict(kwargs)
    if "spec" in named:
        named["spec"] = _stock_bridge_spec(named["spec"])
    elif len(positional) >= 5:
        positional[4] = _stock_bridge_spec(positional[4])
    return _ORIGINAL_LOAD_NONROAD_OBJECTS(*positional, **named)


def install_bridge_render_policy() -> None:
    """Install before later non-road/building wrappers capture generation hooks."""

    global _ORIGINAL_GENERATE_WORLD_OBJECTS, _ORIGINAL_LOAD_NONROAD_OBJECTS, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_GENERATE_WORLD_OBJECTS = _osm.generate_world_objects
    _ORIGINAL_LOAD_NONROAD_OBJECTS = _generator._load_nonroad_objects

    # Direct/API callers and the cached build path both reach the same stock
    # bridge planner. The user's original spec object is never mutated.
    _osm.generate_world_objects = _generate_world_objects
    _generator.generate_world_objects = _generate_world_objects
    _generator._load_nonroad_objects = _load_nonroad_objects
    _INSTALLED = True
