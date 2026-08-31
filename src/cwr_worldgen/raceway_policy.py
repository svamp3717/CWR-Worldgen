# SPDX-License-Identifier: GPL-3.0-or-later
"""OSM raceway classification for the road pipeline.

This module deliberately does not install or re-install road fitting policies.
Road fitter composition is owned exclusively by :mod:`cwr_worldgen.road_pipeline`.
"""
from __future__ import annotations

from dataclasses import replace

from . import asset_mapping as _asset_mapping
from . import generator as _generator
from . import osm as _osm

_INSTALLED = False


def _raceway_default_osm_asset_mapping(original):
    """Add ``highway=raceway`` to the built-in paved-road asset rule."""

    def wrapped(spec, milestone_number: int, *, global_textures=()):
        mapping = original(spec, milestone_number, global_textures=global_textures)
        rules = []
        for rule in mapping.rules:
            if rule.rule_id != "road-paved":
                rules.append(rule)
                continue

            match = []
            for key, values in rule.match:
                if key == "highway" and "raceway" not in values:
                    values = (*values, "raceway")
                match.append((key, values))
            rules.append(replace(rule, match=tuple(match)))
        return replace(mapping, rules=tuple(rules))

    wrapped._cwr_raceway_policy = True  # type: ignore[attr-defined]
    return wrapped


def install_raceway_policy() -> None:
    """Treat OSM motor raceways as supported paved vehicle roads.

    ``highway=raceway`` defaults to the ordinary asphalt family. Explicit
    unpaved ``surface=*`` tags still win, so dirt and motocross circuits remain
    unpaved.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    _osm._MAJOR_HIGHWAYS.add("raceway")
    from . import normalization as _normalization

    _normalization._MAJOR_HIGHWAYS.add("raceway")

    original_mapping = _asset_mapping.default_osm_asset_mapping
    if not getattr(original_mapping, "_cwr_raceway_policy", False):
        wrapped_mapping = _raceway_default_osm_asset_mapping(original_mapping)
        _asset_mapping.default_osm_asset_mapping = wrapped_mapping
        # generator.py imports the function directly, so patch its bound module
        # reference in parallel with the source module.
        _generator.default_osm_asset_mapping = wrapped_mapping

    _INSTALLED = True
