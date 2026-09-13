# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep vector forest context inside late object-generation wrapper chains."""
from __future__ import annotations

from . import forest_vector_performance_policy as _forest
from . import generator as _generator
from . import osm as _osm

_INSTALLED = False


def install_forest_generation_binding_policy() -> None:
    """Bind both namespaces before final building-road policy captures the core.

    ``generator.py`` imported ``generate_world_objects`` by value, while the late
    final-building clearance policy captures ``osm.generate_world_objects`` and
    then rebinds both names. The vector forest context therefore has to be the
    function that late policy captures from *osm*, not only from generator.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _osm.generate_world_objects = _forest._generate_with_vector_forest_context
    _generator.generate_world_objects = _forest._generate_with_vector_forest_context
    _INSTALLED = True
