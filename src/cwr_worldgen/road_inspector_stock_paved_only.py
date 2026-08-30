# SPDX-License-Identifier: GPL-3.0-or-later
"""Do not let retired generated paved helpers suppress grass-wedge findings.

Fresh worlds use stock sil/asf/kos assets only.  Old PBOs can still contain the
short-lived ``paved_fill``, ``paved_miter`` and ``paved_wedge`` experiments.  A
visible turn must not disappear from Road Inspector merely because one of those
objects exists in the WRP.  The final paved-wedge audit therefore accepts only a
real stock straight as third-party coverage.

This is intentionally conservative. Dirt/gravel remains outside the paved audit.
"""
from __future__ import annotations

from . import road_inspector_paved_wedge_audit as _audit


_ORIGINAL_STRICT_SURFACE_CONTAINS = None
_INSTALLED = False


def _stock_surface_contains(road, point: tuple[float, float]) -> bool:
    return bool(
        road.kind == "straight"
        and road.family in _audit._PAVED_FAMILIES
        and _audit._strict_straight_contains(road, point)
    )


def install() -> None:
    global _ORIGINAL_STRICT_SURFACE_CONTAINS, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_STRICT_SURFACE_CONTAINS = _audit._strict_surface_contains
    _audit._strict_surface_contains = _stock_surface_contains
    _INSTALLED = True
