# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep strongly turning paved T nodes on the low fallback-cap path.

A Resistance T junction is rigid: its two main connectors are exactly opposite.
The balanced turning-T chooser can split a modest source-road bend across that
mesh, but Lundby23 demonstrates that accepting a 20.66-degree through-road bend
still leaves the native connectors roughly one to two metres from the fitted
approaches. At that point the existing fallback is visually safer: the actual
approach pieces remain the top road surface and the intersection-edge policy uses
only low same-family fill underneath them.

Tighten only the maximum through-road bend accepted by the already-conservative
balanced chooser. Near-orthogonal/skew Ts with a straight main axis keep their
existing measured native-junction behavior.
"""
from __future__ import annotations

from . import stock_road_skew_orientation_policy as _skew

MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES = 15.0

_INSTALLED = False


def install_stock_road_turning_t_fallback_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if not _skew._INSTALLED:
        raise RuntimeError("stock road skew-orientation policy must install first")
    _skew.MAXIMUM_TURNING_T_MAIN_BEND_DEGREES = (
        MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES
    )
    _INSTALLED = True
