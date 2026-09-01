# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility aliases for the sharp-turn owner.

Exact short-bend fitting now lives in ``stock_road_sharp_turn_policy``.  Keep
these helper aliases temporarily for late consumers while their imports are
retargeted; this module no longer installs or owns road behaviour.
"""
from __future__ import annotations

from .stock_road_sharp_turn_policy import (
    _baseline_short_straights,
    _curve_count,
    _measure_slice,
    _quantised_stock_exit_heading,
    _recover_exact_actions,
    install_stock_road_sharp_exact_policy,
)

__all__ = (
    "_baseline_short_straights",
    "_curve_count",
    "_measure_slice",
    "_quantised_stock_exit_heading",
    "_recover_exact_actions",
    "install_stock_road_sharp_exact_policy",
)
