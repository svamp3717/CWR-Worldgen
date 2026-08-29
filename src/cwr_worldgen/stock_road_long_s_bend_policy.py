# SPDX-License-Identifier: GPL-3.0-or-later
"""Let the exact S-bend fitter cover long junction-to-junction paved roads.

Lundby32 contains a roughly one-kilometre paved run whose source line reverses
curvature several times between two covered junction endpoints. The exact S-bend
policy already has the right safety gates for this case, but its historical
360-metre run cap rejects the road before the connector-locked beam is attempted.
The result is a long sequence of short rotated straight pieces and many exposed
outer-edge wedges.

Raise only the search-length ceiling. The existing exact policy still requires a
stock paved family, junction cover at both ends, a direction reversal, enough
short baseline facets to justify replacement, a 0.60 m source corridor, exact
internal connector continuity, and no more than its small existing piece-count
allowance. Open-ended roads, dirt/gravel roads and ordinary straight runs remain
untouched.
"""
from __future__ import annotations

from . import stock_road_s_bend_exact_policy as _s_exact


MAXIMUM_LONG_EXACT_S_BEND_RUN_METRES = 1200.0

_INSTALLED = False


def install_stock_road_long_s_bend_policy() -> None:
    """Extend the existing exact S-bend search to long covered paved runs."""

    global _INSTALLED
    if _INSTALLED:
        return
    if not _s_exact._INSTALLED:
        raise RuntimeError("stock road exact S-bend policy must install first")
    _s_exact.MAXIMUM_EXACT_S_BEND_RUN_METRES = max(
        float(_s_exact.MAXIMUM_EXACT_S_BEND_RUN_METRES),
        MAXIMUM_LONG_EXACT_S_BEND_RUN_METRES,
    )
    _INSTALLED = True
