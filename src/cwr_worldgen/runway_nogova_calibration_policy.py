# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibrate generated runway-cell backgrounds against Nogova's stock terrain.

Generated runway PAAs replace an entire WRP terrain cell, including the area
outside the runway itself. The first path-aware approximation was still roughly
twice as bright in-game as the neighbouring stock ``o\\t1`` / ``o\\trava*``
terrain. Keep the correction isolated here so the runway renderer remains generic
while the Nogova preset can be tuned from real CWA screenshots.
"""
from __future__ import annotations

from . import runway_surface_policy as _runway


# Screenshot-calibrated source colours. CWA's terrain lighting makes generated
# RGB values appear substantially brighter in-game than the same nominal values
# suggest, so these are intentionally dark. They target the visible luminance of
# the surrounding stock Nogova tiles rather than their unknown source PAA pixels.
NOGOVA_RUNWAY_BACKGROUND_COLOURS: dict[str, tuple[int, int, int]] = {
    r"o\t1.paa": (30, 34, 24),
    r"o\trava2.paa": (39, 45, 29),
    r"o\trava3.paa": (48, 53, 33),
    r"o\pole1.paa": (76, 72, 46),
    r"o\pole2.paa": (63, 61, 36),
    r"o\ps.paa": (90, 82, 56),
    r"o\l1.paa": (50, 49, 45),
    r"o\lom2.paa": (41, 40, 37),
}

_SURFACE_CACHE_V19 = "surface-pipeline-v19-darker-nogova-runway-backgrounds"
_INSTALLED = False


def install_runway_nogova_calibration_policy() -> None:
    """Apply screenshot-calibrated Nogova colours and invalidate older PAAs."""
    global _INSTALLED
    if _INSTALLED:
        return

    _runway._NOGOVA_STOCK_COLOURS.update(NOGOVA_RUNWAY_BACKGROUND_COLOURS)

    # ``runway_cache_key`` resolves this module global at call time, so updating
    # the version after runway policy installation still invalidates v18 output.
    _runway._SURFACE_CACHE_V18 = _SURFACE_CACHE_V19
    _INSTALLED = True
