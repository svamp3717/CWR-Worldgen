# SPDX-License-Identifier: GPL-3.0-or-later
"""Match generated RVW4 terrain quantization to the CWA engine contract.

The CWA/OFP RVW loader stores terrain elevations as signed 16-bit values and
expands them with ``LANDDATA_SCALE``.  The released engine source defines that
scale as 0.03 * 1.5, or exactly 0.045 metres per stored unit.  Worldgen had
historically assumed 0.05 m, so a nominal 96 m plateau serialized as 1920 and
loaded in CWA at only 86.4 m while RVW4 object matrices remained at their
absolute 96 m Y coordinates.

Keep the correction in one late policy while this branch is being validated in
game.  All HeightmapSpec-derived milestones inherit the corrected property, and
Milestone 1 receives the same on-disk contract through WorldSpec.
"""
from __future__ import annotations

from .model import HeightmapSpec, WorldSpec

CWA_RVW4_HEIGHT_SCALE_METRES = 0.045
_INSTALLED = False


def _cwa_height_scale(_self) -> float:
    return CWA_RVW4_HEIGHT_SCALE_METRES


def install_cwa_rvw4_height_scale_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # These classes deliberately expose height_scale as a read-only property.
    # Replace the descriptor rather than adding another user-facing spec knob:
    # RVW4 does not carry a scale field, so CWA itself fixes the value.
    WorldSpec.height_scale = property(_cwa_height_scale)
    HeightmapSpec.height_scale = property(_cwa_height_scale)
    _INSTALLED = True
