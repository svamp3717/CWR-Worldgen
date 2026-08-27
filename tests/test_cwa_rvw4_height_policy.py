# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from cwr_worldgen.cwa_rvw4_height_policy import CWA_RVW4_HEIGHT_SCALE_METRES
from cwr_worldgen.model import HeightmapSpec, WorldSpec
from cwr_worldgen.wrp import quantize_height


def test_world_specs_use_cwa_rvw4_landdata_scale() -> None:
    assert CWA_RVW4_HEIGHT_SCALE_METRES == 0.045
    assert WorldSpec().height_scale == 0.045
    assert HeightmapSpec(heightmap_path=Path("unused.png")).height_scale == 0.045


def test_roadlab_plateau_quantizes_to_cwa_engine_height() -> None:
    scale = WorldSpec().height_scale
    raw = quantize_height(96.0, scale)

    assert raw == 2133
    assert abs(raw * scale - 96.0) <= scale * 0.5 + 1.0e-12
    # The old 0.05 assumption produced 1920, which CWA expands with 0.045 to
    # 86.4 m while RVW4 object matrices remain at their absolute 96 m height.
    assert raw != 1920
