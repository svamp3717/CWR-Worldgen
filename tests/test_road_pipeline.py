# SPDX-License-Identifier: GPL-3.0-or-later
from cwr_worldgen.road_pipeline import ROAD_PIPELINE_STAGES
from cwr_worldgen import stock_road_s_bend_exact_policy as _s_exact


def test_road_pipeline_has_one_unique_declared_order():
    assert len(ROAD_PIPELINE_STAGES) == len(set(ROAD_PIPELINE_STAGES))
    assert ROAD_PIPELINE_STAGES[-1] == "raceway_classification"


def test_cancelled_or_folded_layers_are_not_in_production_pipeline():
    retired = {
        "straight_seam",
        "curve_seam_fallback",
        "intersection_edge",
        "fit_first",
        "long_s_bend",
    }
    assert retired.isdisjoint(ROAD_PIPELINE_STAGES)
    assert ROAD_PIPELINE_STAGES.index("visual_finish") < ROAD_PIPELINE_STAGES.index(
        "final_continuity"
    )
    assert ROAD_PIPELINE_STAGES.index("final_continuity") < ROAD_PIPELINE_STAGES.index(
        "emitted_seam"
    )


def test_folded_long_s_bend_limit_is_preserved():
    assert _s_exact.MAXIMUM_EXACT_S_BEND_RUN_METRES >= 1200.0
