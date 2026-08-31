# SPDX-License-Identifier: GPL-3.0-or-later
from cwr_worldgen.road_pipeline import ROAD_PIPELINE_STAGES


def test_road_pipeline_has_one_unique_declared_order():
    assert len(ROAD_PIPELINE_STAGES) == len(set(ROAD_PIPELINE_STAGES))
    assert ROAD_PIPELINE_STAGES[-1] == "raceway_classification"


def test_cancelled_overlap_layers_are_not_in_production_pipeline():
    cancelled = {
        "straight_seam",
        "curve_seam_fallback",
        "intersection_edge",
        "fit_first",
    }
    assert cancelled.isdisjoint(ROAD_PIPELINE_STAGES)
    assert ROAD_PIPELINE_STAGES.index("visual_finish") < ROAD_PIPELINE_STAGES.index(
        "final_continuity"
    )
    assert ROAD_PIPELINE_STAGES.index("final_continuity") < ROAD_PIPELINE_STAGES.index(
        "emitted_seam"
    )
