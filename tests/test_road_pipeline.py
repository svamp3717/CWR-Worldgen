# SPDX-License-Identifier: GPL-3.0-or-later
from types import SimpleNamespace

from cwr_worldgen.asset_mapping import default_osm_asset_mapping
from cwr_worldgen.normalization import _MAJOR_HIGHWAYS as NORMALIZED_MAJOR_HIGHWAYS
from cwr_worldgen.osm import road_is_dirt, road_is_supported, road_model_for_tags
from cwr_worldgen.road_pipeline import ROAD_PIPELINE_STAGES
from cwr_worldgen import stock_road_s_bend_policy as _s_bend


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
        "single_vertex_bend",
        "wrptool_catalogue",
        "stock_obstacles",
        "stock_relaxation_transaction",
        "emitted_seam_refinement",
        "kodiak_reference",
        "stock_transform",
        "turning_t_fallback",
        "stock_measured_junction",
        "stock_skew",
        "stock_curve_preservation",
        "sharp_exact",
    }
    assert retired.isdisjoint(ROAD_PIPELINE_STAGES)
    assert ROAD_PIPELINE_STAGES.index("visual_finish") < ROAD_PIPELINE_STAGES.index(
        "final_continuity"
    )
    assert ROAD_PIPELINE_STAGES.index("final_continuity") < ROAD_PIPELINE_STAGES.index(
        "emitted_seam"
    )


def test_folded_long_s_bend_limit_is_preserved():
    assert _s_bend.MAXIMUM_EXACT_S_BEND_RUN_METRES >= 1200.0


def _raceway_spec():
    return SimpleNamespace(
        name="raceway_test",
        procedural_gravel_roads=False,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
    )


def test_raceway_classification_is_native_pipeline_data():
    tags = {"highway": "raceway"}
    spec = _raceway_spec()

    assert "raceway" in NORMALIZED_MAJOR_HIGHWAYS
    assert road_is_supported(tags, include_minor=False)
    assert not road_is_dirt(tags)
    assert road_model_for_tags(spec, tags) == spec.paved_road_model

    mapping = default_osm_asset_mapping(spec, 9)
    paved_rule = next(rule for rule in mapping.rules if rule.rule_id == "road-paved")
    highway_values = next(values for key, values in paved_rule.match if key == "highway")
    assert "raceway" in highway_values


def test_explicit_unpaved_raceway_still_uses_dirt():
    tags = {"highway": "raceway", "surface": "dirt"}
    spec = _raceway_spec()

    assert road_is_supported(tags, include_minor=False)
    assert road_is_dirt(tags)
    assert road_model_for_tags(spec, tags) == spec.dirt_road_model
