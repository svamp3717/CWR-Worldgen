from __future__ import annotations

import json
from pathlib import Path


def _weights(values, field):
    return {str(entry[field]): float(entry["weight"]) for entry in values}


def _sweden():
    path = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "country_styles" / "SE_Sweden.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_sweden_ordinary_colours_prefer_red_and_yellow_over_grey():
    document = _sweden()
    for context_name, context in document["contexts"].items():
        materials = context["architectural_details"]["materials"]
        ordinary = _weights(materials["facade_colour_distribution"], "colour")
        timber = _weights(
            materials["wall_material_colour_distributions"]["painted vertical timber cladding"],
            "colour",
        )
        assert ordinary["falun red"] + ordinary["ochre yellow"] >= 48
        assert ordinary["grey"] <= 6
        assert timber["falun red"] + timber["ochre yellow"] >= 58
        assert timber["grey"] <= 5


def test_sweden_rural_houses_are_mostly_painted_timber():
    materials = _sweden()["contexts"]["rural"]["architectural_details"]["materials"]
    walls = _weights(materials["common_wall_material_distribution"], "material")
    assert walls["painted vertical timber cladding"] == 74
    assert walls["stucco/render"] == 22
    assert walls["brick"] == 4


def test_sweden_barns_and_sheds_do_not_drift_grey():
    document = _sweden()
    for context_name, context in document["contexts"].items():
        overrides = context["architectural_details"]["materials"]["building_class_overrides"]
        barn = _weights(overrides["barn"]["facade_colour_distribution"], "colour")
        shed = _weights(overrides["shed"]["facade_colour_distribution"], "colour")
        assert barn["falun red"] >= (78 if context_name == "rural" else 68)
        assert barn["grey"] <= 2
        assert shed["falun red"] >= (58 if context_name == "rural" else 44)
        assert shed["grey"] <= 6


def test_porch_floor_canopy_and_posts_are_not_in_the_generator():
    root = Path(__file__).parents[1]
    source = (root / "src" / "cwr_worldgen" / "osm_house_modeler_upgrade.py").read_text(encoding="utf-8")
    assert "if plan.porch:" not in source
    assert "porch_canopy_texture" not in source
    assert "Porch geometry is intentionally absent" in source


def test_no_porch_p3ds_use_a_new_cache_revision():
    root = Path(__file__).parents[1]
    source = (root / "src" / "cwr_worldgen" / "opening_dimension_policy.py").read_text(encoding="utf-8")
    assert "procedural-building-model-v52-no-porch-geometry" in source
