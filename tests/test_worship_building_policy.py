from pathlib import Path

from cwr_worldgen.building_semantics import is_actual_church, worship_building_class
from cwr_worldgen.osm_house_modeler_styles import StyleChoice, classify_building
from cwr_worldgen import normalization
from cwr_worldgen import procedural_buildings as buildings
from cwr_worldgen.worship_building_policy import (
    apply_global_worship_style,
    install_worship_building_policy,
    load_worship_style_rules,
)


def _choice(**overrides):
    values = dict(
        region_identifier="northern_europe",
        region_name="Northern Europe",
        facade_style="sweden_red",
        roof_style="gabled",
        context="rural",
        family="residential",
        building_class="residential",
        country_code="SE",
        country_name="Sweden",
        country_profile_identifier="se_sweden",
        wall_material="painted timber",
        roof_material="clay/concrete tile",
        wall_thickness_m=0.18,
        colour_palette=("falun red", "white", "ochre yellow"),
        window_spec={"density_multiplier": 1.0},
    )
    values.update(overrides)
    return StyleChoice(**values)


def test_global_worship_semantics_distinguish_religions() -> None:
    assert worship_building_class({"building": "church"}) == "church"
    assert worship_building_class({
        "amenity": "place_of_worship",
        "religion": "christian",
        "denomination": "russian_orthodox",
    }) == "orthodox_church"
    assert worship_building_class({
        "amenity": "place_of_worship",
        "religion": "orthodox",
    }) == "orthodox_church"
    assert worship_building_class({"building": "mosque"}) == "mosque"
    assert worship_building_class({
        "amenity": "place_of_worship",
        "religion": "muslim",
    }) == "mosque"
    assert worship_building_class({"building": "synagogue"}) == "synagogue"
    assert worship_building_class({
        "amenity": "place_of_worship",
        "religion": "jewish",
    }) == "synagogue"
    assert worship_building_class({"building": "temple"}) == "temple"
    assert worship_building_class({"building": "shrine"}) == "shrine"
    assert worship_building_class({"amenity": "place_of_worship"}) == "place_of_worship"


def test_only_christian_worship_uses_church_geometry_family() -> None:
    assert is_actual_church({"building": "church"})
    assert is_actual_church({
        "amenity": "place_of_worship",
        "religion": "christian",
        "denomination": "greek_orthodox",
    })
    assert is_actual_church({
        "amenity": "place_of_worship",
        "religion": "orthodox",
    })
    assert not is_actual_church({"building": "mosque"})
    assert not is_actual_church({"building": "synagogue"})


def test_live_style_classifier_records_worship_class() -> None:
    install_worship_building_policy()
    mosque = classify_building({"building": "mosque"}, 18.0, 26.0)
    synagogue = classify_building({"building": "synagogue"}, 18.0, 26.0)
    assert (mosque.family, mosque.building_class) == ("school", "mosque")
    assert (synagogue.family, synagogue.building_class) == ("school", "synagogue")


def test_cwr_family_keeps_churches_but_not_other_worship() -> None:
    install_worship_building_policy()
    assert buildings._family({"building": "church"}, 18.0, 26.0) == "church"
    assert buildings._family({
        "amenity": "place_of_worship",
        "religion": "christian",
        "denomination": "orthodox",
    }, 18.0, 26.0) == "church"
    assert buildings._family({"building": "mosque"}, 18.0, 26.0) == "school"
    assert buildings._family({"building": "synagogue"}, 18.0, 26.0) == "school"


def test_orthodox_church_can_use_onion_or_dome_roof_geometry() -> None:
    install_worship_building_policy()
    library = buildings.ProceduralBuildingLibrary(world_name="worship-roofs")
    key = library.key_for(
        {
            "amenity": "place_of_worship",
            "religion": "christian",
            "denomination": "orthodox",
        },
        18.0,
        28.0,
    )
    assert key.family == "church"
    assert key.building_class == "orthodox_church"
    assert key.roof_style in {"onion", "dome", "gabled", "hipped"}
    lod = buildings._visual_lod(
        key,
        r"test\worship_wall.paa",
        r"test\worship_roof.paa",
        35.0,
    )
    assert lod.faces
    assert max(point[1] for point in lod.points) > key.height_m
    if key.roof_style in {"onion", "dome"}:
        assert len(lod.points) > 40

    explicit = library.key_for(
        {
            "amenity": "place_of_worship",
            "religion": "christian",
            "denomination": "orthodox",
            "roof:shape": "flat",
        },
        18.0,
        28.0,
    )
    assert explicit.roof_style == "flat"


def test_church_global_style_replaces_residential_red_palette() -> None:
    styled = apply_global_worship_style(
        _choice(),
        {"building": "church", "religion": "christian"},
        width_m=18.0,
        length_m=30.0,
        seed="test-world",
    )
    assert styled.building_class == "church"
    assert styled.facade_style == "worship"
    assert styled.family == "school"
    assert styled.colour_palette
    assert "falun red" not in {value.casefold() for value in styled.colour_palette}
    assert "red" not in {value.casefold() for value in styled.colour_palette}


def test_orthodox_mosque_and_synagogue_use_class_specific_global_rules() -> None:
    cases = (
        (
            {"amenity": "place_of_worship", "religion": "christian", "denomination": "orthodox"},
            "orthodox_church",
            {"onion", "dome", "gabled", "hipped"},
        ),
        ({"building": "mosque"}, "mosque", {"dome", "flat", "hipped"}),
        ({"building": "synagogue"}, "synagogue", {"gabled", "hipped", "dome", "flat"}),
    )
    for tags, expected, roof_styles in cases:
        styled = apply_global_worship_style(
            _choice(), tags, width_m=22.0, length_m=34.0, seed="test-world"
        )
        assert styled.building_class == expected
        assert styled.facade_style == "worship"
        assert styled.roof_style in roof_styles
        assert styled.colour_palette
        assert "falun red" not in {value.casefold() for value in styled.colour_palette}


def test_explicit_osm_worship_appearance_remains_authoritative() -> None:
    choice = _choice(
        wall_material="stone masonry",
        roof_material="slate",
        roof_style="hipped",
        colour_palette=("grey", "white"),
    )
    styled = apply_global_worship_style(
        choice,
        {
            "building": "church",
            "building:material": "stone",
            "building:colour": "blue",
            "roof:material": "slate",
            "roof:shape": "hipped",
        },
        width_m=20.0,
        length_m=32.0,
        seed="test-world",
    )
    assert styled.wall_material == "stone masonry"
    assert styled.roof_material == "slate"
    assert styled.roof_style == "hipped"
    assert styled.colour_palette == ("blue",)


def test_normalization_preserves_explicit_building_and_roof_colours() -> None:
    install_worship_building_policy()
    metadata = set(normalization._BUILDING_METADATA_TAGS)
    assert {
        "building:colour",
        "building:color",
        "roof:colour",
        "roof:color",
    } <= metadata


def test_global_rule_catalogue_contains_no_country_specific_sections() -> None:
    rules = load_worship_style_rules()
    assert {
        "church",
        "orthodox_church",
        "mosque",
        "synagogue",
        "temple",
        "shrine",
        "place_of_worship",
    } <= set(rules)
    text = Path(__file__).resolve().parents[1].joinpath(
        "src", "cwr_worldgen", "data", "worship_building_styles.json"
    ).read_text(encoding="utf-8").casefold()
    assert "sweden" not in text
    assert "falun" not in text
