"""Opt-in stock forest-floor textures, separate from 3D forest undergrowth."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.generator import _ground_texture_paths, _external_ground_texture_paths
from cwr_worldgen.gui import build_milestone9_command, default_gui_values
from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection, GeoPolygon, OsmDataset, OsmPolygonFeature, rasterize_osm,
)
from cwr_worldgen.surface_pass import (
    MATERIAL_INDEX, MILESTONE9_MATERIALS,
    build_surface_pass, surface_texture_wire_paths, external_surface_texture_paths,
)
from cwr_worldgen.terrain import stock_forest_undergrowth_paths
from cwr_worldgen.forest_ground_blends import (
    STOCK_FOREST_BLEND_FAMILIES,
    apply_stock_forest_border_blends,
    stock_forest_blend_path,
)
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


STOCK = {
    "nogova": (r"o\j1.paa", r"o\t1.paa"),
    "everon": (r"eden\pl.paa", r"eden\tn.paa"),
    "malden": (r"abel\pb.paa", r"abel\tt.paa"),
    "kolgujev": (r"cain\u2.paa", r"cain\j9.paa"),
}


@pytest.mark.parametrize("profile,expected", STOCK.items())
def test_stock_forest_ground_interior_and_edge_paths(profile: str, expected: tuple[str, str]) -> None:
    before = surface_texture_wire_paths("example", profile)
    after = surface_texture_wire_paths("example", profile, forest_undergrowth_textures=True)
    assert after[MATERIAL_INDEX["f"]] == expected[0]
    assert after[MATERIAL_INDEX["e"]] == expected[1]
    assert stock_forest_undergrowth_paths(profile, True) == expected
    assert stock_forest_undergrowth_paths(profile, False) is None
    for index, material in enumerate(MILESTONE9_MATERIALS):
        if material.code not in {"f", "e"}:
            assert after[index] == before[index], material.code
    spec = SimpleNamespace(
        name="example", ground_texture_profile=profile,
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=True,
    )
    assert _ground_texture_paths(spec) == after
    if profile != "malden":
        # The original surface palette lists all used stock paths. The
        # production asset-scan optimization separately selects only one
        # representative texture, so do not test that narrow scan as a
        # reflection of the WRP texture table.
        assert set(expected).issubset(external_surface_texture_paths(
            profile, forest_undergrowth_textures=True,
        ))
    else:
        assert external_surface_texture_paths(
            profile, forest_undergrowth_textures=True,
        ) == ()


def test_stock_floor_disabled_or_unsupported_profile_is_unchanged() -> None:
    for profile in (*STOCK, "generated", "desert", "none"):
        baseline = surface_texture_wire_paths("example", profile)
        assert surface_texture_wire_paths("example", profile, forest_undergrowth_textures=False) == baseline
        if profile not in STOCK:
            assert surface_texture_wire_paths(
                "example", profile, forest_undergrowth_textures=True,
            ) == baseline
    assert Milestone9Spec.__dataclass_fields__["forest_undergrowth_textures"].default is False
    assert _Milestone9PlayabilitySpec.__dataclass_fields__["forest_undergrowth_textures"].default is False


def test_legacy_ground_mapping_only_changes_forest_interior() -> None:
    for profile, (floor, _edge) in STOCK.items():
        spec = SimpleNamespace(
            name="example", ground_texture_profile=profile,
            surface_pass_enabled=False, surface_ground_mode="milestone8",
            forest_undergrowth_textures=True,
        )
        updated = _ground_texture_paths(spec)
        original = _ground_texture_paths(SimpleNamespace(
            **{**vars(spec), "forest_undergrowth_textures": False},
        ))
        assert updated[4] == floor  # OSM_MATERIALS[4] = forest
        assert updated[:4] == original[:4]
        assert updated[5:] == original[5:]


def test_common_choices_checkbox_goes_through_gui_and_cli() -> None:
    values = default_gui_values()
    assert values["forest_undergrowth_textures"] is False
    values.update(
        source_dir="source-data/map", output="build/map",
        name="cwr_example", display_name="Example",
    )
    disabled = build_milestone9_command(values, python="python")
    assert "--forest-undergrowth-textures" not in disabled
    values["forest_undergrowth_textures"] = True
    enabled = build_milestone9_command(values, python="python")
    assert enabled.count("--forest-undergrowth-textures") == 1
    options = _parser().parse_args([
        "milestone9", "--source-dir", "input", "--output", "output",
        "--forest-undergrowth-textures",
    ])
    assert options.forest_undergrowth_textures is True
    options_off = _parser().parse_args([
        "milestone9", "--source-dir", "input", "--output", "output",
    ])
    assert options_off.forest_undergrowth_textures is False


def test_forest_classification_has_interior_and_boundary_cells() -> None:
    # An actual forest polygon, unlike an artificial texture-table-only test.
    cells = 32
    projection = BboxProjection.create((0, 0, 0.03, 0.03), 1600)
    corners = ((200, 200), (1350, 200), (1350, 1350), (200, 1350))
    ring = tuple(projection.to_latlon(point) for point in (*corners, corners[0]))
    forest = OsmPolygonFeature("way/forest", {"landuse": "forest"}, (GeoPolygon(ring),))
    dataset = OsmDataset(
        source_generator="forest-ground", element_count=1,
        coastlines=(), water=(), forests=(forest,), farmland=(),
        urban=(), roads=(), barriers=(),
    )
    raster = rasterize_osm(dataset, projection, cells=cells, include_minor_roads=True)
    spec = SimpleNamespace(
        cells=cells, cell_size=50.0, include_minor_roads=True,
        sea_level=0, rock_slope_degrees=30, deterministic_seed="forest-floor",
        surface_forest_edge_cells=1,
    )
    pass_result = build_surface_pass(
        dataset, projection, raster, [50.0] * (cells * cells),
        [2.0] * (cells * cells), spec,
    )
    terrain = np.asarray(pass_result.indices)
    assert np.count_nonzero(terrain == MATERIAL_INDEX["f"]) > 0
    assert np.count_nonzero(terrain == MATERIAL_INDEX["e"]) > 0
    # The option changes *only* the forest & edge WRP texture entries,
    # never forest objects, raster cells, terrain heights or material semantics.
    assert STOCK["nogova"][0] == surface_texture_wire_paths(
        "example", "nogova", forest_undergrowth_textures=True,
    )[MATERIAL_INDEX["f"]]



def test_written_4wvr_contains_forest_floor_and_edge_stock_slots(tmp_path) -> None:
    # The WRP contains true texture paths, not just a preview override.
    spec = SimpleNamespace(
        name="world", ground_texture_profile="nogova",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=True,
    )
    ground_paths = _ground_texture_paths(spec)
    floor_slot = MATERIAL_INDEX["f"] + 1
    edge_slot = MATERIAL_INDEX["e"] + 1
    slots = [floor_slot, edge_slot, floor_slot, edge_slot] * 4
    wrp = tmp_path / "undergrowth.wrp"
    write_rvw4(
        wrp, 4, 4, [10.0] * 16, slots,
        [r"world\data\d.paa", *ground_paths], (),
        height_scale=0.05,
    )
    inspected = inspect_rvw4(wrp, height_scale=0.05)
    assert inspected.texture_slots[floor_slot] == r"o\j1.paa"
    assert inspected.texture_slots[edge_slot] == r"o\t1.paa"


# The four names below are direct matches to original Island Studio atlas
# exports and establish the SW/SE/NW/NE quadrant orientation.
STRAIGHT_BORDERS = {
    "nogova": (
        r"o\j1t1j1t1.pac", r"o\t1j1t1j1.pac",
        r"o\t1t1j1j1.pac", r"o\j1j1t1t1.pac",
    ),
    "everon": (
        r"eden\pltnpltn.paa", r"eden\tnpltnpl.paa",
        r"eden\tntnplpl.paa", r"eden\plpltntn.paa",
    ),
    "malden": (
        r"abel\pbttpbtt.pac", r"abel\ttpbttpb.pac",
        r"abel\ttttpbpb.pac", r"abel\pbpbtttt.pac",
    ),
    "kolgujev": (
        r"cain\u2j9u2j9.paa", r"cain\j9u2j9u2.paa",
        r"cain\j9j9u2u2.paa", r"cain\u2u2j9j9.paa",
    ),
}


def _forest_rect_world(profile: str):
    size = 9
    forest = np.zeros((size, size), dtype=np.bool_)
    forest[2:7, 2:7] = True
    ground = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    ground[2:7, 2:7] = MATERIAL_INDEX["e"]
    ground[3:6, 3:6] = MATERIAL_INDEX["f"]
    before = [r"world\data\d.paa", *surface_texture_wire_paths(
        "world", profile, forest_undergrowth_textures=True,
    )]
    return forest, ground, before


@pytest.mark.parametrize("profile,straight", STRAIGHT_BORDERS.items())
def test_directional_stock_forest_blends_match_original_island(
    profile: str, straight: tuple[str, str, str, str],
) -> None:
    forest, terrain, paths = _forest_rect_world(profile)
    expanded, indices = apply_stock_forest_border_blends(
        terrain.ravel(), forest.ravel(), paths, width=9,
        profile=profile, forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
    )
    image = np.asarray(indices).reshape((9, 9))
    west = expanded[int(image[4, 2])]
    east = expanded[int(image[4, 6])]
    north = expanded[int(image[6, 4])]
    south = expanded[int(image[2, 4])]
    assert (west, east, north, south) == straight
    # Every corner cell uses a verified path with exactly one forest corner.
    for row, col, mask in ((2, 2, 1), (2, 6, 2), (6, 2, 4), (6, 6, 8)):
        assert expanded[int(image[row, col])] == stock_forest_blend_path(profile, mask)
    assert expanded[int(image[4, 4])] == STOCK[profile][0]
    assert expanded[int(image[0, 0])] == paths[MATERIAL_INDEX["g"] + 1]
    # Geometry/material classifications and existing base texture slots survive.
    assert np.count_nonzero(terrain == MATERIAL_INDEX["e"]) == 16
    assert tuple(expanded[:len(paths)]) == tuple(paths)
    assert len(expanded) <= 512
    assert len(expanded) > len(paths)




def test_nogova_border_apron_matches_t1_without_repainting_protected_land() -> None:
    """A j1/t1 transition should meet t1, not unrelated grass or farmland."""
    forest, material, base_paths = _forest_rect_world("nogova")
    dry_grass = MATERIAL_INDEX["h"]
    farmland = MATERIAL_INDEX["a"]
    paved_road = MATERIAL_INDEX["p"]
    material[~forest] = dry_grass
    material[4, 7] = farmland
    material[1, 3] = paved_road

    base_grass_path = base_paths[dry_grass + 1]
    assert "trava2.paa" in base_grass_path
    t1_path = base_paths[MATERIAL_INDEX["e"] + 1]

    paths, indices = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), base_paths,
        width=9, profile="nogova",
        forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(
            MATERIAL_INDEX["g"], dry_grass, MATERIAL_INDEX["j"],
        ),
        enabled=True,
    )
    mapped = np.asarray(indices).reshape(9, 9)
    # Only ordinary ground immediately outside a forest transition is t1.
    for row, col in ((4, 1), (1, 4), (7, 4)):
        assert paths[int(mapped[row, col])] == t1_path
    # Never overwrite farmland, roads or distant ordinary terrain.
    assert paths[int(mapped[4, 7])] == base_paths[farmland + 1]
    assert paths[int(mapped[1, 3])] == base_paths[paved_road + 1]
    assert paths[int(mapped[0, 0])] == base_grass_path
    # Forest edge art remains directionally identical to original Nogova.
    assert paths[int(mapped[4, 2])] == STRAIGHT_BORDERS["nogova"][0]
    assert paths[int(mapped[4, 6])] == STRAIGHT_BORDERS["nogova"][1]

    without_paths, without_indices = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), base_paths,
        width=9, profile="nogova",
        forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        enabled=True,
    )
    without = np.asarray(without_indices).reshape(9, 9)
    assert without_paths[int(without[4, 1])] == base_grass_path
    assert without_paths[int(without[4, 2])] == STRAIGHT_BORDERS["nogova"][0]


def test_nogova_apron_requires_a_real_assigned_blend() -> None:
    forest, material, paths = _forest_rect_world("nogova")
    material[~forest] = MATERIAL_INDEX["h"]
    result, slots = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), paths, width=9,
        profile="nogova", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(MATERIAL_INDEX["h"],),
        maximum_texture_slots=len(paths), enabled=True,
    )
    assert result == tuple(paths)
    actual = np.asarray(slots).reshape(9, 9)
    assert actual[4, 1] == MATERIAL_INDEX["h"] + 1

def test_only_verified_source_texture_names_are_ever_emitted() -> None:
    for profile, family in STOCK_FOREST_BLEND_FAMILIES.items():
        valid_masks = family[4]
        assert len(valid_masks) == {"nogova":14, "everon":13, "malden":12, "kolgujev":14}[profile]
        for mask in range(1, 15):
            path = stock_forest_blend_path(profile, mask)
            assert (path is not None) == (mask in valid_masks)
        # Two uncommon diagonal masks do not exist in source Malden/Everon.
        assert stock_forest_blend_path("malden", 6) is None
        assert stock_forest_blend_path("malden", 9) is None
        assert stock_forest_blend_path("everon", 9) is None
    assert stock_forest_blend_path("nogova", 15) is None


def test_turned_off_forests_leave_original_wrp_indices_untouched() -> None:
    mask, material, paths = _forest_rect_world("nogova")
    result, slots = apply_stock_forest_border_blends(
        material.ravel(), mask.ravel(), paths,
        width=9, profile="nogova", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=False,
    )
    assert result == tuple(paths)
    assert slots == (material.ravel() + 1).tolist()


def test_empty_forest_and_stock_wrp_slot_budget_do_not_generate_fake_paths() -> None:
    forest, material, paths = _forest_rect_world("everon")
    no_forest, empty_indices = apply_stock_forest_border_blends(
        np.full(81, MATERIAL_INDEX["g"], dtype=np.int32),
        np.zeros(81, dtype=np.bool_), paths, width=9,
        profile="everon", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
    )
    assert no_forest == tuple(paths)
    assert set(empty_indices) == {MATERIAL_INDEX["g"] + 1}
    full, result = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), paths, width=9,
        profile="everon", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
        maximum_texture_slots=len(paths),
    )
    assert full == tuple(paths)
    assert max(result) < len(paths)
    with pytest.raises(ValueError, match="dimensions"):
        apply_stock_forest_border_blends(
            material.ravel(), forest.ravel()[:-1], paths,
            width=9, profile="everon", forest_material=MATERIAL_INDEX["f"],
            forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
        )


def test_written_wrp_contains_real_directional_forest_border_paths(tmp_path) -> None:
    forest, material, initial = _forest_rect_world("nogova")
    paths, slots = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), initial, width=9,
        profile="nogova", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
    )
    world = tmp_path / "forest_edges.wrp"
    write_rvw4(
        world, 9, 9, [50.0] * 81, slots, paths, (),
        height_scale=0.05,
    )
    with_world = inspect_rvw4(world, height_scale=0.05)
    for sample in STRAIGHT_BORDERS["nogova"]:
        assert sample in with_world.texture_slots
    assert r"o\j2.paa" not in with_world.texture_slots
