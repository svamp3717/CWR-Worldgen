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


# Directions refer to visible artwork, not raw filename token order.
# Nogova's PAC quadrants display 180 degrees opposite their named positions;
# the other stock palettes retain their existing filename mapping.
STRAIGHT_BORDERS = {
    "nogova": (
        r"o\t1j1t1j1.pac", r"o\j1t1j1t1.pac",
        r"o\j1j1t1t1.pac", r"o\t1t1j1j1.pac",
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




def test_nogova_exterior_blend_fills_an_exposed_forest_corner() -> None:
    """A grass gap next to j1 must receive the matching quartet, not plain t1."""
    forest, material, initial = _forest_rect_world("nogova")
    dry_grass = MATERIAL_INDEX["h"]
    farmland = MATERIAL_INDEX["a"]
    paved_road = MATERIAL_INDEX["p"]
    # Punch an ordinary-grass hole through the west side of forest coverage.
    # Old behavior left a solid j1 cell facing an unblended trava2 cell.
    material[4, 2] = dry_grass
    material[4, 1] = dry_grass  # untouched exterior tile
    material[4, 7] = farmland
    material[1, 3] = paved_road
    baseline, baseline_slots = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), initial,
        width=9, profile="nogova", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"], enabled=True,
    )
    assert baseline[int(np.asarray(baseline_slots).reshape(9, 9)[4, 2])] == initial[dry_grass + 1]

    paths, indices = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), initial,
        width=9, profile="nogova",
        forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(
            MATERIAL_INDEX["g"], dry_grass, MATERIAL_INDEX["j"],
        ),
        enabled=True,
    )
    image = np.asarray(indices).reshape((9, 9))
    # The same WEST boundary quartet used elsewhere on this source forest.
    assert paths[int(image[4, 2])] == STRAIGHT_BORDERS["nogova"][0]
    # Keep the native straight-border direction for all other forest edges.
    assert paths[int(image[4, 6])] == STRAIGHT_BORDERS["nogova"][1]
    assert paths[int(image[6, 4])] == STRAIGHT_BORDERS["nogova"][2]
    # Distant grass, farmland and roads must not be overwritten.
    assert paths[int(image[4, 1])] == initial[dry_grass + 1]
    assert paths[int(image[0, 0])] == initial[MATERIAL_INDEX["g"] + 1]
    assert paths[int(image[4, 7])] == initial[farmland + 1]
    assert paths[int(image[1, 3])] == initial[paved_road + 1]
    assert tuple(paths[:len(initial)]) == tuple(initial)


def test_nogova_exterior_blend_preserves_shared_corner_orientation() -> None:
    """New quarter-blends must meet original borders in WRP cell coordinates."""
    forest, material, initial = _forest_rect_world("nogova")
    material[4, 2] = MATERIAL_INDEX["h"]
    paths, indices = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), initial, width=9,
        profile="nogova", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(MATERIAL_INDEX["h"],),
        enabled=True,
    )
    image = np.asarray(indices, dtype=int).reshape(9, 9)

    def quadrant_mask(row: int, col: int) -> int:
        name = paths[int(image[row, col])].replace("/", "\\").split("\\")[-1]
        if name == "j1.paa":
            return 15
        if name.endswith(".pac") and len(name) == 12:
            stem = name[:8]
            named_mask = sum(1 << corner for corner in range(4)
                             if stem[2 * corner:2 * corner + 2] == "j1")
            # The stock Nogova PAC artwork displays opposite named corners.
            return (
                ((named_mask & 1) << 3) | ((named_mask & 2) << 1)
                | ((named_mask & 4) >> 1) | ((named_mask & 8) >> 3)
            )
        return 0

    # Two bits along each shared edge must agree with the adjacent cell.
    for row in range(9):
        for col in range(9):
            here = quadrant_mask(row, col)
            if col + 1 < 9:
                east = quadrant_mask(row, col + 1)
                assert bool(here & 1) == bool(east & 2)
                assert bool(here & 4) == bool(east & 8)
            if row + 1 < 9:
                north = quadrant_mask(row + 1, col)
                assert bool(here & 1) == bool(north & 4)
                assert bool(here & 2) == bool(north & 8)


def test_nogova_exterior_blends_require_a_real_assigned_border() -> None:
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



def test_nogova_stock_blend_names_match_visible_artwork() -> None:
    """The PAC's image corners are opposite its literal quartet name."""
    visible_corner_to_filename = {
        1: r"o\t1t1t1j1.pac",  # NE forest displayed by SW token
        2: r"o\t1t1j1t1.pac",  # NW forest displayed by SE token
        4: r"o\t1j1t1t1.pac",  # SE forest displayed by NW token
        8: r"o\j1t1t1t1.pac",  # SW forest displayed by NE token
        3: r"o\t1t1j1j1.pac",  # forest to north
        5: r"o\t1j1t1j1.pac",  # forest to east
        10: r"o\j1t1j1t1.pac", # forest to west
        12: r"o\j1j1t1t1.pac", # forest to south
    }
    for mask, expected_path in visible_corner_to_filename.items():
        assert stock_forest_blend_path("nogova", mask) == expected_path
    assert len({
        stock_forest_blend_path("nogova", mask) for mask in range(1, 15)
    }) == 14
    # The other stock texture profiles retain their previous mappings.
    for profile in ("everon", "malden", "kolgujev"):
        namespace, forest, ground, ext, valid_masks = STOCK_FOREST_BLEND_FAMILIES[profile]
        for mask in valid_masks:
            literal = "".join(
                forest if mask & (1 << corner) else ground
                for corner in range(4)
            )
            assert stock_forest_blend_path(profile, mask) == (
                f"{namespace}\{literal}.{ext}"
            )



def test_nogova_protected_ground_has_no_exposed_forest_corner_seams() -> None:
    """Forest/grass blends must share corners even beside roads and farmland."""
    forest, material, paths = _forest_rect_world("nogova")
    # Include holes through the forest plus protected land abutting its edges.
    material[4, 2] = MATERIAL_INDEX["h"]
    material[3, 4] = MATERIAL_INDEX["g"]
    material[4, 7] = MATERIAL_INDEX["a"]
    material[2, 3] = MATERIAL_INDEX["p"]
    material[6, 6] = MATERIAL_INDEX["u"]
    protected = ((4, 7), (2, 3), (6, 6))
    original_protected = {
        (row, col): paths[int(material[row, col]) + 1]
        for row, col in protected
    }

    expanded, slots = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), paths, width=9, profile="nogova",
        forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(
            MATERIAL_INDEX["g"], MATERIAL_INDEX["h"], MATERIAL_INDEX["j"],
        ),
        enabled=True,
    )
    image = np.asarray(slots).reshape((9, 9))
    for (row, col), original in original_protected.items():
        assert expanded[int(image[row, col])] == original

    def visible_forest_mask(row: int, col: int) -> int:
        name = expanded[int(image[row, col])].replace(chr(92), "/").split("/")[-1]
        if name == "j1.paa":
            return 15
        if not (name.endswith(".pac") and len(name) == 12):
            return 0
        filename_bits = sum(
            1 << k for k in range(4)
            if name[2*k:2*k+2] == "j1"
        )
        return (
            ((filename_bits & 1) << 3)
            | ((filename_bits & 2) << 1)
            | ((filename_bits & 4) >> 1)
            | ((filename_bits & 8) >> 3)
        )

    for row in range(9):
        for col in range(9):
            here = visible_forest_mask(row, col)
            if col < 8:
                east = visible_forest_mask(row, col + 1)
                assert bool(here & 1) == bool(east & 2), (row, col, "NE/NW")
                assert bool(here & 4) == bool(east & 8), (row, col, "SE/SW")
            if row < 8:
                north = visible_forest_mask(row + 1, col)
                assert bool(here & 1) == bool(north & 4), (row, col, "NE/SE")
                assert bool(here & 2) == bool(north & 8), (row, col, "NW/SW")



def test_everon_shared_corners_preserve_farms_roads_and_continuity() -> None:
    """Everon's stock pl/tn transition must meet at each shared WRP corner."""
    forest, material, original_paths = _forest_rect_world("everon")
    material[4, 2] = MATERIAL_INDEX["h"]
    material[3, 4] = MATERIAL_INDEX["g"]
    material[4, 7] = MATERIAL_INDEX["a"]
    material[2, 3] = MATERIAL_INDEX["p"]
    material[6, 6] = MATERIAL_INDEX["u"]
    protected = ((4, 7), (2, 3), (6, 6))
    protected_paths = {
        (row, col): original_paths[int(material[row, col]) + 1]
        for row, col in protected
    }
    paths, slots = apply_stock_forest_border_blends(
        material.ravel(), forest.ravel(), original_paths, width=9,
        profile="everon", forest_material=MATERIAL_INDEX["f"],
        forest_edge_material=MATERIAL_INDEX["e"],
        blendable_exterior_materials=(
            MATERIAL_INDEX["g"], MATERIAL_INDEX["h"], MATERIAL_INDEX["j"],
        ), enabled=True,
    )
    grid = np.asarray(slots, dtype=np.int32).reshape(9, 9)
    for (row, col), path in protected_paths.items():
        assert paths[int(grid[row, col])] == path

    def visible_mask(row: int, col: int) -> int:
        path = paths[int(grid[row, col])].replace(chr(92), "/").casefold()
        name = path.split("/")[-1]
        if path == "eden/pl.paa":
            return 15
        if not (path.startswith("eden/") and name.endswith(".paa")
                and len(name) == 12 and "pl" in name):
            return 0
        stem = name[:8]
        return sum(
            1 << corner for corner in range(4)
            if stem[2 * corner:2 * corner + 2] == "pl"
        )

    masks = np.asarray([
        [visible_mask(row, col) for col in range(9)]
        for row in range(9)
    ], dtype=np.uint8)
    assert np.count_nonzero((masks > 0) & (masks < 15)) > 0
    assert not np.any(masks == 9), "Everon has no verified NE+SW diagonal"
    for row in range(9):
        for col in range(9):
            mask = int(masks[row, col])
            if col + 1 < 9:
                east = int(masks[row, col + 1])
                assert bool(mask & 1) == bool(east & 2), (row, col, "NE/NW")
                assert bool(mask & 4) == bool(east & 8), (row, col, "SE/SW")
            if row + 1 < 9:
                north = int(masks[row + 1, col])
                assert bool(mask & 1) == bool(north & 4), (row, col, "NE/SE")
                assert bool(mask & 2) == bool(north & 8), (row, col, "NW/SW")


def test_everon_missing_diagonal_is_resolved_by_shared_vertices() -> None:
    """Never invent Everon's missing mask-9 texture or leave a corner seam."""
    from cwr_worldgen.forest_ground_blends import _consistent_stock_masks

    forest = np.asarray([
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ], dtype=np.bool_)
    plain = _consistent_stock_masks(forest, np.ones_like(forest))
    assert np.any(plain == 9)
    corrected = _consistent_stock_masks(
        forest, np.ones_like(forest), missing_masks=frozenset({9}),
    )
    assert not np.any(corrected == 9)
    for row in range(4):
        for col in range(4):
            mask = int(corrected[row, col])
            if col + 1 < 4:
                east = int(corrected[row, col + 1])
                assert bool(mask & 1) == bool(east & 2)
                assert bool(mask & 4) == bool(east & 8)
            if row + 1 < 4:
                north = int(corrected[row + 1, col])
                assert bool(mask & 1) == bool(north & 4)
                assert bool(mask & 2) == bool(north & 8)


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
