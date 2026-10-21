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
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


STOCK = {
    "nogova": (r"o\j1.paa", r"o\j2.paa"),
    "everon": (r"eden\pl.paa", r"eden\pl.paa"),
    "malden": (r"abel\pb.paa", r"abel\pb.paa"),
    "kolgujev": (r"cain\u2.paa", r"cain\u2.paa"),
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
    assert inspected.texture_slots[edge_slot] == r"o\j2.paa"
