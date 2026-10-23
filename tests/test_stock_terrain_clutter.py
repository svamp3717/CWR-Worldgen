"""Original Eden / Abel / Cain low vegetation with strict tree-preset gating."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    VEGETATION_EVERON, VEGETATION_KOLGUJEV, VEGETATION_MALDEN,
    VEGETATION_NONE, VEGETATION_RESISTANCE_LEAF, VEGETATION_RESISTANCE_PINE,
    build_milestone9_command, default_gui_values,
)
from cwr_worldgen.nogova_terrain_clutter import (
    nogova_clutter_active, terrain_clutter_active, place_nogova_terrain_clutter,
    terrain_clutter_preset,
)
from cwr_worldgen.stock_terrain_clutter_profiles import (
    ABEL_CLUTTER_WEIGHTS, CAIN_CLUTTER_WEIGHTS, EDEN_CLUTTER_WEIGHTS,
    STOCK_CLUTTER_WEIGHTS, stock_clutter_tree_preset,
)

EVERON_BLOCK = r"data3d\les ctverec pruchozi_T1.p3d"
EVERON_TREE = r"data3d\str smrk_medium.p3d"
MALDEN_BLOCK = r"data3d\les_su_ctver_pruchozi.p3d"
MALDEN_TREE = r"data3d\str_fikovnik.p3d"
KOLGUJEV_TREE = r"data3d\str smrk.p3d"
NOGOVA_BLOCK = r"o\tree\les_nw_ctver_pruhozi_T1.p3d"
NOGOVA_TREE = r"o\tree\javor01.p3d"

PRESETS = (
    ("everon", EVERON_BLOCK, EVERON_TREE, EDEN_CLUTTER_WEIGHTS),
    ("malden", MALDEN_BLOCK, MALDEN_TREE, ABEL_CLUTTER_WEIGHTS),
    ("kolgujev", EVERON_BLOCK, KOLGUJEV_TREE, CAIN_CLUTTER_WEIGHTS),
)


def _spec(block, tree, *, enabled=True, maximum=150, seed=11, cells=32):
    return SimpleNamespace(
        forest_tree_model=block,
        forest_single_tree_model=tree,
        nogova_terrain_clutter_enabled=enabled,
        nogova_terrain_clutter_maximum_objects=maximum,
        cells=cells, cell_size=25.0, sea_level=0.0,
        deterministic_seed=seed,
        ground_texture_profile="nogova",  # Ground textures cannot select palette.
    )


def _raster(cells=32, **overrides):
    empty = (False,) * (cells * cells)
    data = {name: empty for name in (
        "water", "roads", "buildings", "urban", "farmland", "forest",
    )}
    data.update(overrides)
    return SimpleNamespace(**data)


@pytest.mark.parametrize("preset,block,tree,weights", PRESETS)
def test_exact_model_weights_match_original_export(
    preset, block, tree, weights,
) -> None:
    assert STOCK_CLUTTER_WEIGHTS[preset] is weights
    observed = dict(weights)
    assert len(observed) == len(weights) and len(weights) >= 8
    assert all(path.startswith("data3d\\") and path.endswith(".p3d")
               and count > 0 for path, count in weights)
    assert sum(weight for _, weight in weights) == {
        "everon": 17805, "malden": 44560, "kolgujev": 27423,
    }[preset]
    expected_signature = {
        "everon": (r"data3d\krovi3.p3d", 4106),
        "malden": (r"data3d\ker trs travy.p3d", 9722),
        "kolgujev": (r"data3d\str_smrcicicek.p3d", 8032),
    }[preset]
    assert expected_signature in weights
    spec = _spec(block, tree)
    assert stock_clutter_tree_preset(spec) == preset
    assert terrain_clutter_preset(spec) == preset
    assert terrain_clutter_active(spec)


@pytest.mark.parametrize("preset,block,tree,weights", PRESETS)
def test_no_matching_tree_preset_no_clutter(
    preset, block, tree, weights,
) -> None:
    wrong_tree = NOGOVA_TREE
    wrong_block = NOGOVA_BLOCK
    assert terrain_clutter_preset(_spec(block, wrong_tree)) is None
    assert terrain_clutter_preset(_spec(wrong_block, tree)) is None
    assert terrain_clutter_preset(_spec(block, tree, enabled=False)) == preset
    assert not terrain_clutter_active(_spec(block, tree, enabled=False))
    # The same tree preset works with *any* ground texture selection.
    for ground in ("everon", "malden", "kolgujev", "nogova", "none"):
        spec = _spec(block, tree)
        spec.ground_texture_profile = ground
        assert terrain_clutter_preset(spec) == preset


@pytest.mark.parametrize("preset,block,tree,weights", PRESETS)
def test_reproducible_dry_natural_ground_only_and_source_models(
    preset, block, tree, weights,
) -> None:
    spec = _spec(block, tree, maximum=12)
    flat = [10.0] * (spec.cells * spec.cells)
    one = place_nogova_terrain_clutter(spec, _raster(), flat, ())
    two = place_nogova_terrain_clutter(spec, _raster(), flat, ())
    assert 0 < len(one) <= 12
    assert one == two
    assert set(obj.model_path for obj in one).issubset(dict(weights))
    assert len(set((obj.x, obj.z) for obj in one)) == len(one)
    assert all(obj.y == pytest.approx(9.975) for obj in one)
    varied = place_nogova_terrain_clutter(
        _spec(block, tree, maximum=12, seed=12), _raster(), flat, (),
    )
    assert one != varied
    assert place_nogova_terrain_clutter(
        _spec(block, tree, enabled=False), _raster(), flat, (),
    ) == ()
    assert place_nogova_terrain_clutter(
        _spec(block, tree, maximum=0), _raster(), flat, (),
    ) == ()


@pytest.mark.parametrize("preset,block,tree,weights", PRESETS)
def test_coarse_masks_submerged_and_steep_terrain_rejected(
    preset, block, tree, weights,
) -> None:
    spec = _spec(block, tree)
    all_true = (True,) * (spec.cells * spec.cells)
    flat = [10.0] * len(all_true)
    for name in ("water", "forest", "farmland", "urban", "buildings", "roads"):
        assert not place_nogova_terrain_clutter(
            spec, _raster(**{name: all_true}), flat, (),
        )
    assert not place_nogova_terrain_clutter(
        spec, _raster(), [-1.0] * len(flat), (),
    )
    hills = [row * 50 + col * 35
             for row in range(spec.cells) for col in range(spec.cells)]
    assert not place_nogova_terrain_clutter(spec, _raster(), hills, ())


def test_gui_checkbox_uses_correct_preset_independent_of_ground_textures() -> None:
    values = default_gui_values()
    assert values["nogova_terrain_clutter"] is False
    for style in (VEGETATION_EVERON, VEGETATION_KOLGUJEV, VEGETATION_MALDEN):
        values["vegetation_style"] = style
        command = build_milestone9_command(values, python="python")
        assert "--terrain-clutter" not in command
        values["nogova_terrain_clutter"] = True
        for ground in ("nogova", "everon", "generated", "malden"):
            values["ground_textures"] = ground
            command = build_milestone9_command(values, python="python")
            assert command.count("--terrain-clutter") == 1
            assert "--nogova-terrain-clutter" not in command
        values["nogova_terrain_clutter"] = False
    values["nogova_terrain_clutter"] = True
    for style in (VEGETATION_RESISTANCE_LEAF, VEGETATION_RESISTANCE_PINE):
        values["vegetation_style"] = style
        command = build_milestone9_command(values, python="python")
        assert "--nogova-terrain-clutter" in command
        assert "--terrain-clutter" not in command
    values["vegetation_style"] = VEGETATION_NONE
    assert "--terrain-clutter" not in build_milestone9_command(values, python="python")
    assert "--nogova-terrain-clutter" not in build_milestone9_command(values, python="python")


def test_generic_cli_flag_preserves_original_nogova_compatibility() -> None:
    parser = _parser()
    args = ["milestone9", "--source-dir", ".", "--name", "world", "--output", "."]
    default = parser.parse_args(args)
    assert default.nogova_terrain_clutter is False
    assert default.nogova_clutter_max_objects == 20000
    for flag in ("--terrain-clutter", "--nogova-terrain-clutter"):
        enabled = parser.parse_args([*args, flag, "--nogova-clutter-max-objects", "3"])
        assert enabled.nogova_terrain_clutter is True
        assert enabled.nogova_clutter_max_objects == 3


def test_original_nogova_family_is_unchanged() -> None:
    nogova = _spec(NOGOVA_BLOCK, NOGOVA_TREE)
    assert terrain_clutter_preset(nogova) == "nogova"
    assert stock_clutter_tree_preset(nogova) is None
    assert nogova_clutter_active(nogova)
