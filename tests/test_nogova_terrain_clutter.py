"""Source-measured Nogova small ground-clutter checkbox and geometry tests."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen.cli import _parser
from cwr_worldgen.gui import (
    default_gui_values, build_milestone9_command,
    VEGETATION_EVERON, VEGETATION_RESISTANCE_LEAF,
    VEGETATION_RESISTANCE_PINE, VEGETATION_NONE,
)
from cwr_worldgen.nogova_terrain_clutter import (
    NOGOVA_CLUTTER_WEIGHTS,
    nogova_tree_preset_selected, nogova_clutter_active,
    place_nogova_terrain_clutter,
)
from cwr_worldgen.osm import WorldObject


LEAF = r"o\tree\les_nw_ctver_pruhozi_T1.p3d"
PINE = r"o\tree\les_nw_jehl_ctver_pruhozi.p3d"


def _spec(*, block=LEAF, single=r"o\tree\javor01.p3d",
          enabled=True, seed=1234, cells=32):
    return SimpleNamespace(
        forest_tree_model=block, forest_single_tree_model=single,
        nogova_terrain_clutter_enabled=enabled,
        deterministic_seed=seed, cells=cells, cell_size=25.0, sea_level=0.0,
        ground_texture_profile="everon",
    )


def _raster(cells=32, **overrides):
    zero = (False,) * (cells * cells)
    attrs = {
        name: zero
        for name in ("water", "roads", "buildings", "urban", "farmland", "forest")
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _heights(cells=32):
    return [10.0] * (cells * cells)


def test_original_nogova_bush_and_sapling_model_weights() -> None:
    observed = dict(NOGOVA_CLUTTER_WEIGHTS)
    assert observed[r"o\tree\dd_bush02d.p3d"] == 27602
    assert observed[r"o\tree\dd_bush01c.p3d"] == 27045
    assert observed[r"o\tree\smrk_maly.p3d"] == 21859
    assert len(observed) == len(NOGOVA_CLUTTER_WEIGHTS)
    assert all(path.startswith("o\\tree\\") and path.endswith(".p3d")
               and weight > 0 for path, weight in NOGOVA_CLUTTER_WEIGHTS)
    # Large trees, buildings, rocks, forest blocks and reeds are separate
    # placement categories, not arbitrary substitutes for terrain clutter.
    assert not any("les_" in path or "rakosi" in path or "hous" in path
                   for path in observed)


@pytest.mark.parametrize(
    ("block", "single", "active"),
    [
        (LEAF, r"o\tree\javor01.p3d", True),
        (PINE, r"o\tree\smrk_maly.p3d", True),
        (r"data3d\les ctverec pruchozi_T1.p3d",
         r"o\tree\javor01.p3d", False),
        (LEAF, r"data3d\str smrk_medium.p3d", False),
        (r"data3d\les ctverec pruchozi_T1.p3d",
         r"data3d\str smrk_medium.p3d", False),
    ],
)
def test_requires_nogova_tree_preset_not_nogova_ground(block, single, active):
    spec = _spec(block=block, single=single)
    assert nogova_tree_preset_selected(spec) is active
    assert nogova_clutter_active(spec) is active
    assert nogova_clutter_active(_spec(block=block, single=single,
                                       enabled=False)) is False


def test_checkbox_defaults_off_and_gui_only_sends_flag_for_resistance_preset():
    values = default_gui_values()
    assert values["nogova_terrain_clutter"] is False
    values["nogova_terrain_clutter"] = True
    values["ground_textures"] = "nogova"
    values["vegetation_style"] = VEGETATION_EVERON
    assert "--nogova-terrain-clutter" not in build_milestone9_command(values, python="python")
    for tree_style in (VEGETATION_RESISTANCE_LEAF, VEGETATION_RESISTANCE_PINE):
        values["vegetation_style"] = tree_style
        command = build_milestone9_command(values, python="python")
        assert command.count("--nogova-terrain-clutter") == 1
    values["vegetation_style"] = VEGETATION_NONE
    assert "--nogova-terrain-clutter" not in build_milestone9_command(values, python="python")
    values["vegetation_style"] = VEGETATION_RESISTANCE_LEAF
    values["nogova_terrain_clutter"] = False
    assert "--nogova-terrain-clutter" not in build_milestone9_command(values, python="python")


def test_cli_flag_is_opt_in_with_hard_object_limit():
    parser = _parser()
    default = parser.parse_args(["milestone9", "--source-dir", ".", "--name",
                                 "testworld", "--output", "."])
    assert default.nogova_terrain_clutter is False
    assert default.nogova_clutter_max_objects is None
    enabled = parser.parse_args(["milestone9", "--source-dir", ".", "--name",
                                 "testworld", "--output", ".",
                                 "--nogova-terrain-clutter",
                                 "--nogova-clutter-max-objects", "99"])
    assert enabled.nogova_terrain_clutter is True
    # A legacy flag remains parseable but no longer imposes any cap.
    assert enabled.nogova_clutter_max_objects == 99


def test_deterministic_seed_changes_without_object_cap():
    spec = _spec()
    raster = _raster()
    elevations = _heights()
    one = place_nogova_terrain_clutter(spec, raster, elevations, ())
    two = place_nogova_terrain_clutter(spec, raster, elevations, ())
    assert one == two
    assert 0 < len(one) <= (int(spec.cells * spec.cell_size // 45)) ** 2
    assert {o.model_path for o in one}.issubset(
        {item[0] for item in NOGOVA_CLUTTER_WEIGHTS}
    )
    assert all(o.y == pytest.approx(9.975) for o in one)
    assert len({(o.x, o.z) for o in one}) == len(one)
    different = place_nogova_terrain_clutter(_spec(seed=1245),
                                             raster, elevations, ())
    assert one != different
    old_saved_setting = _spec()
    old_saved_setting.nogova_terrain_clutter_maximum_objects = 0
    assert place_nogova_terrain_clutter(
        old_saved_setting, raster, elevations, ()
    ) == one  # Deprecated settings cannot cap or disable the new pass.
    assert place_nogova_terrain_clutter(_spec(enabled=False), raster, elevations, ()) == ()


def test_water_urban_forest_roads_fields_and_buildings_are_safe():
    spec = _spec()
    cells = spec.cells
    all_true = (True,) * (cells * cells)
    for key in ("water", "urban", "forest", "roads", "farmland", "buildings"):
        raster = _raster(**{key: all_true})
        assert place_nogova_terrain_clutter(spec, raster, _heights(), ()) == ()
    assert place_nogova_terrain_clutter(spec, _raster(),
                                       [-1.0] * cells * cells, ()) == ()
    # A steep or peaked/unsafe hill is never a grounded shrub candidate.
    slopes = [(z * 50 + x * 35) for z in range(cells)
              for x in range(cells)]
    assert place_nogova_terrain_clutter(spec, _raster(), slopes, ()) == ()


def test_existing_objects_are_kept_and_have_a_clearance():
    spec = _spec()
    base = place_nogova_terrain_clutter(spec, _raster(), _heights(), ())
    assert base
    original = base[0]
    existing = [WorldObject(1, r"o\hous\domek_sedy.p3d",
                            original.x, original.y, original.z, 0.0)]
    result = place_nogova_terrain_clutter(spec, _raster(), _heights(), existing)
    assert all((o.x - original.x) ** 2 + (o.z - original.z) ** 2 >= 7.0 ** 2
               for o in result)
    assert all(o.object_id > existing[0].object_id for o in result)
    assert existing == [WorldObject(1, r"o\hous\domek_sedy.p3d",
                                    original.x, original.y, original.z, 0.0)]


def test_invalid_raster_dimensions_rejected_without_mutating_world():
    spec = _spec()
    broken = _raster(water=[False] * 10)
    with pytest.raises(ValueError, match="mask dimensions"):
        place_nogova_terrain_clutter(spec, broken, _heights(), ())
    with pytest.raises(ValueError, match="terrain dimensions"):
        place_nogova_terrain_clutter(spec, _raster(), [1.0] * 12, ())


def test_pipeline_keeps_extra_objects_in_normal_and_reproducibility_builds():
    source = (Path(__file__).resolve().parents[1] / "src" /
              "cwr_worldgen" / "generator.py").read_text()
    assert "nogova_clutter = place_nogova_terrain_clutter(" in source
    assert "repeat_clutter = place_nogova_terrain_clutter(" in source
    assert "combined_model_usage.update(obj.model_path for obj in nogova_clutter)" in source
    assert '"nogova_terrain_clutter_enabled",' in source
    assert '"nogova_terrain_clutter_maximum_objects"' not in source
