"""Original Eden water, sand and inland texture families and border continuity."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen import generator
from cwr_worldgen.everon_water_blends import (
    _EVERON_SHORE_QUARTETS,
    apply_everon_water_borders,
    everon_shore_blend_path, is_everon_water_blend_path,
)
from cwr_worldgen.surface_pass import MATERIAL_INDEX, STOCK_SURFACE_TEXTURES
from cwr_worldgen.terrain import STOCK_GROUND_TEXTURES
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


def _shore(width: int = 13) -> tuple[list[str], np.ndarray]:
    """A 13x13 water-to-sand-to-grass slice in south-to-north WRP order."""
    texture_paths = [
        r"demo\data\d.paa",
        r"eden\ps.paa", r"eden\sz.paa", r"eden\tn.paa",
        r"eden\zbh.paa",
    ]
    indices = np.full((width, width), 3, dtype=np.int32)
    indices[:, :4] = 1
    indices[:, 4:8] = 2
    return texture_paths, indices


def _corners(path: str) -> tuple[str, str, str, str]:
    """Actual WRP geographic NE, NW, SE, SW for a shoreline tile."""
    stem = path.casefold().split("\\")[-1].split(".")[0]
    return tuple(reversed(tuple(
        stem[index:index + 2] for index in range(0, len(stem), 2)
    ))) if len(stem) == 8 else (stem,) * 4


def test_everon_underwater_and_beach_use_original_stock_paths() -> None:
    assert STOCK_GROUND_TEXTURES["everon"]["w"] == r"Eden\ps.paa"
    assert STOCK_GROUND_TEXTURES["everon"]["s"] == r"Eden\sz.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["w"] == r"Eden\ps.paa"
    for material in ("q", "s", "x"):
        assert STOCK_SURFACE_TEXTURES["everon"][material] == r"Eden\sz.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["g"] == r"Eden\tn.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["a"] == r"Eden\zbh.paa"
    assert STOCK_SURFACE_TEXTURES["nogova"]["w"] != r"Eden\ps.paa"


def test_stock_water_quartet_artwork_uses_opposite_filename_corners() -> None:
    # All verified in the original eden.wrp export; the 180-degree rotation
    # gives exact shared-corner agreement with the stock ground texture grid.
    assert len(_EVERON_SHORE_QUARTETS) == 74
    assert everon_shore_blend_path(1, 0, 1, 0) == r"eden\psszpssz.paa"
    assert everon_shore_blend_path(2, 1, 2, 1) == r"eden\sztnsztn.paa"
    assert everon_shore_blend_path(0, 0, 1, 1) == r"eden\szszpsps.paa"
    assert everon_shore_blend_path(1, 1, 0, 0) == r"eden\pspsszsz.paa"
    assert everon_shore_blend_path(0, 0, 0, 0) == r"eden\ps.paa"
    assert everon_shore_blend_path(2, 2, 2, 2) == r"eden\tn.paa"
    assert everon_shore_blend_path(2, 0, 0, 2) is None  # absent stock quartet
    assert everon_shore_blend_path(0, 1, 2, 2) is not None
    assert everon_shore_blend_path(-1, 1, 1, 1) is None


def test_shoreline_quartets_follow_water_sand_grass_edges() -> None:
    paths, cells = _shore()
    expanded, slots = apply_everon_water_borders(paths, cells.ravel(), width=13)
    output = np.asarray(slots).reshape(13, 13)
    assert expanded[int(output[6, 1])] == r"eden\ps.paa"
    assert expanded[int(output[6, 3])] == r"eden\psszpssz.paa"
    assert expanded[int(output[6, 5])] == r"eden\sz.paa"
    assert expanded[int(output[6, 8])] == r"eden\sztnsztn.paa"
    assert expanded[int(output[6, 11])] == r"eden\tn.paa"
    assert not any("bak" in p for p in expanded[5:])
    # Every tile placed on the shared shoreline must agree exactly on both
    # sides of each geographical WRP vertex, unlike fixed rectangular bands.
    tiles = [[_corners(expanded[slot]) for slot in row] for row in output]
    for row in range(1, 12):
        for col in range(1, 12):
            me = tiles[row][col]
            east = tiles[row][col + 1]
            north = tiles[row + 1][col]
            assert (me[0], me[2]) == (east[1], east[3])
            assert (me[0], me[1]) == (north[2], north[3])


def test_protected_textures_and_single_tile_water_survive() -> None:
    paths, cells = _shore()
    cells[6, 6] = 4  # farmland texture, not ordinary shore ground
    orig = cells[6, 6]
    expanded, slots = apply_everon_water_borders(paths, cells.ravel(), width=13)
    out = np.asarray(slots).reshape(13, 13)
    assert out[6, 6] == orig
    assert expanded[int(out[6, 6])] == r"eden\zbh.paa"

    single = np.full((13, 13), 3, dtype=np.int32)
    single[6, 6] = 1  # A tiny water body cannot disappear into dry grass.
    _, tiny_slots = apply_everon_water_borders(paths, single.ravel(), width=13)
    assert tiny_slots[6 * 13 + 6] == 1


def test_missing_assets_and_texture_table_limits_never_invent_paths() -> None:
    paths, cells = _shore()
    original = cells.ravel().tolist()
    no_slots, unchanged = apply_everon_water_borders(
        paths, original, width=13, maximum_texture_slots=len(paths),
    )
    assert no_slots == tuple(paths)
    assert unchanged == original
    expanded, slots = apply_everon_water_borders(paths, original, width=13)
    assert set(expanded[len(paths):]).issubset({
        rf"eden\{stem}.paa" for stem in _EVERON_SHORE_QUARTETS
    })
    assert len(expanded) <= 512
    assert len(slots) == 13 * 13
    with pytest.raises(ValueError, match="dimensions"):
        apply_everon_water_borders(paths, original, width=12)
    with pytest.raises(ValueError, match="missing paths"):
        apply_everon_water_borders(paths, [99] * 169, width=13)


def test_generated_everon_texture_table_contains_real_shoreline_blends(tmp_path) -> None:
    size = 13
    mats = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    mats[:, :4] = MATERIAL_INDEX["w"]
    mats[:, 4:8] = MATERIAL_INDEX["s"]
    spec = SimpleNamespace(
        name="demo", ground_texture_profile="everon",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=False, cells=size,
        dummy_texture_path=r"demo\data\d.paa",
    )
    raster = SimpleNamespace(forest=np.zeros(size * size, dtype=bool))
    from cwr_worldgen.surface_pass import MILESTONE9_MATERIALS
    paths, slots = generator._stock_ground_texture_table(
        spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
    )
    output = np.asarray(slots).reshape(size, size)
    assert paths[int(output[6, 1])].casefold() == r"eden\ps.paa"
    assert paths[int(output[6, 3])].casefold() == r"eden\psszpssz.paa"
    assert paths[int(output[6, 8])].casefold() == r"eden\sztnsztn.paa"
    wrp = tmp_path / "shore.wrp"
    write_rvw4(
        wrp, size, size, [10.0] * (size * size), slots, paths, (),
        height_scale=0.05,
    )
    parsed = inspect_rvw4(wrp, height_scale=0.05)
    assert set(p.casefold() for p in paths) == set(
        p.casefold() for p in parsed.texture_paths
    )


def test_other_stock_profiles_keep_existing_unchanged_texture_table() -> None:
    size = 13
    from cwr_worldgen.surface_pass import MILESTONE9_MATERIALS
    for profile in ("nogova", "kolgujev", "malden"):
        spec = SimpleNamespace(
            name="demo", ground_texture_profile=profile,
            surface_pass_enabled=True, surface_ground_mode="milestone9",
            forest_undergrowth_textures=False, cells=size,
            dummy_texture_path=r"demo\data\d.paa",
        )
        mats = [MATERIAL_INDEX["w"]] * (size * size)
        raster = SimpleNamespace(forest=[False] * len(mats))
        paths, slots = generator._stock_ground_texture_table(
            spec, mats, raster, MILESTONE9_MATERIALS,
        )
        assert paths == (spec.dummy_texture_path, *generator._ground_texture_paths(spec))
        assert slots == [MATERIAL_INDEX["w"] + 1] * len(mats)


def test_water_border_manifest_paths_are_not_misreported_as_forest() -> None:
    assert is_everon_water_blend_path(r"eden\psszpssz.paa")
    assert is_everon_water_blend_path(r"EDEN\SZTNSZTN.PAA")
    assert not is_everon_water_blend_path(r"eden\pltnpltn.paa")
    assert not is_everon_water_blend_path(r"eden\ps.paa")
    assert not is_everon_water_blend_path(r"o\psszpssz.paa")
