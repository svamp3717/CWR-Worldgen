"""Original Nogova (noe.wrp) ps -> pt -> t1 shoreline ground tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen import generator
from cwr_worldgen.nogova_water_blends import (
    _NOGOVA_COAST_LAND, _NOGOVA_WATER_COAST,
    NOGOVA_COAST_PATH, NOGOVA_LAND_PATH, NOGOVA_WATER_PATH,
    apply_nogova_underwater_borders, is_nogova_water_blend_path,
    nogova_water_blend_path,
)
from cwr_worldgen.surface_pass import (
    MATERIAL_INDEX, MILESTONE9_MATERIALS, STOCK_SURFACE_TEXTURES,
)
from cwr_worldgen.terrain import STOCK_GROUND_TEXTURES
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


def _atlas(size=33):
    paths = [NOGOVA_WATER_PATH, NOGOVA_LAND_PATH, r"o\pole1.paa"]
    slots = np.full((size, size), 1, dtype=np.int32)
    slots[:, :size//2] = 0
    return paths, slots


def _corners(path):
    stem = path.replace("/", "\\").split("\\")[-1].split(".")[0]
    if len(stem) != 8:
        return (stem,) * 4
    return tuple(reversed(tuple(stem[i:i + 2] for i in (0, 2, 4, 6))))


def _assert_seams(paths, indices, *, border=2):
    output = np.asarray(indices)
    height, width = output.shape
    shapes = [[_corners(paths[int(i)]) for i in row] for row in output]
    for row in range(border, height-border-1):
        for col in range(border, width-border-1):
            current = shapes[row][col]
            east, north = shapes[row][col+1], shapes[row+1][col]
            assert (current[0], current[2]) == (east[1], east[3])
            assert (current[0], current[1]) == (north[2], north[3])


def test_stock_noe_water_and_two_complete_quartet_families() -> None:
    # Exact original noe.wrp texture-path table from nogova_textures(1).zip.
    assert len(_NOGOVA_WATER_COAST) == 14
    assert len(_NOGOVA_COAST_LAND) == 14
    for stems, pair in (
        (_NOGOVA_WATER_COAST, ("ps", "pt")),
        (_NOGOVA_COAST_LAND, ("pt", "t1")),
    ):
        possible = {
            "".join(pair[int(bit)] for bit in f"{index:04b}")
            for index in range(1, 15)
        }
        assert stems == possible
    assert STOCK_GROUND_TEXTURES["nogova"]["w"] == NOGOVA_WATER_PATH
    for code in ("g", "s"):
        assert STOCK_GROUND_TEXTURES["nogova"][code] == NOGOVA_LAND_PATH
    assert STOCK_SURFACE_TEXTURES["nogova"]["w"] == NOGOVA_WATER_PATH
    for code in ("g", "q", "s", "x"):
        assert STOCK_SURFACE_TEXTURES["nogova"][code] == NOGOVA_LAND_PATH
    assert is_nogova_water_blend_path(r"o\psptpspt.pac")
    assert is_nogova_water_blend_path(r"o\ptt1ptt1.pac")
    assert not is_nogova_water_blend_path(NOGOVA_WATER_PATH)
    assert not is_nogova_water_blend_path(r"o\t1t1t1j1.pac")
    assert not is_nogova_water_blend_path(r"eden\psptpspt.pac")
    assert nogova_water_blend_path(1, 0, 1, 0) == r"o\psptpspt.pac"
    assert nogova_water_blend_path(2, 1, 2, 1) == r"o\ptt1ptt1.pac"
    assert nogova_water_blend_path(2, 0, 2, 0) is None
    assert nogova_water_blend_path(0, 0, 0, 0) == NOGOVA_WATER_PATH
    assert nogova_water_blend_path(1, 1, 1, 1) == NOGOVA_COAST_PATH
    assert nogova_water_blend_path(2, 2, 2, 2) == NOGOVA_LAND_PATH


def test_straight_nogova_coast_has_two_distinct_underwater_borders() -> None:
    paths, slots = _atlas(size=32)
    expanded, result = apply_nogova_underwater_borders(
        paths, slots.ravel(), width=32,
        underwater_mask=(slots == 0).ravel(),
    )
    output = np.asarray(result).reshape(slots.shape)
    assert expanded[int(output[12, 12])] == NOGOVA_WATER_PATH
    assert expanded[int(output[12, 13])] == r"o\psptpspt.pac"
    assert expanded[int(output[12, 14])] == NOGOVA_COAST_PATH
    assert expanded[int(output[12, 15])] == r"o\ptt1ptt1.pac"
    assert expanded[int(output[12, 16])] == NOGOVA_LAND_PATH
    assert not any(
        is_nogova_water_blend_path(expanded[int(slot)])
        for slot in output[slots != 0]
    )
    _assert_seams(expanded, output)


@pytest.mark.parametrize("shape", ["curved", "narrow", "islands", "random"])
def test_water_ground_border_continuity_without_invented_tiles(shape):
    size = 43
    paths, slots = _atlas(size)
    slots[:] = 1
    if shape == "curved":
        for row in range(size):
            cutoff = 13 + round(4 * np.sin(row * np.pi / 13))
            slots[row, :cutoff] = 0
    elif shape == "narrow":
        slots[5:36, 17:19] = 0
    elif shape == "islands":
        slots[8, 8] = 0
        slots[21:25, 11:16] = 0
        slots[28:31, 20:27] = 0
    else:
        slots[:] = np.where(np.random.default_rng(1122).random(slots.shape) < .37, 0, 1)
    wet = slots == 0
    expanded, result = apply_nogova_underwater_borders(
        paths, slots.ravel(), width=size, underwater_mask=wet.ravel(),
    )
    output = np.asarray(result).reshape(slots.shape)
    _assert_seams(expanded, output)
    assert all(
        item in {
            NOGOVA_WATER_PATH, NOGOVA_COAST_PATH, NOGOVA_LAND_PATH,
        } or is_nogova_water_blend_path(item)
        for item in expanded
    )
    assert all(expanded[int(output[y, x])] == NOGOVA_LAND_PATH
               for y, x in np.argwhere(~wet))


def test_isolated_ps_water_cell_keeps_water_height_but_not_a_square() -> None:
    paths, slots = _atlas(size=25)
    slots[:, :] = 1
    slots[11, 11] = 0
    expanded, result = apply_nogova_underwater_borders(
        paths, slots.ravel(), width=25,
        underwater_mask=(slots == 0).ravel(),
    )
    assert expanded[result[11*25 + 11]] == NOGOVA_LAND_PATH
    assert all(result[i] == 1 for i in range(len(result)))


def test_above_sea_level_or_protected_ground_never_contains_water_art() -> None:
    paths, slots = _atlas(size=33)
    slots[9, 15] = 2   # protected farmland next to submerged sea
    wet = slots == 0
    wet[12, 14] = False   # semantic water on physically dry terrain
    expanded, result = apply_nogova_underwater_borders(
        paths, slots.ravel(), width=33, underwater_mask=wet.ravel(),
    )
    output = np.asarray(result).reshape(slots.shape)
    assert output[9, 15] == 2
    assert expanded[int(output[12, 14])] == NOGOVA_LAND_PATH
    assert all(expanded[int(output[y, x])] not in {
        NOGOVA_WATER_PATH, NOGOVA_COAST_PATH,
    } and not is_nogova_water_blend_path(expanded[int(output[y, x])])
               for y, x in np.argwhere(~wet))
    assert set(expanded[len(paths):]).issubset({
        rf"o\{stem}.pac"
        for stem in _NOGOVA_WATER_COAST | _NOGOVA_COAST_LAND
    })


def test_slot_limit_and_input_dimensions() -> None:
    paths, slots = _atlas(size=25)
    values = slots.ravel()
    unchanged_paths, unchanged_indices = apply_nogova_underwater_borders(
        paths, values, width=25, underwater_mask=(slots == 0).ravel(),
        maximum_texture_slots=len(paths),
    )
    assert unchanged_paths == tuple(paths)
    assert unchanged_indices == values.tolist()
    for args, match in [
        (dict(width=0, underwater_mask=(slots == 0).ravel()), "dimensions"),
        (dict(width=25, underwater_mask=[True]*4), "underwater mask"),
    ]:
        with pytest.raises(ValueError, match=match):
            apply_nogova_underwater_borders(paths, values, **args)
    with pytest.raises(ValueError, match="missing paths"):
        apply_nogova_underwater_borders(
            paths, [999]*625, width=25, underwater_mask=[True]*625,
        )


def test_generator_height_cutoff_and_written_rvw4(tmp_path) -> None:
    size = 33
    heights = np.full((size, size), 6.5)
    heights[:, :17] = 6.0
    mats = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    mats[:, :17] = MATERIAL_INDEX["w"]
    spec = SimpleNamespace(
        name="nogtest", ground_texture_profile="nogova",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        cells=size, sea_level=6.0, forest_undergrowth_textures=False,
        dummy_texture_path=r"nogtest\data\d.paa",
    )
    raster = SimpleNamespace(forest=np.zeros(size*size,dtype=bool))
    with pytest.raises(ValueError, match="final WRP elevations"):
        generator._stock_ground_texture_table(spec, mats.ravel(), raster,
                                              MILESTONE9_MATERIALS)
    paths, indices = generator._stock_ground_texture_table(
        spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
        elevations=heights.ravel(),
    )
    output = np.asarray(indices).reshape((size, size))
    assert paths[int(output[16, 12])] == NOGOVA_WATER_PATH
    assert paths[int(output[16, 14])] == r"o\psptpspt.pac"
    assert paths[int(output[16, 15])] == NOGOVA_COAST_PATH
    assert paths[int(output[16, 16])] == r"o\ptt1ptt1.pac"
    assert paths[int(output[16, 17])] == NOGOVA_LAND_PATH
    assert all(paths[int(output[y, x])] == NOGOVA_LAND_PATH
               for y, x in np.argwhere(heights > spec.sea_level))
    wrp = tmp_path / "nogova_shore.wrp"
    write_rvw4(wrp, size, size, heights.ravel().tolist(), indices, paths, (),
               height_scale=0.05)
    readback = inspect_rvw4(wrp, height_scale=0.05)
    assert {x.casefold() for x in readback.texture_paths} == {
        x.casefold() for x in paths
    }
