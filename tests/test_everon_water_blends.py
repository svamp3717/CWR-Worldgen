"""Verified original Eden ps sand to tn grass shoreline blends."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen import generator
from cwr_worldgen.everon_water_blends import (
    _EVERON_SHORE_QUARTETS,
    _resolve_missing_ps_tn_diagonal,
    apply_everon_water_borders,
    everon_shore_blend_path,
    is_everon_water_blend_path,
)
from cwr_worldgen.surface_pass import (
    MATERIAL_INDEX, MILESTONE9_MATERIALS, STOCK_SURFACE_TEXTURES,
)
from cwr_worldgen.terrain import STOCK_GROUND_TEXTURES
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


def _shore(width: int = 13) -> tuple[list[str], np.ndarray]:
    """13×13 sand-to-grass slice in WRP south-to-north row order."""
    paths = [
        r"demo\data\d.paa", r"eden\ps.paa", r"eden\tn.paa",
        r"eden\zbh.paa",
    ]
    cells = np.full((width, width), 2, dtype=np.int32)
    cells[:, :4] = 1
    return paths, cells


def _corners(path: str) -> tuple[str, str, str, str]:
    """Actual geographic NE/NW/SE/SW artwork corners of source Eden tiles."""
    stem = path.casefold().split("\\")[-1].split(".")[0]
    if len(stem) != 8:
        return (stem,) * 4
    return tuple(reversed(tuple(
        stem[index:index + 2] for index in range(0, 8, 2)
    )))


def test_everon_beach_and_water_use_ps_not_stone_sz() -> None:
    assert STOCK_GROUND_TEXTURES["everon"]["w"] == r"Eden\ps.paa"
    assert STOCK_GROUND_TEXTURES["everon"]["s"] == r"Eden\ps.paa"
    for kind in ("w", "q", "s", "x"):
        assert STOCK_SURFACE_TEXTURES["everon"][kind] == r"Eden\ps.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["g"] == r"Eden\tn.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["a"] == r"Eden\zbh.paa"
    assert r"Eden\sz.paa" not in STOCK_SURFACE_TEXTURES["everon"].values()
    assert STOCK_SURFACE_TEXTURES["nogova"]["w"] != r"Eden\ps.paa"


def test_ps_tn_quartets_are_exact_verified_eden_paths() -> None:
    # Exported texture-paths.csv includes all 13 of these paths, but no
    # geographic grass NE+SW (mask 9) diagonal.
    assert _EVERON_SHORE_QUARTETS == frozenset("""
        pstntnps pspstntn pstntntn tntnpsps pspspstn pstnpstn
        pstnpsps tntnpstn tnpstntn tntntnps tnpstnps tnpspsps
        pspstnps
    """.split())
    assert everon_shore_blend_path(1, 0, 1, 0) == r"eden\pstnpstn.paa"
    assert everon_shore_blend_path(0, 1, 0, 1) == r"eden\tnpstnps.paa"
    assert everon_shore_blend_path(1, 1, 0, 0) == r"eden\pspstntn.paa"
    assert everon_shore_blend_path(0, 0, 1, 1) == r"eden\tntnpsps.paa"
    assert everon_shore_blend_path(0, 0, 0, 0) == r"eden\ps.paa"
    assert everon_shore_blend_path(1, 1, 1, 1) == r"eden\tn.paa"
    assert everon_shore_blend_path(1, 0, 0, 1) is None
    assert everon_shore_blend_path(-1, 0, 1, 1) is None
    assert everon_shore_blend_path(2, 0, 1, 1) is None
    assert all("sz" not in stem for stem in _EVERON_SHORE_QUARTETS)


def test_shared_vertex_resolution_removes_unavailable_diagonal() -> None:
    # An individual geographic cell's NE+SW grass diagonal is absent in
    # Eden. Fix the vertex lattice, not just the name for that cell.
    vertices = np.array([[1, 0], [0, 1]], dtype=np.int8)
    _resolve_missing_ps_tn_diagonal(vertices)
    np.testing.assert_array_equal(vertices, np.array([[0, 0], [0, 1]]))
    two = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ], dtype=np.int8)
    _resolve_missing_ps_tn_diagonal(two)
    ne, nw = two[1:, 1:], two[1:, :-1]
    se, sw = two[:-1, 1:], two[:-1, :-1]
    assert not np.any((ne == 1) & (nw == 0) & (se == 0) & (sw == 1))


def test_straight_sandy_border_blends_into_grass_without_stone() -> None:
    paths, cells = _shore()
    expanded, indices = apply_everon_water_borders(
        paths, cells.ravel(), width=13,
    )
    output = np.asarray(indices).reshape(13, 13)
    assert expanded[int(output[6, 1])] == r"eden\ps.paa"
    assert expanded[int(output[6, 3])] == r"eden\pstnpstn.paa"
    assert expanded[int(output[6, 4])] == r"eden\tn.paa"
    assert expanded[int(output[6, 11])] == r"eden\tn.paa"
    assert all("sz" not in path.casefold() for path in expanded)
    assert len(expanded) == len(paths) + 1

    tiles = [[_corners(expanded[int(slot)]) for slot in row] for row in output]
    # Horizontal and vertical neighbours must agree at both shared corners.
    for row in range(1, 12):
        for col in range(1, 12):
            me = tiles[row][col]
            east = tiles[row][col + 1]
            north = tiles[row + 1][col]
            assert (me[0], me[2]) == (east[1], east[3])
            assert (me[0], me[1]) == (north[2], north[3])


def test_curved_sandy_coast_maintains_corner_continuity() -> None:
    paths, cells = _shore(width=21)
    for row in range(21):
        coast_col = 6 + int(round(2 * np.sin(row * np.pi / 6)))
        cells[row, :] = 2
        cells[row, :coast_col] = 1
    expanded, indices = apply_everon_water_borders(
        paths, cells.ravel(), width=21,
    )
    output = np.asarray(indices).reshape(21, 21)
    assert all("sz" not in p.lower() for p in expanded)
    tiles = [[_corners(expanded[int(slot)]) for slot in row] for row in output]
    # Only original ps, tn and their stock directional paths are valid.
    assert len(expanded) > len(paths) + 1
    for row in range(2, 19):
        for col in range(2, 19):
            cell = tiles[row][col]
            if "zbh" in cell:
                continue
            east = tiles[row][col + 1]
            north = tiles[row + 1][col]
            assert (cell[0], cell[2]) == (east[1], east[3])
            assert (cell[0], cell[1]) == (north[2], north[3])


def test_farmland_is_preserved_and_single_tile_water_survives() -> None:
    paths, cells = _shore()
    cells[6, 6] = 3
    expanded, indices = apply_everon_water_borders(
        paths, cells.ravel(), width=13,
    )
    output = np.asarray(indices).reshape(13, 13)
    assert output[6, 6] == 3
    assert expanded[int(output[6, 6])] == r"eden\zbh.paa"

    isolated = np.full((13, 13), 2, dtype=np.int32)
    isolated[6, 6] = 1
    _, texture_slots = apply_everon_water_borders(
        paths, isolated.ravel(), width=13,
    )
    assert texture_slots[6 * 13 + 6] == 1


def test_no_invented_textures_and_slot_budget() -> None:
    paths, cells = _shore()
    original = cells.ravel().tolist()
    full_paths, unchanged = apply_everon_water_borders(
        paths, original, width=13, maximum_texture_slots=len(paths),
    )
    assert full_paths == tuple(paths)
    assert unchanged == original
    expanded, slots = apply_everon_water_borders(
        paths, original, width=13,
    )
    assert set(expanded[len(paths):]).issubset({
        rf"eden\{stem}.paa" for stem in _EVERON_SHORE_QUARTETS
    })
    assert len(expanded) <= 512
    assert len(slots) == 13 * 13
    with pytest.raises(ValueError, match="dimensions"):
        apply_everon_water_borders(paths, original, width=12)
    with pytest.raises(ValueError, match="missing paths"):
        apply_everon_water_borders(paths, [999] * 169, width=13)


def test_4wvr_export_and_generation_use_exact_ps_tn_paths(tmp_path) -> None:
    size = 13
    mats = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    mats[:, :4] = MATERIAL_INDEX["w"]
    mats[:, 4:6] = MATERIAL_INDEX["s"]
    # Sand and underwater are both ps, so the boundary to tn is only
    # at the last dry shore column, not between w and s.
    spec = SimpleNamespace(
        name="demo", ground_texture_profile="everon",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=False, cells=size,
        dummy_texture_path=r"demo\data\d.paa",
    )
    raster = SimpleNamespace(forest=np.zeros(size * size, dtype=bool))
    paths, slots = generator._stock_ground_texture_table(
        spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
    )
    result = np.asarray(slots).reshape(size, size)
    assert paths[int(result[6, 2])].casefold() == r"eden\ps.paa"
    assert paths[int(result[6, 4])].casefold() == r"eden\ps.paa"
    assert paths[int(result[6, 5])].casefold() == r"eden\pstnpstn.paa"
    assert paths[int(result[6, 8])].casefold() == r"eden\tn.paa"
    assert not any("sz" in path.casefold() for path in paths)
    wrp = tmp_path / "everon_ps_coast.wrp"
    write_rvw4(wrp, size, size, [10.0] * (size * size),
               slots, paths, (), height_scale=0.05)
    parsed = inspect_rvw4(wrp, height_scale=0.05)
    assert {p.casefold() for p in parsed.texture_paths} == {
        p.casefold() for p in paths
    }


def test_other_stock_profiles_keep_existing_ground_texture_tables() -> None:
    size = 13
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
        assert paths == (
            spec.dummy_texture_path, *generator._ground_texture_paths(spec),
        )
        assert slots == [MATERIAL_INDEX["w"] + 1] * len(mats)


def test_water_border_manifest_paths_are_only_ps_tn() -> None:
    assert is_everon_water_blend_path(r"eden\pstnpstn.paa")
    assert is_everon_water_blend_path(r"EDEN\TNPSPSPS.PAA")
    assert not is_everon_water_blend_path(r"eden\psszpssz.paa")
    assert not is_everon_water_blend_path(r"eden\sz.paa")
    assert not is_everon_water_blend_path(r"eden\pltnpltn.paa")
    assert not is_everon_water_blend_path(r"eden\ps.paa")
    assert not is_everon_water_blend_path(r"o\pstnpstn.paa")
