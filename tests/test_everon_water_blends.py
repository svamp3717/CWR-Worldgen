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
    assert STOCK_GROUND_TEXTURES["everon"]["s"] == r"Eden\tn.paa"
    assert STOCK_SURFACE_TEXTURES["everon"]["w"] == r"Eden\ps.paa"
    for kind in ("q", "s", "x"):
        assert STOCK_SURFACE_TEXTURES["everon"][kind] == r"Eden\tn.paa"
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
    # Resolving an absent source filename may expand grass, but must
    # NEVER introduce a ps corner next to dry terrain.
    np.testing.assert_array_equal(vertices, np.array([[1, 0], [1, 1]]))
    two = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ], dtype=np.int8)
    before = two.copy()
    _resolve_missing_ps_tn_diagonal(two)
    assert np.all(two[before == 1] == 1)
    ne, nw = two[1:, 1:], two[1:, :-1]
    se, sw = two[:-1, 1:], two[:-1, :-1]
    assert not np.any((ne == 1) & (nw == 0) & (se == 0) & (sw == 1))


def test_straight_sandy_border_blends_into_grass_without_stone() -> None:
    paths, cells = _shore()
    expanded, indices = apply_everon_water_borders(
        paths, cells.ravel(), width=13,
        underwater_mask=(cells == 1).ravel(),
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
        underwater_mask=(cells == 1).ravel(),
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
        underwater_mask=(cells == 1).ravel(),
    )
    output = np.asarray(indices).reshape(13, 13)
    assert output[6, 6] == 3
    assert expanded[int(output[6, 6])] == r"eden\zbh.paa"

    isolated = np.full((13, 13), 2, dtype=np.int32)
    isolated[6, 6] = 1
    _, texture_slots = apply_everon_water_borders(
        paths, isolated.ravel(), width=13,
        underwater_mask=(isolated == 1).ravel(),
    )
    # A lone submerged 50-metre ps tile with four dry-grass corners
    # cannot be blended seamlessly with any original ps/tn quartet.
    # The only continuous stock tile is tn (terrain remains underwater).
    assert texture_slots[6 * 13 + 6] == 2


def test_no_invented_textures_and_slot_budget() -> None:
    paths, cells = _shore()
    original = cells.ravel().tolist()
    full_paths, unchanged = apply_everon_water_borders(
        paths, original, width=13,
        underwater_mask=(cells == 1).ravel(),
        maximum_texture_slots=len(paths),
    )
    assert full_paths == tuple(paths)
    assert unchanged == original
    expanded, slots = apply_everon_water_borders(
        paths, original, width=13,
        underwater_mask=(cells == 1).ravel(),
    )
    assert set(expanded[len(paths):]).issubset({
        rf"eden\{stem}.paa" for stem in _EVERON_SHORE_QUARTETS
    })
    assert len(expanded) <= 512
    assert len(slots) == 13 * 13
    with pytest.raises(ValueError, match="dimensions"):
        apply_everon_water_borders(
            paths, original, width=12, underwater_mask=(cells == 1).ravel(),
        )
    with pytest.raises(ValueError, match="missing paths"):
        apply_everon_water_borders(
            paths, [999] * 169, width=13,
            underwater_mask=(cells == 1).ravel(),
        )
    with pytest.raises(ValueError, match="underwater mask"):
        apply_everon_water_borders(
            paths, original, width=13, underwater_mask=[True] * 13,
        )


def test_4wvr_export_and_generation_use_exact_ps_tn_paths(tmp_path) -> None:
    size = 13
    mats = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    mats[:, :4] = MATERIAL_INDEX["w"]
    mats[:, 4:6] = MATERIAL_INDEX["s"]
    # The water must be below the plane; dry shore classifications stay tn.
    heights = np.full((size, size), 2.5, dtype=np.float64)
    heights[:, :4] = -2.0
    spec = SimpleNamespace(
        name="demo", ground_texture_profile="everon",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=False, cells=size, sea_level=0.0,
        dummy_texture_path=r"demo\data\d.paa",
    )
    raster = SimpleNamespace(forest=np.zeros(size * size, dtype=bool))
    paths, slots = generator._stock_ground_texture_table(
        spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
        elevations=heights.ravel(),
    )
    result = np.asarray(slots).reshape(size, size)
    assert paths[int(result[6, 2])].casefold() == r"eden\ps.paa"
    assert paths[int(result[6, 3])].casefold() == r"eden\pstnpstn.paa"
    assert paths[int(result[6, 4])].casefold() == r"eden\tn.paa"
    assert paths[int(result[6, 5])].casefold() == r"eden\tn.paa"
    assert paths[int(result[6, 8])].casefold() == r"eden\tn.paa"
    assert not any("sz" in path.casefold() for path in paths)
    wrp = tmp_path / "everon_ps_coast.wrp"
    write_rvw4(wrp, size, size, heights.ravel().tolist(),
               slots, paths, (), height_scale=0.05)
    parsed = inspect_rvw4(wrp, height_scale=0.05)
    assert {p.casefold() for p in parsed.texture_paths} == {
        p.casefold() for p in paths
    }


def test_dry_nogova_land_keeps_unchanged_texture_table() -> None:
    size = 13
    spec = SimpleNamespace(
        name="demo", ground_texture_profile="nogova",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=False, cells=size, sea_level=0.0,
        dummy_texture_path=r"demo\data\d.paa",
    )
    mats = [MATERIAL_INDEX["g"]] * (size * size)
    raster = SimpleNamespace(forest=[False] * len(mats))
    paths, slots = generator._stock_ground_texture_table(
        spec, mats, raster, MILESTONE9_MATERIALS,
        elevations=[10.0] * len(mats),
    )
    assert paths == (
        spec.dummy_texture_path, *generator._ground_texture_paths(spec),
    )
    assert slots == [MATERIAL_INDEX["g"] + 1] * len(mats)


def test_water_border_manifest_paths_are_only_ps_tn() -> None:
    assert is_everon_water_blend_path(r"eden\pstnpstn.paa")
    assert is_everon_water_blend_path(r"EDEN\TNPSPSPS.PAA")
    assert not is_everon_water_blend_path(r"eden\psszpssz.paa")
    assert not is_everon_water_blend_path(r"eden\sz.paa")
    assert not is_everon_water_blend_path(r"eden\pltnpltn.paa")
    assert not is_everon_water_blend_path(r"eden\ps.paa")
    assert not is_everon_water_blend_path(r"o\pstnpstn.paa")


def test_dry_water_semantic_and_above_water_ps_are_never_painted() -> None:
    paths, cells = _shore()
    wet = (cells == 1)
    wet[4, 3] = False
    expanded, slots = apply_everon_water_borders(
        paths, cells.ravel(), width=13, underwater_mask=wet.ravel(),
    )
    result = np.asarray(slots).reshape((13, 13))
    assert expanded[int(result[4, 3])] == r"eden\tn.paa"
    for row, col in np.argwhere(~wet):
        texture = expanded[int(result[row, col])].casefold()
        assert texture != r"eden\ps.paa"
        assert not is_everon_water_blend_path(texture)
    one = np.full((13, 13), 2, dtype=np.int32)
    one[6, 6] = 1
    submerged = np.zeros((13, 13), dtype=np.bool_)
    submerged[6, 6] = True
    _, underwater_slots = apply_everon_water_borders(
        paths, one.ravel(), width=13, underwater_mask=submerged.ravel(),
    )
    _, dry_slots = apply_everon_water_borders(
        paths, one.ravel(), width=13, underwater_mask=np.zeros(169, dtype=bool),
    )
    # All four corners of this one-cell underwater area touch tn,
    # so there is no stock blend that retains ps without hard edges.
    assert underwater_slots[6 * 13 + 6] == 2
    assert dry_slots[6 * 13 + 6] == 2


def test_generator_uses_configured_sea_level_and_requires_final_elevations() -> None:
    size = 13
    spec = SimpleNamespace(
        name="demo", ground_texture_profile="everon",
        surface_pass_enabled=True, surface_ground_mode="milestone9",
        forest_undergrowth_textures=False, cells=size, sea_level=7.5,
        dummy_texture_path=r"demo\data\d.paa",
    )
    mats = np.full((size, size), MATERIAL_INDEX["g"], dtype=np.int32)
    mats[:, 0:4] = MATERIAL_INDEX["w"]
    heights = np.full((size, size), 7.6)
    heights[:, 0:2] = 7.5
    raster = SimpleNamespace(forest=np.zeros(size * size, dtype=bool))
    with pytest.raises(ValueError, match="final WRP elevations"):
        generator._stock_ground_texture_table(
            spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
        )
    paths, slots = generator._stock_ground_texture_table(
        spec, mats.ravel(), raster, MILESTONE9_MATERIALS,
        elevations=heights.ravel(),
    )
    result = np.asarray(slots).reshape((size, size))
    assert paths[int(result[6, 0])].casefold() == r"eden\ps.paa"
    for row in range(size):
        for col in range(size):
            if heights[row, col] > spec.sea_level:
                texture = paths[int(result[row, col])].casefold()
                assert texture != r"eden\ps.paa"
                assert not is_everon_water_blend_path(texture)


def test_isolated_underwater_squares_fall_back_to_seamless_grass() -> None:
    """Screenshot regression: no square ps stamps in a thin coastline."""
    paths, cells = _shore(width=17)
    cells[:, :] = 2
    # Irregular submerged islands alongside a thin 1-cell-wide inlet.
    cells[4, 5] = 1
    cells[9, 11] = 1
    cells[5:10, 7] = 1
    wet = cells == 1
    expanded, texture_slots = apply_everon_water_borders(
        paths, cells.ravel(), width=17, underwater_mask=wet.ravel(),
    )
    output = np.asarray(texture_slots).reshape(cells.shape)
    assert all(expanded[int(output[r, c])] == r"eden\tn.paa"
               for r, c in ((4, 5), (9, 11), (7, 7)))
    # Do not place any stock ps/tn blend over the grass-dry cells.
    for row, col in np.argwhere(~wet):
        assert expanded[int(output[row, col])] == r"eden\tn.paa"
    # Water semantics/elevation are independent of the ground tile.
    assert int(wet.sum()) == 7


def test_narrow_two_cell_water_strip_has_borders_on_both_sides() -> None:
    """A sub-100m inlet must not have rectangular ps end-cap artefacts."""
    paths, cells = _shore(width=17)
    cells[:, :] = 2
    cells[3:14, 6:8] = 1
    wet = cells == 1
    expanded, texture_slots = apply_everon_water_borders(
        paths, cells.ravel(), width=17, underwater_mask=wet.ravel(),
    )
    output = np.asarray(texture_slots).reshape((17, 17))
    assert expanded[int(output[8, 6])] == r"eden\tnpstnps.paa"
    assert expanded[int(output[8, 7])] == r"eden\pstnpstn.paa"
    # The wet edge cells use the verified original PAA family.
    for row, col in np.argwhere(wet):
        path = expanded[int(output[row, col])]
        assert path in {r"eden\ps.paa", r"eden\tn.paa"} or is_everon_water_blend_path(path)
    # Verify shared edge corners for every wet/grass tile, not just that
    # the paths happen to exist in the original Eden archive.
    tiles = [[_corners(expanded[int(slot)]) for slot in row] for row in output]
    for row in range(1, 15):
        for col in range(1, 15):
            this = tiles[row][col]
            east, north = tiles[row][col + 1], tiles[row + 1][col]
            assert (this[0], this[2]) == (east[1], east[3])
            assert (this[0], this[1]) == (north[2], north[3])


def test_ps_border_along_protected_farmland_avoids_full_ps_square() -> None:
    paths, cells = _shore()
    # Farmland lies exactly on a water/grass corner.
    cells[6, 3] = 3
    expanded, slots = apply_everon_water_borders(
        paths, cells.ravel(), width=13, underwater_mask=(cells == 1).ravel(),
    )
    output = np.asarray(slots).reshape(cells.shape)
    assert expanded[int(output[6, 3])] == r"eden\zbh.paa"
    adjacent = expanded[int(output[6, 2])]
    # Adjacent underwater texture must fade towards land, rather than
    # skipping blending because it touches a protected vertex.
    assert adjacent != r"eden\ps.paa"
    assert adjacent == r"eden\tn.paa" or is_everon_water_blend_path(adjacent)
