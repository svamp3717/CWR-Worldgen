"""Verified original Abel and Cain water-ground source WRP borders."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen import generator
from cwr_worldgen.stock_water_ground_blends import (
    STOCK_COAST_FAMILIES,
    _resolve_missing_diagonals,
    apply_stock_underwater_borders,
    is_stock_water_blend_path,
    stock_water_blend_path,
)
from cwr_worldgen.surface_pass import (
    MATERIAL_INDEX, MILESTONE9_MATERIALS, STOCK_SURFACE_TEXTURES,
)
from cwr_worldgen.terrain import STOCK_GROUND_TEXTURES
from cwr_worldgen.wrp import inspect_rvw4, write_rvw4


# These exact original quartet sets were independently obtained by reading
# abel.wrp and cain.wrp texture-paths.csv from Island Studio.
FAMILIES = (
    ("malden", "abel", "pi", "tt", "pac", 14),
    ("kolgujev", "cain", "l4", "j9", "paa", 12),
)


def _scenario(profile: str, size: int = 15):
    family = STOCK_COAST_FAMILIES[profile]
    paths = [r"world\data\d.paa", family.water_path,
             family.land_path, f"{family.namespace}\\farmland.paa"]
    cells = np.full((size, size), 2, dtype=np.int32)
    cells[:, :5] = 1
    return paths, cells


def _corners(path: str) -> tuple[str, str, str, str]:
    stem = path.casefold().split("\\")[-1].split(".")[0]
    if len(stem) != 8:
        return (stem,) * 4
    return tuple(reversed(tuple(stem[i:i+2] for i in (0,2,4,6))))


@pytest.mark.parametrize("profile,namespace,sea,grass,ext,count", FAMILIES)
def test_source_wrps_verify_original_water_ground_and_directional_paths(
    profile, namespace, sea, grass, ext, count,
) -> None:
    family = STOCK_COAST_FAMILIES[profile]
    assert family.namespace == namespace
    assert family.water == sea and family.land == grass
    assert family.extension == ext
    assert len(family.quartets) == count
    for stem in family.quartets:
        assert len(stem) == 8
        assert {stem[i:i+2] for i in (0,2,4,6)} == {sea, grass}
    assert STOCK_GROUND_TEXTURES[profile]["w"] == family.water_path
    assert STOCK_GROUND_TEXTURES[profile]["s"] == family.land_path
    assert STOCK_SURFACE_TEXTURES[profile]["w"] == family.water_path
    for code in ("q", "s", "x", "g", "f"):
        assert STOCK_SURFACE_TEXTURES[profile][code] == family.land_path

    assert stock_water_blend_path(profile, 0, 0, 0, 0) == family.water_path
    assert stock_water_blend_path(profile, 1, 1, 1, 1) == family.land_path
    assert stock_water_blend_path(profile, 1, 0, 1, 0) == (
        f"{namespace}\\{grass}{sea}{grass}{sea}.{ext}"
    )
    assert stock_water_blend_path(profile, 0, 1, 0, 1) == (
        f"{namespace}\\{sea}{grass}{sea}{grass}.{ext}"
    )
    assert stock_water_blend_path(profile, 3, 0, 0, 1) is None
    assert is_stock_water_blend_path(
        profile, f"{namespace}\\{grass}{sea}{grass}{sea}.{ext}"
    )
    assert not is_stock_water_blend_path(profile, family.water_path)
    assert not is_stock_water_blend_path(profile, r"eden\pstnpstn.paa")


@pytest.mark.parametrize("profile", ["malden", "kolgujev"])
def test_underwater_shoreline_uses_exact_blend_only_on_sea_cells(profile):
    paths, cells = _scenario(profile)
    family = STOCK_COAST_FAMILIES[profile]
    expanded, indices = apply_stock_underwater_borders(
        paths, cells.ravel(), profile=profile, width=15,
        underwater_mask=(cells == 1).ravel(),
    )
    out = np.asarray(indices).reshape((15,15))
    assert expanded[int(out[7, 1])] == family.water_path
    assert expanded[int(out[7, 4])] == stock_water_blend_path(profile,1,0,1,0)
    assert expanded[int(out[7, 5])] == family.land_path
    for r,c in np.argwhere(cells != 1):
        assert expanded[int(out[r,c])] == family.land_path
    for row in range(1,13):
        for col in range(1,13):
            me = _corners(expanded[int(out[row,col])])
            east = _corners(expanded[int(out[row,col+1])])
            north = _corners(expanded[int(out[row+1,col])])
            assert (me[0],me[2]) == (east[1],east[3])
            assert (me[0],me[1]) == (north[2],north[3])


@pytest.mark.parametrize("profile", ["malden", "kolgujev"])
def test_curved_coast_blends_without_visible_square_stamps(profile):
    paths, cells = _scenario(profile, size=21)
    cells[:] = 2
    for row in range(21):
        cutoff = 6 + round(2*np.sin(row*np.pi/6))
        cells[row, :cutoff] = 1
    wet = cells == 1
    expanded, indices = apply_stock_underwater_borders(
        paths, cells.ravel(), profile=profile, width=21,
        underwater_mask=wet.ravel(),
    )
    out = np.asarray(indices).reshape(cells.shape)
    assert any(is_stock_water_blend_path(profile, p) for p in expanded)
    for row, col in np.argwhere(~wet):
        assert expanded[int(out[row,col])] == STOCK_COAST_FAMILIES[profile].land_path
    # Adjoining stock quartet edges must have identical geographic corners.
    corners = [[_corners(expanded[int(x)]) for x in row] for row in out]
    for row in range(2,18):
        for col in range(2,18):
            tile = corners[row][col]
            east, north = corners[row][col+1], corners[row+1][col]
            assert (tile[0],tile[2]) == (east[1],east[3])
            assert (tile[0],tile[1]) == (north[2],north[3])


@pytest.mark.parametrize("profile", ["malden", "kolgujev"])
def test_dry_water_classification_and_isolated_sea_ground(profile):
    paths, cells = _scenario(profile)
    wet = cells == 1
    wet[7,4] = False
    expanded, indices = apply_stock_underwater_borders(
        paths, cells.ravel(), profile=profile, width=15,
        underwater_mask=wet.ravel(),
    )
    out = np.asarray(indices).reshape(cells.shape)
    assert expanded[int(out[7,4])] == STOCK_COAST_FAMILIES[profile].land_path
    for row,col in np.argwhere(~wet):
        path = expanded[int(out[row,col])]
        assert path != STOCK_COAST_FAMILIES[profile].water_path
        assert not is_stock_water_blend_path(profile,path)

    isolated = np.full((15,15), 2, dtype=np.int32)
    isolated[7,7] = 1
    expanded, indices = apply_stock_underwater_borders(
        paths, isolated.ravel(), profile=profile, width=15,
        underwater_mask=(isolated==1).ravel(),
    )
    # Original game tiles have no seamless all-grass-corner water centre.
    assert expanded[indices[7*15+7]] == STOCK_COAST_FAMILIES[profile].land_path


@pytest.mark.parametrize("profile", ["malden", "kolgujev"])
def test_protected_ground_and_path_table_limit(profile):
    paths, cells = _scenario(profile)
    wet = cells == 1
    cells[7,4] = 3   # protected farmland on the coast
    no_more, original = apply_stock_underwater_borders(
        paths, cells.ravel(), profile=profile, width=15,
        underwater_mask=wet.ravel(), maximum_texture_slots=len(paths),
    )
    assert no_more == tuple(paths)
    assert original[7*15+4] == 3
    expanded, slots = apply_stock_underwater_borders(
        paths, cells.ravel(), profile=profile, width=15,
        underwater_mask=wet.ravel(),
    )
    assert slots[7*15+4] == 3
    assert all(is_stock_water_blend_path(profile,path)
               for path in expanded[len(paths):])
    assert len(expanded) <= 512

    with pytest.raises(ValueError, match="dimensions"):
        apply_stock_underwater_borders(
            paths, cells.ravel(), profile=profile, width=0,
            underwater_mask=wet.ravel(),
        )
    with pytest.raises(ValueError, match="underwater mask"):
        apply_stock_underwater_borders(
            paths, cells.ravel(), profile=profile, width=15,
            underwater_mask=[True]*14,
        )


def test_cain_missing_both_diagonal_quartets_resolves_shared_corners():
    cain = STOCK_COAST_FAMILIES["kolgujev"]
    assert stock_water_blend_path("kolgujev",1,0,0,1) is None
    assert stock_water_blend_path("kolgujev",0,1,1,0) is None
    vals = np.array([
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,1],
    ], dtype=np.int8)
    before = vals.copy()
    _resolve_missing_diagonals(vals, cain)
    assert np.all(vals[before == 1] == 1)
    ne,nw = vals[1:,1:],vals[1:,:-1]
    se,sw = vals[:-1,1:],vals[:-1,:-1]
    assert not np.any((ne==1)&(nw==0)&(se==0)&(sw==1))
    assert not np.any((ne==0)&(nw==1)&(se==1)&(sw==0))


@pytest.mark.parametrize("profile", ["malden", "kolgujev"])
def test_written_4wvr_includes_only_verified_seabed_border_paths(tmp_path, profile):
    size = 15
    family = STOCK_COAST_FAMILIES[profile]
    material = np.full((size,size), MATERIAL_INDEX["g"],dtype=np.int32)
    material[:,:5] = MATERIAL_INDEX["w"]
    elevations = np.full((size,size), 2.0)
    elevations[:,:5] = -2.0
    spec = SimpleNamespace(
        name="world", ground_texture_profile=profile, cells=size,
        surface_ground_mode="milestone9", surface_pass_enabled=True,
        forest_undergrowth_textures=False, sea_level=0.0,
        dummy_texture_path=r"world\data\d.paa",
    )
    raster = SimpleNamespace(forest=np.zeros(size*size,dtype=bool))
    paths, indices = generator._stock_ground_texture_table(
        spec, material.ravel(), raster, MILESTONE9_MATERIALS,
        elevations=elevations.ravel(),
    )
    out=np.asarray(indices).reshape((size,size))
    assert paths[int(out[7,1])] == family.water_path
    assert paths[int(out[7,4])] == stock_water_blend_path(profile,1,0,1,0)
    assert paths[int(out[7,5])] == family.land_path
    assert not any(is_stock_water_blend_path(profile,paths[int(out[r,c])])
                   for r,c in np.argwhere(elevations>0))
    wrp=tmp_path/f"{profile}.wrp"
    write_rvw4(wrp,size,size,elevations.ravel().tolist(),indices,paths,(),
               height_scale=0.05)
    saved=inspect_rvw4(wrp,height_scale=0.05)
    assert set(x.casefold() for x in saved.texture_paths) == set(
        x.casefold() for x in paths
    )
