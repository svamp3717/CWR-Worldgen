from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmRaster
from cwr_worldgen.terrain_solver import _rvw4_storage_datum_offset, solve_terrain_constraints
from cwr_worldgen.wrp import quantize_height


def _spec():
    return SimpleNamespace(
        height_scale=0.05,
        maximum_grade_adjustment=12.0,
        water_depth=5.0,
        sea_level=0.0,
    )


def _raster(water):
    return SimpleNamespace(water=tuple(bool(value) for value in water))


def test_high_altitude_plateau_is_rebased_below_rvw4_ceiling() -> None:
    elevations = (2500.0, 2700.0, 3100.0, 3353.2277124847787)
    offset = _rvw4_storage_datum_offset(
        elevations, _raster((False,) * len(elevations)), _spec()
    )

    shifted = tuple(value - offset for value in elevations)
    assert offset > 1700.0
    assert max(shifted) <= 32767 * 0.05 - 63.9
    assert min(shifted) > 32.0
    assert tuple(quantize_height(value, 0.05) for value in shifted)


def test_ordinary_altitudes_keep_original_datum() -> None:
    elevations = (100.0, 400.0, 900.0, 1500.0)

    assert (
        _rvw4_storage_datum_offset(
            elevations, _raster((False,) * len(elevations)), _spec()
        )
        == 0.0
    )


def test_vertical_relief_that_cannot_preserve_dry_land_fails_clearly() -> None:
    elevations = (100.0, 900.0, 2000.0, 3353.2277124847787)

    with pytest.raises(ValueError, match="smaller/less vertically extreme area"):
        _rvw4_storage_datum_offset(
            elevations, _raster((False,) * len(elevations)), _spec()
        )


def test_deep_mapped_water_does_not_block_dry_plateau_rebase() -> None:
    elevations = (-2500.0, 2500.0, 3000.0, 3353.2277124847787)
    offset = _rvw4_storage_datum_offset(
        elevations, _raster((True, False, False, False)), _spec()
    )

    assert offset > 1700.0
    assert 2500.0 - offset > 32.0


def test_milestone9_solver_rebases_high_plateau_before_rvw4_quantization() -> None:
    cells = 4
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    dataset = OsmDataset(
        source_generator="high-altitude-rebase",
        element_count=0,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
    )
    raster = OsmRaster(
        cells=cells,
        water=(False,) * (cells * cells),
        forest=(False,) * (cells * cells),
        farmland=(False,) * (cells * cells),
        urban=(False,) * (cells * cells),
        roads=(False,) * (cells * cells),
        buildings=(False,) * (cells * cells),
        high_resolution=cells,
        coastline_seed_count=0,
    )
    spec = _Milestone9PlayabilitySpec(
        name="cwr_high_altitude",
        heightmap_path=Path("unused.png"),
        bbox=(0.0, 0.0, 1.0, 1.0),
        cells=cells,
        cell_size=25.0,
        solver_iterations=1,
        world_edge_blend_cells=0,
        strict_assets=False,
    )
    elevations = tuple(
        2500.0 + (3353.2277124847787 - 2500.0) * index / (cells * cells - 1)
        for index in range(cells * cells)
    )

    report = solve_terrain_constraints(
        elevations,
        dataset,
        projection,
        raster,
        spec,
    )

    assert report.rvw4_storage_rebase
    assert report.vertical_datum_offset > 1700.0
    assert max(report.elevations) < 32767 * spec.height_scale
    assert tuple(
        quantize_height(value, spec.height_scale)
        for value in report.elevations
    )
