from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen.terrain_solver import _rvw4_storage_datum_offset
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
