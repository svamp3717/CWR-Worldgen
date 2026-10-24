"""Grey parking surface is synthetic gravel, not painted stock road pixels."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen import parking_visual_detail_policy as detail
from cwr_worldgen.stock_gravel_parking_texture import (
    STOCK_GRAVEL_PROCEDURAL_RECIPE,
    stock_gravel_colour_material,
)


def _parking(surface="gravel", left=4.0, right=21.0, row0=3.0, row1=20.0):
    return parking._ParkingGeometry(
        osm_key="way/test",
        outer=((left, row0), (right, row0), (right, row1), (left, row1)),
        holes=(),
        centre_x=(left + right) * 0.5,
        centre_z=(row0 + row1) * 0.5,
        half_length=(right - left) * 0.5,
        half_width=(row1 - row0) * 0.5,
        ux=1.0,
        uz=0.0,
        px=0.0,
        pz=1.0,
        surface=surface,
    )


def _spec():
    return SimpleNamespace(cells=2, cell_size=25.0)


def test_parking_recipe_is_procedural_and_uses_stock_gravel_colour() -> None:
    assert parking.PARKING_GRAVEL_RECIPE == STOCK_GRAVEL_PROCEDURAL_RECIPE
    assert parking.PARKING_TEXTURE_CACHE_SCHEMA >= 6
    assert detail.DETAIL_CACHE_SCHEMA >= 4


def test_gravel_parking_preserves_background_and_neutral_tone() -> None:
    base = Image.new("RGB", (64, 64), (12, 40, 6))
    rendered = parking._render_parking_cell(
        cell_index=0, geometries=(_parking(),), spec=_spec(), base=base,
    )
    values = np.asarray(rendered)
    assert tuple(values[0, 0]) == (12, 40, 6)
    assert values[32, 32, 0] == values[32, 32, 1] == values[32, 32, 2]
    assert values[10, 10, 0] == values[10, 10, 1] == values[10, 10, 2]


def test_paved_lots_still_use_previous_asphalt_colours() -> None:
    base = Image.new("RGB", (64, 64), (13, 28, 8))
    rendered = parking._render_parking_cell(
        cell_index=0, geometries=(_parking(surface="paved"),),
        spec=_spec(), base=base,
    )
    values = np.asarray(rendered)
    assert np.array_equal(values[0, 0], (13, 28, 8))
    assert 45.0 < float(values[32, 32].mean()) < 90.0


def test_parked_vehicle_wear_darkens_without_brown_colour() -> None:
    sample = Image.new("RGB", (64, 64), (80, 80, 80))
    actual = detail.apply_parking_visual_detail(
        parking, sample, cell_index=0,
        geometries=(_parking(),), spec=_spec(),
    )
    pixels = np.asarray(actual)
    assert np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
    assert np.array_equal(pixels[:, :, 1], pixels[:, :, 2])
    assert np.array_equal(pixels[0, 0], (80, 80, 80))
    assert int(pixels.min()) < 75


def test_gravel_is_stable_between_cell_slices() -> None:
    x = np.array([[6.125, 26.125, 31.125], [7.125, 27.125, 32.125]])
    z = np.array([[3.75, 3.75, 3.75], [13.75, 13.75, 13.75]])
    full = stock_gravel_colour_material(x, z)
    first = stock_gravel_colour_material(x[:, :1], z[:, :1])
    remainder = stock_gravel_colour_material(x[:, 1:], z[:, 1:])
    assert np.array_equal(full[:, :1], first)
    assert np.array_equal(full[:, 1:], remainder)
