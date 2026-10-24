"""Procedural parking gravel matches stock asf25 COLOUR, never source texture."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen.stock_gravel_parking_texture import (
    STOCK_GRAVEL_PROCEDURAL_RECIPE,
    STOCK_GRAVEL_REFERENCE_GREY_RGB,
    STOCK_GRAVEL_REFERENCE_MODEL,
    _hashed_grain,
    stock_gravel_colour_material,
)


def _grid(*, x0=0.0, z0=0.0, nx=128, nz=128, spacing=0.25):
    xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * spacing
    zs = z0 + (np.arange(nz, dtype=np.float64) + 0.5) * spacing
    return np.meshgrid(xs, zs)


def _lot(*, surface="gravel"):
    return parking._ParkingGeometry(
        osm_key="way/parking",
        outer=((4.0, 4.0), (21.0, 4.0), (21.0, 21.0), (4.0, 21.0)),
        holes=(),
        centre_x=12.5, centre_z=12.5,
        half_length=8.5, half_width=8.5,
        ux=1.0, uz=0.0, px=0.0, pz=1.0, surface=surface,
    )


def test_stock_asf_is_reference_for_colour_only() -> None:
    from cwr_worldgen.model import PlayabilitySpec

    assert STOCK_GRAVEL_REFERENCE_MODEL == r"o\road\asf25.p3d"
    assert PlayabilitySpec.__dataclass_fields__["gravel_road_model"].default == STOCK_GRAVEL_REFERENCE_MODEL
    assert STOCK_GRAVEL_REFERENCE_GREY_RGB == (74, 74, 74)
    assert "procedural" in STOCK_GRAVEL_PROCEDURAL_RECIPE
    assert parking.PARKING_TEXTURE_CACHE_SCHEMA >= 6


def test_procedural_grey_matched_to_asf25_without_copying_road_pixels() -> None:
    x, z = _grid(nx=256, nz=256, spacing=0.23)
    image = stock_gravel_colour_material(x, z)
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.float32
    # Genuine stock asf25 centre sampled at RGB approximately (74,74,74).
    # Preserve a similar overall dark grey, but create independent gravel grain.
    assert abs(float(image.mean()) - 74.0) < 6.0
    assert 4.0 < float(image.std()) < 22.0
    assert np.array_equal(image[:, :, 0], image[:, :, 1])
    assert np.array_equal(image[:, :, 0], image[:, :, 2])
    assert np.any(image[:, :, 0] < 65.0)
    assert np.any(image[:, :, 0] > 85.0)


def test_world_position_not_cell_index_controls_gravel_grain() -> None:
    x, z = _grid(x0=-11.3, z0=32.75)
    once = stock_gravel_colour_material(x, z)
    again = stock_gravel_colour_material(x.copy(), z.copy())
    assert np.array_equal(once, again)
    right = stock_gravel_colour_material(x[:, 64:], z[:, 64:])
    assert np.array_equal(once[:, 64:], right)
    assert not np.array_equal(once, stock_gravel_colour_material(x + 0.28, z))


def test_material_rejects_different_grid_shapes() -> None:
    with pytest.raises(ValueError, match="same-shaped"):
        stock_gravel_colour_material(np.zeros((2, 3)), np.zeros((3, 2)))


def test_hash_grain_works_at_negative_world_positions() -> None:
    x, z = _grid(x0=-240.0, z0=-320.0, nx=5, nz=6)
    values = _hashed_grain(x, z, 2.0, 123)
    assert values.shape == (6, 5)
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_stock_source_not_required_or_used_for_parking(monkeypatch) -> None:
    from cwr_worldgen import runway_exact_background_policy as exact
    from cwr_worldgen import procedural_infrastructure as roads

    def forbidden(*_a, **_kw):
        raise AssertionError("parking must not request original or generated road image")

    monkeypatch.setattr(exact, "_read_external_asset_cached", forbidden)
    monkeypatch.setattr(roads, "create_gravel_junction_texture_image", forbidden)

    base = Image.new("RGB", (128, 128), (29, 100, 20))
    spec = SimpleNamespace(cells=1, cell_size=25.0, asset_roots=("/nonexistent",))
    image = parking._render_parking_cell(
        cell_index=0, geometries=(_lot(),), spec=spec, base=base,
    )
    pixels = np.asarray(image)
    assert tuple(pixels[0, 0]) == (29, 100, 20)
    inside = pixels[55, 55]
    assert inside[0] == inside[1] == inside[2]
    assert int(inside[0]) > 30
    assert int(inside[0]) < 115


def test_paved_lot_does_not_invoke_gravel_material(monkeypatch) -> None:
    def forbidden(*_a, **_kw):
        raise AssertionError("asphalt parking must not load gravel material")

    monkeypatch.setattr(parking, "stock_gravel_colour_material", forbidden)
    rendered = parking._render_parking_cell(
        cell_index=0,
        geometries=(_lot(surface="paved"),),
        spec=SimpleNamespace(cells=1, cell_size=25.0),
        base=Image.new("RGB", (128, 128), (29, 100, 20)),
    )
    result = np.asarray(rendered)
    assert np.array_equal(result[0, 0], (29, 100, 20))
    assert not np.array_equal(result[55, 55], (29, 100, 20))
