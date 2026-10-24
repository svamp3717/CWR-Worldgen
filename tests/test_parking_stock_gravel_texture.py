"""Non-paved parking must use the same neutral aggregate as generated gravel roads."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from cwr_worldgen import parking_surface_policy as parking
from cwr_worldgen import parking_visual_detail_policy as detail
from cwr_worldgen.procedural_infrastructure import (
    GENERATED_GRAVEL_TEXTURE_REPEAT_METRES,
    create_gravel_junction_texture_image,
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
        ux=1.0, uz=0.0, px=0.0, pz=1.0,
        surface=surface,
    )


def _spec():
    return SimpleNamespace(cells=2, cell_size=25.0)


def test_gravel_parking_photo_is_exact_road_junction_artwork() -> None:
    expected = np.asarray(
        create_gravel_junction_texture_image(64).convert("RGB")
    )
    actual = parking._gravel_road_tile_pixels(64)
    assert np.array_equal(actual, expected)
    assert not actual.flags.writeable
    assert parking.PARKING_TEXTURE_CACHE_SCHEMA >= 4
    assert detail.DETAIL_CACHE_SCHEMA >= 4


def test_gravel_material_repeats_at_the_same_world_scale_as_road_uvs() -> None:
    repeat = GENERATED_GRAVEL_TEXTURE_REPEAT_METRES
    xs = np.array([[0.7, 0.7 + repeat, 30.7, 30.7 + repeat]])
    zs = np.array([[0.8, 0.8, 10.8, 10.8]])
    material = parking._gravel_road_material(xs, zs, pixel_scale=0.25)
    assert np.allclose(material[0, 0], material[0, 1], atol=1e-3)
    assert np.allclose(material[0, 2], material[0, 3], atol=1e-3)
    assert np.std(material) > 0


def test_unpaved_lot_is_not_tinted_brown_and_leaves_terrain_outside_untouched(
    monkeypatch,
) -> None:
    sentinel = np.asarray((78.0, 83.0, 88.0), dtype=np.float32)
    monkeypatch.setattr(
        parking, "_gravel_road_material",
        lambda x, z, **_kwargs: np.broadcast_to(
            sentinel, (*x.shape, 3),
        ),
    )
    base = Image.new("RGB", (64, 64), (12, 40, 6))
    rendered = parking._render_parking_cell(
        cell_index=0, geometries=(_parking(),), spec=_spec(), base=base,
    )
    values = np.asarray(rendered)
    assert tuple(values[0, 0]) == (12, 40, 6)
    # The installed parking-detail hook darkens wheel tracks but must not
    # recolour cool grey road aggregate into the old brown parking filler.
    assert tuple(values[32, 32]) != (12, 40, 6)
    assert values[32, 32, 2] > values[32, 32, 1] > values[32, 32, 0]
    assert values[10, 10, 2] > values[10, 10, 1] > values[10, 10, 0]


def test_paved_lots_do_not_consume_gravel_artwork_or_change_paved_colours(
    monkeypatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise AssertionError("paved renderer must not load gravel photo")
    monkeypatch.setattr(parking, "_gravel_road_material", unavailable)
    base = Image.new("RGB", (64, 64), (13, 28, 8))
    rendered = parking._render_parking_cell(
        cell_index=0, geometries=(_parking(surface="paved"),),
        spec=_spec(), base=base,
    )
    pixels = np.asarray(rendered)
    assert np.array_equal(pixels[0, 0], (13, 28, 8))
    assert float(pixels[32, 32].mean()) > 45.0
    assert float(pixels[32, 32].mean()) < 90.0


def test_unsealed_parking_detail_darkens_gravel_without_repainting_it_brown():
    sample = Image.new("RGB", (64, 64), (80, 80, 80))
    actual = detail.apply_parking_visual_detail(
        parking, sample, cell_index=0,
        geometries=(_parking(),), spec=_spec(),
    )
    pixels = np.asarray(actual)
    assert np.all(pixels[:, :, 0] == pixels[:, :, 1])
    assert np.all(pixels[:, :, 1] == pixels[:, :, 2])
    assert np.array_equal(pixels[0, 0], (80, 80, 80))
    assert int(pixels.min()) < 75


def test_gravel_road_material_anchored_to_world_not_cell_origin():
    a_x = np.array([[6.125, 26.125]])
    a_z = np.array([[3.75, 33.75]])
    direct = parking._gravel_road_material(a_x, a_z, pixel_scale=0.35)
    tiled = parking._gravel_road_material(
        a_x + 25.0 - 25.0, a_z + 50.0 - 50.0, pixel_scale=0.35,
    )
    assert np.array_equal(direct, tiled)
