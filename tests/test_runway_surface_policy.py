from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen import generator, surface_pass
from cwr_worldgen.cache import cache_key as raw_cache_key
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.runway_model_policy import (
    DESERT_RUNWAY_END_TEXTURE,
    DESERT_RUNWAY_MIDDLE_TEXTURE,
    DESERT_RUNWAY_START_TEXTURE,
    GRASS_RUNWAY_END_TEXTURE,
    GRASS_RUNWAY_MIDDLE_TEXTURE,
    GRASS_RUNWAY_START_TEXTURE,
    _runway_lods,
    runway_model_path,
    runway_texture_triplet,
)
from cwr_worldgen.runway_surface_policy import (
    install_runway_surface_policy,
    runway_overlay_objects,
)


def _runway_dataset(
    projection: BboxProjection,
    start: tuple[float, float] = (80.0, 20.0),
    end: tuple[float, float] = (80.0, 140.0),
) -> OsmDataset:
    runway = OsmLineFeature(
        "way/49810543",
        {"aeroway": "runway", "surface": "asphalt", "width": "30"},
        tuple(projection.to_latlon(point) for point in (start, end)),
    )
    return OsmDataset(
        source_generator="runway-test",
        element_count=1,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
        aeroway_lines=(runway,),
    )


def _spec(profile: str = "everon"):
    return SimpleNamespace(
        name="wg_runway",
        ground_texture_profile=profile,
        cells=16,
        cell_size=10.0,
        world_size=160.0,
        sea_level=0.0,
    )


def test_runway_uses_verified_nogova_middle_and_end_textures() -> None:
    assert runway_texture_triplet("everon") == (
        r"o\runtr_z.paa",
        r"o\runtr_d.paa",
        r"o\runtr_k.paa",
    )
    assert runway_texture_triplet("desert") == (
        r"o\runpi_z.paa",
        r"o\runpi_d.paa",
        r"o\runpi_k.paa",
    )
    assert GRASS_RUNWAY_START_TEXTURE == r"o\runtr_z.paa"
    assert GRASS_RUNWAY_MIDDLE_TEXTURE == r"o\runtr_d.paa"
    assert GRASS_RUNWAY_END_TEXTURE == r"o\runtr_k.paa"
    assert DESERT_RUNWAY_START_TEXTURE == r"o\runpi_z.paa"
    assert DESERT_RUNWAY_MIDDLE_TEXTURE == r"o\runpi_d.paa"
    assert DESERT_RUNWAY_END_TEXTURE == r"o\runpi_k.paa"


def test_grass_runway_uv_rotates_stock_east_west_artwork_onto_model_axis() -> None:
    visual = _runway_lods("grass", 50.0, 250.0)[0]
    assert [face.texture for face in visual.faces] == [
        GRASS_RUNWAY_START_TEXTURE,
        GRASS_RUNWAY_MIDDLE_TEXTURE,
        GRASS_RUNWAY_END_TEXTURE,
    ]
    start_face = visual.faces[0]
    middle_face = visual.faces[1]
    # Moving from local -Z to +Z changes U, not V: a 90-degree UV rotation.
    assert start_face.vertices[0][2:] == (0.0, 0.0)
    assert start_face.vertices[1][2:] == (1.0, 0.0)
    # 250 m runway = 50 m west cap + 150 m repeated middle + 50 m east cap.
    assert middle_face.vertices[1][2:] == (3.0, 0.0)


def test_desert_runway_keeps_stock_south_north_uv_axis() -> None:
    visual = _runway_lods("desert", 50.0, 250.0)[0]
    assert [face.texture for face in visual.faces] == [
        DESERT_RUNWAY_START_TEXTURE,
        DESERT_RUNWAY_MIDDLE_TEXTURE,
        DESERT_RUNWAY_END_TEXTURE,
    ]
    start_face = visual.faces[0]
    assert start_face.vertices[0][2:] == (0.0, 0.0)
    assert start_face.vertices[1][2:] == (0.0, 1.0)


def test_runway_overlay_rotates_with_osm_bearing_and_uses_generated_model() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    spec = _spec("everon")
    objects = runway_overlay_objects(
        dataset,
        projection,
        (10.0,) * (spec.cells * spec.cells),
        spec,
        starting_id=700,
    )
    assert len(objects) == 1
    runway = objects[0]
    assert runway.object_id == 700
    assert runway.heading_degrees == pytest.approx(0.0, abs=1.0e-6)
    assert runway.model_path == runway_model_path("wg_runway", "everon", 120.0)
    assert runway.model_path.endswith(r"\i\runway_grass_w500_l1200.p3d")


def test_desert_end_roles_are_south_to_north_even_for_reversed_osm_way() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(80.0, 140.0), end=(80.0, 20.0))
    spec = _spec("desert")
    runway = runway_overlay_objects(
        dataset,
        projection,
        (10.0,) * (spec.cells * spec.cells),
        spec,
    )[0]
    # The source way is north->south, but runpi_z is the south cap and runpi_k
    # the north cap, so the generated model is deliberately oriented south->north.
    assert runway.heading_degrees == pytest.approx(0.0, abs=1.0e-6)
    assert runway.model_path.endswith(r"\i\runway_desert_w500_l1200.p3d")


def test_runway_generated_asset_contains_all_three_stock_textures(tmp_path) -> None:
    install_runway_surface_policy()
    library = generator.ProceduralInfrastructureLibrary(
        "wg_runway", cache_enabled=False
    )
    model = runway_model_path("wg_runway", "everon", 250.0)
    library.register_model(model)
    catalogue = tmp_path / "infrastructure.json"
    result = library.write_assets(tmp_path, catalogue)
    assert result.generated_variants == 1
    p3d = tmp_path / "i" / "runway_grass_w500_l2500.p3d"
    payload = p3d.read_bytes()
    for texture in runway_texture_triplet("everon"):
        assert texture.encode("ascii") in payload


def test_line_runway_is_not_left_as_unrotatable_wrp_terrain_texture() -> None:
    install_runway_surface_policy()
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    mask = surface_pass._aeroway_mask(dataset, projection, 16)
    assert not mask.any()


def test_runway_policy_invalidates_sideways_terrain_cache() -> None:
    install_runway_surface_policy()
    payload = {"world": "runway-test"}
    assert generator.cache_key(
        "surface-pipeline-v11-vectorized-material-pass",
        payload,
    ) == raw_cache_key(
        "surface-pipeline-v15-oriented-runway-models",
        payload,
    )
