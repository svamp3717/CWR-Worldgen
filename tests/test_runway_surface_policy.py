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
    RUNWAY_TEXTURE_TILE_METRES,
    _runway_lods,
    runway_model_path,
    runway_texture_triplet,
)
from cwr_worldgen.runway_surface_policy import (
    _runway_tile_plan,
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


def test_grass_runway_tile_rotates_stock_east_west_artwork_onto_model_axis() -> None:
    visual = _runway_lods("grass", "d")[0]
    assert [face.texture for face in visual.faces] == [GRASS_RUNWAY_MIDDLE_TEXTURE]
    face = visual.faces[0]
    # Moving from local -Z to +Z changes U, not V: the required 90-degree UV turn.
    assert face.vertices[0][2:] == (0.0, 0.0)
    assert face.vertices[1][2:] == (1.0, 0.0)


def test_desert_runway_tile_keeps_stock_south_north_uv_axis() -> None:
    visual = _runway_lods("desert", "d")[0]
    assert [face.texture for face in visual.faces] == [DESERT_RUNWAY_MIDDLE_TEXTURE]
    face = visual.faces[0]
    assert face.vertices[0][2:] == (0.0, 0.0)
    assert face.vertices[1][2:] == (0.0, 1.0)


def test_runway_tile_plan_is_gapless_and_uses_z_d_k_roles() -> None:
    # 120 m cannot be represented by exact 50 m stock tiles. Cover it with three
    # adjacent 50 m tiles and split the 30 m excess equally over the two ends.
    plan = _runway_tile_plan(((80.0, 20.0), (80.0, 140.0)), "everon")
    assert [role for role, _start, _end in plan] == ["z", "d", "k"]
    assert len(plan) == 3
    for (_role_a, _start_a, end_a), (_role_b, start_b, _end_b) in zip(plan, plan[1:]):
        assert end_a == pytest.approx(start_b, abs=1.0e-7)
    assert plan[0][1] == pytest.approx((80.0, 5.0), abs=1.0e-7)
    assert plan[-1][2] == pytest.approx((80.0, 155.0), abs=1.0e-7)


def test_runway_overlay_rotates_each_stock_tile_with_osm_bearing() -> None:
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
    assert len(objects) == 3
    assert [obj.object_id for obj in objects] == [700, 701, 702]
    assert [obj.heading_degrees for obj in objects] == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-6)
    assert [obj.model_path for obj in objects] == [
        runway_model_path("wg_runway", "everon", "z"),
        runway_model_path("wg_runway", "everon", "d"),
        runway_model_path("wg_runway", "everon", "k"),
    ]
    assert [obj.z for obj in objects] == pytest.approx([30.0, 80.0, 130.0], abs=1.0e-6)
    assert RUNWAY_TEXTURE_TILE_METRES == 50.0


def test_grass_end_roles_are_west_to_east_even_for_reversed_osm_way() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(140.0, 80.0), end=(20.0, 80.0))
    spec = _spec("everon")
    objects = runway_overlay_objects(
        dataset,
        projection,
        (10.0,) * (spec.cells * spec.cells),
        spec,
    )
    assert [obj.model_path.rsplit("\\", 1)[-1] for obj in objects] == [
        "runway_grass_z.p3d",
        "runway_grass_d.p3d",
        "runway_grass_k.p3d",
    ]
    assert [obj.heading_degrees for obj in objects] == pytest.approx([90.0, 90.0, 90.0], abs=1.0e-6)
    assert [obj.x for obj in objects] == pytest.approx([30.0, 80.0, 130.0], abs=1.0e-6)


def test_desert_end_roles_are_south_to_north_even_for_reversed_osm_way() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(80.0, 140.0), end=(80.0, 20.0))
    spec = _spec("desert")
    objects = runway_overlay_objects(
        dataset,
        projection,
        (10.0,) * (spec.cells * spec.cells),
        spec,
    )
    assert [obj.model_path.rsplit("\\", 1)[-1] for obj in objects] == [
        "runway_desert_z.p3d",
        "runway_desert_d.p3d",
        "runway_desert_k.p3d",
    ]
    assert [obj.heading_degrees for obj in objects] == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-6)
    assert [obj.z for obj in objects] == pytest.approx([30.0, 80.0, 130.0], abs=1.0e-6)


def test_short_runway_uses_one_middle_tile_instead_of_overlapping_end_caps() -> None:
    plan = _runway_tile_plan(((20.0, 50.0), (50.0, 50.0)), "everon")
    assert len(plan) == 1
    assert plan[0][0] == "d"
    assert plan[0][1] == pytest.approx((10.0, 50.0), abs=1.0e-7)
    assert plan[0][2] == pytest.approx((60.0, 50.0), abs=1.0e-7)


def test_runway_generated_assets_reference_each_stock_role_texture(tmp_path) -> None:
    install_runway_surface_policy()
    library = generator.ProceduralInfrastructureLibrary(
        "wg_runway", cache_enabled=False
    )
    for role in ("z", "d", "k"):
        library.register_model(runway_model_path("wg_runway", "everon", role))
    catalogue = tmp_path / "infrastructure.json"
    result = library.write_assets(tmp_path, catalogue)
    assert result.generated_variants == 3

    expected = {
        "z": GRASS_RUNWAY_START_TEXTURE,
        "d": GRASS_RUNWAY_MIDDLE_TEXTURE,
        "k": GRASS_RUNWAY_END_TEXTURE,
    }
    for role, texture in expected.items():
        p3d = tmp_path / "i" / f"runway_grass_{role}.p3d"
        assert p3d.is_file()
        payload = p3d.read_bytes()
        assert texture.encode("ascii") in payload


def test_line_runway_is_not_left_as_unrotatable_wrp_terrain_texture() -> None:
    install_runway_surface_policy()
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    mask = surface_pass._aeroway_mask(dataset, projection, 16)
    assert not mask.any()


def test_runway_policy_invalidates_sideways_and_single_model_surface_cache() -> None:
    install_runway_surface_policy()
    payload = {"world": "runway-test"}
    assert generator.cache_key(
        "surface-pipeline-v11-vectorized-material-pass",
        payload,
    ) == raw_cache_key(
        "surface-pipeline-v16-oriented-runway-tiles",
        payload,
    )
