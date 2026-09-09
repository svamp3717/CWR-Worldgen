from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import generator, surface_pass
from cwr_worldgen.cache import cache_key as raw_cache_key
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.runway_surface_policy import (
    DESERT_RUNWAY_TEXTURE,
    GRASS_RUNWAY_TEXTURE,
    RUNWAY_MATERIAL_CODE,
    apply_runway_material_indices,
    install_runway_surface_policy,
)


def _runway_dataset(projection: BboxProjection) -> OsmDataset:
    runway = OsmLineFeature(
        "way/49810543",
        {"aeroway": "runway", "surface": "asphalt", "width": "30"},
        tuple(
            projection.to_latlon(point)
            for point in ((20.0, 80.0), (140.0, 80.0))
        ),
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


def _surface_spec(profile: str):
    return SimpleNamespace(
        name="wg_runway",
        ground_texture_profile=profile,
        surface_pass_enabled=True,
        surface_ground_mode="milestone9",
    )


def test_runway_material_uses_requested_stock_texture_per_profile() -> None:
    install_runway_surface_policy()
    index = surface_pass.MATERIAL_INDEX[RUNWAY_MATERIAL_CODE]

    for profile in ("generated", "everon", "nogova", "malden"):
        paths = surface_pass.surface_texture_wire_paths("wg_runway", profile)
        assert paths[index] == GRASS_RUNWAY_TEXTURE
        assert GRASS_RUNWAY_TEXTURE in surface_pass.external_surface_texture_paths(profile)
        assert generator._ground_texture_paths(_surface_spec(profile))[index] == GRASS_RUNWAY_TEXTURE

    desert_paths = surface_pass.surface_texture_wire_paths("wg_runway", "desert")
    assert desert_paths[index] == DESERT_RUNWAY_TEXTURE
    assert DESERT_RUNWAY_TEXTURE in surface_pass.external_surface_texture_paths("desert")
    assert generator._ground_texture_paths(_surface_spec("desert"))[index] == DESERT_RUNWAY_TEXTURE


def test_runway_overlay_replaces_generic_paved_cells() -> None:
    install_runway_surface_policy()
    cells = 16
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    raster = SimpleNamespace(
        water=(False,) * (cells * cells),
        buildings=(False,) * (cells * cells),
    )
    spec = SimpleNamespace(cells=cells, sea_level=0.0)
    paved = surface_pass.MATERIAL_INDEX["p"]
    runway = surface_pass.MATERIAL_INDEX[RUNWAY_MATERIAL_CODE]

    result = apply_runway_material_indices(
        (paved,) * (cells * cells),
        dataset,
        projection,
        raster,
        (10.0,) * (cells * cells),
        spec,
    )

    assert result.count(runway) > 0
    assert result.count(paved) > 0


def test_runway_overlay_does_not_paint_renderable_water() -> None:
    install_runway_surface_policy()
    cells = 16
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    raster = SimpleNamespace(
        water=(True,) * (cells * cells),
        buildings=(False,) * (cells * cells),
    )
    spec = SimpleNamespace(cells=cells, sea_level=0.0)
    paved = surface_pass.MATERIAL_INDEX["p"]

    result = apply_runway_material_indices(
        (paved,) * (cells * cells),
        dataset,
        projection,
        raster,
        (-1.0,) * (cells * cells),
        spec,
    )

    assert result == (paved,) * (cells * cells)


def test_runway_policy_invalidates_old_surface_pipeline_cache() -> None:
    install_runway_surface_policy()
    payload = {"world": "runway-test"}
    assert generator.cache_key(
        "surface-pipeline-v11-vectorized-material-pass",
        payload,
    ) == raw_cache_key(
        "surface-pipeline-v12-stock-runway-texture",
        payload,
    )
