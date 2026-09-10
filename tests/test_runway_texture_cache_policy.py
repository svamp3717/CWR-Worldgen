from __future__ import annotations

import json
from types import SimpleNamespace

from cwr_worldgen import surface_pass
from cwr_worldgen.build_cache_policy import BUILD_CACHE_DIRNAME, build_cache_dir
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
import cwr_worldgen.runway_surface_policy as runway
from cwr_worldgen.runway_texture_cache_policy import (
    RUNWAY_GROUND_CACHE_DIRNAME,
    _runway_cache_dir,
)


def _dataset(projection: BboxProjection) -> OsmDataset:
    feature = OsmLineFeature(
        "way/cache-runway",
        {"aeroway": "runway", "surface": "asphalt", "width": "30"},
        tuple(
            projection.to_latlon(point)
            for point in ((85.0, 20.0), (85.0, 140.0))
        ),
    )
    return OsmDataset(
        source_generator="runway-cache-test",
        element_count=1,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
        aeroway_lines=(feature,),
    )


def _spec(cache_dir):
    return SimpleNamespace(
        name="wg_runway",
        ground_texture_profile="nogova",
        surface_pass_enabled=True,
        surface_ground_mode="milestone9",
        deterministic_seed="runway-cache-tests",
        cells=16,
        cell_size=10.0,
        world_size=160.0,
        sea_level=0.0,
        cache_dir=cache_dir,
        cache_enabled=True,
        cache_refresh=False,
        asset_roots=(),
        deploy_mod_dir=None,
    )


def _base_paths() -> tuple[str, ...]:
    return (
        r"wg_runway\data\d.paa",
        *surface_pass.surface_texture_wire_paths("wg_runway", "nogova"),
    )


def test_runway_cache_survives_disposable_build_cache_cleanup(tmp_path) -> None:
    build_dir = tmp_path / "build"
    routed_cache = build_cache_dir(build_dir)
    spec = _spec(routed_cache)

    runway_cache = _runway_cache_dir(tmp_path / "world", spec)

    assert runway_cache == build_dir.resolve() / RUNWAY_GROUND_CACHE_DIRNAME
    assert BUILD_CACHE_DIRNAME not in runway_cache.parts


def test_one_runway_uses_exactly_start_middle_end_and_reuses_cached_paas(
    tmp_path, monkeypatch
) -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _dataset(projection)
    cache_dir = tmp_path / "persistent-cache"
    source_dir = tmp_path / "world"
    source_dir.mkdir()
    spec = _spec(cache_dir)
    grass_index = surface_pass.MATERIAL_INDEX["g"] + 1
    base_indices = (grass_index,) * (spec.cells * spec.cells)

    revised_indices, revised_paths, generated = runway.apply_generated_runway_texture_table(
        source_dir,
        dataset,
        projection,
        spec,
        base_indices,
        _base_paths(),
    )

    assert len(generated) == 3
    assert [path.rsplit("\\", 1)[-1][-5] for path in generated] == ["z", "d", "k"]
    touched = runway.runway_texture_cell_indices(dataset, projection, spec)
    assert len({revised_indices[index] for index in touched}) == 3
    assert len(revised_paths) == len(_base_paths()) + 3

    report = json.loads((source_dir / "runway-textures.json").read_text(encoding="utf-8"))
    assert report["strategy"] == "three-textures-per-runway"
    assert report["runway_count"] == 1
    assert report["cache_hits"] == 0
    assert report["cache_misses"] == 3
    assert (cache_dir / RUNWAY_GROUND_CACHE_DIRNAME).is_dir()

    for path in generated:
        (source_dir / path.rsplit("\\", 1)[-1]).unlink()

    def fail_render(*args, **kwargs):
        raise AssertionError("cached runway textures must not be repainted")

    monkeypatch.setattr(runway, "_render_runway_cell", fail_render)
    second_indices, second_paths, second_generated = runway.apply_generated_runway_texture_table(
        source_dir,
        dataset,
        projection,
        spec,
        base_indices,
        _base_paths(),
    )

    assert second_generated == generated
    assert second_indices == revised_indices
    assert second_paths == revised_paths
    second_report = json.loads(
        (source_dir / "runway-textures.json").read_text(encoding="utf-8")
    )
    assert second_report["cache_hits"] == 3
    assert second_report["cache_misses"] == 0
    assert all(
        (source_dir / path.rsplit("\\", 1)[-1]).is_file()
        for path in second_generated
    )
