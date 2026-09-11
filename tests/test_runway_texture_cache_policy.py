from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

from cwr_worldgen import surface_pass
from cwr_worldgen.build_cache_policy import BUILD_CACHE_DIRNAME, build_cache_dir
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
import cwr_worldgen.runway_surface_policy as runway
from cwr_worldgen.runway_texture_cache_policy import _runway_cache_dir
from cwr_worldgen.shared_cache_policy import SHARED_CACHE_DIRNAME
from cwr_worldgen.shared_runway_cache_policy import RUNWAY_TEXTURE_CACHE_DIRNAME


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


def test_runway_cache_uses_cross_world_shared_cache_root(tmp_path) -> None:
    world_a = tmp_path / "world-a"
    world_b = tmp_path / "world-b"
    spec_a = _spec(build_cache_dir(world_a))
    spec_b = _spec(build_cache_dir(world_b))

    runway_cache_a = _runway_cache_dir(tmp_path / "source-a", spec_a)
    runway_cache_b = _runway_cache_dir(tmp_path / "source-b", spec_b)
    expected = (
        tmp_path.resolve()
        / SHARED_CACHE_DIRNAME
        / RUNWAY_TEXTURE_CACHE_DIRNAME
    )

    assert runway_cache_a == expected
    assert runway_cache_b == expected
    assert BUILD_CACHE_DIRNAME not in runway_cache_a.parts
    assert ".cwr-worldgen-runway-cache" not in runway_cache_a.parts


def test_wide_runway_keeps_per_cell_alignment_and_reuses_cached_paas(
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

    touched = runway.runway_texture_cell_indices(dataset, projection, spec)
    assert touched
    # A 30 m-wide runway on 10 m cells spans several columns. The old triplet
    # optimization reused one local image across those columns and visibly
    # created parallel runways. At least one row must retain distinct local
    # texture slots across its touched cells.
    rows: dict[int, list[int]] = defaultdict(list)
    for cell_index in touched:
        row, _column = divmod(cell_index, spec.cells)
        rows[row].append(cell_index)
    multi_cell_rows = [values for values in rows.values() if len(values) >= 2]
    assert multi_cell_rows
    assert any(
        len({revised_indices[index] for index in values}) >= 2
        for values in multi_cell_rows
    )

    assert len(generated) > 3
    assert len(revised_paths) == len(_base_paths()) + len(generated)
    report = json.loads((source_dir / "runway-textures.json").read_text(encoding="utf-8"))
    assert report["strategy"] == "per-cell-content-addressed-cache"
    assert report["runway_count"] == 1
    assert report["runway_cells"] == len(touched)
    assert report["cache_hits"] == 0
    assert report["cache_misses"] == len(touched)
    expected_cache = (
        cache_dir.resolve()
        / SHARED_CACHE_DIRNAME
        / RUNWAY_TEXTURE_CACHE_DIRNAME
    )
    assert expected_cache.is_dir()
    assert report["cache_directory"] == str(expected_cache)

    for path in generated:
        (source_dir / path.rsplit("\\", 1)[-1]).unlink()

    def fail_render(*args, **kwargs):
        raise AssertionError("cached runway cells must not be repainted")

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
    assert second_report["cache_hits"] == len(touched)
    assert second_report["cache_misses"] == 0
    assert all(
        (source_dir / path.rsplit("\\", 1)[-1]).is_file()
        for path in second_generated
    )
