from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import runway_surface_policy as runway
from cwr_worldgen import surface_pass
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.runway_texture_cache_policy import (
    RUNWAY_GROUND_CACHE_DIRNAME,
    install_runway_texture_cache_policy,
)


def _dataset(projection: BboxProjection) -> OsmDataset:
    runway_line = OsmLineFeature(
        "way/cache-test",
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
        aeroway_lines=(runway_line,),
    )


def _spec(cache_dir: Path):
    return SimpleNamespace(
        name="wg_runway",
        ground_texture_profile="everon",
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
    )


def _base_paths() -> tuple[str, ...]:
    return (
        r"wg_runway\data\d.paa",
        *surface_pass.surface_texture_wire_paths("wg_runway", "everon"),
    )


def test_runway_uses_three_cached_role_textures_and_restores_without_repaint(
    tmp_path, monkeypatch
) -> None:
    install_runway_texture_cache_policy()
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _dataset(projection)
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "world"
    source_dir.mkdir()
    spec = _spec(cache_dir)
    grass_index = surface_pass.MATERIAL_INDEX["g"] + 1
    base_indices = (grass_index,) * (spec.cells * spec.cells)
    base_paths = _base_paths()
    touched = runway.runway_texture_cell_indices(dataset, projection, spec)

    first_indices, first_paths, generated = runway.apply_generated_runway_texture_table(
        source_dir,
        dataset,
        projection,
        spec,
        base_indices,
        base_paths,
    )

    assert len(generated) == 3
    assert [path.rsplit("\\", 1)[-1][-5] for path in generated] == ["z", "d", "k"]
    assert len(first_paths) == len(base_paths) + 3
    assert len({first_indices[index] for index in touched}) == 3
    assert all(first_indices[index] >= len(base_paths) for index in touched)

    first_report = json.loads((source_dir / "runway-textures.json").read_text())
    assert first_report["strategy"] == "three-textures-per-runway"
    assert first_report["runway_count"] == 1
    assert first_report["generated_runway_textures"] == 3
    assert first_report["cache_hits"] == 0
    assert first_report["cache_misses"] == 3
    assert (cache_dir / RUNWAY_GROUND_CACHE_DIRNAME).is_dir()

    first_bytes = {
        path.rsplit("\\", 1)[-1]: (source_dir / path.rsplit("\\", 1)[-1]).read_bytes()
        for path in generated
    }
    for filename in first_bytes:
        (source_dir / filename).unlink()

    def repaint_forbidden(*_args, **_kwargs):
        raise AssertionError("cached runway texture should not be repainted")

    monkeypatch.setattr(runway, "_render_runway_cell", repaint_forbidden)
    second_indices, second_paths, second_generated = runway.apply_generated_runway_texture_table(
        source_dir,
        dataset,
        projection,
        spec,
        base_indices,
        base_paths,
    )

    assert second_indices == first_indices
    assert second_paths == first_paths
    assert second_generated == generated
    for filename, expected in first_bytes.items():
        assert (source_dir / filename).read_bytes() == expected

    second_report = json.loads((source_dir / "runway-textures.json").read_text())
    assert second_report["cache_hits"] == 3
    assert second_report["cache_misses"] == 0
