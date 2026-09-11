from __future__ import annotations

from pathlib import Path

from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen.build_cache_policy import build_cache_dir
from cwr_worldgen.building_asset_budget_policy import _texture_cache_tasks
from cwr_worldgen.building_texture_cache_policy import (
    BUILDING_TEXTURE_CACHE_DIRNAME,
    _shared_texture_path,
    shared_building_texture_cache_dir,
)
from cwr_worldgen.postbuild_cleanup import cleanup_build_outputs


def _variant() -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        family="residential",
        roof_style="gabled",
        width_m=10.0,
        length_m=8.0,
        height_m=6.0,
        regional_style="western_stucco",
        texture_style_token="western_stucco|stucco|cream",
        wall_material="stucco",
        roof_material="tile",
    )


def test_building_paa_cache_survives_gui_cleanup_and_restores_without_render(
    tmp_path,
) -> None:
    build_dir = tmp_path / "build"
    cache_dir = build_cache_dir(build_dir)
    local = cache_dir / "procedural-assets" / "texture-key.paa"
    destination = tmp_path / "first.paa"
    payload = b"shared-building-texture"

    def producer(path: Path) -> None:
        path.write_bytes(payload)

    first_hit = pb.restore_or_create_file(
        cache_path=local,
        destination=destination,
        producer=producer,
        enabled=True,
        refresh=False,
    )
    assert not first_hit
    shared_root = shared_building_texture_cache_dir(cache_dir)
    assert shared_root == build_dir.resolve() / BUILDING_TEXTURE_CACHE_DIRNAME
    shared = shared_root / "paa" / local.name
    assert shared.read_bytes() == payload

    cleanup_build_outputs(build_dir)
    assert not cache_dir.exists()
    assert shared.read_bytes() == payload

    second_destination = tmp_path / "second.paa"

    def render_forbidden(_path: Path) -> None:
        raise AssertionError("shared building texture cache should avoid repainting")

    second_hit = pb.restore_or_create_file(
        cache_path=local,
        destination=second_destination,
        producer=render_forbidden,
        enabled=True,
        refresh=False,
    )
    assert second_hit
    assert local.read_bytes() == payload
    assert second_destination.read_bytes() == payload


def test_p3d_cache_entries_are_not_promoted_to_shared_texture_cache(tmp_path) -> None:
    cache_dir = build_cache_dir(tmp_path / "build")
    model = cache_dir / "procedural-assets" / "model-key.p3d"
    assert _shared_texture_path(model) is None


def test_texture_prewarm_hydrates_build_local_cache_from_shared_store(tmp_path) -> None:
    build_dir = tmp_path / "build"
    cache_dir = build_cache_dir(build_dir)
    library = pb.ProceduralBuildingLibrary(
        world_name="SharedTexturePrewarm",
        cache_dir=cache_dir,
        cache_enabled=True,
    )
    library._usage[_variant()] = 1

    total, misses = _texture_cache_tasks(library, pb)
    assert total >= 4
    wall = next(task for task in misses if task.kind == "wall")
    shared = _shared_texture_path(wall.cache_path)
    assert shared is not None
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(b"already-rendered-paa")

    total_after, misses_after = _texture_cache_tasks(library, pb)
    assert total_after == total
    assert len(misses_after) == len(misses) - 1
    assert wall.cache_path.read_bytes() == b"already-rendered-paa"
