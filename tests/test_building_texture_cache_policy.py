from __future__ import annotations

from pathlib import Path

from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen.build_cache_policy import build_cache_dir
from cwr_worldgen.building_asset_budget_policy import _texture_cache_tasks
from cwr_worldgen.building_texture_cache_policy import _shared_texture_path
from cwr_worldgen.postbuild_cleanup import cleanup_build_outputs
from cwr_worldgen.shared_cache_policy import (
    BUILDING_TEXTURE_CACHE_DIRNAME,
    SHARED_CACHE_DIRNAME,
    shared_building_texture_cache_dir,
)


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


def test_building_paa_cache_is_shared_between_worlds_and_survives_cleanup(
    tmp_path,
) -> None:
    world_a = tmp_path / "WorldA"
    world_b = tmp_path / "WorldB"
    cache_a = build_cache_dir(world_a)
    cache_b = build_cache_dir(world_b)
    local_a = cache_a / "procedural-assets" / "texture-key.paa"
    local_b = cache_b / "procedural-assets" / "texture-key.paa"
    first_destination = tmp_path / "first.paa"
    payload = b"shared-building-texture"

    def producer(path: Path) -> None:
        path.write_bytes(payload)

    first_hit = pb.restore_or_create_file(
        cache_path=local_a,
        destination=first_destination,
        producer=producer,
        enabled=True,
        refresh=False,
    )
    assert not first_hit

    shared_root = shared_building_texture_cache_dir(cache_a)
    assert shared_root == (
        tmp_path.resolve()
        / SHARED_CACHE_DIRNAME
        / BUILDING_TEXTURE_CACHE_DIRNAME
    )
    assert shared_building_texture_cache_dir(cache_b) == shared_root
    shared = shared_root / "paa" / local_a.name
    assert shared.read_bytes() == payload

    cleanup_build_outputs(world_a)
    assert not cache_a.exists()
    assert shared.read_bytes() == payload

    second_destination = tmp_path / "second.paa"

    def render_forbidden(_path: Path) -> None:
        raise AssertionError("World B should reuse World A's shared building texture")

    second_hit = pb.restore_or_create_file(
        cache_path=local_b,
        destination=second_destination,
        producer=render_forbidden,
        enabled=True,
        refresh=False,
    )
    assert second_hit
    assert local_b.read_bytes() == payload
    assert second_destination.read_bytes() == payload


def test_p3d_cache_entries_are_not_promoted_to_shared_texture_cache(tmp_path) -> None:
    cache_dir = build_cache_dir(tmp_path / "WorldA")
    model = cache_dir / "procedural-assets" / "model-key.p3d"
    assert _shared_texture_path(model) is None


def test_texture_prewarm_hydrates_different_world_from_shared_store(tmp_path) -> None:
    world_a_cache = build_cache_dir(tmp_path / "WorldA")
    world_b_cache = build_cache_dir(tmp_path / "WorldB")

    library_a = pb.ProceduralBuildingLibrary(
        world_name="SharedTexturePrewarmA",
        cache_dir=world_a_cache,
        cache_enabled=True,
    )
    library_a._usage[_variant()] = 1
    total_a, misses_a = _texture_cache_tasks(library_a, pb)
    assert total_a >= 4
    wall_a = next(task for task in misses_a if task.kind == "wall")
    shared = _shared_texture_path(wall_a.cache_path)
    assert shared is not None
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(b"already-rendered-paa")

    library_b = pb.ProceduralBuildingLibrary(
        world_name="SharedTexturePrewarmB",
        cache_dir=world_b_cache,
        cache_enabled=True,
    )
    library_b._usage[_variant()] = 1
    total_b, misses_b = _texture_cache_tasks(library_b, pb)
    wall_b_path = world_b_cache / "procedural-assets" / wall_a.cache_path.name

    assert total_b == total_a
    assert len(misses_b) == len(misses_a) - 1
    assert wall_b_path.read_bytes() == b"already-rendered-paa"
