from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen import surface_pass
from cwr_worldgen import terrain
from cwr_worldgen.stock_desert_surface_policy import (
    DESERT_STOCK_SURFACE_TEXTURES,
    install_stock_desert_surface_policy,
)


def _spec(*, surface_ground: bool = True):
    return SimpleNamespace(
        name="test_desert",
        ground_texture_profile="desert",
        surface_pass_enabled=surface_ground,
        surface_ground_mode="milestone9" if surface_ground else "milestone8",
    )


def test_desert_milestone9_texture_table_uses_only_stock_game_paths() -> None:
    install_stock_desert_surface_policy()
    paths = generator._ground_texture_paths(_spec())

    assert len(paths) == len(surface_pass.MILESTONE9_MATERIALS)
    assert paths == tuple(
        DESERT_STOCK_SURFACE_TEXTURES[material.code]
        for material in surface_pass.MILESTONE9_MATERIALS
    )
    assert all(path.casefold().startswith(("eden\\", "o\\")) for path in paths)
    assert all("test_desert\\" not in path.casefold() for path in paths)


def test_desert_external_dependencies_are_stock_and_deduplicated() -> None:
    install_stock_desert_surface_policy()
    paths = generator._external_ground_texture_paths(_spec())

    assert paths
    assert len(paths) == len(set(path.casefold() for path in paths))
    assert set(path.casefold() for path in paths) == {
        path.casefold() for path in DESERT_STOCK_SURFACE_TEXTURES.values()
    }


def test_desert_surface_writer_generates_no_ground_paas(tmp_path) -> None:
    install_stock_desert_surface_policy()
    written = surface_pass.write_surface_textures(
        tmp_path,
        "test_desert",
        "desert",
        "seed",
        16,
    )

    assert written == ()
    data_dir = tmp_path / "data"
    assert not data_dir.exists() or not tuple(data_dir.glob("*.paa"))


def test_desert_profile_suppresses_legacy_generated_texture_predicate() -> None:
    install_stock_desert_surface_policy()
    profile = generator._ground_texture_profile(_spec())

    # Keep the public/manifest value Desert while satisfying the old generator's
    # hard-coded {Everon,Nogova} stock membership test. This prevents the local
    # PAA list from being populated before the terrain-table stage.
    assert str(profile) == "desert"
    assert profile == "desert"
    assert profile in {"everon", "nogova"}
    generated = tuple(
        material.code
        for material in surface_pass.MILESTONE9_MATERIALS
        if profile not in {"everon", "nogova"}
        or getattr(material, "everon_path", None) is None
    )
    assert generated == ()


def test_legacy_desert_ground_path_also_resolves_to_stock_texture() -> None:
    install_stock_desert_surface_policy()
    assert terrain.ground_texture_path("test_desert", "s", "desert") == r"o\ps.paa"
    assert terrain.ground_texture_path("test_desert", "d", "desert") == r"Eden\bak\bah.pac"


def test_non_desert_profile_keeps_normal_generator_identity() -> None:
    install_stock_desert_surface_policy()
    spec = SimpleNamespace(ground_texture_profile="generated")
    assert generator._ground_texture_profile(spec) == "generated"
