from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen.malden_classic_policy import (
    _StockMaldenProfile,
    _wrp_texture_usage,
    resolved_malden_surface_textures,
)
from cwr_worldgen.paa import write_solid_dxt1_paa
from cwr_worldgen import single_runway_background_policy as single
from cwr_worldgen.wrp import write_rvw4


def _spec(game_root, *, surface=True):
    return SimpleNamespace(
        name="wg_malden_test",
        ground_texture_profile="malden",
        surface_pass_enabled=surface,
        surface_ground_mode="milestone9",
        asset_roots=(game_root,),
        deploy_mod_dir=None,
    )


def _write_stock_malden_fixture(root):
    worlds = root / "Worlds"
    landtext = root / "LandText"
    worlds.mkdir(parents=True)
    landtext.mkdir(parents=True)

    paths = (
        r"LandText\mal_grass.paa",
        r"LandText\mal_sand.paa",
        r"LandText\mal_rock.paa",
        r"LandText\mal_earth.paa",
    )
    colours = (
        (109, 118, 70),
        (184, 162, 109),
        (110, 105, 96),
        (120, 91, 59),
    )
    for wire, colour in zip(paths, colours):
        write_solid_dxt1_paa(
            root.joinpath(*wire.split("\\")),
            colour=colour,
        )

    # Keep all four source archetypes common enough to survive the transition
    # candidate cutoff while making grass the ordinary dominant Malden tile.
    indices = (
        0, 0, 0, 0,
        0, 0, 0, 1,
        1, 1, 1, 2,
        2, 2, 3, 3,
    )
    write_rvw4(
        worlds / "abel.wrp",
        4,
        4,
        (0.0,) * 16,
        indices,
        paths,
        (),
        height_scale=0.05,
    )
    return paths


def test_malden_resolver_uses_original_abel_wrp_texture_table(tmp_path) -> None:
    paths = _write_stock_malden_fixture(tmp_path)
    mapping = resolved_malden_surface_textures(_spec(tmp_path))

    assert mapping is not None
    assert mapping["g"] == paths[0]
    assert mapping["a"] == paths[0]
    assert mapping["b"] == paths[0]
    assert mapping["c"] == paths[0]
    assert mapping["s"] == paths[1]
    assert mapping["x"] == paths[1]
    assert mapping["r"] == paths[2]
    assert mapping["k"] == paths[2]
    assert mapping["d"] == paths[3]
    assert mapping["p"] == paths[3]


def test_malden_stock_profile_routes_generator_like_everon_classic(tmp_path) -> None:
    paths = _write_stock_malden_fixture(tmp_path)
    spec = _spec(tmp_path)

    profile = generator._ground_texture_profile(spec)
    ground_paths = generator._ground_texture_paths(spec)

    assert isinstance(profile, _StockMaldenProfile)
    assert str(profile) == "malden"
    # Compatibility marker makes the existing generator's "is this stock?"
    # predicate take the same branch as Everon without renaming the preset.
    assert profile in {"everon", "nogova"}
    assert ground_paths[3] == paths[0]  # grass
    assert ground_paths[9] == paths[0]  # farmland light
    assert ground_paths[10] == paths[0]  # farmland dark
    assert ground_paths[2] == paths[1]  # sand
    assert ground_paths[5] == paths[2]  # rock
    assert all(r"\data\" not in path.casefold() for path in ground_paths)
    assert single.runway_background_texture_path(spec) == paths[0]
    assert generator._external_ground_texture_paths(spec) == (paths[0],)


def test_malden_without_original_island_keeps_generated_fallback(tmp_path) -> None:
    spec = _spec(tmp_path)

    assert resolved_malden_surface_textures(spec) is None
    assert generator._ground_texture_profile(spec) == "malden"
    paths = generator._ground_texture_paths(spec)
    assert any(path.casefold().startswith(r"wg_malden_test\data") for path in paths)


def test_stock_wrp_usage_parser_ranks_original_texture_frequency(tmp_path) -> None:
    paths = _write_stock_malden_fixture(tmp_path)
    usage = _wrp_texture_usage(tmp_path / "Worlds" / "abel.wrp")

    assert usage[0] == (paths[0], 7)
    assert dict(usage)[paths[1]] == 4
    assert dict(usage)[paths[2]] == 3
    assert dict(usage)[paths[3]] == 2
