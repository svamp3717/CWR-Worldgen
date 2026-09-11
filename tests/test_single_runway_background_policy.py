from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cwr_worldgen.fast_asset_scan_policy as fast
import cwr_worldgen.runway_exact_background_policy as exact
from cwr_worldgen.single_runway_background_policy import (
    _filter_scan_assets,
    _standard_pbos,
    external_runway_texture_paths,
    runway_background_texture_path,
)


def _spec(profile: str, name: str = "wg_test") -> SimpleNamespace:
    return SimpleNamespace(name=name, ground_texture_profile=profile)


def test_one_runway_background_texture_per_preset() -> None:
    assert runway_background_texture_path(_spec("nogova")) == r"o\t1.paa"
    assert runway_background_texture_path(_spec("everon")) == r"Eden\zbh.paa"
    assert runway_background_texture_path(_spec("desert")) == r"o\ps.paa"
    assert runway_background_texture_path(_spec("malden")) == r"wg_test\data\g.paa"
    assert runway_background_texture_path(_spec("generated")) == r"wg_test\data\g.paa"

    assert external_runway_texture_paths(_spec("nogova")) == (r"o\t1.paa",)
    assert external_runway_texture_paths(_spec("everon")) == (r"Eden\zbh.paa",)
    assert external_runway_texture_paths(_spec("desert")) == (r"o\ps.paa",)
    assert external_runway_texture_paths(_spec("malden")) == ()
    assert external_runway_texture_paths(_spec("generated")) == ()


def test_scan_filter_keeps_stock_models_and_only_allowed_ground_textures(tmp_path: Path) -> None:
    game = tmp_path / "CWA"
    (game / "Res" / "AddOns").mkdir(parents=True)
    (game / "Res" / "AddOns" / "O.pbo").write_bytes(b"")

    selected = (
        r"o\road\sil25.p3d",
        r"wg_test\i\gravel25.p3d",
        r"o\t1.paa",
        r"o\trava2.paa",
        r"o\pole1.paa",
        r"wg_test\i\g.paa",
    )

    assert _filter_scan_assets(fast, (game,), selected) == (
        r"o\road\sil25.p3d",
        r"o\t1.paa",
    )


def test_standard_pbo_lookup_does_not_recurse(tmp_path: Path) -> None:
    game = tmp_path / "CWA"
    deep = game / "weird" / "nested"
    deep.mkdir(parents=True)
    (deep / "O.pbo").write_bytes(b"")

    assert _standard_pbos(exact, game, "o") == ()

    normal = game / "Res" / "AddOns"
    normal.mkdir(parents=True, exist_ok=True)
    expected = normal / "O.pbo"
    expected.write_bytes(b"")
    assert _standard_pbos(exact, game, "o") == (expected,)
