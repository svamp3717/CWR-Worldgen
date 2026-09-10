from __future__ import annotations

from pathlib import Path

from cwr_worldgen.game_folder_gui_policy import (
    GAME_FOLDER_STATE_KEY,
    merge_game_folder_asset_roots,
    remembered_game_folder,
)


def test_remembered_game_folder_restores_existing_directory(tmp_path: Path) -> None:
    game = tmp_path / "Cold War Assault"
    game.mkdir()
    assert remembered_game_folder({GAME_FOLDER_STATE_KEY: str(game)}) == str(game.resolve())


def test_remembered_game_folder_ignores_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "moved-game"
    assert remembered_game_folder({GAME_FOLDER_STATE_KEY: str(missing)}) == ""


def test_game_folder_is_injected_once_before_optional_asset_roots(tmp_path: Path) -> None:
    game = tmp_path / "game"
    other = tmp_path / "unpacked"
    game.mkdir()
    other.mkdir()
    roots = merge_game_folder_asset_roots([str(other)], str(game))
    assert roots == [str(game.resolve()), str(other)]

    duplicate = merge_game_folder_asset_roots(roots, str(game))
    assert duplicate == roots


def test_empty_game_folder_leaves_asset_roots_unchanged() -> None:
    assert merge_game_folder_asset_roots(["one", "two"], "") == ["one", "two"]
