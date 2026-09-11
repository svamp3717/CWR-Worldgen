from __future__ import annotations

from pathlib import Path

from cwr_worldgen.game_folder_gui_policy import (
    GAME_FOLDER_STATE_KEY,
    merge_game_folder_asset_roots,
    preferred_gui_window_size,
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


def test_gui_uses_roomier_window_on_1080p_desktop() -> None:
    assert preferred_gui_window_size(1920, 1080) == (1240, 920)


def test_gui_window_clamps_to_smaller_screen() -> None:
    width, height = preferred_gui_window_size(1366, 768)
    assert width == 1240
    assert height == 668
    assert width <= 1366
    assert height <= 768
