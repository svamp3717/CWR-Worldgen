from unittest.mock import patch

from cwr_worldgen.grid_default_policy import (
    DEFAULT_TERRAIN_CELL_SIZE_METRES,
    _with_default_cli_cell_size,
)


def test_cli_injects_50m_for_new_world_and_source_commands() -> None:
    assert DEFAULT_TERRAIN_CELL_SIZE_METRES == 50.0
    assert _with_default_cli_cell_size(["fetch-sources", "--source-dir", "source", "--center", "0", "0"])[-2:] == ["--cell-size", "50.0"]
    assert _with_default_cli_cell_size(["milestone1", "--output", "build"])[-2:] == ["--cell-size", "50.0"]


def test_explicit_cli_cell_size_is_never_overridden() -> None:
    args = ["fetch-sources", "--source-dir", "source", "--center", "0", "0", "--cell-size", "25"]
    assert _with_default_cli_cell_size(args) == args


def test_regrid_cells_only_mode_keeps_inference() -> None:
    args = ["regrid-sources", "--source-dir", "a", "--output-source-dir", "b", "--cells", "512"]
    assert _with_default_cli_cell_size(args) == args


def test_gui_entry_uses_50m_wizard_defaults(monkeypatch, tmp_path) -> None:
    import cwr_worldgen.gui as gui
    import cwr_worldgen.gui_entry as gui_entry

    captured = {}
    monkeypatch.setattr(gui, "DEFAULT_GUI_CELL_SIZE_METRES", 25.0)
    monkeypatch.setattr(gui, "default_gui_values", gui.default_gui_values)
    monkeypatch.setenv("CWR_WORLDGEN_RUNTIME_DIR", "@CWR-Milestone9")
    monkeypatch.setenv("CWR_WORLDGEN_GUI_STATE", str(tmp_path / "gui-state.json"))

    def fake_gui_main(_args):
        captured.update(gui.default_gui_values())
        return 0

    with patch.object(gui, "main", fake_gui_main), patch.object(gui_entry, "_configure_gui", lambda _gui, _base: None):
        assert gui_entry.main([]) == 0
    assert float(captured["fetch_cell_size"]) == 50.0
    assert gui.DEFAULT_GUI_CELL_SIZE_METRES == 50.0
