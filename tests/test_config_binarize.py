"""Regression coverage for optional CWA RAP v4 config.bin packaging."""
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen.config_binarize import (
    binarize_config, binarize_config_text, pack_binarized_runtime_pbo,
    parse_cpp_config,
)
from cwr_island_studio.world_metadata import parse_world_config
from cwr_worldgen.pbo import pack_directory, read_pbo
from cwr_worldgen.templates import render_config
from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import binarize_rvw4


def cfg(animated: bool = False) -> str:
    return render_config(
        SimpleNamespace(
            name="test_island", display_name="Test Island",
            centre=1600, cell_size=50, surface_pass_enabled=False,
        ),
        milestone=9,
        town_names=(
            SimpleNamespace(
                class_name="FirstTown", name="East Port",
                x=525, z=1375,
            ),
            SimpleNamespace(
                class_name="SecondTown", name="North Village",
                x=2075, z=2245,
            ),
        ),
        animated_building_models=(
            (r"test_island\structures\door_house.p3d",)
            if animated else ()
        ),
    )


def _wrp(path: Path) -> None:
    write_rvw4(
        path, 4, 4, [0.05 * n for n in range(16)],
        [0] * 16, ["test_island\\ground.paa"],
        [WorldObject(1, r"data3d\strom.p3d", 80, 0, 70)],
        height_scale=0.05,
    )


def test_encoded_rap_header_matches_real_cwa_engine_fixture() -> None:
    # The CWR-CE ParamFile::SerializeBin empty config fixture is
    # magic, version int32 4, root name index0, empty root name,
    # empty base, entry-count varint zero.
    empty = binarize_config_text("")
    assert empty == b"\x00raP\x04\x00\x00\x00\x00\x00\x00\x00"


def test_rap_configuration_preserves_grid_names_door_classes_and_values() -> None:
    for with_doors in (False, True):
        original = cfg(with_doors)
        encoded = binarize_config_text(original)
        assert encoded[:8] == b"\x00raP\x04\x00\x00\x00"
        assert len(encoded) < len(original.encode("utf-8"))
        text_metadata = parse_world_config(original, "test_island")
        binary_metadata = parse_world_config(encoded, "test_island")
        assert binary_metadata.land_grid == text_metadata.land_grid == 50
        assert binary_metadata.locations == text_metadata.locations
        assert binary_metadata.source == "config.cpp"
        assert b"CfgVehicles" in encoded if with_doors else b"CfgWorlds" in encoded
        if with_doors:
            assert b"OpenDoor1" in encoded
            assert b"door_house.p3d" in encoded
            assert b"this animate" in encoded


def test_missing_or_unsupported_constructs_are_rejected_without_partial_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        binarize_config_text('class CfgWorlds { LandGrid = 50 * 2; };')
    with pytest.raises(ValueError, match="inherits"):
        binarize_config_text("class CfgWorlds { class island: MissingBase {}; };")
    with pytest.raises(ValueError, match="duplicate"):
        binarize_config_text("class Test { landGrid=50; landGrid=25; };")
    path = tmp_path / "config.cpp"
    path.write_text("class CfgVehicles { class Foo {}; };")
    target = tmp_path / "config.bin"
    with pytest.raises(ValueError, match="CfgWorlds"):
        binarize_config(path, target)
    assert not target.exists()


def test_source_config_stays_editable_and_game_pbo_contains_only_binary(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = source_dir / "config.cpp"
    source_config.write_text(cfg(), encoding="utf-8")
    original_config = source_config.read_bytes()
    source_wrp = source_dir / "test_island.wrp"
    _wrp(source_wrp)
    original_wrp = source_wrp.read_bytes()
    output = tmp_path / "test_island.pbo"
    pack_directory(source_dir, output, backend="python")
    binary = binarize_config(source_config, tmp_path / "config.bin")
    assert binary.exists() and binary.read_bytes()[:4] == b"\x00raP"
    pack_binarized_runtime_pbo(
        source_dir, source_config, binary, output, backend="python",
    )
    members = {entry.name.lower(): entry.data for entry in read_pbo(output)}
    assert "config.bin" in members
    assert "config.cpp" not in members
    assert members["config.bin"] == binary.read_bytes()
    assert members["test_island.wrp"] == original_wrp
    assert source_config.read_bytes() == original_config
    assert source_wrp.read_bytes() == original_wrp
    assert not (source_dir / "config.bin").exists()


def test_combined_config_and_world_binarization_packs_once_without_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = source_dir / "config.cpp"
    source_config.write_text(cfg())
    source_wrp = source_dir / "test_island.wrp"
    _wrp(source_wrp)
    original_config, original_wrp = source_config.read_bytes(), source_wrp.read_bytes()
    binary_config = binarize_config(source_config, tmp_path / "config.bin")
    binary_wrp = tmp_path / "test_island.binarized.wrp"
    binarize_rvw4(source_wrp, binary_wrp, height_scale=0.05)
    output = tmp_path / "game.pbo"
    pack_directory(source_dir, output, backend="python")
    calls = []
    import cwr_worldgen.pbo as packer
    original_pack = packer.pack_directory

    def counted(*args, **kwargs):
        calls.append(1)
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(packer, "pack_directory", counted)
    pack_binarized_runtime_pbo(
        source_dir, source_config, binary_config, output,
        binarized_wrp=binary_wrp, backend="python",
    )
    assert len(calls) == 1
    members = {entry.name.lower(): entry.data for entry in read_pbo(output)}
    assert members["config.bin"][:8] == b"\x00raP\x04\x00\x00\x00"
    assert members["test_island.wrp"][:8] == b"OPRW\x03\x00\x00\x00"
    assert "config.cpp" not in members
    assert source_config.read_bytes() == original_config
    assert source_wrp.read_bytes() == original_wrp


def test_pack_failure_restores_sources_and_previous_pbo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_config = source_dir / "config.cpp"
    source_config.write_text(cfg())
    original_config = source_config.read_bytes()
    source_wrp = source_dir / "test_island.wrp"
    _wrp(source_wrp)
    original_wrp = source_wrp.read_bytes()
    binary_config = binarize_config(source_config, tmp_path / "config.bin")
    binary_wrp = tmp_path / "test_island.binarized.wrp"
    binarize_rvw4(source_wrp, binary_wrp, height_scale=0.05)
    game = tmp_path / "game.pbo"
    game.write_bytes(b"previous game PBO")
    import cwr_worldgen.pbo as packer

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(packer, "pack_directory", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        pack_binarized_runtime_pbo(
            source_dir, source_config, binary_config, game,
            binarized_wrp=binary_wrp, backend="python",
        )
    assert source_config.read_bytes() == original_config
    assert source_wrp.read_bytes() == original_wrp
    assert game.read_bytes() == b"previous game PBO"
    assert not (source_dir / "config.bin").exists()


def test_advanced_option_persists_and_reaches_milestone9_cli(tmp_path: Path) -> None:
    from cwr_worldgen.gui import (
        default_gui_values, defaults_with_recent_source, load_gui_state,
        save_gui_state, build_milestone9_command, WorldgenGui,
    )
    from cwr_worldgen.cli import _parser

    defaults = default_gui_values()
    assert defaults["binarize_config"] is False
    defaults.update({
        "source_dir": str(tmp_path), "output": str(tmp_path / "out"),
        "name": "test_island", "display_name": "Test Island",
    })
    assert "--binarize-config" not in build_milestone9_command(defaults, python="python")
    defaults["binarize_config"] = True
    command = build_milestone9_command(defaults, python="python")
    assert "--binarize-config" in command
    args = _parser().parse_args(command[command.index("milestone9"):])
    assert args.binarize_config is True

    class Checkbox:
        def __init__(self):
            self.value = False
            self.callbacks = []
        def get(self):
            return self.value
        def set(self, value):
            self.value = value
            for callback in self.callbacks:
                callback()
        def trace_add(self, mode, callback):
            self.callbacks.append(callback)

    fake = object.__new__(WorldgenGui)
    fake.state_path = tmp_path / "gui-state.json"
    check = Checkbox()
    fake.vars = {"binarize_config": check}
    WorldgenGui._install_binarized_config_state_persistence(fake)
    check.set(True)
    assert load_gui_state(fake.state_path)["binarize_config"] is True
    assert defaults_with_recent_source(default_gui_values(), load_gui_state(fake.state_path))["binarize_config"] is True
    check.set(False)
    assert load_gui_state(fake.state_path)["binarize_config"] is False
