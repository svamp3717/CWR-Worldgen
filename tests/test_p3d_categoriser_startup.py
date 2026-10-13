"""Regression coverage for initial PBO/IFA selection and JSON output choices."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import categorise_p3d_models as cli
import p3d_categoriser_startup as startup
from p3d_categoriser_app import Classification
from p3d_categoriser_storage import incomplete_state_path, save_split_state
from cwr_worldgen.pbo import PboEntry, write_pbo


def _archive(tmp_path: Path, name: str = "houses.ifa") -> Path:
    path = tmp_path / name
    write_pbo(path, (PboEntry("house.p3d", b"fixture"),))
    return path


@pytest.mark.parametrize("extension", [".pbo", ".ifa", ".pbo.zst", ".ifa.zst"])
def test_picker_accepts_model_archives_and_wrappers(tmp_path: Path, extension: str) -> None:
    source = tmp_path / ("models" + extension)
    source.write_bytes(b"not needed until scan")
    output = tmp_path / "labels.json"
    assert startup.validate_startup_selection([source], output) is None


def test_picker_accepts_folder_and_loose_p3d(tmp_path: Path) -> None:
    folder = tmp_path / "Addons"
    folder.mkdir()
    loose = tmp_path / "wall.p3d"
    loose.write_bytes(b"fixture")
    assert startup.validate_startup_selection([folder, loose], tmp_path / "a.json") is None


@pytest.mark.parametrize("bad_name", ["models.zip", "texture.paa", "world.wrp"])
def test_picker_rejects_unrelated_file_types(tmp_path: Path, bad_name: str) -> None:
    bad = tmp_path / bad_name
    bad.write_bytes(b"abc")
    assert "Unsupported model source" in startup.validate_startup_selection(
        [bad], tmp_path / "models.json"
    )


def test_picker_rejects_missing_archives_missing_output_and_directory_as_output(
    tmp_path: Path,
) -> None:
    source = _archive(tmp_path)
    assert "Select at least one" in startup.validate_startup_selection(
        [], tmp_path / "categories.json"
    )
    assert "does not exist" in startup.validate_startup_selection(
        [tmp_path / "gone.ifa"], tmp_path / "categories.json"
    )
    assert "Choose a .json" in startup.validate_startup_selection(
        [source], tmp_path / "category.txt"
    )
    assert "not a file" in startup.validate_startup_selection([source], tmp_path)


def test_main_no_arguments_cancel_does_not_write_json(tmp_path: Path, monkeypatch) -> None:
    chosen = Mock(return_value=None)
    monkeypatch.setattr(cli, "prompt_for_startup_selection", chosen)
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 0
    chosen.assert_called_once_with(default_output=Path("p3d-model-categories.json"))
    assert list(tmp_path.glob("*.json")) == []


def test_cli_direct_archive_does_not_open_selection_gui(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _archive(tmp_path)
    out = tmp_path / "categories.json"
    monkeypatch.setattr(
        cli, "prompt_for_startup_selection",
        Mock(side_effect=AssertionError("CLI input should bypass setup")),
    )
    monkeypatch.setattr(cli.measure, "count_models", lambda *_args: 0)
    monkeypatch.setattr(cli, "preview_models", Mock(return_value=iter(())))
    root = Mock()
    monkeypatch.setattr(cli.tk, "Tk", Mock(return_value=root))
    launcher = Mock()
    monkeypatch.setattr(cli, "SplitStateCategoriserApp", launcher)

    assert cli.main([str(source), "-o", str(out)]) == 0
    assert launcher.call_args.kwargs["output"] == out
    root.mainloop.assert_called_once()


def test_gui_selection_sets_inputs_output_and_loads_existing_resume_json(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _archive(tmp_path)
    output = tmp_path / "reviewed.json"
    key = r"houses\house.p3d"
    save_split_state(
        output,
        categories=["Custom"],
        state={key: Classification(["Custom"], "Urban", True)},
        resume_model_path=key,
    )
    choice = startup.StartupSelection((source,), output)
    monkeypatch.setattr(
        cli, "prompt_for_startup_selection",
        Mock(return_value=choice),
    )
    calls = []
    monkeypatch.setattr(
        cli.measure, "count_models",
        lambda inputs, includes: calls.append((inputs, includes)) or 1,
    )
    monkeypatch.setattr(cli, "preview_models", Mock(return_value=iter(())))
    root = Mock()
    monkeypatch.setattr(cli.tk, "Tk", Mock(return_value=root))
    launcher = Mock()
    monkeypatch.setattr(cli, "SplitStateCategoriserApp", launcher)

    assert cli.main([]) == 0
    assert calls == [([source], [])]
    kwargs = launcher.call_args.kwargs
    assert kwargs["output"] == output
    assert kwargs["categories"] == ["Custom"]
    assert kwargs["resume_model_path"] == key
    assert kwargs["state"][key].categories == ["Custom"]
    assert kwargs["texture_resolver"].inputs == (source,)
    root.mainloop.assert_called_once_with()


def test_initial_output_can_be_set_from_cli_even_when_input_uses_gui(
    tmp_path: Path, monkeypatch,
) -> None:
    desired = tmp_path / "my-custom-file.json"
    picker = Mock(return_value=None)
    monkeypatch.setattr(cli, "prompt_for_startup_selection", picker)
    assert cli.main(["-o", str(desired)]) == 0
    picker.assert_called_once_with(default_output=desired)


def test_cancel_of_setup_does_not_load_or_modify_existing_json(
    tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "categories.json"
    output.write_text('{"old": "data"}', encoding="utf-8")
    original = output.read_bytes()
    monkeypatch.setattr(cli, "prompt_for_startup_selection", Mock(return_value=None))
    assert cli.main(["-o", str(output)]) == 0
    assert output.read_bytes() == original



def test_texture_archive_startup_validation_allows_multiple_individual_files(
    tmp_path: Path,
) -> None:
    models = _archive(tmp_path, "models.pbo")
    textures = [
        tmp_path / "textures.pbo",
        tmp_path / "extra.ifa",
        tmp_path / "wrapped.pbo.zst",
        tmp_path / "wrapped.ifa.zst",
    ]
    for path in textures:
        path.touch()

    assert startup.validate_startup_selection(
        (models,), tmp_path / "categories.json", textures
    ) is None
    assert "does not exist" in startup.validate_startup_selection(
        (models,), tmp_path / "categories.json", (tmp_path / "missing.pbo",)
    )
    wrong_file = tmp_path / "wrong.zip"
    wrong_file.touch()
    assert "Unsupported texture archive" in startup.validate_startup_selection(
        (models,), tmp_path / "categories.json", (wrong_file,)
    )
    assert "Unsupported texture archive" in startup.validate_startup_selection(
        (models,), tmp_path / "categories.json", (tmp_path,)
    )


def test_startup_texture_picker_collects_distinct_archives_separately_from_models(
    tmp_path: Path, monkeypatch,
) -> None:
    model = _archive(tmp_path, "house.pbo")
    a = _archive(tmp_path, "walls.ifa")
    b = _archive(tmp_path, "decorations.pbo")

    dialog = startup.StartupSelectionDialog.__new__(startup.StartupSelectionDialog)
    dialog.root = object()
    dialog.inputs = [model]
    dialog.texture_sources = []
    dialog._refresh_texture_sources = Mock()
    picked = []

    def choose(**kwargs):
        picked.append(kwargs)
        return (str(a), str(b), str(a))

    monkeypatch.setattr(startup.filedialog, "askopenfilenames", choose)
    dialog.select_texture_archives()
    assert dialog.inputs == [model]
    assert dialog.texture_sources == [a, b]
    assert len(picked) == 1
    assert "*.ifa" in " ".join(pattern for _, pattern in picked[0]["filetypes"])
    dialog._refresh_texture_sources.assert_called_once_with()

    dialog.texture_source_list = SimpleNamespace(curselection=lambda: (0,))
    dialog.remove_selected_textures()
    assert dialog.texture_sources == [b]
    assert dialog.inputs == [model]


def test_gui_texture_archives_do_not_extend_categoriser_model_scan(
    tmp_path: Path, monkeypatch,
) -> None:
    model = _archive(tmp_path, "models.pbo")
    extra_pbo = _archive(tmp_path, "tex1.pbo")
    extra_ifa = _archive(tmp_path, "tex2.ifa")
    selected = startup.StartupSelection(
        (model,), tmp_path / "categories.json", (extra_pbo, extra_ifa)
    )
    monkeypatch.setattr(cli, "prompt_for_startup_selection", Mock(return_value=selected))
    observed = []
    monkeypatch.setattr(
        cli.measure, "count_models",
        lambda inputs, includes: observed.append(("count", tuple(inputs))) or 0,
    )
    monkeypatch.setattr(
        cli, "preview_models",
        lambda inputs, includes, max_points, *, lod_quality: observed.append(
            ("preview", tuple(inputs), lod_quality)
        ) or iter(()),
    )
    root = Mock()
    monkeypatch.setattr(cli.tk, "Tk", Mock(return_value=root))
    launch = Mock()
    monkeypatch.setattr(cli, "SplitStateCategoriserApp", launch)

    assert cli.main([]) == 0
    assert observed == [("count", (model,)), ("preview", (model,), "fast")]
    assert launch.call_args.kwargs["texture_resolver"].inputs == (
        model, extra_pbo, extra_ifa
    )
    assert launch.call_args.kwargs["initial_texture_sources"] == (
        extra_pbo, extra_ifa
    )



def test_gui_startup_lod_preference_is_passed_into_model_stream(
    tmp_path: Path, monkeypatch,
) -> None:
    model = _archive(tmp_path, "models.pbo")
    selection = startup.StartupSelection(
        (model,), tmp_path / "categories.json", (), "full",
    )
    monkeypatch.setattr(cli, "prompt_for_startup_selection", Mock(return_value=selection))
    monkeypatch.setattr(cli.measure, "count_models", lambda *_args: 1)
    preview = Mock(return_value=iter(()))
    monkeypatch.setattr(cli, "preview_models", preview)
    monkeypatch.setattr(cli.tk, "Tk", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "SplitStateCategoriserApp", Mock())

    assert cli.main([]) == 0
    assert preview.call_args.kwargs["lod_quality"] == "full"


def test_cli_preview_lod_defaults_fast_and_supports_full(
    tmp_path: Path, monkeypatch,
) -> None:
    archive = _archive(tmp_path, "models.pbo")
    monkeypatch.setattr(cli.measure, "count_models", lambda *_args: 0)
    preview = Mock(return_value=iter(()))
    monkeypatch.setattr(cli, "preview_models", preview)
    monkeypatch.setattr(cli.tk, "Tk", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "SplitStateCategoriserApp", Mock())

    assert cli.main([str(archive)]) == 0
    assert preview.call_args.kwargs["lod_quality"] == "fast"
    assert cli.main([str(archive), "--preview-lod", "full"]) == 0
    assert preview.call_args.kwargs["lod_quality"] == "full"
