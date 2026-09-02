from pathlib import Path

from cwr_worldgen.gui_entry import (
    CONSOLE_LOG_FILENAME,
    ROAD_INSPECTOR_CHILD_CODE,
    ROAD_INSPECTOR_CLI_MARKER,
    _run_road_inspector_postbuild,
    console_log_paths,
    generated_mod_folder,
    generated_world_pbo,
    managed_replacement,
    mirror_console_log_fragment,
    road_inspector_postbuild_command,
)
from cwr_worldgen.postbuild_cleanup import (
    CLEANUP_BUILD_AFTER_BUILD,
    CLEANUP_ONLY_MARKER,
    cleanup_build_outputs,
    postbuild_cleanup_command,
)


def test_frozen_worker_dispatch_happens_before_gui_import() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "gui_entry.py").read_text(encoding="utf-8")
    freeze_index = source.index("_multiprocessing.freeze_support()")
    main_index = source.index("def main(")
    gui_import_index = source.index("from . import gui")
    assert freeze_index < main_index < gui_import_index


def test_managed_replacement_updates_untouched_value() -> None:
    value, managed = managed_replacement("build/my_world", "build/my_world", "build/new_world")
    assert value == "build/new_world"
    assert managed == "build/new_world"


def test_managed_replacement_preserves_user_custom_value() -> None:
    value, managed = managed_replacement("D:/Worlds/custom", "build/my_world", "build/new_world")
    assert value == "D:/Worlds/custom"
    assert managed is None


def test_managed_replacement_can_compare_normalized_paths() -> None:
    value, managed = managed_replacement(
        "BUILD/My_World",
        "build/my_world",
        "build/new_world",
        normalizer=str.casefold,
    )
    assert value == "build/new_world"
    assert managed == "build/new_world"


def test_console_log_paths_targets_source_and_build(tmp_path: Path) -> None:
    source = tmp_path / "source-data" / "world"
    output = tmp_path / "build" / "world"
    assert console_log_paths(source, output) == (
        source / CONSOLE_LOG_FILENAME,
        output / CONSOLE_LOG_FILENAME,
    )


def test_console_log_paths_deduplicates_same_folder(tmp_path: Path) -> None:
    folder = tmp_path / "world"
    assert console_log_paths(folder, folder) == (folder / CONSOLE_LOG_FILENAME,)


def test_console_log_mirror_starts_fresh_and_appends(tmp_path: Path) -> None:
    source_log, build_log = console_log_paths(tmp_path / "source", tmp_path / "build")
    source_log.parent.mkdir(parents=True)
    source_log.write_text("old run\n", encoding="utf-8")
    initialized: set[str] = set()
    mirror_console_log_fragment((source_log, build_log), "> command\n", "> command\n", initialized)
    mirror_console_log_fragment((source_log, build_log), "progress\n", "> command\nprogress\n", initialized)
    expected = "> command\nprogress\n"
    assert source_log.read_text(encoding="utf-8") == expected
    assert build_log.read_text(encoding="utf-8") == expected


def test_console_log_mirror_restores_full_transcript_after_folder_cleanup(tmp_path: Path) -> None:
    target = tmp_path / "build" / CONSOLE_LOG_FILENAME
    initialized: set[str] = set()
    mirror_console_log_fragment((target,), "first\n", "first\n", initialized)
    target.unlink()
    mirror_console_log_fragment((target,), "second\n", "first\nsecond\n", initialized)
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


def test_generated_mod_folder_finds_worldgen_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "CWR-Worldgen"
    (runtime / "Addons").mkdir(parents=True)
    (runtime / "Anims").mkdir()
    assert generated_mod_folder(tmp_path) == runtime.resolve()


def test_generated_mod_folder_accepts_runtime_root_directly(tmp_path: Path) -> None:
    (tmp_path / "Addons").mkdir()
    (tmp_path / "Anims").mkdir()
    assert generated_mod_folder(tmp_path) == tmp_path.resolve()


def test_generated_mod_folder_requires_addons_and_anims(tmp_path: Path) -> None:
    incomplete = tmp_path / "CWR-Worldgen"
    (incomplete / "Addons").mkdir(parents=True)
    assert generated_mod_folder(tmp_path) is None


def test_generated_mod_folder_prefers_worldgen_when_multiple_exist(tmp_path: Path) -> None:
    other = tmp_path / "OtherRuntime"
    (other / "Addons").mkdir(parents=True)
    (other / "Anims").mkdir()
    worldgen = tmp_path / "CWR-Worldgen"
    (worldgen / "Addons").mkdir(parents=True)
    (worldgen / "Anims").mkdir()
    assert generated_mod_folder(tmp_path) == worldgen.resolve()


def test_generated_world_pbo_uses_exact_world_name(tmp_path: Path) -> None:
    runtime = tmp_path / "CWR-Worldgen"
    addons = runtime / "Addons"
    addons.mkdir(parents=True)
    (runtime / "Anims").mkdir()
    expected = addons / "wg_demo.pbo"
    expected.write_bytes(b"pbo")
    (addons / "other.pbo").write_bytes(b"other")
    assert generated_world_pbo(tmp_path, "wg_demo") == expected.resolve()


def test_source_gui_postbuild_command_avoids_runpy_reentry(tmp_path: Path) -> None:
    command = road_inspector_postbuild_command(tmp_path / "build", "wg_demo", frozen=False, executable="python-test")
    assert command == [
        "python-test", "-c", ROAD_INSPECTOR_CHILD_CODE,
        "--build-dir", str(tmp_path / "build"), "--world-name", "wg_demo",
    ]
    assert "-m" not in command


def test_frozen_gui_postbuild_command_reenters_executable(tmp_path: Path) -> None:
    command = road_inspector_postbuild_command(tmp_path / "build", "wg_demo", frozen=True, executable="CWR-Worldgen.exe")
    assert command == [
        "CWR-Worldgen.exe", ROAD_INSPECTOR_CLI_MARKER,
        "--build-dir", str(tmp_path / "build"), "--world-name", "wg_demo",
    ]


def test_missing_postbuild_pbo_is_nonfatal(tmp_path: Path) -> None:
    assert _run_road_inspector_postbuild(["--build-dir", str(tmp_path), "--world-name", "wg_missing"]) == 0
    assert (tmp_path / "road-inspector" / "error.txt").is_file()


def test_cleanup_removes_only_source_and_normalized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    normalized = tmp_path / "normalized"
    runtime = tmp_path / "CWR-Worldgen"
    inspector = tmp_path / "road-inspector"
    source.mkdir()
    normalized.mkdir()
    runtime.mkdir()
    inspector.mkdir()
    (source / "world.wrp").write_text("temporary", encoding="utf-8")
    (normalized / "roads.geojson").write_text("temporary", encoding="utf-8")
    final_pbo = tmp_path / "wg_demo.pbo"
    final_pbo.write_bytes(b"pbo")
    log = tmp_path / CONSOLE_LOG_FILENAME
    log.write_text("log", encoding="utf-8")

    removed = cleanup_build_outputs(tmp_path)

    assert removed == (source, normalized)
    assert not source.exists()
    assert not normalized.exists()
    assert runtime.is_dir()
    assert inspector.is_dir()
    assert final_pbo.read_bytes() == b"pbo"
    assert log.read_text(encoding="utf-8") == "log"


def test_source_cleanup_command_reuses_quiet_child_launcher(tmp_path: Path) -> None:
    command = postbuild_cleanup_command(tmp_path / "build", frozen=False, executable="python-test")
    assert command == [
        "python-test", "-c", ROAD_INSPECTOR_CHILD_CODE,
        CLEANUP_ONLY_MARKER, "--build-dir", str(tmp_path / "build"),
    ]


def test_frozen_cleanup_command_reenters_existing_marker(tmp_path: Path) -> None:
    command = postbuild_cleanup_command(tmp_path / "build", frozen=True, executable="CWR-Worldgen.exe")
    assert command == [
        "CWR-Worldgen.exe", ROAD_INSPECTOR_CLI_MARKER,
        CLEANUP_ONLY_MARKER, "--build-dir", str(tmp_path / "build"),
    ]


def test_cleanup_checkbox_is_default_on_and_above_inspector() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "cwr_worldgen" / "postbuild_cleanup.py"
    ).read_text(encoding="utf-8")
    assert f'values[CLEANUP_BUILD_AFTER_BUILD] = True' in source
    assert "Delete temporary build files after a successful build" in source
    assert 'cleanup.pack(anchor="w", before=inspector)' in source
    assert 'CLEANUP_DIR_NAMES = ("source", "normalized")' in source


def test_gui_entry_contains_inspector_checkbox_pipeline_map_and_road_type_filter() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen" / "gui_entry.py").read_text(encoding="utf-8")
    assert "Run Road Inspector after a successful build" in source
    assert '"run_road_inspector_after_build"' in source
    assert '"Running Road Inspector"' in source
    assert "road_inspector_postbuild_command(" in source
    assert '<svg id="map"></svg>' in source
    assert "medium +" in source
    assert "Copy coords" in source
    assert "document.execCommand('copy')" in source
    assert 'id="roadtype"' in source
    assert '>paved<' in source
    assert '>gravel<' in source
    assert '>dirt<' in source
