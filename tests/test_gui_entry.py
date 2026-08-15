from pathlib import Path

from cwr_worldgen.gui_entry import generated_mod_folder, managed_replacement


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
