from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen.gui_entry import (
    LEGACY_WORLD_NAME_PREFIXES,
    WORLD_NAME_PREFIX,
    _configure_gui,
    generated_world_identifier,
)


def _slugify(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split()) or "my_world"


def _fake_gui() -> SimpleNamespace:
    class DummyWorldgenGui:
        pass

    return SimpleNamespace(
        application_base_dir=lambda: Path.cwd(),
        resolve_gui_path=lambda value: Path(value),
        default_gui_path=lambda value: str(value),
        slugify_world_name=_slugify,
        default_gui_values=lambda: {
            "name": "cwr_my_world",
            "display_name": "My CWA World",
            "output": "build/my_world",
            "source_dir": "source-data/my_world",
        },
        defaults_with_recent_source=lambda defaults, _state: dict(defaults),
        suggested_world_values=lambda *_args, **_kwargs: {},
        WorldgenGui=DummyWorldgenGui,
    )


def test_generated_world_identifier_uses_wg_prefix_and_migrates_legacy_prefixes() -> None:
    assert WORLD_NAME_PREFIX == "wg_"
    assert set(LEGACY_WORLD_NAME_PREFIXES) == {"cwa_", "cwr_"}
    assert generated_world_identifier("alpine_test") == "wg_alpine_test"
    assert generated_world_identifier("wg_alpine_test") == "wg_alpine_test"
    assert generated_world_identifier("cwa_alpine_test") == "wg_alpine_test"
    assert generated_world_identifier("cwr_alpine_test") == "wg_alpine_test"


def test_official_gui_entrypoint_defaults_and_suggestions_use_wg_prefix(tmp_path: Path) -> None:
    gui = _fake_gui()
    _configure_gui(gui, tmp_path)

    assert gui.default_gui_values()["name"] == "wg_my_world"
    suggested = gui.suggested_world_values(
        "Alpine Test",
        source_mode="new",
        source_dir="existing-source",
    )
    assert suggested["name"] == "wg_alpine_test"
    assert suggested["output"] == str(tmp_path / "build" / "alpine_test")
    assert suggested["source_dir"] == str(tmp_path / "source-data" / "alpine_test")


def test_recent_auto_generated_legacy_name_is_migrated_but_manual_name_is_preserved(tmp_path: Path) -> None:
    gui = _fake_gui()
    _configure_gui(gui, tmp_path)

    migrated = gui.defaults_with_recent_source(
        {"name": "cwa_old_world2", "output": str(tmp_path / "build" / "cwa_old_world2")},
        {},
    )
    assert migrated["name"] == "wg_old_world2"
    assert migrated["output"] == str(tmp_path / "build" / "old_world2")

    manual = gui.defaults_with_recent_source(
        {"name": "custom_world", "output": str(tmp_path / "build" / "custom_world")},
        {},
    )
    assert manual["name"] == "custom_world"
