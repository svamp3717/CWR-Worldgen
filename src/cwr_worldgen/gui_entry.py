# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

FROZEN_CLI_MARKER = "--cwr-cli"
OVERTURE_CLI_MARKER = "--cwr-overture"
CONSOLE_LOG_FILENAME = "cwr-worldgen-console.log"


def storage_base_dir() -> Path:
    """Return the folder that owns build, source-data, and config."""
    if not bool(getattr(sys, "frozen", False)):
        return Path.cwd().resolve()
    executable = Path(sys.executable).resolve()
    if (
        sys.platform == "darwin"
        and executable.parent.name == "MacOS"
        and executable.parent.parent.name == "Contents"
    ):
        return executable.parents[3]
    return executable.parent


def managed_replacement(
    current: str,
    managed: str | None,
    replacement: str,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> tuple[str, str | None]:
    """Replace an auto-managed value, but stop tracking a user-edited value."""
    if managed is None:
        return current, None
    normalize = normalizer or (lambda value: value)
    if normalize(current) != normalize(managed):
        return current, None
    return replacement, replacement


def console_log_paths(
    source_dir: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, ...]:
    """Return de-duplicated console-log targets for source and build folders."""
    targets: list[Path] = []
    seen: set[str] = set()
    for raw_root in (source_dir, output_dir):
        if raw_root is None or not str(raw_root).strip():
            continue
        target = Path(raw_root).expanduser() / CONSOLE_LOG_FILENAME
        key = os.path.normcase(os.path.abspath(str(target)))
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return tuple(targets)


def mirror_console_log_fragment(
    targets: Iterable[Path],
    fragment: str,
    transcript: str,
    initialized: set[str],
) -> None:
    """Mirror a GUI log fragment, restoring the full transcript after folder cleanup."""
    for raw_target in targets:
        target = Path(raw_target)
        key = os.path.normcase(os.path.abspath(str(target)))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            reset = key not in initialized or not target.is_file()
            mode = "w" if reset else "a"
            payload = transcript if reset else fragment
            with target.open(mode, encoding="utf-8", newline="") as stream:
                stream.write(payload)
            initialized.add(key)
        except OSError:
            # A diagnostic mirror must never be allowed to break the build itself.
            continue


def generated_mod_folder(output_dir: Path) -> Path | None:
    """Find the generated runtime folder that directly contains Addons and Anims."""
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return None
    if (root / "Addons").is_dir() and (root / "Anims").is_dir():
        return root.resolve()
    try:
        candidates = [
            child.resolve()
            for child in root.iterdir()
            if child.is_dir()
            and (child / "Addons").is_dir()
            and (child / "Anims").is_dir()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.casefold() == "cwr-worldgen":
            return candidate
    candidates.sort(key=lambda path: path.name.casefold())
    return candidates[0]


def _install_frozen_dem_cache(base_dir: Path) -> None:
    """Make dem-stitcher localize remote DEM tiles before merging them."""
    import dem_stitcher

    original_stitch_dem = dem_stitcher.stitch_dem
    if bool(getattr(original_stitch_dem, "_cwr_local_cache", False)):
        return

    def stitch_dem_with_local_cache(*args: Any, **kwargs: Any):
        if kwargs.get("dst_tile_dir") is None:
            dem_name = kwargs.get("dem_name")
            if dem_name is None and len(args) > 1:
                dem_name = args[1]
            cache_dir = base_dir / "source-data" / ".dem-stitcher-cache" / str(dem_name or "dem")
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["dst_tile_dir"] = cache_dir
        return original_stitch_dem(*args, **kwargs)

    stitch_dem_with_local_cache._cwr_local_cache = True  # type: ignore[attr-defined]
    dem_stitcher.stitch_dem = stitch_dem_with_local_cache


def _ensure_cli_streams() -> None:
    """Provide harmless stdio streams in PyInstaller windowed processes."""
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _run_bundled_overture_cli(args: list[str]) -> int:
    """Run CWR's isolated Overture Python-API worker without the upstream CLI."""
    _ensure_cli_streams()
    from .overture import run_overture_worker

    try:
        return run_overture_worker(args)
    except Exception as exc:
        print(f"CWR Worldgen Overture worker failed: {exc}", file=sys.stderr)
        return 1


def _configure_gui(gui: Any, base_dir: Path) -> None:
    """Apply stable storage and world-name synchronization to the GUI module."""
    gui.application_base_dir = lambda: base_dir

    def normalize_path(value: str) -> str:
        if not value:
            return ""
        return os.path.normcase(str(gui.resolve_gui_path(value)))

    def generated_values(display_name: str) -> dict[str, str]:
        slug = gui.slugify_world_name(display_name)
        return {
            "name": "cwr_" + slug if not slug.startswith("cwr_") else slug,
            "output": gui.default_gui_path(Path("build") / slug),
            "source_dir": gui.default_gui_path(Path("source-data") / slug),
        }

    original_class = gui.WorldgenGui

    class SyncedWorldgenGui(original_class):
        """Keep generated paths aligned, mirror logs, and expose the finished runtime."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._auto_world_guard = True
            self._auto_display_name = ""
            self._console_log_text = ""
            self._console_log_initialized: set[str] = set()
            self._managed_world_values: dict[str, str | None] = {
                "name": None,
                "output": None,
                "source_dir": None,
            }
            super().__init__(*args, **kwargs)
            self._clear_remembered_deploy_default()
            self._auto_display_name = str(self.vars["display_name"].get())
            self._arm_auto_world_values()
            self._auto_world_guard = False
            self.open_generated_mod_button = gui.ttk.Button(
                self.page_frames[gui.PROGRESS_STEP_INDEX],
                text="Open generated files folder",
                command=self._open_generated_mod_folder,
            )
            self._update_navigation()

        def _clear_remembered_deploy_default(self) -> None:
            """Start deployment disabled and migrate away any remembered machine path."""
            self.vars["deploy_to_mod_folder"].set(False)
            self.vars["deploy_mod_dir"].set("")
            try:
                state = gui.load_gui_state(self.state_path)
                had_deploy_path = bool(str(state.get("last_deploy_mod_dir", "")).strip())
                deploy_was_enabled = bool(state.get("deploy_to_mod_folder", False))
                if had_deploy_path or deploy_was_enabled:
                    state.pop("state_version", None)
                    state.pop("last_deploy_mod_dir", None)
                    state["deploy_to_mod_folder"] = False
                    gui.save_gui_state(self.state_path, state)
            except OSError:
                pass

        def _console_log_targets(self) -> tuple[Path, ...]:
            vars_map = getattr(self, "vars", {})
            source_var = vars_map.get("source_dir")
            output_var = vars_map.get("output")
            source_text = str(source_var.get()).strip() if source_var is not None else ""
            output_text = str(output_var.get()).strip() if output_var is not None else ""
            source_dir = gui.resolve_gui_path(source_text) if source_text else None
            output_dir = gui.resolve_gui_path(output_text) if output_text else None
            return console_log_paths(source_dir, output_dir)

        def _start_pipeline(self, *args: Any, **kwargs: Any) -> None:
            if self.process is None and not self._pipeline_active:
                self._console_log_text = ""
                self._console_log_initialized.clear()
            super()._start_pipeline(*args, **kwargs)

        def _append_log(self, text: str) -> None:
            super()._append_log(text)
            self._console_log_text += text
            mirror_console_log_fragment(
                self._console_log_targets(),
                text,
                self._console_log_text,
                self._console_log_initialized,
            )

        def _generated_mod_folder(self) -> Path | None:
            output_text = str(self.vars["output"].get()).strip()
            if not output_text:
                return None
            return generated_mod_folder(gui.resolve_gui_path(output_text))

        def _open_generated_mod_folder(self) -> None:
            runtime = self._generated_mod_folder()
            if runtime is None:
                gui.messagebox.showinfo(
                    gui.APP_TITLE,
                    "No completed generated folder containing both Addons and Anims was found.",
                )
                return
            self._open_path(str(runtime))

        def _update_navigation(self) -> None:
            super()._update_navigation()
            button = getattr(self, "open_generated_mod_button", None)
            if button is None:
                return
            show_button = (
                self._operation_success
                and self._pipeline_kind == "build"
                and self.process is None
                and not self._pipeline_active
                and self._generated_mod_folder() is not None
            )
            if show_button:
                if not button.winfo_manager():
                    button.pack(anchor="e", padx=4, pady=(8, 4))
            elif button.winfo_manager():
                button.pack_forget()

        def _arm_auto_world_values(self) -> None:
            display_name = str(self.vars["display_name"].get())
            generated = generated_values(display_name)
            defaults = gui.default_gui_values()

            current_name = str(self.vars["name"].get()).strip()
            if current_name in {str(defaults["name"]), generated["name"]}:
                self._managed_world_values["name"] = current_name
            else:
                self._managed_world_values["name"] = None

            current_output = str(self.vars["output"].get()).strip()
            if normalize_path(current_output) in {
                normalize_path(str(defaults["output"])),
                normalize_path(generated["output"]),
            }:
                self._managed_world_values["output"] = current_output
            else:
                self._managed_world_values["output"] = None

            current_source = str(self.vars["source_dir"].get()).strip()
            if str(self.vars["source_mode"].get()) == "new" and normalize_path(current_source) in {
                normalize_path(str(defaults["source_dir"])),
                normalize_path(generated["source_dir"]),
            }:
                self._managed_world_values["source_dir"] = current_source
            else:
                self._managed_world_values["source_dir"] = None

        def _sync_auto_world_values(self) -> None:
            if self._auto_world_guard or "display_name" not in self.vars:
                return
            display_name = str(self.vars["display_name"].get())
            if display_name == self._auto_display_name:
                return

            generated = generated_values(display_name)
            current_name = str(self.vars["name"].get()).strip()
            new_name, managed_name = managed_replacement(
                current_name,
                self._managed_world_values["name"],
                generated["name"],
            )
            self._managed_world_values["name"] = managed_name
            if new_name != current_name:
                self.vars["name"].set(new_name)

            current_output = str(self.vars["output"].get()).strip()
            new_output, managed_output = managed_replacement(
                current_output,
                self._managed_world_values["output"],
                generated["output"],
                normalizer=normalize_path,
            )
            self._managed_world_values["output"] = managed_output
            if normalize_path(new_output) != normalize_path(current_output):
                self.vars["output"].set(new_output)

            if str(self.vars["source_mode"].get()) == "new":
                current_source = str(self.vars["source_dir"].get()).strip()
                new_source, managed_source = managed_replacement(
                    current_source,
                    self._managed_world_values["source_dir"],
                    generated["source_dir"],
                    normalizer=normalize_path,
                )
                self._managed_world_values["source_dir"] = managed_source
                if normalize_path(new_source) != normalize_path(current_source):
                    self.vars["source_dir"].set(new_source)
                    self._sync_source_paths()
            else:
                self._managed_world_values["source_dir"] = None

            self._auto_display_name = display_name

        def _refresh_views(self) -> None:
            self._sync_auto_world_values()
            super()._refresh_views()

        def _suggest_names(self) -> None:
            super()._suggest_names()
            self._auto_display_name = str(self.vars["display_name"].get())
            generated = generated_values(self._auto_display_name)
            self._managed_world_values["name"] = str(self.vars["name"].get()).strip()
            self._managed_world_values["output"] = str(self.vars["output"].get()).strip()
            self._managed_world_values["source_dir"] = (
                str(self.vars["source_dir"].get()).strip()
                if str(self.vars["source_mode"].get()) == "new"
                else None
            )
            # Keep this reference useful even if a future GUI revision changes
            # how the suggestion button derives its values.
            _ = generated

    gui.WorldgenGui = SyncedWorldgenGui


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    base_dir = storage_base_dir()
    os.environ.setdefault("CWR_WORLDGEN_GUI_STATE", str(base_dir / "config" / "gui-state.json"))
    os.environ.setdefault("CWR_WORLDGEN_RUNTIME_DIR", "CWR-Worldgen")

    if args and args[0] == OVERTURE_CLI_MARKER:
        return _run_bundled_overture_cli(args[1:])

    frozen = bool(getattr(sys, "frozen", False))
    if frozen and args and args[0] == FROZEN_CLI_MARKER:
        _install_frozen_dem_cache(base_dir)

    from . import gui

    _configure_gui(gui, base_dir)
    return gui.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
