# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional GUI cleanup of reproducible build intermediates."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

CLEANUP_BUILD_AFTER_BUILD = "cleanup_build_after_build"
CLEANUP_ONLY_MARKER = "--cleanup-build-files"
CLEANUP_DESCRIPTION = "Deleting temporary build files"
CLEANUP_DIR_NAMES = ("source", "normalized")
_INSTALLED = False


def cleanup_build_outputs(build_dir: str | Path) -> tuple[Path, ...]:
    """Delete only the known reproducible build directories."""
    root = Path(build_dir).expanduser().resolve()
    removed: list[Path] = []
    for name in CLEANUP_DIR_NAMES:
        target = root / name
        try:
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            else:
                continue
            removed.append(target)
        except OSError as exc:
            print(f"Build cleanup warning: could not remove {target}: {exc}")
    return tuple(removed)


def _run_cleanup_only(args: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, required=True)
    options = parser.parse_args(args)
    removed = cleanup_build_outputs(options.build_dir)
    if removed:
        print("Build cleanup: removed " + ", ".join(path.name for path in removed))
    else:
        print("Build cleanup: no temporary build directories remained")
    return 0


def postbuild_cleanup_command(
    build_dir: str | Path,
    *,
    frozen: bool | None = None,
    executable: str | None = None,
) -> list[str]:
    """Use the existing GUI child launcher for the cleanup-only post-build job."""
    from . import gui_entry

    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    launcher = executable or sys.executable
    command = [launcher, gui_entry.ROAD_INSPECTOR_CLI_MARKER] if frozen else [
        launcher,
        "-c",
        gui_entry.ROAD_INSPECTOR_CHILD_CODE,
    ]
    command.extend((CLEANUP_ONLY_MARKER, "--build-dir", str(build_dir)))
    return command


def install_postbuild_cleanup() -> None:
    """Add a default-on cleanup checkbox and a final cleanup pipeline job."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import gui_entry

    original_runner = gui_entry._run_road_inspector_postbuild

    def postbuild_runner(args: list[str]) -> int:
        if args and args[0] == CLEANUP_ONLY_MARKER:
            return _run_cleanup_only(args[1:])
        return original_runner(args)

    gui_entry._run_road_inspector_postbuild = postbuild_runner

    original_configure = gui_entry._configure_gui

    def configure_gui(gui: Any, base_dir: Path) -> None:
        original_configure(gui, base_dir)

        original_defaults = gui.default_gui_values

        def default_gui_values() -> dict[str, object]:
            values = original_defaults()
            values[CLEANUP_BUILD_AFTER_BUILD] = True
            return values

        gui.default_gui_values = default_gui_values
        original_class = gui.WorldgenGui

        class CleanupWorldgenGui(original_class):
            def _build_world_page(self) -> None:
                super()._build_world_page()
                page = self.page_frames[-1]
                children = page.winfo_children()
                body = getattr(children[0], "body", None) if children else None
                if body is None:
                    return
                checks = next(
                    (
                        child for child in body.winfo_children()
                        if str(child.cget("text")) == "Post-build checks"
                    ),
                    None,
                )
                if checks is None:
                    return
                inspector = next(
                    (
                        child for child in checks.winfo_children()
                        if str(child.cget("text")) == "Run Road Inspector after a successful build"
                    ),
                    None,
                )
                cleanup = gui.ttk.Checkbutton(
                    checks,
                    text="Delete temporary build files after a successful build",
                    variable=self._var(CLEANUP_BUILD_AFTER_BUILD, True, boolean=True),
                )
                if inspector is None:
                    cleanup.pack(anchor="w")
                else:
                    cleanup.pack(anchor="w", before=inspector)

            def _start_pipeline(
                self,
                jobs: list[tuple[list[str], str]],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                cleanup_var = self.vars.get(CLEANUP_BUILD_AFTER_BUILD)
                cleanup_enabled = (
                    kwargs.get("kind") == "build"
                    and cleanup_var is not None
                    and bool(cleanup_var.get())
                )
                if not cleanup_enabled:
                    super()._start_pipeline(jobs, *args, **kwargs)
                    return

                jobs = list(jobs)
                inspector_var = self.vars.get("run_road_inspector_after_build")
                inspector_enabled = inspector_var is not None and bool(inspector_var.get())
                output_text = str(self.vars["output"].get()).strip()
                world_name = str(self.vars["name"].get()).strip()

                # The parent GUI normally appends Road Inspector itself. When
                # cleanup is enabled, append it here first so cleanup is always
                # the final job and cannot remove intermediates before checks.
                if inspector_enabled and output_text and world_name:
                    already_added = any(
                        gui_entry.ROAD_INSPECTOR_CLI_MARKER in command
                        or gui_entry.ROAD_INSPECTOR_CHILD_CODE in command
                        for command, _description in jobs
                    )
                    if not already_added:
                        jobs.append((
                            gui_entry.road_inspector_postbuild_command(
                                gui.resolve_gui_path(output_text), world_name
                            ),
                            "Running Road Inspector",
                        ))

                if output_text and not any(
                    description == CLEANUP_DESCRIPTION for _command, description in jobs
                ):
                    jobs.append((
                        postbuild_cleanup_command(gui.resolve_gui_path(output_text)),
                        CLEANUP_DESCRIPTION,
                    ))

                if inspector_var is None or not inspector_enabled:
                    super()._start_pipeline(jobs, *args, **kwargs)
                    return

                # Prevent the parent wrapper from appending a second inspector
                # after our final cleanup job. Restore the user's checkbox value
                # immediately after the pipeline has been queued.
                inspector_var.set(False)
                try:
                    super()._start_pipeline(jobs, *args, **kwargs)
                finally:
                    inspector_var.set(True)

        gui.WorldgenGui = CleanupWorldgenGui

    gui_entry._configure_gui = configure_gui
    _INSTALLED = True
