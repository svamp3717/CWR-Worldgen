# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import sys
import threading
import traceback
from types import TracebackType
from typing import Any


CRASH_LOG_FILENAME = "cwr-worldgen-crash.log"


def _crash_log_path() -> Path:
    configured = os.environ.get("CWR_WORLDGEN_CRASH_LOG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / CRASH_LOG_FILENAME


def _format_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
    *,
    source: str,
) -> str:
    header = (
        "\n"
        "============================================================\n"
        f"CWR-Worldgen Python error ({source})\n"
        "============================================================\n"
    )
    return header + "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )


def _report_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
    *,
    source: str,
) -> None:
    report = _format_exception(
        exc_type,
        exc_value,
        exc_traceback,
        source=source,
    )

    # Console builds and LAUNCH-GUI.cmd both have a real stderr stream.
    try:
        print(report, file=sys.stderr, flush=True)
    except Exception:
        pass

    path = _crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(report)
            if not report.endswith("\n"):
                stream.write("\n")
    except OSError:
        pass


def _install_exception_hooks() -> None:
    def process_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        _report_exception(
            exc_type,
            exc_value,
            exc_traceback,
            source="main thread",
        )

    sys.excepthook = process_hook

    if hasattr(threading, "excepthook"):
        def thread_hook(args: threading.ExceptHookArgs) -> None:
            thread_name = getattr(args.thread, "name", None) or "worker thread"
            _report_exception(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                source=f"thread: {thread_name}",
            )

        threading.excepthook = thread_hook


def _install_tk_exception_hook() -> None:
    # Tkinter normally catches callback exceptions inside mainloop and merely
    # writes them to stderr. Override the standard callback reporter so the
    # traceback is both visible in the console and persisted to the crash log.
    import tkinter as tk

    def report_callback_exception(
        self: Any,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        _report_exception(
            exc_type,
            exc_value,
            exc_traceback,
            source="Tk callback",
        )

    tk.Tk.report_callback_exception = report_callback_exception  # type: ignore[assignment]



def _install_auto_enable_existing_mod_folder() -> None:
    """Enable deployment automatically whenever a remembered mod folder is set.

    The existing map-picker extension deliberately restored the remembered path
    while forcing deploy_to_mod_folder back to False. This wrapper keeps the
    remembered-path behavior but makes the checkbox follow the presence of a
    folder, which is what the GUI label implies to a normal human being.
    """
    from . import gui

    original_class = gui.WorldgenGui
    if bool(getattr(original_class, "_cwr_auto_enable_existing_mod_folder", False)):
        return

    class AutoEnableExistingModFolderGui(original_class):
        _cwr_auto_enable_existing_mod_folder = True

        def _enable_existing_mod_folder_if_set(self) -> None:
            vars_map = getattr(self, "vars", {})
            folder_var = vars_map.get("deploy_mod_dir")
            enabled_var = vars_map.get("deploy_to_mod_folder")
            if folder_var is None or enabled_var is None:
                return

            folder = str(folder_var.get()).strip()
            enabled = bool(folder)
            enabled_var.set(enabled)

            state_path = getattr(self, "state_path", None)
            if state_path is not None:
                try:
                    gui.update_gui_state(
                        state_path,
                        {
                            "last_deploy_mod_dir": folder,
                            "deploy_to_mod_folder": enabled,
                        },
                    )
                except OSError:
                    pass

            update_controls = getattr(self, "_update_deploy_controls", None)
            if callable(update_controls):
                update_controls()

        def _restore_remembered_mod_folder(self) -> None:
            # Preserve the existing remembered-folder restoration first.
            restore = getattr(super(), "_restore_remembered_mod_folder", None)
            if callable(restore):
                restore()
            self._enable_existing_mod_folder_if_set()

        def _browse(self, key: str, kind: str) -> None:
            super()._browse(key, kind)
            if key == "deploy_mod_dir":
                self._enable_existing_mod_folder_if_set()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # gui_entry performs its own startup safety reset after constructing
            # the wrapped class. Running once at idle makes the final checkbox
            # state follow the final restored folder value.
            self.after_idle(self._enable_existing_mod_folder_if_set)

    gui.WorldgenGui = AutoEnableExistingModFolderGui


def main(argv: list[str] | None = None) -> int:
    # Enables traceback output for fatal native signals where Python can still
    # report them, in addition to ordinary Python exception handling below.
    try:
        faulthandler.enable(all_threads=True)
    except (RuntimeError, OSError):
        pass

    _install_exception_hooks()

    try:
        _install_tk_exception_hook()
        # Importing gui_entry first lets the normal package extensions install
        # their GUI subclasses. Then layer the existing-mod-folder behavior on
        # top without replacing the normal GUI implementation.
        from .gui_entry import main as gui_main
        _install_auto_enable_existing_mod_folder()
        return int(gui_main(argv))
    except KeyboardInterrupt:
        print("\nCWR-Worldgen interrupted by user.", file=sys.stderr, flush=True)
        return 130
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr, flush=True)
        return 1
    except BaseException:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        assert exc_type is not None and exc_value is not None
        _report_exception(
            exc_type,
            exc_value,
            exc_traceback,
            source="startup/runtime",
        )
        try:
            print(
                f"Crash log: {_crash_log_path().resolve()}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
