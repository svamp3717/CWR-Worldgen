"""Provide writable output streams for windowed PyInstaller P3D tools.

On Windows, PyInstaller --windowed sets sys.stdout/sys.stderr to None.  The
P3D tools report scan warnings on stderr, so route those streams to a log
before importing the GUI and model-processing modules.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


def configure_windowed_logging(tool_name: str) -> None:
    """Log print output for frozen GUI programs without console streams."""
    if not getattr(sys, "frozen", False):
        return
    if sys.stdout is not None and sys.stderr is not None:
        return

    # The one-file extraction directory is temporary and must never hold logs.
    base = os.environ.get("LOCALAPPDATA")
    log_dir = (Path(base) if base else Path.home() / ".cwr-worldgen") / "CWR-Worldgen" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = (log_dir / f"{tool_name}.log").open(
            "a", encoding="utf-8", buffering=1
        )
    except OSError:
        # A read-only home directory should not stop the GUI from starting.
        stream = open(os.devnull, "w", encoding="utf-8")

    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def dismiss_splash(root=None) -> None:
    """Close PyInstaller's bootloader splash once the GUI is ready to draw."""
    if root is not None:
        root.update_idletasks()
    try:
        import pyi_splash
    except ImportError:
        # Running from Python source without a PyInstaller splash.
        return
    try:
        pyi_splash.close()
    except (OSError, RuntimeError):
        # A terminated splash process must not stop the actual tool.
        pass
