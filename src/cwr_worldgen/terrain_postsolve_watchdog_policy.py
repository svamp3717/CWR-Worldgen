# SPDX-License-Identifier: GPL-3.0-or-later
"""Dump a traceback when post-solve terrain wrappers stop making progress."""
from __future__ import annotations

import faulthandler
import sys
from typing import Any, Callable

from . import terrain_solver as _terrain

_INSTALLED = False
_ORIGINAL_SOLVE_TERRAIN: Any = None
_WATCHDOG_SECONDS = 300.0


def _cancel_watchdog() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except (AttributeError, RuntimeError):
        pass


def _arm_watchdog() -> None:
    _cancel_watchdog()
    try:
        faulthandler.dump_traceback_later(
            _WATCHDOG_SECONDS,
            repeat=False,
            file=sys.stderr,
        )
    except (AttributeError, RuntimeError, ValueError):
        pass


def _solve_with_postsolve_watchdog(*args: Any, **kwargs: Any) -> Any:
    original_callback: Callable[[int, str], None] | None = kwargs.get(
        "progress_callback"
    )
    armed = False

    def progress(percent: int, stage: str) -> None:
        nonlocal armed
        text = str(stage)
        if text == "Terrain constraint solution ready":
            armed = True
            _arm_watchdog()
            print(
                "[terrain-postsolve] watchdog armed after core terrain solver; "
                f"a Python traceback will be printed after {int(_WATCHDOG_SECONDS)}s "
                "if bridge/causeway/abutment post-processing does not return",
                flush=True,
            )
        elif armed:
            armed = False
            _cancel_watchdog()

        if original_callback is not None:
            original_callback(percent, stage)

    kwargs["progress_callback"] = progress
    try:
        return _ORIGINAL_SOLVE_TERRAIN(*args, **kwargs)
    finally:
        _cancel_watchdog()


def install_terrain_postsolve_watchdog_policy() -> None:
    global _INSTALLED, _ORIGINAL_SOLVE_TERRAIN
    if _INSTALLED:
        return
    _ORIGINAL_SOLVE_TERRAIN = _terrain.solve_terrain_constraints
    _terrain.solve_terrain_constraints = _solve_with_postsolve_watchdog
    _INSTALLED = True
