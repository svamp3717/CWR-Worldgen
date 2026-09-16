# SPDX-License-Identifier: GPL-3.0-or-later
"""Expose silent work after the stock-road fitter reports completion."""
from __future__ import annotations

import faulthandler
import sys

from . import generator as _generator
from . import playability as _p

_WATCHDOG_SECONDS = 300.0
_COMPLETION_PREFIX = "Stock road fitting complete:"
_INSTALLED = False
_ORIGINAL_FIT = None


def _cancel_watchdog() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except (AttributeError, RuntimeError, ValueError):
        pass


def _arm_watchdog() -> bool:
    _cancel_watchdog()
    try:
        faulthandler.dump_traceback_later(
            _WATCHDOG_SECONDS,
            repeat=False,
            file=sys.stderr,
        )
    except (AttributeError, RuntimeError, ValueError):
        return False
    print(
        "[road-postfit] watchdog armed after stock road fitting; "
        "a Python traceback will be printed after 5 minutes if final road "
        "post-processing does not return",
        flush=True,
    )
    return True


def _fit_with_postfit_watchdog(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    armed = False

    def progress(value: int, message: str) -> None:
        nonlocal armed
        if progress_callback is not None:
            progress_callback(value, message)
        if value >= 100 and str(message).startswith(_COMPLETION_PREFIX):
            armed = _arm_watchdog()

    try:
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress if progress_callback is not None else None,
        )
    finally:
        if armed:
            _cancel_watchdog()


def install_road_postfit_watchdog_policy() -> None:
    """Install outside the complete road wrapper chain."""
    global _INSTALLED, _ORIGINAL_FIT
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit_with_postfit_watchdog
    _generator.fit_road_objects = _fit_with_postfit_watchdog
    _INSTALLED = True
