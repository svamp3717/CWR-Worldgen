# SPDX-License-Identifier: GPL-3.0-or-later
"""PyInstaller GUI entry point with multiprocessing dispatch before package import.

PyInstaller starts ``spawn`` workers by re-executing the frozen application.
The worker command line must be intercepted before importing ``cwr_worldgen``;
otherwise package-level policy installation runs during the bootstrap pass and
can leave spawned procedural-asset workers with a different runtime patch stack
than normal Python workers.
"""
from __future__ import annotations

import multiprocessing as _multiprocessing

# PyInstaller replaces freeze_support() with worker-aware dispatch logic. In a
# normal Python interpreter this is a harmless no-op, which also keeps this
# script usable as the build-analysis entry point on every supported platform.
_multiprocessing.freeze_support()

# Frozen distributions ship editable JSON in a sibling config directory. Mirror
# it into PyInstaller's extracted package data before importing cwr_worldgen so
# every existing catalogue/style loader sees the user's files transparently.
from external_json_runtime import overlay_external_json as _overlay_external_json

_overlay_external_json()

# The frozen entry must use the guarded launcher. Besides reporting uncaught
# exceptions, it provides a persistent log stream for windowed builds where
# PyInstaller intentionally supplies no console stdout/stderr handles.
from cwr_worldgen.debug_entry import main


if __name__ == "__main__":
    raise SystemExit(main())
