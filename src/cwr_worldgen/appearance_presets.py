# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility hooks retained for older v5 bootstrap launchers.

Appearance presets now live directly in :mod:`cwr_worldgen.gui`, so launchers
no longer need to rewrite defaults, combobox values, preset application, or
command construction at runtime.  The older bootstrap also carried deployment
folder behavior; that small compatibility hook remains here so an existing
bootstrap can still install it without duplicating the native preset logic.
"""
from __future__ import annotations

from typing import Any


def install_gui_extensions() -> None:
    """Install only the legacy deployment-folder compatibility behavior."""
    from . import gui

    if bool(getattr(gui, "_cwr_deployment_extensions_v5", False)):
        return

    original_class = gui.WorldgenGui

    class DeploymentCompatibleWorldgenGui(original_class):
        def _browse(self, key: str, kind: str) -> None:
            super()._browse(key, kind)
            if key != "deploy_mod_dir":
                return
            update_controls = getattr(self, "_update_deploy_controls", None)
            if callable(update_controls):
                update_controls()

    gui.WorldgenGui = DeploymentCompatibleWorldgenGui
    gui._cwr_deployment_extensions_v5 = True
    # Older bootstraps may check this historical marker after installation.
    gui._cwr_startup_safe_presets_v5 = True
