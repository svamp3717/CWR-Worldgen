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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # map_picker_coords may schedule remembered-folder restoration
            # during super().__init__. Queue this afterwards so a restored
            # folder finishes with deployment enabled.
            self.after_idle(self._enable_deployment_when_folder_is_set)

        def _enable_deployment_when_folder_is_set(self) -> None:
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

        def _browse(self, key: str, kind: str) -> None:
            super()._browse(key, kind)
            if key == "deploy_mod_dir":
                self._enable_deployment_when_folder_is_set()

    gui.WorldgenGui = DeploymentCompatibleWorldgenGui
    gui._cwr_deployment_extensions_v5 = True
    # Older bootstraps may check this historical marker after installation.
    gui._cwr_startup_safe_presets_v5 = True
