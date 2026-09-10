# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent game-installation selection for the desktop GUI.

The asset utility already allowed a game directory to be added for one session,
but generated runway textures now benefit from having the original game artwork
available on every build.  Keep one dedicated game-installation folder in the
per-user GUI state, restore it on startup, and add it to the build's asset roots
for every appearance preset.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Mapping


GAME_FOLDER_STATE_KEY = "last_game_folder"
GAME_FOLDER_VALUE_KEY = "game_folder"
_INSTALLED = False


def remembered_game_folder(state: Mapping[str, object]) -> str:
    """Return the remembered existing game folder, or an empty string."""
    text = str(state.get(GAME_FOLDER_STATE_KEY, "")).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_dir():
        return ""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def merge_game_folder_asset_roots(
    roots: list[str] | tuple[str, ...], game_folder: str
) -> list[str]:
    """Add the dedicated game folder once, preserving existing root order."""
    result = [str(value) for value in roots if str(value).strip()]
    text = str(game_folder or "").strip()
    if not text:
        return result
    path = Path(text).expanduser()
    try:
        normalized = str(path.resolve())
    except OSError:
        normalized = str(path)
    folded = normalized.casefold()
    if all(str(value).casefold() != folded for value in result):
        result.insert(0, normalized)
    return result


def _persist_game_folder(gui: Any, instance: Any) -> None:
    variable = getattr(instance, "vars", {}).get(GAME_FOLDER_VALUE_KEY)
    if variable is None:
        return
    text = str(variable.get()).strip()
    if text:
        path = Path(text).expanduser()
        try:
            text = str(path.resolve())
        except OSError:
            text = str(path)
    try:
        gui.update_gui_state(instance.state_path, {GAME_FOLDER_STATE_KEY: text})
    except OSError:
        return


def _install_into_gui(gui: Any) -> None:
    if getattr(gui, "_cwr_game_folder_policy", False):
        return

    original_defaults = gui.default_gui_values
    original_recent = gui.defaults_with_recent_source
    original_build_world_page = gui.WorldgenGui._build_world_page
    original_collect = gui.WorldgenGui._collect_build_values
    original_browse = gui.WorldgenGui._browse
    original_add_asset_folder = gui.WorldgenGui._add_asset_folder
    original_candidates = gui.WorldgenGui._candidate_game_roots
    original_close = gui.WorldgenGui._on_close
    original_load_profile = gui.WorldgenGui._load_profile

    @wraps(original_defaults)
    def default_gui_values() -> dict[str, object]:
        values = original_defaults()
        values.setdefault(GAME_FOLDER_VALUE_KEY, "")
        return values

    @wraps(original_recent)
    def defaults_with_recent_source(
        defaults: dict[str, object], state: Mapping[str, object]
    ) -> dict[str, object]:
        values = original_recent(defaults, state)
        game_folder = remembered_game_folder(state)
        if game_folder:
            values[GAME_FOLDER_VALUE_KEY] = game_folder
        return values

    @wraps(original_build_world_page)
    def build_world_page(self) -> None:
        original_build_world_page(self)
        # The original page is a ScrollFrame containing its public ``body``.
        # Add a separate section below World identity instead of coupling this
        # small persistent setting to that function's internal grid row numbers.
        page = self.page_frames[-1]
        scroll = next(
            (child for child in page.winfo_children() if hasattr(child, "body")),
            None,
        )
        if scroll is None:
            return
        body = scroll.body
        box = gui.ttk.LabelFrame(
            body,
            text="Game installation",
            style="Section.TLabelframe",
            padding=12,
        )
        box.pack(fill="x", pady=(12, 0))
        entry = self._entry_row(
            box,
            0,
            "Game installation folder",
            GAME_FOLDER_VALUE_KEY,
            browse="directory",
        )
        gui.ttk.Label(
            box,
            text=(
                "Remembered between restarts. Used as an asset root for every preset so "
                "stock terrain textures such as O.pbo/Eden.pbo can be read when available."
            ),
            style="Hint.TLabel",
            wraplength=700,
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, 4))
        entry.bind("<FocusOut>", lambda _event: _persist_game_folder(gui, self))

    @wraps(original_collect)
    def collect_build_values(self) -> dict[str, object]:
        values = original_collect(self)
        variable = self.vars.get(GAME_FOLDER_VALUE_KEY)
        game_folder = str(variable.get()).strip() if variable is not None else ""
        values["asset_roots"] = merge_game_folder_asset_roots(
            list(values.get("asset_roots", []) or []), game_folder
        )
        if game_folder:
            try:
                values[GAME_FOLDER_VALUE_KEY] = str(Path(game_folder).expanduser().resolve())
            except OSError:
                values[GAME_FOLDER_VALUE_KEY] = game_folder
        return values

    @wraps(original_browse)
    def browse(self, key: str, kind: str) -> None:
        before = (
            str(self.vars[key].get()).strip()
            if key in self.vars else ""
        )
        original_browse(self, key, kind)
        if key == GAME_FOLDER_VALUE_KEY:
            after = str(self.vars[key].get()).strip()
            if after != before:
                _persist_game_folder(gui, self)

    @wraps(original_add_asset_folder)
    def add_asset_folder(self) -> None:
        before = tuple(self.asset_roots)
        original_add_asset_folder(self)
        added = [value for value in self.asset_roots if value not in before]
        if added and GAME_FOLDER_VALUE_KEY in self.vars:
            # Keep the long-standing Asset tools button compatible with the new
            # dedicated field. The most recently selected directory becomes the
            # remembered game installation as well as an optional asset root.
            self.vars[GAME_FOLDER_VALUE_KEY].set(added[-1])
            _persist_game_folder(gui, self)

    @wraps(original_candidates)
    def candidate_game_roots(self) -> list[Path]:
        values: list[Path] = []
        variable = self.vars.get(GAME_FOLDER_VALUE_KEY)
        if variable is not None:
            text = str(variable.get()).strip()
            if text:
                values.append(Path(text).expanduser())
        values.extend(original_candidates(self))
        unique: list[Path] = []
        seen: set[str] = set()
        for value in values:
            key = str(value).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        return unique

    @wraps(original_close)
    def on_close(self) -> None:
        _persist_game_folder(gui, self)
        original_close(self)

    @wraps(original_load_profile)
    def load_profile(self) -> None:
        original_load_profile(self)
        # Profiles may carry a game folder in their normal values. Persist it so
        # reopening the application does not silently forget that selection.
        _persist_game_folder(gui, self)

    gui.default_gui_values = default_gui_values
    gui.defaults_with_recent_source = defaults_with_recent_source
    gui.WorldgenGui._build_world_page = build_world_page
    gui.WorldgenGui._collect_build_values = collect_build_values
    gui.WorldgenGui._browse = browse
    gui.WorldgenGui._add_asset_folder = add_asset_folder
    gui.WorldgenGui._candidate_game_roots = candidate_game_roots
    gui.WorldgenGui._on_close = on_close
    gui.WorldgenGui._load_profile = load_profile
    gui._cwr_game_folder_policy = True


def install_game_folder_gui_policy() -> None:
    """Patch the official GUI configuration hook with persistent game-folder UI."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import gui_entry

    original_configure = gui_entry._configure_gui

    @wraps(original_configure)
    def configure_gui(gui: Any, base_dir: Path) -> None:
        original_configure(gui, base_dir)
        _install_into_gui(gui)

    gui_entry._configure_gui = configure_gui
    _INSTALLED = True
