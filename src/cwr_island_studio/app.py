# SPDX-License-Identifier: GPL-3.0-or-later
"""Island Studio: responsive terrain and object editing for large RVW4 worlds."""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, Future
import threading
from pathlib import Path
import math
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

from PIL import Image, ImageTk

from .core import IslandDocument, MapObject
from .oprw import open_island_document
from .world_metadata import load_loose_world_metadata, MapLocation
from .object_layers import is_forest_model, object_marker_mode
from .cursor_info import cursor_readout
from .roads import is_road_model, plan_straight_road, road_segment
from .navigation import center_on_position, pan_with_arrow, parse_game_position, zoom_around
from .model_bounds import ModelBounds, load_model_bounds, footprint_corners, footprint_contains
from .pbo_textures import PboTextures
from .pbo_worlds import PboWorldArchive
from .asset_sources import append_unique, remove_selected
from .textures import clear_archive_texture_cache, clear_loose_texture_cache, render_real_textures
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cwr-worldgen")
except PackageNotFoundError:
    __version__ = "dev"


def world_source_kind(path: Path | str) -> str:
    """Identify a loose WRP or PBO/IFA container without opening its contents."""
    name = str(path).casefold()
    if name.endswith(".wrp"):
        return "wrp"
    if name.endswith((".pbo", ".ifa", ".pbo.zst", ".ifa.zst")):
        return "archive"
    raise ValueError("Choose a .wrp, .pbo, .ifa, .pbo.zst or .ifa.zst file")


class _DaemonExecutor:
    """Short-lived GUI tasks must not keep the process alive after Quit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._futures: set[Future] = set()

    def submit(self, function: Callable[[], object]) -> Future:
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Background executor has shut down")
            self._futures.add(future)
        future.add_done_callback(self._forget)

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = function()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=run, name="island-studio-worker", daemon=True).start()
        return future

    def _forget(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = False) -> None:
        with self._lock:
            self._closed = True
            pending = tuple(self._futures)
            self._futures.clear()
        if cancel_futures:
            for future in pending:
                future.cancel()
        # Active threads are daemon threads; Quit never joins running I/O.


class IslandStudio:
    """Viewport-centred editor with bounded rendering and background I/O."""

    def __init__(self, root: tk.Tk, initial_path: Path | None = None) -> None:
        self.root = root
        self.root.title(f"CWR Island Studio {__version__}")
        self.root.geometry("1280x820")
        self.root.minsize(870, 560)
        self.root.configure(background="#18251d")
        self.document: IslandDocument | None = None
        self._executor = _DaemonExecutor()
        self._archive_cancel = threading.Event()
        self._closing = False
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._index_future: Future | None = None
        self._index_cancel: threading.Event | None = None
        self._model_future: Future | None = None
        self._model_bounds: dict[str, ModelBounds | None] = {}
        self._model_stamp = 0
        self._waiting_pbos: PboTextures | None = None
        self._busy = False
        self._edit_buttons: list[ttk.Button] = []
        self._edit_entries: list[ttk.Entry] = []

        self._render_pending = False
        self._image: ImageTk.PhotoImage | None = None
        self._visible: tuple[MapObject, ...] = ()
        self.selected: int | None = None
        self._map_locations: tuple[MapLocation, ...] = ()
        self.view_x = 0.0
        self.view_z = 0.0
        self.zoom = 1.0
        self._pan_origin: tuple[int, int, float, float] | None = None
        self._last_brush: tuple[int, int] | None = None
        self._hover_screen: tuple[float, float] | None = None
        self._road_anchor: tuple[float, float] | None = None
        self._road_cursor: tuple[float, float] | None = None

        self.cell_size = tk.StringVar(value="50")
        self.auto_grid = tk.BooleanVar(value=True)
        self.show_locations = tk.BooleanVar(value=True)
        self.height_scale = tk.StringVar(value="1")
        self.layer = tk.StringVar(value="Elevation")
        self.tool = tk.StringVar(value="Select")
        # View a newly opened loose world safely until editing is requested.
        self.readonly_checked = tk.BooleanVar(value=True)
        self.show_objects = tk.BooleanVar(value=True)
        self.show_model_footprints = tk.BooleanVar(value=False)
        self.show_forests = tk.BooleanVar(value=False)
        self.show_roads = tk.BooleanVar(value=True)
        self.water_level = tk.StringVar(value="0")
        self.asset_roots: tuple[Path, ...] = ()
        self.selected_pbos: tuple[Path, ...] = ()
        self.pbo_textures: PboTextures | None = None
        self._loading_pbos = False
        self.texture_folder_label = tk.StringVar(value="No texture/model folders selected")
        self.pbo_label = tk.StringVar(value="No PBO archives selected")
        self.road_model = tk.StringVar(value=r"o\road\sil25.p3d")
        self.brush_radius = tk.StringVar(value="75")
        self.brush_strength = tk.StringVar(value="1")
        self.paint_slot = tk.StringVar(value="0")
        self.model_path = tk.StringVar(value=r"o\road\sil25.p3d")
        self.obj_x = tk.StringVar(value="0")
        self.obj_y = tk.StringVar(value="0")
        self.obj_z = tk.StringVar(value="0")
        self.obj_heading = tk.StringVar(value="0")
        self.metadata = tk.StringVar(value="Open a WRP or an island from a PBO to start")
        self.status = tk.StringVar(value="Ready. The source file will not be modified.")
        self.cursor_info = tk.StringVar(value="")
        self._last_game_position: str | None = None
        self._build()
        self._update_edit_access()
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)
        self.root.after(120, self._poll)
        if initial_path:
            self.root.after(250, lambda: self.open_path(initial_path))

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(9, 7))
        header.pack(fill="x")
        ttk.Button(header, text="Open World", command=self._open_dialog).pack(side="left", padx=3)
        self.save_button = ttk.Button(header, text="Save As", command=self._save_dialog)
        self.save_button.pack(side="left", padx=3)
        self.readonly_toggle = ttk.Checkbutton(
            header, text="Read-only", variable=self.readonly_checked,
            command=self._toggle_read_only,
        )
        self.readonly_toggle.pack(side="left", padx=(8, 4))
        ttk.Button(header, text="Fit map", command=self._fit).pack(side="left", padx=3)
        ttk.Button(header, text="Go to Position", command=self._go_to_position_dialog).pack(
            side="left", padx=(7, 3),
        )
        ttk.Button(header, text="−", width=3, command=lambda: self._zoom_button(0.8)).pack(side="left", padx=(12, 2))
        ttk.Button(header, text="+", width=3, command=lambda: self._zoom_button(1.25)).pack(side="left", padx=2)
        ttk.Label(header, text="Grid fallback (m)").pack(side="left", padx=(12, 2))
        ttk.Entry(header, textvariable=self.cell_size, width=6).pack(side="left")
        ttk.Checkbutton(
            header, text="Auto LandGrid", variable=self.auto_grid,
        ).pack(side="left", padx=(4, 0))
        ttk.Label(header, text="Height scale").pack(side="left", padx=(13, 2))
        ttk.Entry(header, textvariable=self.height_scale, width=6).pack(side="left")
        ttk.Label(header, textvariable=self.metadata).pack(side="right", padx=8)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(body)
        sidebar = ttk.Frame(body, width=280)
        body.add(canvas_frame, weight=5)
        body.add(sidebar, weight=0)
        side_canvas = tk.Canvas(sidebar, borderwidth=0, highlightthickness=0, width=266)
        side_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(side_canvas, padding=12)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        self.canvas = tk.Canvas(
            canvas_frame, background="#23352b",
            borderwidth=0, highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._queue_draw())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e, 1.2))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e, 1 / 1.2))
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonPress-1>", self._left_click)
        self.canvas.bind("<B1-Motion>", self._left_drag)
        self.canvas.bind("<Motion>", self._mouse_move)
        self.canvas.bind("<Leave>", self._mouse_leave)
        self.canvas.bind("<Enter>", lambda event: self._update_hover(event.x, event.y))
        self.canvas.bind("<Escape>", lambda _e: self._cancel_road())

        ttk.Label(side, text="View", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Combobox(
            side, textvariable=self.layer,
            state="readonly", values=("Elevation", "Texture slots", "Actual textures", "Road preview"),
        ).pack(fill="x", pady=(4, 7))
        self.layer.trace_add("write", lambda *_: self._layer_changed())
        ttk.Checkbutton(
            side, text="Show world objects", variable=self.show_objects,
            command=self._queue_draw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            side, text="Show objects at model size", variable=self.show_model_footprints,
            command=self._queue_draw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            side, text="Show forests (green overlay)", variable=self.show_forests,
            command=self._queue_draw,
        ).pack(anchor="w")
        ttk.Label(
            side, text="Footprints work without object dots. Forests show "
                       "tree and forest model placements from the WRP. "
                       "Missing model sizes stay as dots only when dots or "
                       "forests are enabled.", wraplength=230,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(
            side, text="Show roads as lines", variable=self.show_roads,
            command=self._queue_draw,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(
            side, text="Show place names (config.cpp)", variable=self.show_locations,
            command=self._queue_draw,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(side, text="Water level, metres").pack(anchor="w")
        water_entry = ttk.Entry(side, textvariable=self.water_level, width=8)
        water_entry.pack(fill="x", pady=(0, 4))
        water_entry.bind("<Return>", lambda _e: self._queue_draw())
        water_entry.bind("<FocusOut>", lambda _e: self._queue_draw())
        ttk.Label(side, text="Texture / model folders").pack(anchor="w", pady=(7, 2))
        self.folder_list = tk.Listbox(
            side, height=4, selectmode="extended", exportselection=False,
        )
        self.folder_list.pack(fill="x", pady=(0, 3))
        folder_buttons = ttk.Frame(side)
        folder_buttons.pack(fill="x")
        ttk.Button(folder_buttons, text="Add folder…", command=self._choose_texture_root).pack(
            side="left", fill="x", expand=True,
        )
        ttk.Button(folder_buttons, text="Remove", command=self._remove_folders).pack(
            side="left", padx=(4, 0),
        )
        ttk.Button(folder_buttons, text="Clear", command=self._clear_folders).pack(
            side="left", padx=(4, 0),
        )
        ttk.Label(side, textvariable=self.texture_folder_label, wraplength=230).pack(
            anchor="w", pady=(3, 6),
        )
        ttk.Label(side, text="PBO archives (textures and models)").pack(anchor="w", pady=(3, 2))
        self.pbo_list = tk.Listbox(
            side, height=4, selectmode="extended", exportselection=False,
        )
        self.pbo_list.pack(fill="x", pady=(0, 3))
        archive_buttons = ttk.Frame(side)
        archive_buttons.pack(fill="x")
        ttk.Button(archive_buttons, text="Add PBOs…", command=self._choose_pbos).pack(
            side="left", fill="x", expand=True,
        )
        ttk.Button(archive_buttons, text="Remove", command=self._remove_pbos).pack(
            side="left", padx=(4, 0),
        )
        ttk.Button(archive_buttons, text="Clear", command=self._clear_pbos).pack(
            side="left", padx=(4, 0),
        )
        ttk.Label(side, textvariable=self.pbo_label, wraplength=230).pack(
            anchor="w", pady=(3, 8),
        )
        ttk.Separator(side).pack(fill="x", pady=(4, 9))

        ttk.Label(side, text="Edit tool", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.edit_tool_combo = ttk.Combobox(
            side, textvariable=self.tool, state="readonly",
            values=("Select", "Raise terrain", "Lower terrain", "Paint texture", "Draw road line", "Erase road"),
        )
        self.edit_tool_combo.pack(fill="x", pady=(4, 7))
        for label, var in (
            ("Brush radius, metres", self.brush_radius),
            ("Elevation step, metres", self.brush_strength),
            ("Texture slot (0–511)", self.paint_slot),
        ):
            ttk.Label(side, text=label).pack(anchor="w", pady=(3, 0))
            entry = ttk.Entry(side, textvariable=var)
            entry.pack(fill="x", pady=(0, 4))
            self._edit_entries.append(entry)

        ttk.Label(side, text="Straight road model").pack(anchor="w", pady=(8, 0))
        self.edit_road_combo = ttk.Combobox(
            side, textvariable=self.road_model,
            values=(r"o\road\sil25.p3d", r"o\road\asf25.p3d",
                    r"o\road\ces25.p3d", r"o\road\kos25.p3d",
                    r"o\road\sil12.p3d", r"o\road\sil6.p3d"),
        )
        self.edit_road_combo.pack(fill="x")
        ttk.Label(side, text="Road: click start/end; Esc cancels", wraplength=230).pack(anchor="w", pady=(2, 0))
        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=13)
        ttk.Label(side, text="Selected object", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        for label, var in (
            ("Model path", self.model_path),
            ("X (metres)", self.obj_x),
            ("Height (metres)", self.obj_y),
            ("Z (metres)", self.obj_z),
            ("Heading (degrees)", self.obj_heading),
        ):
            ttk.Label(side, text=label).pack(anchor="w", pady=(4, 0))
            entry = ttk.Entry(side, textvariable=var)
            entry.pack(fill="x", pady=(0, 2))
            self._edit_entries.append(entry)
        for label, callback, pad in (
            ("Apply object changes", self._apply_object, (10, 3)),
            ("Add object at map centre", self._add_object, (3, 3)),
            ("Remove selected object", self._remove_object, (3, 3)),
        ):
            button = ttk.Button(side, text=label, command=callback)
            button.pack(fill="x", pady=pad)
            self._edit_buttons.append(button)

        footer = ttk.Frame(self.root, padding=(8, 5))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Label(footer, textvariable=self.cursor_info, anchor="e").pack(side="right", padx=(12, 2))
        ttk.Label(
            footer, text="Arrows: pan   Shift+arrows: faster   + / −: zoom   Right-drag: pan",
        ).pack(side="bottom", anchor="e")
        self.root.bind("<Control-o>", lambda _e: self._open_dialog())
        self.root.bind("<Control-Shift-S>", lambda _e: self._save_dialog())
        self.root.bind("<Control-s>", lambda _e: self._save_dialog())
        self.root.bind("<Control-g>", lambda _e: self._go_to_position_dialog())
        # Keep text entry usable for negative coordinates and numeric settings.
        for sequence, factor in (
            ("<KeyPress-plus>", 1.25), ("<KeyPress-equal>", 1.25),
            ("<KeyPress-minus>", 0.8), ("<KeyPress-underscore>", 0.8),
            ("<KP_Add>", 1.25), ("<KP_Subtract>", 0.8),
            ("<Control-plus>", 1.25), ("<Control-equal>", 1.25),
            ("<Control-minus>", 0.8), ("<Control-KP_Add>", 1.25),
            ("<Control-KP_Subtract>", 0.8),
        ):
            self.root.bind(sequence, lambda event, amount=factor: self._zoom_hotkey(event, amount))
        for direction in ("Left", "Right", "Up", "Down"):
            self.root.bind(
                f"<{direction}>",
                lambda event, key=direction: self._pan_hotkey(event, key),
            )

    def _go_to_position_dialog(self) -> None:
        """Navigate to an in-game position in both loose and archived maps."""
        if self.document is None or self._busy:
            return
        if self._last_game_position is None:
            left, top, right, bottom = self._extent()
            initial = f"[{(left + right) / 2:.1f}, {(top + bottom) / 2:.1f}, 0]"
        else:
            initial = self._last_game_position
        pasted = simpledialog.askstring(
            "Go to Position",
            "Paste coordinates from the game console:\n"
            "[X, Y, altitude] or X, Y\n"
            "Game Y is north/south; altitude is ignored in this 2D view.",
            initialvalue=initial,
            parent=self.root,
        )
        if pasted is not None:
            self._go_to_position(pasted)

    def _go_to_position(self, value: str) -> None:
        doc = self.document
        if doc is None or self._busy:
            return
        try:
            game_x, game_y, _altitude = parse_game_position(value)
            if not (0 <= game_x < doc.world_width and
                    0 <= game_y < doc.world_height):
                raise ValueError(
                    f"Position is outside the island (X 0–{doc.world_width:g}, "
                    f"Y 0–{doc.world_height:g} metres)"
                )
            new_x, new_z = center_on_position(
                game_x, game_y, max(1, self.canvas.winfo_width()),
                max(1, self.canvas.winfo_height()), self.zoom,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid game position", str(exc))
            return
        self.view_x, self.view_z = new_x, new_z
        self._last_game_position = value.strip()
        self._refresh_hover()
        self._queue_draw()
        self.status.set(
            f"Map centered at game X {game_x:,.2f}, Y {game_y:,.2f} m"
        )

    def _num(self, value: tk.StringVar, name: str) -> float:
        try:
            number = float(value.get())
            if not math.isfinite(number):
                raise ValueError()
            return number
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite number") from exc

    def _retire_current_world(self) -> None:
        """Detach the old map before loading another one.

        The old object-index worker may still hold mmap views. Cancelling
        indexing and retiring its document after that worker finishes prevents
        Windows file locks and races with the Tk drawing thread. Cleanup must
        never block the UI.
        """
        previous = self.document
        if previous is None:
            return
        index_future = self._index_future
        if self._index_cancel is not None:
            self._index_cancel.set()
        self._index_future = None
        self._index_cancel = None
        self.document = None
        self._map_locations = ()
        self._reset_model_bounds()
        self.selected = None
        self._visible = ()
        self._image = None
        self.canvas.delete("all")
        self.metadata.set("Opening world…")
        self._update_edit_access()

        def release(_future: Future | None = None) -> None:
            # A Future callback is sometimes invoked immediately on the
            # caller's thread if it just completed. Always close elsewhere.
            def close_document() -> None:
                try:
                    previous.close()
                except Exception as exc:
                    self._messages.put(("cleanup_error", str(exc)))

            threading.Thread(
                target=close_document,
                name="island-studio-old-world-cleanup",
                daemon=True,
            ).start()

        if index_future is not None and not index_future.done():
            index_future.add_done_callback(release)
        else:
            release()

    def _open_pbo_path(self, path: Path) -> None:
        if self._busy:
            return
        if self.document is not None and self.document.changed:
            if not messagebox.askyesno(
                "Discard pending edits?",
                "Opening another world will discard unsaved edits. Continue?",
            ):
                return
        try:
            cell_size = self._num(self.cell_size, "Grid metres")
            height_scale = self._num(self.height_scale, "Height scale")
            if cell_size <= 0 or height_scale <= 0:
                raise ValueError("Grid metres and height scale must be positive")
        except ValueError as exc:
            messagebox.showerror("Invalid map settings", str(exc))
            return
        self._busy = True
        if self.document is not None:
            self._retire_current_world()
        self._archive_cancel = threading.Event()
        cancel_event = self._archive_cancel
        self.status.set(f"Indexing WRP entries in {Path(path).name}…")

        def inspect() -> None:
            try:
                archive = PboWorldArchive(
                    Path(path), cancel_event=cancel_event,
                    progress=lambda message: self._messages.put(("archive_progress", message)),
                )
                if cancel_event.is_set():
                    archive.close()
                    return
                self._messages.put(("archive_catalog", (archive, cell_size, height_scale)))
            except CancelledError:
                pass
            except Exception as exc:
                if not cancel_event.is_set():
                    self._messages.put(("error", str(exc)))

        self._executor.submit(inspect)

    def _select_archive_member(
        self, archive: PboWorldArchive, cell_size: float, height_scale: float,
    ) -> None:
        """Choose a map without nesting a blocking Tk wait_window event loop."""
        names = sorted(archive.members, key=str.casefold)
        if len(names) == 1:
            self._open_archive_member(archive, names[0], cell_size, height_scale)
            return
        self.status.set(
            f"Found {len(names):,} WRP entries in {archive.path.name}. "
            "Choose an island from the separate window."
        )
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose a WRP from the PBO")
        dialog.geometry("550x380")
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"{len(names)} WRP files found. Select one to view read-only:").pack(anchor="w")
        selection = tk.Listbox(frame, exportselection=False)
        selection.pack(fill="both", expand=True, pady=8)
        for name in names:
            selection.insert("end", name)
        selection.selection_set(0)
        selected = False

        def close_dialog() -> None:
            dialog.destroy()
            if not selected:
                archive.close()
                if not self._closing:
                    self._busy = False
                    self.status.set("Archive opening cancelled")

        def accept(_event=None) -> None:
            nonlocal selected
            indices = selection.curselection()
            if not indices:
                return
            selected = True
            member = names[int(indices[0])]
            dialog.destroy()
            if self._closing:
                archive.close()
                return
            self._open_archive_member(archive, member, cell_size, height_scale)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Open read-only", command=accept).pack(side="right", padx=4)
        ttk.Button(buttons, text="Cancel", command=close_dialog).pack(side="right")
        selection.bind("<Double-Button-1>", accept)
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.lift()
        selection.focus_set()

    def _open_archive_member(
        self, archive: PboWorldArchive, member: str,
        cell_size: float, height_scale: float,
    ) -> None:
        self.status.set(f"Loading read-only island {member}…")
        # Tk variables must be read on the UI thread, not inside a worker.
        auto_grid = bool(self.auto_grid.get()) if hasattr(self, "auto_grid") else True

        def load() -> None:
            try:
                doc = archive.open_world(
                    member, cell_size=cell_size, height_scale=height_scale,
                    auto_grid=auto_grid,
                )
                if self._archive_cancel.is_set():
                    doc.close()
                    return
                self._messages.put(("open", doc))
            except CancelledError:
                pass
            except Exception as exc:
                if not self._archive_cancel.is_set():
                    self._messages.put(("error", str(exc)))

        self._executor.submit(load)

    def _update_edit_access(self) -> None:
        doc = self.document
        locked = doc is None or doc.read_only
        forced = bool(doc is not None and doc.read_only_forced)
        self.readonly_checked.set(doc.read_only if doc is not None else True)
        self.readonly_toggle.configure(
            state="disabled" if forced or doc is None or self._busy else "normal",
        )
        self.save_button.configure(state="disabled" if locked else "normal")
        self.edit_tool_combo.configure(state="disabled" if locked else "readonly")
        self.edit_road_combo.configure(state="disabled" if locked else "normal")
        if locked:
            self.tool.set("Select")
            self._road_anchor = self._road_cursor = None
        for button in self._edit_buttons:
            button.configure(state="disabled" if locked else "normal")
        for entry in self._edit_entries:
            entry.configure(state="readonly" if locked else "normal")

    def _update_document_label(self) -> None:
        doc = self.document
        if doc is None:
            return
        suffix = (
            f" | OPRW v{doc.oprw_version} | READ-ONLY"
            if hasattr(doc, "oprw_version")
            else " | READ-ONLY PBO" if doc.read_only_forced
            else " | READ-ONLY" if doc.read_only else ""
        )
        grid_note = f" | {doc.cell_size:g} m/cell"
        if getattr(doc, "grid_source", "manual/default") not in ("manual/default", "manual/default (50 m for CWA OPRW)"):
            grid_note += " (config)"
        labels = len(getattr(doc, "map_locations", ()))
        if labels:
            grid_note += f" | {labels:,} places"
        self.metadata.set(
            f"{doc.width:,}×{doc.height:,} | {doc.object_count:,} objects"
            + suffix + grid_note
        )
        self.root.title(
            f"CWR Island Studio | {doc.source_label}"
            + (" [read-only]" if doc.read_only else "")
        )

    def _toggle_read_only(self) -> None:
        """Lock/unlock loose maps; PBO worlds can never be unlocked."""
        doc = self.document
        if doc is None or self._busy or doc.read_only_forced:
            self._update_edit_access()
            return
        doc.read_only = bool(self.readonly_checked.get())
        self._update_edit_access()
        self._update_document_label()
        if doc.read_only:
            self.status.set(
                "Read-only enabled. Editing and Save As disabled."
                + (" Unsaved edits are retained." if doc.changed else "")
            )
        else:
            self.status.set("Editing enabled for loose WRP. Save As preserves the source.")
        self._queue_draw()

    def _open_dialog(self) -> None:
        if self._busy:
            return
        chosen = filedialog.askopenfilename(
            title="Open a world from WRP or PBO",
            filetypes=[
                ("World files", "*.wrp *.pbo *.ifa *.pbo.zst *.ifa.zst"),
                ("Loose WRP", "*.wrp"),
                ("PBO / IFA archives", "*.pbo *.ifa *.pbo.zst *.ifa.zst"),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            self.open_path(Path(chosen))

    def open_path(self, path: Path) -> None:
        if self._busy:
            return
        try:
            kind = world_source_kind(path)
        except ValueError as exc:
            messagebox.showerror("Unsupported world file", str(exc))
            return
        if kind == "archive":
            self._open_pbo_path(path)
            return
        if self.document and self.document.changed:
            if not messagebox.askyesno(
                "Discard pending edits?",
                "Opening another world will discard unsaved edits. Continue?",
            ):
                return
        try:
            cell_size = self._num(self.cell_size, "Grid metres")
            height_scale = self._num(self.height_scale, "Height scale")
            if cell_size <= 0 or height_scale <= 0:
                raise ValueError("Grid metres and height scale must be positive")
        except ValueError as exc:
            messagebox.showerror("Invalid map settings", str(exc))
            return
        self._busy = True
        if self.document is not None:
            self._retire_current_world()
        self.status.set(f"Opening {path.name}…")
        # Tk StringVar/BooleanVar reads from a daemon thread can deadlock
        # while another world is opening or the window is closing.
        auto_grid = bool(self.auto_grid.get()) if hasattr(self, "auto_grid") else True
        def load() -> None:
            try:
                metadata = load_loose_world_metadata(path)
                with path.open("rb") as source:
                    magic = source.read(4)
                use_config = auto_grid and magic == b"4WVR" and metadata.land_grid is not None
                doc = open_island_document(
                    path,
                    cell_size=metadata.land_grid if use_config else cell_size,
                    height_scale=height_scale,
                )
                doc.map_locations = metadata.locations
                doc.grid_source = metadata.source if use_config else "manual/default"
                self._messages.put(("open", doc))
            except Exception as exc:
                self._messages.put(("error", str(exc)))
        self._executor.submit(load)

    def _accept_document(self, doc: IslandDocument) -> None:
        if self.document:
            self._retire_current_world()
        self.document = doc
        self._map_locations = tuple(getattr(doc, "map_locations", ()))
        # Loose WRPs start protected by default; the user can deliberately
        # uncheck Read-only. An archived WRP is permanently protected.
        doc.read_only = True
        self.selected = None
        self.cursor_info.set("")
        self._road_anchor = None
        self._road_cursor = None
        self._busy = False
        self._update_edit_access()
        self._update_document_label()
        self._fit()
        self.status.set(
            "Binarized OPRW opened read-only. Building object index…"
            if hasattr(doc, "oprw_version")
            else "PBO WRP opened read-only. Building object index…"
            if doc.read_only_forced
            else "Loose WRP opened read-only. Uncheck Read-only to edit. Building object index…"
        )
        self._index_cancel = threading.Event()
        cancel_event = self._index_cancel
        self._index_future = self._executor.submit(
            lambda: doc.build_object_index(cancel_event=cancel_event)
        )

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            # Bound work per tick; a huge PBO directory can produce hundreds
            # of progress events and must not monopolize Tk's event loop.
            for _ in range(80):
                kind, value = self._messages.get_nowait()
                if kind == "open":
                    self._accept_document(value)
                elif kind == "archive_progress":
                    self.status.set(str(value))
                elif kind == "cleanup_error":
                    # A retired world must never interrupt the new world's load.
                    pass
                elif kind == "archive_catalog":
                    archive, cell_size, height_scale = value
                    try:
                        self._select_archive_member(archive, cell_size, height_scale)
                    except Exception as exc:
                        archive.close()
                        self._busy = False
                        messagebox.showerror("Archive WRP", str(exc))
                elif kind == "saved":
                    self._busy = False
                    self.status.set(f"Exported successfully: {value}")
                elif kind == "saving":
                    self.status.set(str(value))
                elif kind == "pbos":
                    # A model-bounds worker may still be reading the old PBO.
                    # Swap the archive only after it has finished.
                    if self._model_future is not None and not self._model_future.done():
                        self._waiting_pbos = value
                    else:
                        self._install_pbos(value)
                elif kind == "pbo_error":
                    self._loading_pbos = False
                    # Leave the previous, working PBO set selected on failure.
                    self._refresh_source_lists()
                    self.status.set(f"PBO selection failed: {value}")
                    messagebox.showerror("Texture PBOs", str(value))
                elif kind == "error":
                    self._busy = False
                    self.status.set(str(value))
                    messagebox.showerror("Island Studio", str(value))
        except queue.Empty:
            pass
        if self._index_future is not None and self._index_future.done():
            try:
                self._index_future.result()
            except CancelledError:
                pass
            except Exception as exc:
                self.status.set(f"Object index unavailable: {exc}")
            else:
                if self.document:
                    self.status.set(
                        f"Ready: {self.document.object_count:,} objects indexed. "
                        (f"Binarized OPRW v{self.document.oprw_version}. Viewing and inspection only."
                         if hasattr(self.document, "oprw_version")
                         else "Read-only WRP from PBO. Viewing and inspection only."
                         if self.document.read_only_forced
                         else "Read-only protection enabled. Uncheck to edit."
                         if self.document.read_only
                         else "Editing enabled. Save As preserves the original WRP.")
                    )
                self._queue_draw()
            self._index_future = None
            self._index_cancel = None
        if self._model_future is not None and self._model_future.done():
            try:
                stamp, results = self._model_future.result()
                if stamp == self._model_stamp:
                    self._model_bounds.update(results)
            except Exception as exc:
                self.status.set(f"Some model footprints were unavailable: {exc}")
            self._model_future = None
            if self._waiting_pbos is not None:
                pending = self._waiting_pbos
                self._waiting_pbos = None
                self._install_pbos(pending)
            self._queue_draw()
        if not self._closing:
            self.root.after(150, self._poll)

    def _model_key(self, path: str) -> str:
        return path.replace("/", "\\").strip("\\").casefold()

    def _reset_model_bounds(self) -> None:
        self._model_stamp += 1
        self._model_bounds.clear()

    def _install_pbos(self, archives: PboTextures) -> None:
        self._loading_pbos = False
        self._reset_model_bounds()
        clear_archive_texture_cache()
        if self.pbo_textures is not None:
            self.pbo_textures.close()
        self.pbo_textures = archives
        self.selected_pbos = archives.archives
        self._refresh_source_lists()
        self.pbo_label.set(
            f"{len(archives.archives):,} PBOs; {len(archives.members):,} textures; "
            f"{len(archives.models):,} models"
        )
        self.status.set("PBO model geometry available for scaled markers.")
        self._queue_draw()

    def _request_model_bounds(self, objects: tuple[MapObject, ...]) -> None:
        if self._model_future is not None or self._loading_pbos:
            return
        if not self.asset_roots and self.pbo_textures is None:
            return
        names: dict[str, str] = {}
        for obj in objects:
            key = self._model_key(obj.model_path)
            if key not in self._model_bounds and key not in names:
                names[key] = obj.model_path
            if len(names) >= 48:
                break
        if not names:
            return
        stamp = self._model_stamp
        folders, archives = self.asset_roots, self.pbo_textures

        def measure() -> tuple[int, dict[str, ModelBounds | None]]:
            found: dict[str, ModelBounds | None] = {}
            for key, virtual_path in names.items():
                try:
                    found[key] = load_model_bounds(virtual_path, folders, archives)
                except (OSError, ValueError):
                    found[key] = None
            return stamp, found

        self._model_future = self._executor.submit(measure)

    def _fit(self) -> None:
        if not self.document:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.zoom = min(
            width / self.document.world_width,
            height / self.document.world_height,
        ) * 0.96
        self.view_x = -(width / self.zoom - self.document.world_width) / 2
        self.view_z = -(height / self.zoom - self.document.world_height) / 2
        self._refresh_hover()
        self._queue_draw()

    def _queue_draw(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.root.after(60, self._draw)

    def _extent(self) -> tuple[float, float, float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            self.view_x, self.view_z,
            self.view_x + width / self.zoom,
            self.view_z + height / self.zoom,
        )

    def _draw(self) -> None:
        self._render_pending = False
        if not self.document:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 4 or height < 4:
            return
        left, top, right, bottom = self._extent()
        try:
            view = self.layer.get()
            if view == "Actual textures":
                fallback = self.document.viewport(
                    left, top, right, bottom, width, height, textures=True,
                )
                slots, x, z = self.document.sample_viewport(
                    left, top, right, bottom, width, height, textures=True,
                )
                pixels, loaded = render_real_textures(
                    slots, x, z, self.document.texture_paths,
                    self.asset_roots, fallback, self.document.cell_size,
                    archives=self.pbo_textures,
                )
                if not self.asset_roots and self.pbo_textures is None:
                    self.status.set("Choose a texture folder or PBOs to show real images")
                elif loaded == 0:
                    self.status.set("No matching texture files found; showing texture-slot colours")
            else:
                pixels = self.document.viewport(
                    left, top, right, bottom, width, height,
                    textures=view == "Texture slots",
                    water_level=self._num(self.water_level, "Water level"),
                )
            image = Image.fromarray(pixels, "RGB")
            self._image = ImageTk.PhotoImage(image, master=self.canvas)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._image)
            if self.show_roads.get() or view == "Road preview":
                for obj in self.document.visible_objects(
                    left, top, right, bottom, limit=3200, roads_only=True,
                ):
                    segment = road_segment(obj.x, obj.z, obj.heading, obj.model_path)
                    if segment is None:
                        sx, sz = self._to_screen(obj.x, obj.z)
                        self.canvas.create_oval(sx - 2.5, sz - 2.5, sx + 2.5, sz + 2.5,
                                                fill="#ed974a", outline="")
                    else:
                        a, b = segment
                        x0, z0 = self._to_screen(*a)
                        x1, z1 = self._to_screen(*b)
                        self.canvas.create_line(
                            x0, z0, x1, z1, fill="#ed974a",
                            width=max(2, min(5, self.zoom * 3)),
                        )
            # These controls are independent: hiding generic dots must not
            # hide true-size outlines or the optional forest overlay.
            show_dots = self.show_objects.get()
            show_real = self.show_model_footprints.get()
            show_forests = self.show_forests.get()
            normal = (
                self.document.visible_objects(
                    left, top, right, bottom, limit=2700,
                )
                if show_dots or show_real else ()
            )
            forest = (
                self.document.visible_objects(
                    left, top, right, bottom, limit=2600, forest_only=True,
                )
                if show_forests else ()
            )
            # Selection covers every object displayed by either layer.
            candidates = tuple({
                obj.index: obj for obj in (*normal, *forest)
            }.values())
            if show_real or show_forests:
                self._request_model_bounds(candidates)
            selectable = {obj.index: obj for obj in forest}

            # Forest blocks and individual vegetation are shown in green.
            # This is a WRP-object overlay, not reconstructed OSM polygons.
            for obj in forest:
                sx, sz = self._to_screen(obj.x, obj.z)
                selected = obj.index == self.selected
                bounds = self._model_bounds.get(self._model_key(obj.model_path))
                corners = None
                extent = 0.0
                if bounds is not None:
                    corners = footprint_corners(
                        bounds, obj.x, obj.z, obj.heading,
                        basis=(obj.basis_xx, obj.basis_xz, obj.basis_zx, obj.basis_zz),
                    )
                    screen = [self._to_screen(*point) for point in corners]
                    extent = max(
                        max(p[0] for p in screen) - min(p[0] for p in screen),
                        max(p[1] for p in screen) - min(p[1] for p in screen),
                    )
                if corners is not None and extent >= 2.0:
                    self.canvas.create_polygon(
                        *(v for point in screen for v in point),
                        fill="#37784d", stipple="gray75",
                        outline="#65df92" if selected else "#328b51",
                        width=2 if selected else 1,
                    )
                else:
                    size = 5 if selected else 2.5
                    self.canvas.create_oval(
                        sx - size, sz - size, sx + size, sz + size,
                        fill="#65df92" if selected else "#37874d",
                        outline="",
                    )

            for obj in normal:
                # Forest objects already have their own green markers.
                if show_forests and is_forest_model(obj.model_path):
                    continue
                sx, sz = self._to_screen(obj.x, obj.z)
                selected = obj.index == self.selected
                colour = "#ffcf6d" if selected else "#e9ece2"
                bounds = (
                    self._model_bounds.get(self._model_key(obj.model_path))
                    if show_real else None
                )
                corners = None
                extent = 0.0
                if bounds is not None:
                    corners = footprint_corners(
                        bounds, obj.x, obj.z, obj.heading,
                        basis=(obj.basis_xx, obj.basis_xz, obj.basis_zx, obj.basis_zz),
                    )
                    screen = [self._to_screen(*point) for point in corners]
                    extent = max(
                        max(p[0] for p in screen) - min(p[0] for p in screen),
                        max(p[1] for p in screen) - min(p[1] for p in screen),
                    )
                mode = object_marker_mode(show_dots, show_real, bounds is not None, extent)
                if mode == "footprint":
                    selectable[obj.index] = obj
                    self.canvas.create_polygon(
                        *(v for point in screen for v in point),
                        outline=colour, fill="", width=2 if selected else 1,
                    )
                elif mode == "dot":
                    selectable[obj.index] = obj
                    radius = 5 if selected else 1.65
                    self.canvas.create_oval(
                        sx - radius, sz - radius, sx + radius, sz + radius,
                        outline=colour, fill=colour,
                    )
            self._visible = tuple(selectable.values())
            if getattr(self, "show_locations", None) is not None and self.show_locations.get():
                # The places are CfgWorlds names, not WRP object records.
                occupied: set[tuple[int, int]] = set()
                printed = 0
                for place in self._map_locations:
                    if not (left <= place.x <= right and top <= place.z <= bottom):
                        continue
                    sx, sz = self._to_screen(place.x, place.z)
                    tile = (int(sx // 105), int(sz // 21))
                    if tile in occupied:
                        continue
                    occupied.add(tile)
                    self.canvas.create_oval(
                        sx - 2, sz - 2, sx + 2, sz + 2,
                        fill="#fff2af", outline="#18251d",
                    )
                    self.canvas.create_text(
                        sx + 6 + 1, sz - 8 + 1, text=place.name,
                        anchor="sw", fill="#15221c",
                        font=("TkDefaultFont", 9, "bold"),
                    )
                    self.canvas.create_text(
                        sx + 6, sz - 8, text=place.name,
                        anchor="sw", fill="#fff5cb",
                        font=("TkDefaultFont", 9, "bold"),
                    )
                    printed += 1
                    if printed >= 125:
                        break
            if self._road_anchor is not None:
                first = self._to_screen(*self._road_anchor)
                self.canvas.create_oval(
                    first[0] - 4, first[1] - 4, first[0] + 4, first[1] + 4,
                    outline="#ffe18b", width=2,
                )
                if self._road_cursor is not None:
                    second = self._to_screen(*self._road_cursor)
                    self.canvas.create_line(
                        *first, *second, fill="#ffe18b", dash=(6, 4), width=2,
                    )
                    try:
                        planned = plan_straight_road(
                            self._road_anchor, self._road_cursor, self.road_model.get(),
                            maximum_pieces=10000,
                        )
                        for placement in planned[:250]:
                            centre = self._to_screen(placement.x, placement.z)
                            self.canvas.create_oval(
                                centre[0] - 2, centre[1] - 2,
                                centre[0] + 2, centre[1] + 2,
                                fill="#f9f0a1", outline="",
                            )
                    except ValueError:
                        pass
        except Exception as exc:
            self.status.set(f"Preview error: {exc}")

    def _refresh_source_lists(self) -> None:
        self.folder_list.delete(0, "end")
        for folder in self.asset_roots:
            self.folder_list.insert("end", str(folder))
        self.texture_folder_label.set(
            f"{len(self.asset_roots)} selected folders. Last added takes priority."
            if self.asset_roots else "No texture/model folders selected"
        )
        self.pbo_list.delete(0, "end")
        for archive in self.selected_pbos:
            self.pbo_list.insert("end", str(archive))
        if not self._loading_pbos:
            self.pbo_label.set(
                f"{len(self.selected_pbos)} PBOs selected. Last added takes priority."
                if self.selected_pbos else "No PBO archives selected"
            )

    def _set_folders(self, folders: tuple[Path, ...]) -> None:
        if folders == self.asset_roots:
            return
        self.asset_roots = folders
        self._reset_model_bounds()
        clear_loose_texture_cache()
        self._refresh_source_lists()
        self.status.set(
            f"{len(folders)} texture/model folders selected. "
            "Loose folders take precedence over PBO archives."
        )
        self._queue_draw()

    def _choose_texture_root(self) -> None:
        # Tk's standard folder picker selects one directory at a time; every
        # selection appends, allowing as many directories as needed.
        folder = filedialog.askdirectory(title="Add a texture or model folder")
        if folder:
            self._set_folders(append_unique(self.asset_roots, (folder,)))

    def _remove_folders(self) -> None:
        selected = self.folder_list.curselection()
        if selected:
            self._set_folders(remove_selected(self.asset_roots, selected))

    def _clear_folders(self) -> None:
        self._set_folders(())

    def _schedule_pbos(self, paths: tuple[Path, ...]) -> None:
        if self._loading_pbos:
            self.status.set("Wait for current PBO indexing to finish")
            return
        if paths == self.selected_pbos:
            return
        if not paths:
            if self._model_future is not None and not self._model_future.done():
                self.status.set("Wait for model measurement to finish before clearing PBOs")
                return
            self._reset_model_bounds()
            clear_archive_texture_cache()
            if self.pbo_textures is not None:
                self.pbo_textures.close()
            self.pbo_textures = None
            self.selected_pbos = ()
            self._refresh_source_lists()
            self.status.set("All PBO archives deselected")
            self._queue_draw()
            return

        self._loading_pbos = True
        self.pbo_label.set(f"Indexing {len(paths)} selected PBOs…")
        self.status.set("Reading selected PBO headers in the background…")

        def index() -> None:
            try:
                archives = PboTextures(paths)
                self._messages.put(("pbos", archives))
            except Exception as exc:
                self._messages.put(("pbo_error", str(exc)))

        self._executor.submit(index)

    def _choose_pbos(self) -> None:
        if self._loading_pbos:
            return
        chosen = filedialog.askopenfilenames(
            title="Add texture or model PBO archives",
            filetypes=[
                ("Game archives", "*.pbo *.pbo.zst *.ifa *.ifa.zst"),
                ("PBO archives", "*.pbo"),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            self._schedule_pbos(append_unique(self.selected_pbos, chosen))

    def _remove_pbos(self) -> None:
        selected = self.pbo_list.curselection()
        if selected:
            self._schedule_pbos(remove_selected(self.selected_pbos, selected))

    def _clear_pbos(self) -> None:
        self._schedule_pbos(())

    def _layer_changed(self) -> None:
        self._refresh_hover()
        self._queue_draw()

    def _update_hover(self, screen_x: float, screen_z: float) -> None:
        self._hover_screen = (screen_x, screen_z)
        self._refresh_hover()

    def _refresh_hover(self) -> None:
        if self._hover_screen is None or self.document is None:
            self.cursor_info.set("")
            return
        x, z = self._to_world(*self._hover_screen)
        self.cursor_info.set(cursor_readout(self.document, x, z, self.layer.get()))

    def _mouse_leave(self, _event: tk.Event) -> None:
        self._hover_screen = None
        self.cursor_info.set("")

    def _mouse_move(self, event: tk.Event) -> None:
        self._update_hover(event.x, event.y)
        if self._road_anchor is not None and self.tool.get() == "Draw road line":
            self._road_cursor = self._to_world(event.x, event.y)
            self._queue_draw()

    def _cancel_road(self) -> None:
        self._road_anchor = self._road_cursor = None
        self.status.set("Road stroke cancelled")
        self._queue_draw()

    def _road_action(self, position: tuple[float, float], mode: str) -> None:
        if self.document is None or self.document.read_only:
            return
        x, z = position
        if mode == "Erase road":
            roads = self.document.visible_objects(
                x - 25, z - 25, x + 25, z + 25,
                limit=4000, roads_only=True,
            )
            nearest = min(roads, key=lambda r: (r.x - x) ** 2 + (r.z - z) ** 2,
                          default=None)
            if nearest is not None and math.hypot(nearest.x - x, nearest.z - z) <= 15:
                self.document.delete_object(nearest.index)
                self.status.set(f"Removed road object ID {nearest.object_id}")
                self._queue_draw()
            return
        if self._road_anchor is None:
            self._road_anchor = position
            self._road_cursor = position
            self.status.set("Road start set. Move to preview, then click the end point.")
            self._queue_draw()
            return
        try:
            placed = plan_straight_road(
                self._road_anchor, position, self.road_model.get(),
            )
            for piece in placed:
                col = min(self.document.width - 1, max(0, int(piece.x / self.document.cell_size)))
                row = min(self.document.height - 1, max(0, int(piece.z / self.document.cell_size)))
                y = self.document.terrain_at(col, row)
                self.document.add_object(
                    self.road_model.get(), x=piece.x, y=y, z=piece.z,
                    heading=piece.heading,
                )
            self.status.set(f"Placed {len(placed):,} straight road pieces (unsaved)")
            self._road_anchor = self._road_cursor = None
            self._queue_draw()
        except (ValueError, OverflowError) as exc:
            self.status.set(str(exc))

    def _to_screen(self, x: float, z: float) -> tuple[float, float]:
        return (x - self.view_x) * self.zoom, (z - self.view_z) * self.zoom

    def _to_world(self, sx: float, sz: float) -> tuple[float, float]:
        return self.view_x + sx / self.zoom, self.view_z + sz / self.zoom

    def _zoom_at_point(self, x: float, y: float, factor: float) -> None:
        if not self.document:
            return
        fitted = min(
            max(1, self.canvas.winfo_width()) / self.document.world_width,
            max(1, self.canvas.winfo_height()) / self.document.world_height,
        )
        self.view_x, self.view_z, self.zoom = zoom_around(
            self.view_x, self.view_z, self.zoom,
            x, y, factor, fitted * 0.35, 8.0,
        )
        self._refresh_hover()
        self._queue_draw()

    def _pointer_canvas_point(self) -> tuple[float, float]:
        px, py = self.root.winfo_pointerxy()
        x = px - self.canvas.winfo_rootx()
        y = py - self.canvas.winfo_rooty()
        if not (0 <= x < self.canvas.winfo_width() and
                0 <= y < self.canvas.winfo_height()):
            x = self.canvas.winfo_width() / 2
            y = self.canvas.winfo_height() / 2
        return x, y

    def _zoom_button(self, factor: float) -> None:
        self._zoom_at_point(*self._pointer_canvas_point(), factor)

    def _zoom_hotkey(self, event: tk.Event, factor: float) -> str | None:
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, ttk.Spinbox)):
            return None
        self._zoom_button(factor)
        return "break"

    def _pan_hotkey(self, event: tk.Event, direction: str) -> str | None:
        """Pan unless a focused editor control needs its own arrow keys."""
        if self.document is None or self._busy:
            return None
        focused = self.root.focus_get()
        if isinstance(focused, (
            tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Listbox,
            tk.Spinbox, ttk.Spinbox, tk.Scale, ttk.Scale,
            ttk.Treeview, tk.Scrollbar, ttk.Scrollbar,
        )):
            return None
        shift_held = bool(event.state & 0x0001)
        self.view_x, self.view_z = pan_with_arrow(
            self.view_x, self.view_z, self.zoom, direction,
            fast=shift_held,
        )
        self._refresh_hover()
        self._queue_draw()
        return "break"

    def _zoom_at(self, event: tk.Event, factor: float) -> None:
        self._zoom_at_point(event.x, event.y, factor)

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(event, 1.25 if event.delta > 0 else 0.8)

    def _pan_start(self, event: tk.Event) -> None:
        self._pan_origin = (event.x, event.y, self.view_x, self.view_z)

    def _pan_move(self, event: tk.Event) -> None:
        if self._pan_origin is None:
            return
        x, y, vx, vz = self._pan_origin
        self.view_x = vx - (event.x - x) / self.zoom
        self.view_z = vz - (event.y - y) / self.zoom
        self._update_hover(event.x, event.y)
        self._queue_draw()

    def _left_click(self, event: tk.Event) -> None:
        self._last_brush = None
        self._left_action(event)

    def _left_drag(self, event: tk.Event) -> None:
        if self.tool.get() in ("Raise terrain", "Lower terrain", "Paint texture"):
            self._left_action(event)

    def _left_action(self, event: tk.Event) -> None:
        if not self.document or self._busy:
            return
        mode = self.tool.get()
        if self.document.read_only and mode != "Select":
            return
        if mode in ("Draw road line", "Erase road"):
            position = self._to_world(event.x, event.y)
            if 0 <= position[0] < self.document.world_width and 0 <= position[1] < self.document.world_height:
                self._road_action(position, mode)
            return
        if mode == "Select":
            nearest = None
            distance = 12 ** 2
            world_x, world_z = self._to_world(event.x, event.y)
            for obj in self._visible:
                sx, sz = self._to_screen(obj.x, obj.z)
                d2 = (event.x - sx) ** 2 + (event.y - sz) ** 2
                bounds = (
                    self._model_bounds.get(self._model_key(obj.model_path))
                    if self.show_model_footprints.get() else None
                )
                if bounds is not None and footprint_contains(
                    footprint_corners(
                        bounds, obj.x, obj.z, obj.heading,
                        basis=(obj.basis_xx, obj.basis_xz, obj.basis_zx, obj.basis_zz),
                    ), world_x, world_z,
                ):
                    d2 = min(d2, 1.0)
                if d2 < distance:
                    distance, nearest = d2, obj
            if nearest:
                self._select(nearest.index)
            else:
                self.selected = None
                self._queue_draw()
            return
        world_x, world_z = self._to_world(event.x, event.y)
        if not (0 <= world_x < self.document.world_width
                and 0 <= world_z < self.document.world_height):
            return
        # A drag crossing the same terrain grid position only needs one edit.
        grid = (round(world_x / self.document.cell_size),
                round(world_z / self.document.cell_size))
        if grid == self._last_brush:
            return
        self._last_brush = grid
        try:
            radius = self._num(self.brush_radius, "Brush radius")
            strength = abs(self._num(self.brush_strength, "Elevation step"))
            texture = None
            change = 0.0
            if mode == "Paint texture":
                raw = self.paint_slot.get().strip()
                texture = int(raw)
            elif mode == "Raise terrain":
                change = strength
            elif mode == "Lower terrain":
                change = -strength
            count = self.document.paint(
                world_x, world_z, radius=radius,
                raise_metres=change, texture=texture,
            )
            self.status.set(f"{mode}: {count:,} terrain points touched (unsaved)")
            self._update_hover(event.x, event.y)
            self._queue_draw()
        except ValueError as exc:
            self.status.set(str(exc))

    def _select(self, index: int) -> None:
        if not self.document:
            return
        obj = self.document.object_at(index)
        self.selected = index
        self.model_path.set(obj.model_path)
        self.obj_x.set(f"{obj.x:.4f}")
        self.obj_y.set(f"{obj.y:.4f}")
        self.obj_z.set(f"{obj.z:.4f}")
        self.obj_heading.set(f"{obj.heading:.3f}")
        self.status.set(f"Selected record {index:,}, object ID {obj.object_id:,}")
        self._queue_draw()

    def _apply_object(self) -> None:
        if not self.document or self.document.read_only or self.selected is None or self._busy:
            return
        try:
            obj = self.document.object_at(self.selected)
            x = self._num(self.obj_x, "X")
            y = self._num(self.obj_y, "Height")
            z = self._num(self.obj_z, "Z")
            heading = self._num(self.obj_heading, "Heading")
            if not (0 <= x < self.document.world_width
                    and 0 <= z < self.document.world_height):
                raise ValueError("object must remain inside the map")
            self.document.move_object(
                obj.index, x=x, y=y, z=z, heading=heading,
            )
            self.document.set_model(obj.index, self.model_path.get().strip())
        except (ValueError, IndexError) as exc:
            messagebox.showerror("Invalid object edit", str(exc))
            return
        self.status.set("Object edited. Use Save As to export changes.")
        self._queue_draw()

    def _add_object(self) -> None:
        if not self.document or self.document.read_only or self._busy:
            return
        left, top, right, bottom = self._extent()
        x = min(self.document.world_width - 0.001, max(0.0, (left + right) / 2))
        z = min(self.document.world_height - 0.001, max(0.0, (top + bottom) / 2))
        column = min(self.document.width - 1, max(0, int(x / self.document.cell_size)))
        row = min(self.document.height - 1, max(0, int(z / self.document.cell_size)))
        y = self.document.terrain_at(column, row)
        try:
            index = self.document.add_object(
                self.model_path.get().strip(), x=x, y=y, z=z,
                heading=self._num(self.obj_heading, "Heading"),
            )
        except ValueError as exc:
            messagebox.showerror("Cannot add object", str(exc))
            return
        self._select(index)

    def _remove_object(self) -> None:
        if not self.document or self.document.read_only or self.selected is None or self._busy:
            return
        try:
            self.document.delete_object(self.selected)
        except (IndexError, ValueError) as exc:
            messagebox.showerror("Cannot remove object", str(exc))
            return
        self.selected = None
        self.status.set("Object removed from export (source unchanged)")
        self._queue_draw()

    def _save_dialog(self) -> None:
        if not self.document or self.document.read_only or self._busy:
            return
        default = self.document.path.with_name(self.document.path.stem + "_edited.wrp")
        chosen = filedialog.asksaveasfilename(
            title="Export edited WRP (source is preserved)",
            initialdir=str(default.parent), initialfile=default.name,
            defaultextension=".wrp",
            filetypes=[("World maps", "*.wrp")],
        )
        if not chosen:
            return
        dest = Path(chosen).resolve()
        if dest == self.document.path:
            messagebox.showerror("Source protected", "Choose a different filename from the original island.")
            return
        if dest.exists() and not messagebox.askyesno(
            "Overwrite export?", f"Replace the existing export?\n{dest}",
        ):
            return
        self._busy = True
        doc = self.document
        self.status.set("Exporting edited world…")
        def task() -> None:
            try:
                saved = doc.export(
                    dest, progress=lambda message: self._messages.put(("saving", message)),
                )
                self._messages.put(("saved", str(saved)))
            except Exception as exc:
                self._messages.put(("error", str(exc)))
        self._executor.submit(task)

    def _shutdown(self) -> None:
        if self._closing:
            return
        if self.document and self.document.changed and not messagebox.askyesno(
            "Unsaved edits", "Close without exporting your edits?",
        ):
            return
        self._closing = True
        self._archive_cancel.set()
        if self.document is not None:
            self._retire_current_world()
        elif getattr(self, "_index_cancel", None) is not None:
            self._index_cancel.set()
        # Cancelling queued jobs and detaching running daemon workers allows
        # Quit to close immediately, even during PBO/WRP indexing.
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._waiting_pbos is not None:
            self._waiting_pbos.close()
            self._waiting_pbos = None
        # World cleanup is scheduled by _retire_current_world without
        # waiting for a still-running indexer or blocking Windows file removal.
        clear_archive_texture_cache()
        if self.pbo_textures is not None and (
            self._model_future is None or self._model_future.done()
        ):
            self.pbo_textures.close()
            self.pbo_textures = None
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast large-world RVW4 island editor")
    parser.add_argument("wrp", nargs="?", type=Path, help="optional .wrp, .pbo or .ifa file to open")
    args = parser.parse_args(argv)
    root = tk.Tk()
    IslandStudio(root, args.wrp)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
