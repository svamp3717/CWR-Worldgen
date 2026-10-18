# SPDX-License-Identifier: GPL-3.0-or-later
"""Island Studio: responsive terrain and object editing for large RVW4 worlds."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import math
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from .core import IslandDocument, MapObject
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cwr-worldgen")
except PackageNotFoundError:
    __version__ = "dev"


class IslandStudio:
    """Viewport-centred editor with bounded rendering and background I/O."""

    def __init__(self, root: tk.Tk, initial_path: Path | None = None) -> None:
        self.root = root
        self.root.title(f"CWR Island Studio {__version__}")
        self.root.geometry("1280x820")
        self.root.minsize(870, 560)
        self.root.configure(background="#18251d")
        self.document: IslandDocument | None = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="island-studio")
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._index_future: Future | None = None
        self._busy = False
        self._render_pending = False
        self._image: ImageTk.PhotoImage | None = None
        self._visible: tuple[MapObject, ...] = ()
        self.selected: int | None = None
        self.view_x = 0.0
        self.view_z = 0.0
        self.zoom = 1.0
        self._pan_origin: tuple[int, int, float, float] | None = None
        self._last_brush: tuple[int, int] | None = None

        self.cell_size = tk.StringVar(value="50")
        self.height_scale = tk.StringVar(value="1")
        self.layer = tk.StringVar(value="Elevation")
        self.tool = tk.StringVar(value="Select")
        self.show_objects = tk.BooleanVar(value=True)
        self.brush_radius = tk.StringVar(value="75")
        self.brush_strength = tk.StringVar(value="1")
        self.paint_slot = tk.StringVar(value="0")
        self.model_path = tk.StringVar(value=r"o\road\sil25.p3d")
        self.obj_x = tk.StringVar(value="0")
        self.obj_y = tk.StringVar(value="0")
        self.obj_z = tk.StringVar(value="0")
        self.obj_heading = tk.StringVar(value="0")
        self.metadata = tk.StringVar(value="Open a loose 4WVR .wrp island to start")
        self.status = tk.StringVar(value="Ready. The source file will not be modified.")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)
        self.root.after(120, self._poll)
        if initial_path:
            self.root.after(250, lambda: self.open_path(initial_path))

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(9, 7))
        header.pack(fill="x")
        ttk.Button(header, text="Open WRP", command=self._open_dialog).pack(side="left", padx=3)
        ttk.Button(header, text="Save As", command=self._save_dialog).pack(side="left", padx=3)
        ttk.Button(header, text="Fit map", command=self._fit).pack(side="left", padx=3)
        ttk.Label(header, text="Grid metres").pack(side="left", padx=(20, 2))
        ttk.Entry(header, textvariable=self.cell_size, width=6).pack(side="left")
        ttk.Label(header, text="Height scale").pack(side="left", padx=(13, 2))
        ttk.Entry(header, textvariable=self.height_scale, width=6).pack(side="left")
        ttk.Label(header, textvariable=self.metadata).pack(side="right", padx=8)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(body)
        side = ttk.Frame(body, padding=12, width=270)
        body.add(canvas_frame, weight=5)
        body.add(side, weight=0)
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

        ttk.Label(side, text="View", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Combobox(
            side, textvariable=self.layer,
            state="readonly", values=("Elevation", "Texture slots"),
        ).pack(fill="x", pady=(4, 7))
        self.layer.trace_add("write", lambda *_: self._queue_draw())
        ttk.Checkbutton(
            side, text="Show world objects", variable=self.show_objects,
            command=self._queue_draw,
        ).pack(anchor="w", pady=(0, 13))

        ttk.Label(side, text="Edit tool", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Combobox(
            side, textvariable=self.tool, state="readonly",
            values=("Select", "Raise terrain", "Lower terrain", "Paint texture"),
        ).pack(fill="x", pady=(4, 7))
        for label, var in (
            ("Brush radius, metres", self.brush_radius),
            ("Elevation step, metres", self.brush_strength),
            ("Texture slot (0–511)", self.paint_slot),
        ):
            ttk.Label(side, text=label).pack(anchor="w", pady=(3, 0))
            ttk.Entry(side, textvariable=var).pack(fill="x", pady=(0, 4))

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
            ttk.Entry(side, textvariable=var).pack(fill="x", pady=(0, 2))
        ttk.Button(side, text="Apply object changes", command=self._apply_object).pack(fill="x", pady=(10, 3))
        ttk.Button(side, text="Add object at map centre", command=self._add_object).pack(fill="x", pady=3)
        ttk.Button(side, text="Remove selected object", command=self._remove_object).pack(fill="x", pady=3)

        footer = ttk.Frame(self.root, padding=(8, 5))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Label(footer, text="Wheel: zoom   Right-drag: pan   Left-click: select / paint").pack(side="right")
        self.root.bind("<Control-o>", lambda _e: self._open_dialog())
        self.root.bind("<Control-Shift-S>", lambda _e: self._save_dialog())
        self.root.bind("<Control-s>", lambda _e: self._save_dialog())

    def _num(self, value: tk.StringVar, name: str) -> float:
        try:
            number = float(value.get())
            if not math.isfinite(number):
                raise ValueError()
            return number
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite number") from exc

    def _open_dialog(self) -> None:
        if self._busy:
            return
        chosen = filedialog.askopenfilename(
            title="Open an island WRP",
            filetypes=[("RVW4 worlds", "*.wrp"), ("All files", "*.*")],
        )
        if chosen:
            self.open_path(Path(chosen))

    def open_path(self, path: Path) -> None:
        if self._busy:
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
        self.status.set(f"Opening {path.name}…")
        def load() -> None:
            try:
                doc = IslandDocument(path, cell_size=cell_size, height_scale=height_scale)
                self._messages.put(("open", doc))
            except Exception as exc:
                self._messages.put(("error", str(exc)))
        self._executor.submit(load)

    def _accept_document(self, doc: IslandDocument) -> None:
        if self._index_future is not None and not self._index_future.done():
            self._index_future.result()
        if self.document:
            self.document.close()
        self.document = doc
        self.selected = None
        self._busy = False
        self.metadata.set(
            f"{doc.width:,}×{doc.height:,}  |  {doc.object_count:,} objects"
        )
        self.root.title(f"CWR Island Studio | {doc.path.name}")
        self._fit()
        self.status.set("Map ready. Building the background object index…")
        self._index_future = self._executor.submit(doc.build_object_index)

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self._messages.get_nowait()
                if kind == "open":
                    self._accept_document(value)
                elif kind == "saved":
                    self._busy = False
                    self.status.set(f"Exported successfully: {value}")
                elif kind == "saving":
                    self.status.set(str(value))
                elif kind == "error":
                    self._busy = False
                    self.status.set(str(value))
                    messagebox.showerror("Island Studio", str(value))
        except queue.Empty:
            pass
        if self._index_future is not None and self._index_future.done():
            try:
                self._index_future.result()
            except Exception as exc:
                self.status.set(f"Object index unavailable: {exc}")
            else:
                if self.document:
                    self.status.set(
                        f"Ready: {self.document.object_count:,} objects indexed. "
                        "Edits stay in memory until Save As."
                    )
                self._queue_draw()
            self._index_future = None
        self.root.after(150, self._poll)

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
            pixels = self.document.viewport(
                left, top, right, bottom, width, height,
                textures=self.layer.get() == "Texture slots",
            )
            image = Image.fromarray(pixels, "RGB")
            self._image = ImageTk.PhotoImage(image, master=self.canvas)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._image)
            if self.show_objects.get():
                self._visible = self.document.visible_objects(
                    left, top, right, bottom, limit=2700,
                )
                for obj in self._visible:
                    sx, sz = self._to_screen(obj.x, obj.z)
                    selected = obj.index == self.selected
                    colour = "#ffcf6d" if selected else "#e9ece2"
                    radius = 5 if selected else 1.65
                    self.canvas.create_oval(
                        sx - radius, sz - radius, sx + radius, sz + radius,
                        outline=colour, fill=colour,
                    )
            else:
                self._visible = ()
        except Exception as exc:
            self.status.set(f"Preview error: {exc}")

    def _to_screen(self, x: float, z: float) -> tuple[float, float]:
        return (x - self.view_x) * self.zoom, (z - self.view_z) * self.zoom

    def _to_world(self, sx: float, sz: float) -> tuple[float, float]:
        return self.view_x + sx / self.zoom, self.view_z + sz / self.zoom

    def _zoom_at(self, event: tk.Event, factor: float) -> None:
        if not self.document:
            return
        wx, wz = self._to_world(event.x, event.y)
        fitted = min(
            max(1, self.canvas.winfo_width()) / self.document.world_width,
            max(1, self.canvas.winfo_height()) / self.document.world_height,
        )
        self.zoom = min(max(self.zoom * factor, fitted * 0.35), 8.0)
        self.view_x, self.view_z = wx - event.x / self.zoom, wz - event.y / self.zoom
        self._queue_draw()

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
        self._queue_draw()

    def _left_click(self, event: tk.Event) -> None:
        self._last_brush = None
        self._left_action(event)

    def _left_drag(self, event: tk.Event) -> None:
        if self.tool.get() != "Select":
            self._left_action(event)

    def _left_action(self, event: tk.Event) -> None:
        if not self.document or self._busy:
            return
        mode = self.tool.get()
        if mode == "Select":
            nearest = None
            distance = 12 ** 2
            for obj in self._visible:
                sx, sz = self._to_screen(obj.x, obj.z)
                d2 = (event.x - sx) ** 2 + (event.y - sz) ** 2
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
        if not self.document or self.selected is None or self._busy:
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
        if not self.document or self._busy:
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
        if not self.document or self.selected is None or self._busy:
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
        if not self.document or self._busy:
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
        if self._busy:
            messagebox.showwarning("Working", "Wait for the current read or export to finish.")
            return
        if self.document and self.document.changed and not messagebox.askyesno(
            "Unsaved edits", "Close without exporting your edits?",
        ):
            return
        if self._index_future and not self._index_future.done():
            self._index_future.result()
        if self.document:
            self.document.close()
        self._executor.shutdown(wait=False)
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast large-world RVW4 island editor")
    parser.add_argument("wrp", nargs="?", type=Path, help="optional .wrp file to open")
    args = parser.parse_args(argv)
    root = tk.Tk()
    IslandStudio(root, args.wrp)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
