"""Tk/Matplotlib UI for browsing and categorising textured P3D models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterator, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

import measure_p3d_models as measure
from p3d_preview_geometry import PreviewModel
from cwr_worldgen.pbo import is_pbo_path
from p3d_texture_sources import TextureResolver
from p3d_texture_render import MipmapCache, render_textured_model


PLACEMENTS = ("Urban", "Rural", "Both")


@dataclass(slots=True)
class Classification:
    categories: list[str]
    placement: str = ""
    reviewed: bool = False
    width_m: float | None = None
    length_m: float | None = None
    height_m: float | None = None
    aspect_ratio: float | None = None
    origin_to_bottom_m: float | None = None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _measurement_values(model: PreviewModel) -> dict[str, float]:
    measurement = model.measurement
    return {
        "width_m": float(measurement.width_m),
        "length_m": float(measurement.length_m),
        "height_m": float(measurement.height_m),
        "aspect_ratio": float(measurement.aspect_ratio),
        "origin_to_bottom_m": float(measurement.origin_to_bottom_m),
    }


def load_state(path: Path) -> tuple[dict[str, Classification], list[str]]:
    if not path.exists():
        return {}, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read existing classification file {path}: {exc}") from exc
    result: dict[str, Classification] = {}
    categories = [str(v) for v in raw.get("categories", []) if str(v).strip()]
    for item in raw.get("models", []):
        if not isinstance(item, dict) or "model_path" not in item:
            continue
        key = measure._canonical_model_path(str(item["model_path"]))
        placement = str(item.get("placement", "")).strip()
        if placement not in PLACEMENTS:
            placement = ""
        result[key] = Classification(
            categories=[str(v) for v in item.get("categories", [])],
            placement=placement,
            reviewed=bool(item.get("reviewed", True)),
            width_m=_optional_float(item.get("width_m")),
            length_m=_optional_float(item.get("length_m")),
            height_m=_optional_float(item.get("height_m")),
            aspect_ratio=_optional_float(item.get("aspect_ratio")),
            origin_to_bottom_m=_optional_float(item.get("origin_to_bottom_m")),
        )
    return result, categories


class CategoriserApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        model_iter: Iterator[PreviewModel | measure.ModelFailure],
        texture_resolver: TextureResolver,
        output: Path,
        categories: Sequence[str],
        state: dict[str, Classification],
        total_models: int | None = None,
    ) -> None:
        self.root = root
        self.model_iter = model_iter
        self.texture_resolver = texture_resolver
        self.mipmap_cache = MipmapCache()
        # Browsing more texture archives must not restart the model stream or
        # modify category JSON; keep its source list separate from the models.
        self._base_texture_sources = tuple(texture_resolver.inputs)
        self._extra_texture_sources: list[Path] = []
        self._related_texture_paths: tuple[str, ...] = ()
        self.output = output
        self.categories = list(categories)
        self.state = state
        self.total_models = total_models
        self.models: list[PreviewModel] = []
        self.failures: list[measure.ModelFailure] = []
        self.index = -1
        self.current: PreviewModel | None = None
        self.exhausted = False
        self._updating_checks = False
        self.azim = 35.0
        self.elev = 25.0

        root.title("P3D Model Categoriser")
        root.geometry("1500x920")
        root.minsize(1050, 700)
        self._build_ui()
        root.bind("<Left>", lambda _e: self.previous_model())
        root.bind("<Right>", lambda _e: self.next_model())
        root.bind("<Control-s>", lambda _e: self.save_state())
        root.bind("<Control-f>", self._focus_model_search)
        root.bind("<Control-Shift-C>", lambda _e: self.copy_current_path())
        root.bind("<Control-r>", lambda _e: self._reset_view())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.next_model(mark_current=False)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        self.path_var = tk.StringVar(master=self.root, value="Loading model...")
        self.progress_var = tk.StringVar(master=self.root, value="")
        ttk.Label(
            header,
            textvariable=self.path_var,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.progress_var).grid(row=0, column=1, sticky="e")

        search_row = ttk.Frame(header)
        search_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        search_row.columnconfigure(1, weight=1)
        ttk.Label(search_row, text="Find model:").grid(row=0, column=0, padx=(0, 6))
        self.search_var = tk.StringVar(master=self.root, value="")
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.search_entry.bind("<Return>", lambda _event: self.find_next_model())
        ttk.Button(search_row, text="Find next", command=self.find_next_model).grid(
            row=0, column=2, padx=(6, 0)
        )
        ttk.Button(
            search_row, text="×", width=3, command=lambda: self.search_var.set("")
        ).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(
            search_row, text="Searches model paths (Ctrl+F)"
        ).grid(row=0, column=4, padx=(9, 0))

        frame = ttk.Frame(outer)
        frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.ax_preview = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        side = ttk.Frame(outer, padding=(8, 4))
        side.grid(row=1, column=1, sticky="ns")
        ttk.Label(side, text="Categories", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        self.category_vars: dict[str, tk.BooleanVar] = {}
        for category in self.categories:
            var = tk.BooleanVar(master=self.root, value=False)
            self.category_vars[category] = var
            ttk.Checkbutton(
                side,
                text=category,
                variable=var,
                command=self._category_changed,
            ).pack(anchor="w", fill="x", pady=2)

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        ttk.Label(side, text="Placement", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.placement_var = tk.StringVar(master=self.root, value="")
        for placement in PLACEMENTS:
            label = {
                "Urban": "Urban only",
                "Rural": "Rural only",
                "Both": "Both / anywhere",
            }[placement]
            ttk.Radiobutton(
                side,
                text=label,
                value=placement,
                variable=self.placement_var,
                command=self._placement_changed,
            ).pack(anchor="w", fill="x", pady=1)
        ttk.Button(
            side,
            text="Clear placement",
            command=self._clear_placement,
        ).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            side,
            text=(
                "Urban = towns/cities only; Rural = countryside only; Both = valid in either. "
                "Clear placement removes the saved model entirely when no categories are selected."
            ),
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        ttk.Label(side, text="View", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.skip_textures_var = tk.BooleanVar(master=self.root, value=False)
        ttk.Checkbutton(
            side,
            text="Skip textures (faster)",
            variable=self.skip_textures_var,
            command=self._redraw,
        ).pack(anchor="w", fill="x", pady=(4, 2))
        row = ttk.Frame(side)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="↶ 15°", command=lambda: self._rotate(-15)).pack(side=tk.LEFT)
        ttk.Button(row, text="15° ↷", command=lambda: self._rotate(15)).pack(
            side=tk.LEFT, padx=4
        )
        row2 = ttk.Frame(side)
        row2.pack(fill="x")
        ttk.Button(row2, text="Tilt +", command=lambda: self._tilt(10)).pack(side=tk.LEFT)
        ttk.Button(row2, text="Tilt -", command=lambda: self._tilt(-10)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(row2, text="Zoom +", command=lambda: self._zoom_by(1.25)).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(row2, text="Zoom −", command=lambda: self._zoom_by(1 / 1.25)).pack(
            side=tk.LEFT, padx=4
        )
        row3 = ttk.Frame(side)
        row3.pack(fill="x", pady=(5, 0))
        ttk.Button(row3, text="Reset view", command=self._reset_view).pack(
            side=tk.LEFT
        )
        ttk.Button(row3, text="Copy model path", command=self.copy_current_path).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        self.info_var = tk.StringVar(master=self.root, value="")
        self.texture_var = tk.StringVar(master=self.root, value="")
        self.status_var = tk.StringVar(master=self.root, value="")
        ttk.Label(side, textvariable=self.info_var, justify=tk.LEFT, wraplength=300).pack(
            anchor="w"
        )
        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        ttk.Label(side, text="Referenced textures", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w"
        )
        ttk.Label(side, textvariable=self.texture_var, justify=tk.LEFT, wraplength=300).pack(
            anchor="w", pady=(2, 0)
        )
        texture_box = ttk.Frame(side)
        texture_box.pack(fill=tk.X, pady=(4, 0))
        self.texture_list = tk.Listbox(
            texture_box, height=6, width=38, exportselection=False, selectmode=tk.BROWSE
        )
        self.texture_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        texture_scroll = ttk.Scrollbar(
            texture_box, orient=tk.VERTICAL, command=self.texture_list.yview
        )
        texture_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.texture_list.configure(yscrollcommand=texture_scroll.set)
        self.texture_list.bind("<Double-1>", self.preview_selected_texture)
        texture_actions = ttk.Frame(side)
        texture_actions.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(
            texture_actions, text="Preview texture", command=self.preview_selected_texture
        ).pack(side=tk.LEFT)
        ttk.Button(
            texture_actions, text="Copy texture path", command=self.copy_selected_texture_path
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(
            side, text="Add texture sources (does not change the model scan):",
            wraplength=300,
        ).pack(anchor="w", pady=(9, 0))
        source_actions = ttk.Frame(side)
        source_actions.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(
            source_actions, text="Add PBO / IFA…", command=self.select_texture_archives
        ).pack(side=tk.LEFT)
        ttk.Button(
            source_actions, text="Add folder…", command=self.select_texture_folder
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(
            side, text="Clear added texture sources", command=self.clear_added_texture_sources
        ).pack(anchor="w", pady=(4, 0))
        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        ttk.Label(side, textvariable=self.status_var, wraplength=300).pack(anchor="w")

        nav = ttk.Frame(outer)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        nav.columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav, text="◀ Previous", command=self.previous_model)
        self.prev_button.grid(row=0, column=0, padx=(0, 6))
        ttk.Label(
            nav,
            text="Left/Right navigate • Ctrl+S saves • category/placement changes autosave",
        ).grid(row=0, column=1)
        self.next_button = ttk.Button(nav, text="Next ▶", command=self.next_model)
        self.next_button.grid(row=0, column=2, padx=(6, 0))

    def _busy(self, text: str) -> None:
        self.progress_var.set(text)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()

    def _unbusy(self) -> None:
        self.root.configure(cursor="")
        if self.current is not None:
            self._update_progress_counter()

    def _update_progress_counter(self) -> None:
        if self.current is None or self.index < 0:
            return
        if self.total_models is not None:
            counter = f"Model {self.index + 1:,} / {self.total_models:,}"
        else:
            suffix = " +" if not self.exhausted else ""
            counter = f"Model {self.index + 1:,} / {len(self.models):,}{suffix}"
        self.progress_var.set(
            f"{counter} • failures skipped: {len(self.failures):,}"
        )

    def _load_next(self) -> PreviewModel | None:
        while not self.exhausted:
            try:
                item = next(self.model_iter)
            except StopIteration:
                self.exhausted = True
                return None
            if isinstance(item, measure.ModelFailure):
                self.failures.append(item)
                continue
            self.models.append(item)
            existing = self.state.get(item.model_path)
            if existing is not None:
                for field, value in _measurement_values(item).items():
                    setattr(existing, field, value)
            return item
        return None

    def _current_categories(self) -> list[str]:
        return [name for name, var in self.category_vars.items() if var.get()]

    def _current_placement(self) -> str:
        placement = self.placement_var.get().strip()
        return placement if placement in PLACEMENTS else ""

    def _commit(self, reviewed: bool) -> None:
        if self.current is None or self._updating_checks:
            return
        categories = self._current_categories()
        placement = self._current_placement()
        if not categories and not placement:
            # Empty means genuinely unclassified. Remove the state entry rather
            # than persisting a reviewed-but-empty record in the companion JSON.
            self.state.pop(self.current.model_path, None)
            return
        old = self.state.get(self.current.model_path, Classification([]))
        self.state[self.current.model_path] = Classification(
            categories=categories,
            placement=placement,
            reviewed=reviewed or old.reviewed,
            **_measurement_values(self.current),
        )

    def _clear_placement(self) -> None:
        if self._updating_checks or self.current is None:
            return
        if not self.placement_var.get():
            return
        self.placement_var.set("")
        self._commit(True)
        self.save_state()

    def _category_changed(self) -> None:
        if self._updating_checks or self.current is None:
            return
        self._commit(True)
        self.save_state()

    def _placement_changed(self) -> None:
        if self._updating_checks or self.current is None:
            return
        self._commit(True)
        self.save_state()

    def _rotate(self, amount: float) -> None:
        self.azim = (self.azim + amount) % 360
        self._redraw()

    def _tilt(self, amount: float) -> None:
        self.elev = max(-80, min(80, self.elev + amount))
        self._redraw()

    def _redraw(self) -> None:
        if self.current is None:
            return
        skip_textures = bool(self.skip_textures_var.get())
        self._busy("Rendering geometry..." if skip_textures else "Rendering textured model...")
        try:
            self._draw_model(self.current)
        finally:
            self._unbusy()

    def next_model(self, *, mark_current: bool = True) -> None:
        if mark_current and self.current is not None:
            self._commit(True)
            self.save_state()
        target = self.index + 1
        if target >= len(self.models):
            self._busy("Loading next model...")
            try:
                model = self._load_next()
            except (OSError, ValueError) as exc:
                print(f"[scan error] {exc}", file=sys.stderr, flush=True)
                messagebox.showerror("Scan error", str(exc))
                return
            finally:
                self._unbusy()
            if model is None:
                summary = f"End of scan: {len(self.models)} model(s), {len(self.failures)} failure(s)"
                print(f"[scan] {summary}", file=sys.stderr, flush=True)
                self.progress_var.set(summary)
                self.next_button.configure(state=tk.DISABLED)
                return
        self.index = target
        self._show_model(self.models[self.index])

    def previous_model(self) -> None:
        if self.current is not None:
            self._commit(True)
            self.save_state()
        if self.index <= 0:
            return
        self.index -= 1
        self._show_model(self.models[self.index])

    def _show_model(self, model: PreviewModel) -> None:
        self.current = model
        self.path_var.set(model.model_path)
        self._update_progress_counter()
        classification = self.state.get(model.model_path, Classification([]))
        selected = set(classification.categories)
        self._updating_checks = True
        try:
            for category, var in self.category_vars.items():
                var.set(category in selected)
            self.placement_var.set(classification.placement)
        finally:
            self._updating_checks = False

        m = model.measurement
        placement_text = classification.placement or "Unspecified"
        self.info_var.set(
            f"Format: {m.format} v{m.version}\n"
            f"LOD: {m.lod}\n"
            f"Preview: {'bounds only' if model.render_mode == 'bounds only' else model.render_mode}\n"
            f"Vertices: {model.original_vertex_count:,}\n"
            f"Faces: {len(model.faces):,}\n"
            f"Width: {m.width_m:g} m\n"
            f"Height: {m.height_m:g} m\n"
            f"Length: {m.length_m:g} m\n"
            f"Footprint: {m.footprint_area_m2:g} m²\n"
            f"Placement: {placement_text}\n"
            f"Reviewed: {'yes' if classification.reviewed else 'no'}"
        )
        names = [measure._canonical_model_path(n) for n in model.texture_paths if n]
        self.texture_var.set(
            "No texture references in this LOD"
            if not names
            else f"{len(names)} referenced\n"
            + "\n".join(names[:8])
            + (f"\n… +{len(names)-8} more" if len(names) > 8 else "")
        )
        self.prev_button.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL)
        self._redraw()

    def _preview_render_size(self) -> tuple[int, int]:
        """Render near the canvas' actual size to avoid wasted raster work."""
        try:
            widget = self.canvas.get_tk_widget()
            width, height = int(widget.winfo_width()), int(widget.winfo_height())
        except (AttributeError, TypeError, ValueError, tk.TclError):
            return (960, 720)
        if width <= 2 or height <= 2:
            return (960, 720)
        return (
            max(480, min(1200, round(width * 1.25))),
            max(360, min(840, round(height * 1.25))),
        )

    def _render_model(self, model: PreviewModel, *, renderer, zoom: float = 1.0) -> None:
        """Render through the browser's mesh, mipmap and bounds-only paths.

        The session UI supplies its renderer for backwards-compatible tests and
        scroll-zoom behavior; this shared implementation keeps both views in
        sync with the modern Arma browser.
        """
        self.ax_preview.clear()
        self.status_var.set("")

        if getattr(model, "render_mode", "") == "bounds only":
            points = model.points - model.points.mean(axis=0)
            azim, elev = np.deg2rad(self.azim), np.deg2rad(self.elev)
            horizontal = np.cos(azim) * points[:, 0] - np.sin(azim) * points[:, 2]
            depth = np.sin(azim) * points[:, 0] + np.cos(azim) * points[:, 2]
            vertical = np.cos(elev) * points[:, 1] - np.sin(elev) * depth
            for a, b in model.edges:
                self.ax_preview.plot(
                    (horizontal[a], horizontal[b]),
                    (vertical[a], vertical[b]),
                    linewidth=1.5,
                )
            extent = max(float(np.ptp(horizontal)), float(np.ptp(vertical)), 0.1)
            half = 0.6 * extent / max(float(zoom), 0.05)
            self.ax_preview.set_xlim(-half, half)
            self.ax_preview.set_ylim(-half, half)
            self.ax_preview.set_aspect("equal", adjustable="box")
            self.ax_preview.set_title(
                f"ODOL v{model.measurement.version} • bounds only, NOT model geometry"
            )
            self.status_var.set("Model bounds preview only; polygon mesh not decoded.")
            self.canvas.draw_idle()
            return

        if not model.faces:
            self.ax_preview.text(
                0.5,
                0.5,
                "Textured preview unavailable\nNo polygon topology was parsed for this model.",
                ha="center",
                va="center",
                transform=self.ax_preview.transAxes,
                fontsize=13,
            )
            self.ax_preview.axis("off")
            self.figure.tight_layout(pad=1.2)
            self.canvas.draw_idle()
            return

        width, height = self._preview_render_size()
        image, hits, misses = renderer(
            model.points,
            model.faces,
            self.texture_resolver,
            model.source,
            width=width,
            height=height,
            azim_deg=self.azim,
            elev_deg=self.elev,
            zoom=zoom,
            load_textures=not self.skip_textures_var.get(),
            mipmap_cache=getattr(self, "mipmap_cache", None),
        )
        self.ax_preview.imshow(image, interpolation="hanning", resample=True)
        if self.skip_textures_var.get():
            self.ax_preview.set_title(
                f"Geometry preview • textures skipped • az {self.azim:.0f}° / "
                f"el {self.elev:.0f}° • zoom {zoom:.2f}×"
            )
            self.status_var.set(
                "Texture lookup skipped for faster model browsing. "
                "Mouse wheel still zooms the geometry preview."
            )
        else:
            self.ax_preview.set_title(
                f"Textured model • az {self.azim:.0f}° / el {self.elev:.0f}° • "
                f"zoom {zoom:.2f}×"
            )
            if misses:
                self.status_var.set(
                    f"Texture sampling: {hits} textured triangle(s), "
                    f"{misses} untextured triangle(s). See console for missing assets."
                )
            else:
                self.status_var.set(
                    f"Texture sampling: {hits} textured triangle(s)."
                )
        self.ax_preview.axis("off")
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def _draw_model(self, model: PreviewModel) -> None:
        self._render_model(
            model,
            renderer=render_textured_model,
            zoom=getattr(self, "zoom", 1.0),
        )

    def save_state(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": 2,
            "categories": self.categories,
            "placements": list(PLACEMENTS),
            "reviewed_count": sum(1 for x in self.state.values() if x.reviewed),
            "classified_count": sum(1 for x in self.state.values() if x.categories),
            "placement_count": sum(1 for x in self.state.values() if x.placement),
            "models": [
                {
                    "model_path": key,
                    "categories": self.state[key].categories,
                    "placement": self.state[key].placement,
                    "reviewed": self.state[key].reviewed,
                    "width_m": self.state[key].width_m,
                    "length_m": self.state[key].length_m,
                    "height_m": self.state[key].height_m,
                    "aspect_ratio": self.state[key].aspect_ratio,
                    "origin_to_bottom_m": self.state[key].origin_to_bottom_m,
                }
                for key in sorted(self.state)
            ],
            "failures": [asdict(x) for x in self.failures],
        }
        temp = self.output.with_name(self.output.name + ".tmp")
        temp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.output)
        self.status_var.set(f"Saved: {self.output}")

    def close(self) -> None:
        try:
            if self.current is not None:
                self._commit(False)
            self.save_state()
        except OSError as exc:
            print(f"[save error] {exc}", file=sys.stderr, flush=True)
            if not messagebox.askyesno(
                "Could not save",
                f"Could not save classifications:\n{exc}\n\nClose anyway?",
            ):
                return
        self.root.destroy()
