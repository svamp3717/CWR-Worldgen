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
        initial_texture_sources: Sequence[Path] = (),
    ) -> None:
        self.root = root
        self.model_iter = model_iter
        self.texture_resolver = texture_resolver
        self.mipmap_cache = MipmapCache()
        # Browsing more texture archives must not restart the model stream or
        # modify category JSON; keep its source list separate from the models.
        # Startup-selected texture archives remain removable with the normal
        # Clear added texture sources control, just like sources added later.
        extra_sources = list(dict.fromkeys(
            Path(path).expanduser() for path in initial_texture_sources
        ))
        self._base_texture_sources = tuple(
            path for path in texture_resolver.inputs if path not in extra_sources
        )
        self._extra_texture_sources: list[Path] = extra_sources
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
        root.bind("<Control-Shift-S>", lambda _e: self.change_json_output())
        root.bind("<Control-f>", self._focus_model_search)
        root.bind("<Control-Shift-C>", lambda _e: self.copy_current_path())
        root.bind("<Control-r>", lambda _e: self._reset_view())
        root.bind("<Control-l>", lambda _e: self.show_scanned_models())
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

        output_row = ttk.Frame(header)
        output_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        output_row.columnconfigure(1, weight=1)
        ttk.Label(output_row, text="JSON output:").grid(
            row=0, column=0, sticky="w", padx=(0, 7)
        )
        self.output_var = tk.StringVar(master=self.root, value=str(self.output))
        ttk.Label(output_row, textvariable=self.output_var).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            output_row, text="Change output JSON…", command=self.change_json_output
        ).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(output_row, text="Save now", command=self.save_state).grid(
            row=0, column=3, padx=(6, 0)
        )

        search_row = ttk.Frame(header)
        search_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        search_row.columnconfigure(1, weight=1)
        ttk.Label(search_row, text="Find model:").grid(row=0, column=0, padx=(0, 6))
        self.search_var = tk.StringVar(master=self.root, value="")
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.search_entry.bind("<Return>", lambda _event: self.find_next_model())
        ttk.Button(search_row, text="Find next", command=self.find_next_model).grid(
            row=0, column=2, padx=(6, 0)
        )
        ttk.Button(search_row, text="Scanned models…", command=self.show_scanned_models).grid(
            row=0, column=3, padx=(6, 0)
        )
        ttk.Button(
            search_row, text="×", width=3, command=lambda: self.search_var.set("")
        ).grid(row=0, column=4, padx=(4, 0))
        ttk.Label(
            search_row, text="Searches model paths (Ctrl+F)"
        ).grid(row=0, column=5, padx=(9, 0))

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

        # Categorisation plus independent texture browsing needs more room
        # than the original CWA-only form. Keep the controls usable on laptop
        # screens rather than allowing buttons below the window to be clipped.
        side_container = ttk.Frame(outer)
        side_container.grid(row=1, column=1, sticky="nsew")
        side_canvas = tk.Canvas(
            side_container, width=335, highlightthickness=0, borderwidth=0
        )
        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_scroll = ttk.Scrollbar(
            side_container, orient=tk.VERTICAL, command=side_canvas.yview
        )
        side_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side = ttk.Frame(side_canvas, padding=(8, 4))
        sidebar_window = side_canvas.create_window(
            (0, 0), window=side, anchor="nw"
        )
        side.bind(
            "<Configure>",
            lambda _event: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        side_canvas.bind(
            "<Configure>",
            lambda event: side_canvas.itemconfigure(
                sidebar_window, width=event.width
            ),
        )
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
            source_actions, text="Add PBOs / IFAs…", command=self.select_texture_archives
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
        names = tuple(dict.fromkeys(
            measure._canonical_model_path(n) for n in model.texture_paths if n
        ))
        self._related_texture_paths = names
        self.texture_var.set(
            f"{len(names)} unique texture reference(s)"
            if names else "No texture references in this LOD"
        )
        if hasattr(self, "texture_list"):
            self.texture_list.delete(0, tk.END)
            for name in names:
                self.texture_list.insert(tk.END, name)
            if names:
                self.texture_list.selection_set(0)
        self.prev_button.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL)
        self._redraw()

    def _focus_model_search(self, _event=None) -> str:
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, tk.END)
        return "break"

    def _select_model_index(self, target: int) -> None:
        """Use existing category/resume semantics when jumping between models."""
        if not 0 <= target < len(self.models):
            return
        if self.current is not None and self.index != target:
            self._commit(True)
            self.save_state()
        self.index = target
        self._show_model(self.models[target])
        self.save_state()

    def find_next_model(self) -> None:
        """Search by model path, streaming forward without rendering every model.

        Loaded models are searched first. Scanning proceeds only when needed,
        and wraps around previously loaded models after reaching the end.
        """
        needle = measure._canonical_model_path(self.search_var.get()).strip()
        if not needle:
            self.status_var.set("Enter a model name or path to search.")
            return
        begin = getattr(self, "_begin_navigation_loading", None)
        end = getattr(self, "_finish_navigation_loading", None)
        if callable(begin) and not begin(
            "Searching model paths…",
            f"Searching for {needle} (new models are scanned without rendering).",
        ):
            return
        try:
            target = next(
                (i for i in range(self.index + 1, len(self.models))
                 if needle in self.models[i].model_path.casefold()), None
            )
            while target is None and not self.exhausted:
                model = self._load_next()
                if model is None:
                    break
                if needle in model.model_path.casefold():
                    target = len(self.models) - 1
                    break
                if len(self.models) % 25 == 0:
                    progress = getattr(self, "_set_navigation_status", None)
                    if callable(progress):
                        progress(
                            "Searching model paths…",
                            f"Scanned {len(self.models):,} models • {model.model_path}",
                        )
            if target is None:
                target = next(
                    (i for i in range(0, min(self.index + 1, len(self.models)))
                     if needle in self.models[i].model_path.casefold()), None
                )
            if target is None:
                self.status_var.set(f"No matching P3D model found: {needle}")
                return
            self._select_model_index(target)
            self.status_var.set(
                f"Found model {target + 1:,}: {self.models[target].model_path}"
            )
        except (OSError, ValueError) as exc:
            print(f"[model search error] {exc}", file=sys.stderr, flush=True)
            self.status_var.set(f"Model search failed: {exc}")
        finally:
            if callable(end):
                end()

    def show_scanned_models(self) -> None:
        """Browsable, searchable index of models reached in this session.

        The categoriser remains streaming; the list deliberately does not
        read every unopened model just to show a catalogue.
        """
        if getattr(self, "_model_list_popup", None) is not None:
            try:
                if self._model_list_popup.winfo_exists():
                    self._model_list_popup.lift()
                    return
            except tk.TclError:
                pass
        popup = tk.Toplevel(self.root)
        popup.title("Scanned P3D models")
        popup.geometry("850x530")
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(2, weight=1)
        self._model_list_popup = popup
        ttk.Label(
            popup,
            text=f"{len(self.models):,} models scanned so far. "
                 "Double-click to jump; use Find next to search unopened models.",
            wraplength=800,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        query = tk.StringVar(master=popup, value="")
        field = ttk.Entry(popup, textvariable=query)
        field.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        table_frame = ttk.Frame(popup)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=10)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(
            table_frame, columns=("index", "path", "reviewed"),
            show="headings", selectmode="browse",
        )
        for column, label, width in (
            ("index", "#", 65), ("path", "Model path", 600),
            ("reviewed", "Reviewed", 95),
        ):
            tree.heading(column, text=label)
            tree.column(column, width=width, stretch=column == "path")
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)

        def refresh(*_args) -> None:
            term = measure._canonical_model_path(query.get())
            children = tree.get_children()
            if children:
                tree.delete(*children)
            for i, model in enumerate(self.models):
                if term in model.model_path:
                    saved = self.state.get(model.model_path)
                    tree.insert(
                        "", tk.END, iid=str(i),
                        values=(i + 1, model.model_path,
                                "Yes" if saved and saved.reviewed else "No"),
                    )

        def jump(_event=None) -> None:
            selected = tree.selection()
            if not selected:
                return
            index = int(selected[0])
            popup.destroy()
            self._model_list_popup = None
            # SessionCategoriserApp supports jumping with loading feedback;
            # the base categoriser can navigate its already loaded entries too.
            jump_to = getattr(self, "jump_to_model_number", None)
            if callable(jump_to):
                jump_to(index + 1)
            else:
                self._select_model_index(index)

        tree.bind("<Double-1>", jump)
        query.trace_add("write", refresh)
        field.bind("<Return>", lambda _event: jump())
        ttk.Button(popup, text="Jump to selected model", command=jump).grid(
            row=3, column=0, sticky="e", padx=10, pady=8
        )
        refresh()
        field.focus_set()

    def _selected_texture_path(self) -> str | None:
        if not hasattr(self, "texture_list"):
            return None
        selected = self.texture_list.curselection()
        paths = getattr(self, "_related_texture_paths", ())
        if not selected or selected[0] >= len(paths):
            self.status_var.set("Select a referenced texture first.")
            return None
        return paths[selected[0]]

    def preview_selected_texture(self, _event=None) -> None:
        """Show the actual decoded PAA/PAC referenced by the current model."""
        if self.current is None:
            return
        path = self._selected_texture_path()
        if path is None:
            return
        self.status_var.set(f"Loading texture {path}…")
        try:
            image = self.texture_resolver.load_rgba(path, self.current.source)
            if image is None:
                self.status_var.set(
                    f"Texture not found: {path}. Add its PBO/IFA or game folder."
                )
                return
            previous = getattr(self, "_texture_popup", None)
            if previous is not None:
                try:
                    previous.destroy()
                except tk.TclError:
                    pass
            popup = tk.Toplevel(self.root)
            popup.title(f"PAA/PAC texture • {path}")
            popup.geometry("850x650")
            popup.columnconfigure(0, weight=1)
            popup.rowconfigure(1, weight=1)
            ttk.Label(
                popup,
                text=f"{path}  •  {image.shape[1]} × {image.shape[0]} pixels",
            ).grid(row=0, column=0, sticky="w", padx=8, pady=7)
            figure = Figure(figsize=(8, 6), dpi=100)
            ax = figure.add_subplot(1, 1, 1)
            ax.imshow(image, interpolation="nearest")
            ax.axis("off")
            figure.tight_layout(pad=0.5)
            canvas = FigureCanvasTkAgg(figure, master=popup)
            canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
            canvas.draw_idle()
            popup._texture_canvas = canvas
            self._texture_popup = popup
            self.status_var.set(f"Texture loaded: {path}")
        except (OSError, ValueError, measure.ModelReadError, tk.TclError) as exc:
            self.status_var.set(f"Could not preview texture {path}: {exc}")
            messagebox.showerror("Texture preview failed", str(exc), parent=self.root)

    def _copy_path(self, path: str, label: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.root.update_idletasks()
        except tk.TclError as exc:
            self.status_var.set(f"Could not copy {label} path: {exc}")
            return
        self.status_var.set(f"Copied {label} path: {path}")

    def copy_current_path(self) -> None:
        if self.current is not None:
            self._copy_path(self.current.model_path, "model")

    def copy_selected_texture_path(self) -> None:
        path = self._selected_texture_path()
        if path is not None:
            self._copy_path(path, "texture")

    def _set_extra_texture_sources(self, paths: Sequence[Path]) -> None:
        """Replace only the texture resolver; the model iterator stays intact."""
        proposed = list(self._extra_texture_sources)
        existing = set(self._base_texture_sources) | set(proposed)
        for source in paths:
            path = Path(source).expanduser()
            if not path.exists() or not (path.is_dir() or
                                           (path.is_file() and is_pbo_path(path))):
                self.status_var.set(f"Not a PBO/IFA archive or folder: {path}")
                return
            if path not in existing:
                proposed.append(path)
                existing.add(path)
        if proposed == self._extra_texture_sources:
            self.status_var.set("All selected texture sources are already loaded.")
            return
        self._extra_texture_sources = proposed
        self.texture_resolver = TextureResolver(
            (*self._base_texture_sources, *self._extra_texture_sources)
        )
        self.mipmap_cache.clear()
        if self.current is not None:
            self._redraw()
        self.status_var.set(
            f"Added texture sources: {len(self._extra_texture_sources)}. "
            "Model scan and classifications unchanged."
        )

    def select_texture_archives(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Add texture PBO / IFA archives (select multiple)",
            filetypes=(
                ("PBO / IFA archives", "*.pbo *.pbo.zst *.ifa *.ifa.zst"),
                ("PBO archives", "*.pbo *.pbo.zst"),
                ("Iron Front IFA archives", "*.ifa *.ifa.zst"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._set_extra_texture_sources(tuple(Path(path) for path in selected))

    def select_texture_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self.root, title="Add folder containing PAA/PAC texture archives"
        )
        if folder:
            self._set_extra_texture_sources((Path(folder),))

    def clear_added_texture_sources(self) -> None:
        if not self._extra_texture_sources:
            self.status_var.set("No additional texture sources to clear.")
            return
        self._extra_texture_sources.clear()
        self.texture_resolver = TextureResolver(self._base_texture_sources)
        self.mipmap_cache.clear()
        if self.current is not None:
            self._redraw()
        self.status_var.set("Additional texture sources cleared. Model scan unchanged.")

    def change_json_output(self) -> None:
        """Save As without resetting the model stream or existing classifications.

        Explicitly confirm replacing a pre-existing JSON or its reviewed
        .unclassified.json companion rather than overwriting another session.
        """
        chosen = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose categoriser classification JSON output",
            defaultextension=".json",
            initialdir=str(self.output.parent),
            initialfile=self.output.name,
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not chosen:
            return
        new_output = Path(chosen).expanduser()
        if new_output.suffix.casefold() != ".json":
            messagebox.showerror(
                "JSON output required",
                "Choose a filename ending with .json.", parent=self.root,
            )
            return
        try:
            same_output = new_output.resolve() == self.output.resolve()
        except OSError:
            same_output = new_output == self.output
        if not same_output:
            from p3d_categoriser_storage import incomplete_state_path
            companion = incomplete_state_path(new_output)
            existing = [path for path in (new_output, companion) if path.exists()]
            if existing and not messagebox.askyesno(
                "Replace existing classifications?",
                "The selected output already has saved classification data:\n\n"
                + "\n".join(str(path) for path in existing)
                + "\n\nReplace it with your current session?",
                parent=self.root,
            ):
                return

        previous = self.output
        self.output = new_output
        try:
            if self.current is not None:
                self._commit(False)
            self.save_state()
        except (OSError, ValueError) as exc:
            self.output = previous
            messagebox.showerror(
                "Could not save classification JSON", str(exc), parent=self.root,
            )
            return
        if hasattr(self, "output_var"):
            self.output_var.set(str(new_output))

    def _zoom_by(self, factor: float) -> None:
        if self.current is None:
            return
        self.zoom = max(0.65, min(5.0, getattr(self, "zoom", 1.0) * factor))
        self._redraw()

    def _reset_view(self) -> None:
        if self.current is None:
            return
        self.azim, self.elev, self.zoom = 35.0, 25.0, 1.0
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
