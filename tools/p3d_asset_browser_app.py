"""Interactive browser for P3D models and PAA/PAC textures."""
from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from p3d_asset_browser_catalog import (
    BrowserAsset,
    BrowserCatalogue,
    filter_assets,
    load_model,
    load_texture,
    model_texture_paths,
    scan_catalogue,
    texture_users,
)
from p3d_texture_io import TextureResolver
from p3d_texture_render import render_textured_model
from cwr_worldgen.assets import canonical_asset_path


class AssetBrowserApp:
    """Model/texture browser derived from the P3D categoriser preview shell."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.sources: list[Path] = []
        self.catalogue = BrowserCatalogue((), (), {})
        self.texture_resolver = TextureResolver(())
        self.current: BrowserAsset | None = None
        self.current_model = None
        self.azim = 35.0
        self.elev = 25.0
        self.zoom = 1.0
        self._row_assets: dict[str, BrowserAsset] = {}
        self._related_assets: dict[str, BrowserAsset] = {}
        self._scan_queue: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self._scan_generation = 0
        self._scan_in_progress = False
        self._source_buttons: list[ttk.Button] = []

        root.title("CWR P3D Model & Texture Browser")
        root.geometry("1650x980")
        root.minsize(1180, 720)
        self._build_ui()
        root.bind("<Control-o>", lambda _event: self.select_pbo())
        root.bind("<Control-f>", lambda _event: self.search_entry.focus_set())
        root.bind("<Control-Shift-C>", lambda _event: self.copy_current_path())
        root.bind("<Escape>", lambda _event: self.search_var.set(""))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        select_pbo_button = ttk.Button(toolbar, text="Select PBO…", command=self.select_pbo)
        select_pbo_button.pack(side=tk.LEFT)
        add_folder_button = ttk.Button(toolbar, text="Add folder…", command=self.select_folder)
        add_folder_button.pack(side=tk.LEFT, padx=(6, 0))
        clear_sources_button = ttk.Button(
            toolbar, text="Clear sources", command=self.clear_sources
        )
        clear_sources_button.pack(side=tk.LEFT, padx=(6, 14))
        self._source_buttons.extend(
            (select_pbo_button, add_folder_button, clear_sources_button)
        )
        self.sources_var = tk.StringVar(master=self.root, value="No source selected")
        ttk.Label(toolbar, textvariable=self.sources_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_var = tk.StringVar(
            master=self.root,
            value="Select a PBO or folder to browse models and textures.",
        )
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=(12, 0))
        self.scan_progress = ttk.Progressbar(
            toolbar,
            mode="indeterminate",
            length=190,
        )

        sidebar = ttk.Frame(outer, padding=(0, 0, 8, 0))
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.rowconfigure(3, weight=1)
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(
            sidebar,
            text="Asset catalogue",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        search_row = ttk.Frame(sidebar)
        search_row.grid(row=1, column=0, sticky="ew", pady=(7, 5))
        search_row.columnconfigure(0, weight=1)
        self.search_var = tk.StringVar(master=self.root, value="")
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var, width=48)
        self.search_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(search_row, text="×", width=3, command=lambda: self.search_var.set("")).grid(
            row=0, column=1, padx=(4, 0)
        )
        self.search_var.trace_add("write", lambda *_args: self._refresh_lists())

        self.count_var = tk.StringVar(master=self.root, value="0 models • 0 textures")
        ttk.Label(sidebar, textvariable=self.count_var).grid(row=2, column=0, sticky="w")

        self.notebook = ttk.Notebook(sidebar)
        self.notebook.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
        self.model_tree = self._make_asset_tree(self.notebook, "Models")
        self.texture_tree = self._make_asset_tree(self.notebook, "Textures")

        main = ttk.Frame(outer)
        main.grid(row=1, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(main)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(11, 7.5), dpi=100)
        self.ax_preview = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        mpl_toolbar = NavigationToolbar2Tk(self.canvas, preview_frame, pack_toolbar=False)
        mpl_toolbar.update()
        mpl_toolbar.grid(row=1, column=0, sticky="ew")
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

        controls = ttk.Frame(main)
        controls.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(controls, text="↶ 15°", command=lambda: self._rotate(-15)).pack(side=tk.LEFT)
        ttk.Button(controls, text="15° ↷", command=lambda: self._rotate(15)).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(controls, text="Tilt +", command=lambda: self._tilt(10)).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Button(controls, text="Tilt -", command=lambda: self._tilt(-10)).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(controls, text="Reset view", command=self._reset_view).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        self.skip_textures_var = tk.BooleanVar(master=self.root, value=False)
        ttk.Checkbutton(
            controls,
            text="Skip textures (faster)",
            variable=self.skip_textures_var,
            command=self._redraw_current_model,
        ).pack(side=tk.LEFT, padx=(18, 0))

        details = ttk.Frame(main)
        details.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)

        info_box = ttk.LabelFrame(details, text="Selected asset", padding=7)
        info_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.info_var = tk.StringVar(master=self.root, value="Nothing selected")
        ttk.Label(info_box, textvariable=self.info_var, justify=tk.LEFT, wraplength=600).pack(
            anchor="w", fill=tk.X
        )
        self.copy_path_button = ttk.Button(
            info_box,
            text="Copy asset path",
            command=self.copy_current_path,
            state=tk.DISABLED,
        )
        self.copy_path_button.pack(anchor="w", pady=(8, 0))
        ttk.Label(
            info_box,
            text="Copies the in-game path used by Worldgen, not the PBO source chain.",
            wraplength=600,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(3, 0))

        related_box = ttk.LabelFrame(details, text="Related assets • double-click to jump", padding=5)
        related_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        related_box.columnconfigure(0, weight=1)
        related_box.rowconfigure(0, weight=1)
        self.related_tree = ttk.Treeview(
            related_box,
            columns=("kind", "path"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        self.related_tree.heading("kind", text="Type")
        self.related_tree.heading("path", text="Path")
        self.related_tree.column("kind", width=80, stretch=False)
        self.related_tree.column("path", width=520, stretch=True)
        self.related_tree.grid(row=0, column=0, sticky="nsew")
        related_scroll = ttk.Scrollbar(
            related_box, orient=tk.VERTICAL, command=self.related_tree.yview
        )
        related_scroll.grid(row=0, column=1, sticky="ns")
        self.related_tree.configure(yscrollcommand=related_scroll.set)
        self.related_tree.bind("<Double-1>", self._jump_related)

        self._show_empty_preview()

    def _make_asset_tree(self, notebook: ttk.Notebook, title: str) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(
            frame,
            columns=("path", "size"),
            show="headings",
            selectmode="browse",
            height=28,
        )
        tree.heading("path", text=title[:-1] if title.endswith("s") else title)
        tree.heading("size", text="Size")
        tree.column("path", width=455, stretch=True)
        tree.column("size", width=78, stretch=False, anchor="e")
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        tree.bind("<<TreeviewSelect>>", self._asset_selected)
        tree.bind("<Double-1>", self._asset_selected)
        notebook.add(frame, text=title)
        return tree

    @staticmethod
    def _format_size(value: int) -> str:
        value = max(0, int(value))
        if value < 1024:
            return f"{value} B"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} KiB"
        return f"{value / (1024 * 1024):.1f} MiB"

    def _busy(self, text: str) -> None:
        self.status_var.set(text)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()

    def _unbusy(self, text: str = "") -> None:
        self.root.configure(cursor="")
        if text:
            self.status_var.set(text)

    def _set_scan_active(self, active: bool) -> None:
        self._scan_in_progress = bool(active)
        state = tk.DISABLED if active else tk.NORMAL
        for button in self._source_buttons:
            button.configure(state=state)
        if active:
            self.scan_progress.pack(side=tk.RIGHT, padx=(10, 0))
            self.scan_progress.start(12)
            self.root.configure(cursor="watch")
        else:
            self.scan_progress.stop()
            self.scan_progress.pack_forget()
            self.root.configure(cursor="")
        self.root.update_idletasks()

    def select_pbo(self) -> None:
        if self._scan_in_progress:
            self.status_var.set("Asset scan already in progress.")
            return
        values = filedialog.askopenfilenames(
            parent=self.root,
            title="Select PBO archive(s)",
            filetypes=(
                ("PBO archives", "*.pbo"),
                ("Zstd-wrapped PBO archives", "*.pbo.zst"),
                ("All files", "*.*"),
            ),
        )
        if values:
            self._add_sources(Path(value) for value in values)

    def select_folder(self) -> None:
        if self._scan_in_progress:
            self.status_var.set("Asset scan already in progress.")
            return
        value = filedialog.askdirectory(
            parent=self.root,
            title="Select asset folder",
            mustexist=True,
        )
        if value:
            self._add_sources((Path(value),))

    def _add_sources(self, paths) -> None:
        known = {str(path.resolve()).casefold() for path in self.sources if path.exists()}
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                continue
            try:
                key = str(path.resolve()).casefold()
            except OSError:
                key = str(path).casefold()
            if key not in known:
                self.sources.append(path)
                known.add(key)
        self._scan_sources()

    def clear_sources(self) -> None:
        self.sources.clear()
        self.catalogue = BrowserCatalogue((), (), {})
        self.texture_resolver = TextureResolver(())
        self.current = None
        self.current_model = None
        self.sources_var.set("No source selected")
        self._refresh_lists()
        self._clear_related()
        self.info_var.set("Nothing selected")
        self.copy_path_button.configure(text="Copy asset path", state=tk.DISABLED)
        self.status_var.set("Select a PBO or folder to browse models and textures.")
        self._show_empty_preview()

    def _scan_sources(self) -> None:
        if not self.sources:
            self.clear_sources()
            return
        if self._scan_in_progress:
            return

        sources = tuple(self.sources)
        self._scan_generation += 1
        generation = self._scan_generation
        self.status_var.set("Scanning models and textures…")
        self._set_scan_active(True)

        def worker() -> None:
            try:
                catalogue = scan_catalogue(sources)
            except Exception as exc:
                self._scan_queue.put((generation, "error", exc))
            else:
                self._scan_queue.put((generation, "success", (sources, catalogue)))

        threading.Thread(
            target=worker,
            name="p3d-asset-browser-scan",
            daemon=True,
        ).start()
        self.root.after(50, self._poll_scan_queue)

    def _poll_scan_queue(self) -> None:
        handled = False
        while True:
            try:
                generation, kind, payload = self._scan_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._scan_generation:
                continue
            handled = True
            self._set_scan_active(False)
            if kind == "error":
                exc = payload
                messagebox.showerror("Asset scan failed", str(exc), parent=self.root)
                self.status_var.set(f"Scan failed: {exc}")
                continue
            sources, catalogue = payload
            self.catalogue = catalogue
            self.texture_resolver = TextureResolver(sources)
            self.current = None
            self.current_model = None
            self.sources_var.set(
                " • ".join(path.name or str(path) for path in sources[-3:])
                + (f" • +{len(sources) - 3} more" if len(sources) > 3 else "")
            )
            self._refresh_lists()
            self._clear_related()
            self._show_empty_preview()
            self.info_var.set("Select a model or texture from the catalogue.")
            self.status_var.set(
                f"Indexed {len(catalogue.models):,} model(s) and "
                f"{len(catalogue.textures):,} texture(s)."
            )

        if self._scan_in_progress and not handled:
            self.root.after(50, self._poll_scan_queue)

    def _refresh_lists(self) -> None:
        query = self.search_var.get() if hasattr(self, "search_var") else ""
        models = filter_assets(self.catalogue, query, ("model",))
        textures = filter_assets(self.catalogue, query, ("texture",))
        self._row_assets.clear()
        for tree, values, prefix in (
            (self.model_tree, models, "m"),
            (self.texture_tree, textures, "t"),
        ):
            for child in tree.get_children():
                tree.delete(child)
            for index, item in enumerate(values):
                iid = f"{prefix}{index}"
                self._row_assets[iid] = item
                tree.insert(
                    "",
                    tk.END,
                    iid=iid,
                    values=(item.path, self._format_size(item.size)),
                )
        self.count_var.set(
            f"{len(models):,} / {len(self.catalogue.models):,} models • "
            f"{len(textures):,} / {len(self.catalogue.textures):,} textures"
        )

    def _asset_selected(self, event=None) -> None:
        tree = event.widget if event is not None else None
        if tree not in {self.model_tree, self.texture_tree}:
            return
        selection = tree.selection()
        if not selection:
            return
        item = self._row_assets.get(selection[0])
        if item is None:
            return
        self._show_asset(item)

    def _show_asset(self, item: BrowserAsset) -> None:
        self.current = item
        self.zoom = 1.0
        self.copy_path_button.configure(
            text="Copy P3D path" if item.kind == "model" else "Copy texture path",
            state=tk.NORMAL,
        )
        if item.kind == "model":
            self._show_model(item)
        else:
            self._show_texture(item)

    def _show_model(self, item: BrowserAsset) -> None:
        self._busy(f"Loading model {item.path}…")
        try:
            model = load_model(item, self.catalogue)
            self.current_model = model
            self._draw_model(model)
            texture_paths = tuple(
                canonical_asset_path(value)
                for value in model.texture_paths
                if value
            )
            self._set_related_textures(texture_paths)
            m = model.measurement
            self.info_var.set(
                f"MODEL\n{item.path}\n\n"
                f"Source: {item.source}\n"
                f"Size: {self._format_size(item.size)}\n"
                f"Dimensions: {m.width_m:.2f} × {m.length_m:.2f} × {m.height_m:.2f} m\n"
                f"Textures referenced: {len(texture_paths):,}"
            )
            self._unbusy(f"Model loaded: {item.path}")
        except Exception as exc:
            self.current_model = None
            self._show_error("Model preview failed", exc)
            self._unbusy(f"Could not preview {item.path}")

    def _draw_model(self, model) -> None:
        self.ax_preview.clear()
        if not model.faces:
            points = model.points
            if len(points):
                self.ax_preview.scatter(points[:, 0], points[:, 2], s=1)
                self.ax_preview.set_aspect("equal", adjustable="box")
            self.ax_preview.set_title(f"{model.model_path} • vertex preview")
            self.canvas.draw_idle()
            return
        image, hits, misses = render_textured_model(
            model.points,
            model.faces,
            self.texture_resolver,
            model.source,
            width=1400,
            height=980,
            azim_deg=self.azim,
            elev_deg=self.elev,
            zoom=self.zoom,
            load_textures=not self.skip_textures_var.get(),
        )
        self.ax_preview.imshow(image, interpolation="nearest")
        if self.skip_textures_var.get():
            self.ax_preview.set_title(
                f"{model.model_path} • geometry preview • textures skipped"
            )
        else:
            self.ax_preview.set_title(
                f"{model.model_path} • {hits} textured face hit(s)"
                + (f" • {misses} missing" if misses else "")
            )
        self.ax_preview.axis("off")
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def _show_texture(self, item: BrowserAsset) -> None:
        self._busy(f"Loading texture {item.path}…")
        try:
            image = load_texture(item, self.catalogue)
            self.current_model = None
            self.ax_preview.clear()
            self.ax_preview.imshow(image, interpolation="nearest")
            self.ax_preview.set_title(
                f"{item.path} • {image.shape[1]} × {image.shape[0]}"
            )
            self.ax_preview.axis("off")
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw_idle()
            users = texture_users(item.path, self.catalogue)
            self._set_related_models(users)
            self.info_var.set(
                f"TEXTURE\n{item.path}\n\n"
                f"Source: {item.source}\n"
                f"Size: {self._format_size(item.size)}\n"
                f"Decoded: {image.shape[1]} × {image.shape[0]}\n"
                f"Indexed model users: {len(users):,}"
            )
            self._unbusy(f"Texture loaded: {item.path}")
        except Exception as exc:
            self.current_model = None
            self._show_error("Texture preview failed", exc)
            self._unbusy(f"Could not preview {item.path}")

    def _show_error(self, title: str, exc: Exception) -> None:
        self.ax_preview.clear()
        self.ax_preview.text(
            0.5,
            0.5,
            f"{title}\n\n{exc}",
            ha="center",
            va="center",
            transform=self.ax_preview.transAxes,
            wrap=True,
        )
        self.ax_preview.axis("off")
        self.canvas.draw_idle()
        self.info_var.set(f"{title}: {exc}")

    def _show_empty_preview(self) -> None:
        self.ax_preview.clear()
        self.ax_preview.text(
            0.5,
            0.5,
            "Select PBO… or Add folder…\n\n"
            "Then click any model or texture in the catalogue.",
            ha="center",
            va="center",
            transform=self.ax_preview.transAxes,
            fontsize=14,
        )
        self.ax_preview.axis("off")
        self.canvas.draw_idle()

    def _clear_related(self) -> None:
        self._related_assets.clear()
        for child in self.related_tree.get_children():
            self.related_tree.delete(child)

    def _set_related_textures(self, paths: tuple[str, ...]) -> None:
        self._clear_related()
        by_path = {item.path: item for item in self.catalogue.textures}
        by_basename: dict[str, list[BrowserAsset]] = {}
        for item in self.catalogue.textures:
            by_basename.setdefault(item.path.rsplit("\\", 1)[-1], []).append(item)
        for index, path in enumerate(dict.fromkeys(paths)):
            asset = by_path.get(path)
            if asset is None:
                matches = by_basename.get(path.rsplit("\\", 1)[-1], [])
                asset = matches[0] if len(matches) == 1 else None
            iid = f"r{index}"
            if asset is not None:
                self._related_assets[iid] = asset
            self.related_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=("Texture" if asset else "Missing", path),
            )

    def _set_related_models(self, models: tuple[BrowserAsset, ...]) -> None:
        self._clear_related()
        for index, asset in enumerate(models):
            iid = f"r{index}"
            self._related_assets[iid] = asset
            self.related_tree.insert("", tk.END, iid=iid, values=("Model", asset.path))

    def _jump_related(self, _event=None) -> None:
        selection = self.related_tree.selection()
        if not selection:
            return
        asset = self._related_assets.get(selection[0])
        if asset is not None:
            self.jump_to(asset)

    def jump_to(self, asset: BrowserAsset) -> None:
        self.search_var.set("")
        tree = self.model_tree if asset.kind == "model" else self.texture_tree
        self.notebook.select(0 if asset.kind == "model" else 1)
        target = None
        for iid, candidate in self._row_assets.items():
            if (
                candidate.kind == asset.kind
                and candidate.path == asset.path
                and candidate.source == asset.source
            ):
                target = iid
                break
        if target is not None:
            tree.selection_set(target)
            tree.focus(target)
            tree.see(target)
        self._show_asset(asset)

    def _rotate(self, delta: float) -> None:
        if self.current_model is None:
            return
        self.azim = (self.azim + delta) % 360.0
        self._draw_model(self.current_model)

    def _tilt(self, delta: float) -> None:
        if self.current_model is None:
            return
        self.elev = max(-80.0, min(80.0, self.elev + delta))
        self._draw_model(self.current_model)

    def _reset_view(self) -> None:
        self.azim, self.elev, self.zoom = 35.0, 25.0, 1.0
        if self.current_model is not None:
            self._draw_model(self.current_model)

    def _redraw_current_model(self) -> None:
        if self.current_model is None:
            return
        self._draw_model(self.current_model)
        if self.skip_textures_var.get():
            self.status_var.set(
                "Texture lookup skipped for faster model browsing."
            )
        elif self.current is not None:
            self.status_var.set(f"Model loaded: {self.current.path}")

    def copy_current_path(self) -> None:
        item = self.current
        if item is None:
            return
        path = item.path
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.root.update_idletasks()
        except tk.TclError as exc:
            self.status_var.set(f"Could not copy path: {exc}")
            return
        kind = "P3D" if item.kind == "model" else "texture"
        self.status_var.set(f"Copied {kind} path: {path}")

    def _on_scroll_zoom(self, event) -> None:
        if self.current_model is None:
            return
        step = getattr(event, "step", 0)
        button = getattr(event, "button", None)
        if button == "up" or step > 0:
            self.zoom = min(8.0, self.zoom * 1.25)
        elif button == "down" or step < 0:
            self.zoom = max(0.2, self.zoom / 1.25)
        else:
            return
        self._draw_model(self.current_model)


def launch() -> int:
    root = tk.Tk()
    AssetBrowserApp(root)
    root.mainloop()
    return 0
