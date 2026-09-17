"""Tk/Matplotlib UI for browsing and categorising textured P3D models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterator, Sequence

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

import measure_p3d_models as measure
from p3d_preview_geometry import PreviewModel
from p3d_texture_io import TextureResolver
from p3d_texture_render import render_textured_model

DEFAULT_MAX_RENDER_EDGES = 40_000


@dataclass(slots=True)
class Classification:
    categories: list[str]
    reviewed: bool = False


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
        result[key] = Classification([str(v) for v in item.get("categories", [])], bool(item.get("reviewed", True)))
    return result, categories


def _equal_2d(ax, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    span = max(xmax-xmin, ymax-ymin, 1e-5)
    half = span * 0.56
    ax.set_xlim((xmin+xmax)/2-half, (xmin+xmax)/2+half)
    ax.set_ylim((ymin+ymax)/2-half, (ymin+ymax)/2+half)
    ax.set_aspect("equal", adjustable="box")


def _segments(points: np.ndarray, edges: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    if len(edges) > DEFAULT_MAX_RENDER_EDGES:
        picks = np.linspace(0, len(edges)-1, DEFAULT_MAX_RENDER_EDGES, dtype=np.int64)
        edges = edges[picks]
    if len(edges) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    return np.stack((points[edges[:,0]][:,axes], points[edges[:,1]][:,axes]), axis=1)


class CategoriserApp:
    def __init__(
        self, root: tk.Tk, *, model_iter: Iterator[PreviewModel | measure.ModelFailure],
        texture_resolver: TextureResolver, output: Path, categories: Sequence[str],
        state: dict[str, Classification],
    ) -> None:
        self.root, self.model_iter, self.texture_resolver = root, model_iter, texture_resolver
        self.output, self.categories, self.state = output, list(categories), state
        self.models: list[PreviewModel] = []
        self.failures: list[measure.ModelFailure] = []
        self.index, self.current, self.exhausted = -1, None, False
        self._updating_checks = False
        self.azim, self.elev = 35.0, 25.0
        root.title("CWR P3D Model Categoriser")
        root.geometry("1500x920")
        root.minsize(1100, 720)
        self._build_ui()
        root.bind("<Left>", lambda _e: self.previous_model())
        root.bind("<Right>", lambda _e: self.next_model())
        root.bind("<Control-s>", lambda _e: self.save_state())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.next_model(mark_current=False)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8); outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1); outer.rowconfigure(1, weight=1)
        header = ttk.Frame(outer); header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,6)); header.columnconfigure(0, weight=1)
        self.path_var, self.progress_var = tk.StringVar(value="Loading model..."), tk.StringVar()
        ttk.Label(header, textvariable=self.path_var, font=("TkDefaultFont",11,"bold")).grid(row=0,column=0,sticky="w")
        ttk.Label(header, textvariable=self.progress_var).grid(row=0,column=1,sticky="e")

        frame = ttk.Frame(outer); frame.grid(row=1,column=0,sticky="nsew",padx=(0,8)); frame.rowconfigure(0,weight=1); frame.columnconfigure(0,weight=1)
        self.figure = Figure(figsize=(11,7.5), dpi=100)
        self.ax_preview = self.figure.add_subplot(2,2,1); self.ax_top = self.figure.add_subplot(2,2,2)
        self.ax_front = self.figure.add_subplot(2,2,3); self.ax_side = self.figure.add_subplot(2,2,4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame); self.canvas.get_tk_widget().grid(row=0,column=0,sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, frame, pack_toolbar=False); toolbar.update(); toolbar.grid(row=1,column=0,sticky="ew")

        side = ttk.Frame(outer, padding=(8,4)); side.grid(row=1,column=1,sticky="ns")
        ttk.Label(side,text="Categories",font=("TkDefaultFont",11,"bold")).pack(anchor="w",pady=(0,6))
        self.category_vars: dict[str, tk.BooleanVar] = {}
        for category in self.categories:
            var = tk.BooleanVar(False); self.category_vars[category] = var
            ttk.Checkbutton(side,text=category,variable=var,command=self._category_changed).pack(anchor="w",fill="x",pady=2)
        ttk.Separator(side,orient=tk.HORIZONTAL).pack(fill="x",pady=10)
        ttk.Label(side,text="View",font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        row = ttk.Frame(side); row.pack(fill="x",pady=4)
        ttk.Button(row,text="↶ 15°",command=lambda:self._rotate(-15)).pack(side=tk.LEFT)
        ttk.Button(row,text="15° ↷",command=lambda:self._rotate(15)).pack(side=tk.LEFT,padx=4)
        row2 = ttk.Frame(side); row2.pack(fill="x")
        ttk.Button(row2,text="Tilt +",command=lambda:self._tilt(10)).pack(side=tk.LEFT)
        ttk.Button(row2,text="Tilt -",command=lambda:self._tilt(-10)).pack(side=tk.LEFT,padx=4)
        ttk.Separator(side,orient=tk.HORIZONTAL).pack(fill="x",pady=10)
        self.info_var, self.texture_var, self.status_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Label(side,textvariable=self.info_var,justify=tk.LEFT,wraplength=300).pack(anchor="w")
        ttk.Separator(side,orient=tk.HORIZONTAL).pack(fill="x",pady=10)
        ttk.Label(side,text="Textures",font=("TkDefaultFont",10,"bold")).pack(anchor="w")
        ttk.Label(side,textvariable=self.texture_var,justify=tk.LEFT,wraplength=300).pack(anchor="w")
        ttk.Separator(side,orient=tk.HORIZONTAL).pack(fill="x",pady=10)
        ttk.Label(side,textvariable=self.status_var,wraplength=300).pack(anchor="w")

        nav = ttk.Frame(outer); nav.grid(row=2,column=0,columnspan=2,sticky="ew",pady=(8,0)); nav.columnconfigure(1,weight=1)
        self.prev_button = ttk.Button(nav,text="◀ Previous",command=self.previous_model); self.prev_button.grid(row=0,column=0,padx=(0,6))
        ttk.Label(nav,text="Left/Right navigate • Ctrl+S saves • checkbox changes autosave").grid(row=0,column=1)
        self.next_button = ttk.Button(nav,text="Next ▶",command=self.next_model); self.next_button.grid(row=0,column=2,padx=(6,0))

    def _busy(self, text: str) -> None:
        self.progress_var.set(text); self.root.configure(cursor="watch"); self.root.update_idletasks()

    def _unbusy(self) -> None:
        self.root.configure(cursor="")

    def _load_next(self) -> PreviewModel | None:
        while not self.exhausted:
            try: item = next(self.model_iter)
            except StopIteration: self.exhausted = True; return None
            if isinstance(item, measure.ModelFailure): self.failures.append(item); continue
            self.models.append(item); return item
        return None

    def _current_categories(self) -> list[str]:
        return [name for name,var in self.category_vars.items() if var.get()]

    def _commit(self, reviewed: bool) -> None:
        if self.current is None or self._updating_checks: return
        old = self.state.get(self.current.model_path, Classification([]))
        self.state[self.current.model_path] = Classification(self._current_categories(), reviewed or old.reviewed)

    def _category_changed(self) -> None:
        if self._updating_checks or self.current is None: return
        self._commit(True); self.save_state()

    def _rotate(self, amount: float) -> None:
        self.azim = (self.azim + amount) % 360; self._redraw()

    def _tilt(self, amount: float) -> None:
        self.elev = max(-80,min(80,self.elev+amount)); self._redraw()

    def _redraw(self) -> None:
        if self.current is None: return
        self._busy("Rendering textured model...")
        try: self._draw_model(self.current)
        finally: self._unbusy()

    def next_model(self, *, mark_current: bool=True) -> None:
        if mark_current and self.current is not None: self._commit(True); self.save_state()
        target = self.index + 1
        if target >= len(self.models):
            self._busy("Loading next model...")
            try: model = self._load_next()
            except (OSError,ValueError) as exc:
                print(f"[scan error] {exc}",file=sys.stderr,flush=True); messagebox.showerror("Scan error",str(exc)); return
            finally: self._unbusy()
            if model is None:
                summary=f"End of scan: {len(self.models)} model(s), {len(self.failures)} failure(s)"
                print(f"[scan] {summary}",file=sys.stderr,flush=True); self.progress_var.set(summary); self.next_button.configure(state=tk.DISABLED); return
        self.index=target; self._show_model(self.models[self.index])

    def previous_model(self) -> None:
        if self.current is not None: self._commit(True); self.save_state()
        if self.index <= 0: return
        self.index -= 1; self._show_model(self.models[self.index])

    def _show_model(self, model: PreviewModel) -> None:
        self.current=model; self.path_var.set(model.model_path)
        suffix=" +" if not self.exhausted else ""
        self.progress_var.set(f"Model {self.index+1}/{len(self.models)}{suffix} • failures skipped: {len(self.failures)}")
        classification=self.state.get(model.model_path,Classification([])); selected=set(classification.categories)
        self._updating_checks=True
        try:
            for category,var in self.category_vars.items(): var.set(category in selected)
        finally: self._updating_checks=False
        m=model.measurement
        self.info_var.set(
            f"Format: {m.format} v{m.version}\nVertices: {model.original_vertex_count:,}\nFaces: {len(model.faces):,}\n"
            f"Edges: {len(model.edges):,}\nPreview: {model.render_mode}\nWidth: {m.width_m:g} m\nHeight: {m.height_m:g} m\n"
            f"Length: {m.length_m:g} m\nFootprint: {m.footprint_area_m2:g} m²\nReviewed: {'yes' if classification.reviewed else 'no'}"
        )
        names=[measure._canonical_model_path(n) for n in model.texture_paths if n]
        self.texture_var.set("No texture references in this LOD" if not names else f"{len(names)} referenced\n"+"\n".join(names[:8])+(f"\n… +{len(names)-8} more" if len(names)>8 else ""))
        self.prev_button.configure(state=tk.NORMAL if self.index>0 else tk.DISABLED); self.next_button.configure(state=tk.NORMAL)
        self._redraw()

    def _draw_model(self, model: PreviewModel) -> None:
        points=model.points; x,height,depth=points[:,0],points[:,1],points[:,2]
        for ax in (self.ax_preview,self.ax_top,self.ax_front,self.ax_side): ax.clear()
        if model.faces:
            image,hits,misses=render_textured_model(points,model.faces,self.texture_resolver,model.source,width=760,height=540,azim_deg=self.azim,elev_deg=self.elev)
            self.ax_preview.imshow(image); self.ax_preview.set_title(f"Textured perspective • az {self.azim:.0f}° / el {self.elev:.0f}°"); self.ax_preview.axis("off")
            if misses: self.status_var.set(f"Texture sampling: {hits} face hit(s), {misses} missing/unreadable face texture(s). See console.")
        else:
            self.ax_preview.scatter(x,depth,s=1,alpha=.55); self.ax_preview.set_title("Topology unavailable: vertex fallback"); _equal_2d(self.ax_preview,x,depth)
        if len(model.edges):
            self.ax_top.add_collection(LineCollection(_segments(points,model.edges,(0,2)),linewidths=.45))
            self.ax_front.add_collection(LineCollection(_segments(points,model.edges,(0,1)),linewidths=.45))
            self.ax_side.add_collection(LineCollection(_segments(points,model.edges,(2,1)),linewidths=.45))
        else:
            self.ax_top.scatter(x,depth,s=1,alpha=.55); self.ax_front.scatter(x,height,s=1,alpha=.55); self.ax_side.scatter(depth,height,s=1,alpha=.55)
        for ax,title,xv,yv,xlab,ylab in (
            (self.ax_top,"Top: X / Z",x,depth,"X","Z"),(self.ax_front,"Front: X / Y",x,height,"X","Y height"),(self.ax_side,"Side: Z / Y",depth,height,"Z","Y height")):
            ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylabel(ylab); _equal_2d(ax,xv,yv)
        self.figure.tight_layout(pad=2); self.canvas.draw_idle()

    def save_state(self) -> None:
        self.output.parent.mkdir(parents=True,exist_ok=True)
        report={"schema":1,"categories":self.categories,
                "reviewed_count":sum(1 for x in self.state.values() if x.reviewed),
                "classified_count":sum(1 for x in self.state.values() if x.categories),
                "models":[{"model_path":k,"categories":self.state[k].categories,"reviewed":self.state[k].reviewed} for k in sorted(self.state)],
                "failures":[asdict(x) for x in self.failures]}
        temp=self.output.with_name(self.output.name+".tmp"); temp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); temp.replace(self.output)
        self.status_var.set(f"Saved: {self.output}")

    def close(self) -> None:
        try:
            if self.current is not None: self._commit(False)
            self.save_state()
        except OSError as exc:
            print(f"[save error] {exc}",file=sys.stderr,flush=True)
            if not messagebox.askyesno("Could not save",f"Could not save classifications:\n{exc}\n\nClose anyway?"): return
        self.root.destroy()
