#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactively preview and categorise OFP/CWA P3D models.

The preview parser mirrors the OFP-is-not-dead CWR-CE P3D loaders closely enough
for catalogue work: ODOL v6/v7 and MLOD/SP3X first-LOD vertices *and polygon
faces* are read, then faces are rendered as a shaded mesh/wireframe. This makes
models recognisable instead of showing a cloud of unrelated vertices.

Examples::

    python tools/categorise_p3d_models.py "C:\\Games\\ColdWarAssault"

    python tools/categorise_p3d_models.py O.pbo Data3D.pbo \
        --include "o\\hous\\*.p3d" --output stock-model-categories.json

Dependencies::

    python -m pip install matplotlib numpy

Tkinter is included with the normal Windows Python installer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import struct
import sys
from typing import Iterator, Sequence

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Tkinter is required. On Windows, install Python from python.org with Tcl/Tk enabled."
    ) from exc

try:
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This tool needs matplotlib and numpy. Install them with: "
        "python -m pip install matplotlib numpy"
    ) from exc

try:
    import measure_p3d_models as measure
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import measure_p3d_models.py. Put this script beside it in tools/."
    ) from exc


DEFAULT_CATEGORIES = (
    "Residential",
    "Commercial",
    "Industrial",
    "Agricultural",
    "Military",
    "Civic / Public",
    "Religious",
    "Infrastructure",
    "Ruins",
    "Prop / Misc",
)

DEFAULT_MAX_PREVIEW_POINTS = 25_000
DEFAULT_MAX_RENDER_EDGES = 40_000
DEFAULT_MAX_SURFACE_FACES = 10_000
_MAX_FACE_COUNT = 2_000_000
_MAX_TEXTURE_COUNT = 100_000
_I16 = struct.Struct("<h")
_I32 = struct.Struct("<i")
_U8 = struct.Struct("<B")
_MLOD_FACE = struct.Struct("<32si" + "iiff" * 4 + "i")


@dataclass(slots=True)
class PreviewModel:
    model_path: str
    source: str
    measurement: measure.ModelMeasurement
    points: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    edges: np.ndarray
    original_vertex_count: int
    render_mode: str


@dataclass(slots=True)
class Classification:
    categories: list[str]
    reviewed: bool = False


def _read_i16(stream: io.BytesIO, label: str) -> int:
    return _I16.unpack(measure._read_exact(stream, _I16.size, label))[0]


def _read_i32(stream: io.BytesIO, label: str) -> int:
    return _I32.unpack(measure._read_exact(stream, _I32.size, label))[0]


def _read_u8(stream: io.BytesIO, label: str) -> int:
    return _U8.unpack(measure._read_exact(stream, _U8.size, label))[0]


def _read_odol_faces(stream: io.BytesIO, point_count: int) -> tuple[tuple[int, ...], ...]:
    """Read first-LOD ODOL face topology using the CWR-CE ODOL layout."""
    measure._read_exact(stream, 48, "ODOL LOD bounds")

    texture_count = measure._read_u32(stream, "ODOL texture count")
    if texture_count > _MAX_TEXTURE_COUNT:
        raise measure.ModelReadError(f"implausible ODOL texture count {texture_count}")
    for index in range(texture_count):
        measure._read_cstring(stream, f"ODOL texture {index}")

    for label in ("ODOL MLOD edge indices", "ODOL vertex edge indices"):
        count = measure._read_u32(stream, f"{label} count")
        measure._read_odol_array(stream, count, 2, label)

    face_count = measure._read_u32(stream, "ODOL face count")
    _offset_to_sections = measure._read_u32(stream, "ODOL section offset")
    if face_count > _MAX_FACE_COUNT:
        raise measure.ModelReadError(f"implausible ODOL face count {face_count}")

    faces: list[tuple[int, ...]] = []
    for face_index in range(face_count):
        measure._read_u32(stream, f"ODOL face {face_index} flags")
        _read_i16(stream, f"ODOL face {face_index} texture index")
        vertex_count = _read_u8(stream, f"ODOL face {face_index} vertex count")
        if vertex_count > 64:
            raise measure.ModelReadError(
                f"implausible ODOL face {face_index} vertex count {vertex_count}"
            )
        raw = measure._read_exact(
            stream, vertex_count * 2, f"ODOL face {face_index} vertex indices"
        )
        indices = struct.unpack("<" + "H" * vertex_count, raw) if vertex_count else ()
        if vertex_count in (3, 4) and all(index < point_count for index in indices):
            faces.append(tuple(int(index) for index in indices))
    return tuple(faces)


def _read_mlod_faces(
    stream: io.BytesIO, point_count: int, face_count: int
) -> tuple[tuple[int, ...], ...]:
    """Read MLOD/SP3X faces as documented by CWR-CE's MLOD loader."""
    if face_count < 0 or face_count > _MAX_FACE_COUNT:
        raise measure.ModelReadError(f"implausible SP3X face count {face_count}")

    faces: list[tuple[int, ...]] = []
    for face_index in range(face_count):
        values = _MLOD_FACE.unpack(
            measure._read_exact(stream, _MLOD_FACE.size, f"SP3X face {face_index}")
        )
        vertex_count = int(values[1])
        if vertex_count not in (3, 4):
            continue
        indices = tuple(int(values[2 + i * 4]) for i in range(vertex_count))
        if all(0 <= index < point_count for index in indices):
            if vertex_count == 3:
                indices = (indices[1], indices[0], indices[2])
            else:
                indices = (indices[1], indices[0], indices[3], indices[2])
            faces.append(indices)
    return tuple(faces)


def _extract_first_lod_geometry(
    data: bytes, *, model_path: str, source: str
) -> tuple[measure.ModelMeasurement, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Read first-LOD vertices and polygon topology from an ODOL or MLOD P3D."""
    if len(data) < 12:
        raise measure.ModelReadError("P3D is too small to contain a valid header")

    signature = data[:4]
    stream = io.BytesIO(data)
    faces: tuple[tuple[int, ...], ...] = ()

    if signature == b"ODOL":
        measure._read_exact(stream, 4, "ODOL signature")
        version = measure._read_u32(stream, "ODOL version")
        lod_count = measure._read_u32(stream, "ODOL LOD count")
        if version not in {6, 7}:
            raise measure.ModelReadError(
                f"unsupported ODOL version {version}; expected OFP/CWA v6/v7"
            )
        if lod_count <= 0 or lod_count > 128:
            raise measure.ModelReadError(f"implausible ODOL LOD count {lod_count}")

        flag_count = measure._read_u32(stream, "ODOL point-flag count")
        measure._read_odol_array(stream, flag_count, 4, "ODOL point flags")
        uv_count = measure._read_u32(stream, "ODOL UV count")
        measure._read_odol_array(stream, uv_count, 8, "ODOL UV coordinates")
        point_count = measure._read_u32(stream, "ODOL point count")
        if point_count <= 0 or point_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible ODOL point count {point_count}")
        if flag_count != point_count or uv_count != point_count:
            raise measure.ModelReadError(
                "ODOL first-LOD vertex table count mismatch "
                f"(flags={flag_count}, uv={uv_count}, points={point_count})"
            )
        raw_points = measure._read_exact(
            stream, point_count * measure._VEC3.size, "ODOL points"
        )
        points = np.frombuffer(raw_points, dtype="<f4").reshape((-1, 3)).copy()
        normal_count = measure._read_u32(stream, "ODOL normal count")
        if normal_count != point_count:
            raise measure.ModelReadError(
                f"ODOL first-LOD normal count {normal_count} does not match {point_count} vertices"
            )
        measure._read_exact(stream, normal_count * measure._VEC3.size, "ODOL normals")

        try:
            faces = _read_odol_faces(stream, point_count)
        except (measure.ModelReadError, struct.error) as exc:
            if version == 7:
                raise
            print(
                f"[preview warning] {model_path}: ODOL v6 face table could not be read: {exc}; "
                "falling back to vertices",
                file=sys.stderr,
                flush=True,
            )
            faces = ()
        format_name = "ODOL"

    elif signature == b"MLOD":
        measure._read_exact(stream, 4, "MLOD signature")
        version_raw = measure._read_u32(stream, "MLOD version")
        version_major = version_raw & 0xFF
        version_minor = (version_raw >> 8) & 0xFF
        version = version_major * 10 + version_minor
        lod_count = measure._read_u32(stream, "MLOD LOD count")
        if lod_count <= 0 or lod_count > 128:
            raise measure.ModelReadError(f"implausible MLOD LOD count {lod_count}")

        lod_signature = measure._read_exact(stream, 4, "MLOD first-LOD signature")
        if lod_signature != b"SP3X":
            raise measure.ModelReadError(
                f"unsupported MLOD LOD format {lod_signature!r}; expected SP3X"
            )
        head_size = _read_i32(stream, "SP3X header size")
        _sp3x_version = _read_i32(stream, "SP3X version")
        point_count = _read_i32(stream, "SP3X point count")
        normal_count = _read_i32(stream, "SP3X normal count")
        face_count = _read_i32(stream, "SP3X face count")
        _flags = _read_i32(stream, "SP3X flags")
        if head_size < 28 or head_size > 4096:
            raise measure.ModelReadError(f"implausible SP3X header size {head_size}")
        if head_size > 28:
            measure._read_exact(stream, head_size - 28, "SP3X extra header")
        if point_count <= 0 or point_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible SP3X point count {point_count}")
        if normal_count < 0 or normal_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible SP3X normal count {normal_count}")

        raw = measure._read_exact(stream, point_count * 16, "SP3X points")
        records = np.frombuffer(
            raw,
            dtype=np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("flags", "<i4")]
            ),
        )
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(
            np.float32, copy=False
        )
        measure._read_exact(stream, normal_count * measure._VEC3.size, "SP3X normals")
        faces = _read_mlod_faces(stream, point_count, face_count)
        format_name = "MLOD/SP3X"

    else:
        raise measure.ModelReadError(f"unsupported P3D signature {signature!r}")

    measurement = measure._measurement(
        model_path=model_path,
        source=source,
        format_name=format_name,
        version=version,
        lod="first",
        points=points,
    )
    return measurement, points, faces


def _decimate_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _collect_edges(faces: Sequence[Sequence[int]]) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for face in faces:
        count = len(face)
        for index in range(count):
            a = int(face[index])
            b = int(face[(index + 1) % count])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edge_set.add((a, b))
    if not edge_set:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(sorted(edge_set), dtype=np.int32)


def _sample_rows(array: np.ndarray, maximum: int) -> np.ndarray:
    if len(array) <= maximum:
        return array
    indices = np.linspace(0, len(array) - 1, maximum, dtype=np.int64)
    return array[indices]


def _sample_faces(
    faces: tuple[tuple[int, ...], ...], maximum: int
) -> tuple[tuple[int, ...], ...]:
    if len(faces) <= maximum:
        return faces
    indices = np.linspace(0, len(faces) - 1, maximum, dtype=np.int64)
    return tuple(faces[int(index)] for index in indices)


def _resolve_inputs(raw_inputs: Sequence[Path]) -> list[Path]:
    inputs = [path.expanduser() for path in raw_inputs]
    if not inputs or all(path.exists() for path in inputs):
        return inputs
    if len(inputs) > 1:
        joined = Path(" ".join(str(path) for path in inputs)).expanduser()
        if joined.exists():
            return [joined]
    return inputs


def _preview_models(
    inputs: Sequence[Path], patterns: Sequence[str], max_points: int
) -> Iterator[PreviewModel | measure.ModelFailure]:
    for model_path, source, data in measure._iter_models(inputs, patterns):
        try:
            measurement, points, faces = _extract_first_lod_geometry(
                data, model_path=model_path, source=source
            )
            edges = _collect_edges(faces)
            if faces:
                preview_points = points.astype(np.float32, copy=False)
                render_mode = "mesh"
            else:
                preview_points = _decimate_points(points, max_points)
                render_mode = "vertex fallback"
            yield PreviewModel(
                model_path=model_path,
                source=source,
                measurement=measurement,
                points=preview_points,
                faces=faces,
                edges=edges,
                original_vertex_count=len(points),
                render_mode=render_mode,
            )
        except (
            measure.ModelReadError,
            OSError,
            struct.error,
            OverflowError,
            IndexError,
            ValueError,
        ) as exc:
            failure = measure.ModelFailure(model_path=model_path, source=source, error=str(exc))
            print(
                f"[model error] {failure.model_path}\n"
                f"  source: {failure.source}\n"
                f"  error:  {failure.error}",
                file=sys.stderr,
                flush=True,
            )
            yield failure


def _load_state(path: Path) -> tuple[dict[str, Classification], list[str]]:
    if not path.exists():
        return {}, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read existing classification file {path}: {exc}") from exc

    result: dict[str, Classification] = {}
    saved_categories = [str(value) for value in raw.get("categories", []) if str(value).strip()]
    for item in raw.get("models", []):
        if not isinstance(item, dict) or "model_path" not in item:
            continue
        model_path = measure._canonical_model_path(str(item["model_path"]))
        result[model_path] = Classification(
            categories=[str(value) for value in item.get("categories", [])],
            reviewed=bool(item.get("reviewed", True)),
        )
    return result, saved_categories


def _set_2d_equal(ax, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    span = max(xmax - xmin, ymax - ymin, 1.0e-5)
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    half = span * 0.56
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


def _edge_segments_2d(points: np.ndarray, edges: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    sampled = _sample_rows(edges, DEFAULT_MAX_RENDER_EDGES)
    if len(sampled) == 0:
        return np.empty((0, 2, 2), dtype=np.float32)
    p0 = points[sampled[:, 0]][:, axes]
    p1 = points[sampled[:, 1]][:, axes]
    return np.stack((p0, p1), axis=1)


def _edge_segments_3d(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    sampled = _sample_rows(edges, DEFAULT_MAX_RENDER_EDGES)
    if len(sampled) == 0:
        return np.empty((0, 2, 3), dtype=np.float32)
    display = points[:, [0, 2, 1]]
    return np.stack((display[sampled[:, 0]], display[sampled[:, 1]]), axis=1)


class CategoriserApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        model_iter: Iterator[PreviewModel | measure.ModelFailure],
        output: Path,
        categories: Sequence[str],
        state: dict[str, Classification],
    ) -> None:
        self.root = root
        self.model_iter = model_iter
        self.output = output
        self.categories = list(categories)
        self.state = state
        self.models: list[PreviewModel] = []
        self.failures: list[measure.ModelFailure] = []
        self.index = -1
        self.current: PreviewModel | None = None
        self.exhausted = False
        self._updating_checks = False

        self.root.title("CWR P3D Model Categoriser")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 700)
        self._build_ui()
        self.root.bind("<Left>", lambda _event: self.previous_model())
        self.root.bind("<Right>", lambda _event: self.next_model())
        self.root.bind("<Control-s>", lambda _event: self.save_state())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.next_model(mark_current=False)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        self.path_var = tk.StringVar(value="Loading model...")
        self.progress_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.path_var, font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, textvariable=self.progress_var).grid(row=0, column=1, sticky="e")

        preview_frame = ttk.Frame(outer)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax3d = self.figure.add_subplot(2, 2, 1, projection="3d")
        self.ax_top = self.figure.add_subplot(2, 2, 2)
        self.ax_front = self.figure.add_subplot(2, 2, 3)
        self.ax_side = self.figure.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, preview_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        side = ttk.Frame(outer, padding=(8, 4))
        side.grid(row=1, column=1, sticky="ns")
        ttk.Label(side, text="Categories", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        self.category_vars: dict[str, tk.BooleanVar] = {}
        for category in self.categories:
            var = tk.BooleanVar(value=False)
            self.category_vars[category] = var
            ttk.Checkbutton(side, text=category, variable=var, command=self._category_changed).pack(
                anchor="w", fill="x", pady=2
            )

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=12)
        self.surface_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            side,
            text="Shaded surfaces",
            variable=self.surface_var,
            command=lambda: self._draw_model(self.current) if self.current is not None else None,
        ).pack(anchor="w")

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=12)
        self.info_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.info_var, justify=tk.LEFT).pack(anchor="w")
        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill="x", pady=12)
        self.save_status_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.save_status_var, wraplength=270).pack(anchor="w")

        nav = ttk.Frame(outer)
        nav.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        nav.columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav, text="◀ Previous", command=self.previous_model)
        self.prev_button.grid(row=0, column=0, padx=(0, 6))
        ttk.Label(
            nav,
            text="Left/Right navigate • Ctrl+S saves • checkbox changes autosave",
        ).grid(row=0, column=1)
        self.next_button = ttk.Button(nav, text="Next ▶", command=self.next_model)
        self.next_button.grid(row=0, column=2, padx=(6, 0))

    def _set_busy(self, text: str) -> None:
        self.progress_var.set(text)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()

    def _clear_busy(self) -> None:
        self.root.configure(cursor="")

    def _load_next_from_generator(self) -> PreviewModel | None:
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
            return item
        return None

    def _current_categories(self) -> list[str]:
        return [name for name, var in self.category_vars.items() if var.get()]

    def _commit_current(self, *, reviewed: bool) -> None:
        if self.current is None or self._updating_checks:
            return
        existing = self.state.get(self.current.model_path, Classification([]))
        self.state[self.current.model_path] = Classification(
            categories=self._current_categories(), reviewed=reviewed or existing.reviewed
        )

    def _category_changed(self) -> None:
        if self._updating_checks or self.current is None:
            return
        self._commit_current(reviewed=True)
        self.save_state()

    def next_model(self, *, mark_current: bool = True) -> None:
        if mark_current and self.current is not None:
            self._commit_current(reviewed=True)
            self.save_state()
        target = self.index + 1
        if target >= len(self.models):
            self._set_busy("Loading next model...")
            try:
                model = self._load_next_from_generator()
            except (OSError, ValueError) as exc:
                print(f"[scan error] {exc}", file=sys.stderr, flush=True)
                messagebox.showerror("Scan error", str(exc))
                return
            finally:
                self._clear_busy()
            if model is None:
                summary = f"End of scan: {len(self.models)} model(s), {len(self.failures)} failure(s)"
                print(f"[scan] {summary}", file=sys.stderr, flush=True)
                self.progress_var.set(
                    f"End of scan • {len(self.models)} models • {len(self.failures)} failures"
                )
                self.next_button.configure(state=tk.DISABLED)
                return
        self.index = target
        self._show_model(self.models[self.index])

    def previous_model(self) -> None:
        if self.current is not None:
            self._commit_current(reviewed=True)
            self.save_state()
        if self.index <= 0:
            return
        self.index -= 1
        self._show_model(self.models[self.index])

    def _show_model(self, model: PreviewModel) -> None:
        self.current = model
        self.path_var.set(model.model_path)
        suffix = " +" if not self.exhausted else ""
        self.progress_var.set(
            f"Model {self.index + 1}/{len(self.models)}{suffix} • failures skipped: {len(self.failures)}"
        )
        classification = self.state.get(model.model_path, Classification([]))
        selected = set(classification.categories)
        self._updating_checks = True
        try:
            for category, var in self.category_vars.items():
                var.set(category in selected)
        finally:
            self._updating_checks = False

        m = model.measurement
        reviewed = "yes" if classification.reviewed else "no"
        self.info_var.set(
            f"Format: {m.format} v{m.version}\n"
            f"Vertices: {model.original_vertex_count:,}\n"
            f"Faces: {len(model.faces):,}\n"
            f"Edges: {len(model.edges):,}\n"
            f"Preview: {model.render_mode}\n"
            f"Width: {m.width_m:g} m\n"
            f"Height: {m.height_m:g} m\n"
            f"Length: {m.length_m:g} m\n"
            f"Footprint: {m.footprint_area_m2:g} m²\n"
            f"Reviewed: {reviewed}"
        )
        self.prev_button.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL)
        self._draw_model(model)

    def _draw_model(self, model: PreviewModel | None) -> None:
        if model is None:
            return
        points = model.points
        x, height, depth = points[:, 0], points[:, 1], points[:, 2]
        for ax in (self.ax3d, self.ax_top, self.ax_front, self.ax_side):
            ax.clear()

        if len(model.edges):
            segments3d = _edge_segments_3d(points, model.edges)
            if self.surface_var.get() and model.faces:
                display = points[:, [0, 2, 1]]
                sampled_faces = _sample_faces(model.faces, DEFAULT_MAX_SURFACE_FACES)
                polygons = [display[np.asarray(face, dtype=np.int32)] for face in sampled_faces]
                self.ax3d.add_collection3d(
                    Poly3DCollection(polygons, alpha=0.22, linewidths=0.15)
                )
            self.ax3d.add_collection3d(Line3DCollection(segments3d, linewidths=0.45))

            top_segments = _edge_segments_2d(points, model.edges, (0, 2))
            front_segments = _edge_segments_2d(points, model.edges, (0, 1))
            side_segments = _edge_segments_2d(points, model.edges, (2, 1))
            self.ax_top.add_collection(LineCollection(top_segments, linewidths=0.45))
            self.ax_front.add_collection(LineCollection(front_segments, linewidths=0.45))
            self.ax_side.add_collection(LineCollection(side_segments, linewidths=0.45))
        else:
            self.ax3d.scatter(x, depth, height, s=1.0, alpha=0.55, depthshade=False)
            self.ax_top.scatter(x, depth, s=1.0, alpha=0.55)
            self.ax_front.scatter(x, height, s=1.0, alpha=0.55)
            self.ax_side.scatter(depth, height, s=1.0, alpha=0.55)

        self.ax3d.set_title("Perspective mesh (drag to rotate)")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Z")
        self.ax3d.set_zlabel("Y height")
        self.ax3d.view_init(elev=24, azim=-55)

        m = model.measurement
        span = max(m.max_x - m.min_x, m.max_z - m.min_z, m.max_y - m.min_y, 1.0e-5)
        x_mid = (m.min_x + m.max_x) / 2.0
        z_mid = (m.min_z + m.max_z) / 2.0
        y_mid = (m.min_y + m.max_y) / 2.0
        half = span * 0.56
        self.ax3d.set_xlim(x_mid - half, x_mid + half)
        self.ax3d.set_ylim(z_mid - half, z_mid + half)
        self.ax3d.set_zlim(y_mid - half, y_mid + half)
        try:
            self.ax3d.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

        self.ax_top.set_title("Top: X / Z")
        self.ax_top.set_xlabel("X")
        self.ax_top.set_ylabel("Z")
        _set_2d_equal(self.ax_top, x, depth)
        self.ax_front.set_title("Front: X / Y")
        self.ax_front.set_xlabel("X")
        self.ax_front.set_ylabel("Y height")
        _set_2d_equal(self.ax_front, x, height)
        self.ax_side.set_title("Side: Z / Y")
        self.ax_side.set_xlabel("Z")
        self.ax_side.set_ylabel("Y height")
        _set_2d_equal(self.ax_side, depth, height)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def save_state(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": 1,
            "categories": self.categories,
            "reviewed_count": sum(1 for item in self.state.values() if item.reviewed),
            "classified_count": sum(1 for item in self.state.values() if item.categories),
            "models": [
                {
                    "model_path": model_path,
                    "categories": self.state[model_path].categories,
                    "reviewed": self.state[model_path].reviewed,
                }
                for model_path in sorted(self.state)
            ],
            "failures": [asdict(item) for item in self.failures],
        }
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        temp = self.output.with_name(self.output.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(self.output)
        self.save_status_var.set(f"Saved: {self.output}")

    def close(self) -> None:
        try:
            if self.current is not None:
                self._commit_current(reviewed=False)
            self.save_state()
        except OSError as exc:
            print(f"[save error] {exc}", file=sys.stderr, flush=True)
            if not messagebox.askyesno(
                "Could not save", f"Could not save classifications:\n{exc}\n\nClose anyway?"
            ):
                return
        self.root.destroy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview OFP/CWA P3D first-LOD meshes and assign model categories interactively."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="P3D, PBO, or directory to scan")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only include canonical model paths matching GLOB. May be repeated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("p3d-model-categories.json"),
        help="Classification JSON file. Existing classifications are loaded automatically.",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        metavar="NAME",
        help="Use a custom category checkbox. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--max-preview-points",
        type=int,
        default=DEFAULT_MAX_PREVIEW_POINTS,
        metavar="N",
        help=(
            "Maximum points drawn only when polygon topology cannot be read "
            f"(default: {DEFAULT_MAX_PREVIEW_POINTS:,})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_preview_points < 100:
        print("error: --max-preview-points must be at least 100", file=sys.stderr)
        return 2

    inputs = _resolve_inputs(args.inputs)
    missing = [path for path in inputs if not path.exists()]
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        message = (
            "Input path does not exist:\n"
            f"{joined}\n\n"
            "If the path contains spaces, put the whole path in double quotes."
        )
        print(f"error: {message}", file=sys.stderr)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Scan path not found", message, parent=root)
            root.destroy()
        except tk.TclError:
            pass
        return 2

    try:
        state, saved_categories = _load_state(args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    categories = args.categories or saved_categories or list(DEFAULT_CATEGORIES)
    categories = list(dict.fromkeys(category.strip() for category in categories if category.strip()))
    if not categories:
        print("error: at least one category is required", file=sys.stderr)
        return 2

    iterator = _preview_models(inputs, args.include, args.max_preview_points)
    root = tk.Tk()
    app = CategoriserApp(
        root,
        model_iter=iterator,
        output=args.output,
        categories=categories,
        state=state,
    )
    root.mainloop()
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
