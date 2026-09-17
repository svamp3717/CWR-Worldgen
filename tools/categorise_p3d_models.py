#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactively preview and categorise OFP/CWA P3D models.

Place this file next to ``measure_p3d_models.py`` in the repository's ``tools``
directory. It reuses that script's PBO reader, LZSS decompressor, path matching,
and measurement helpers.

The preview uses the first LOD vertex table. The existing native reader does not
parse polygon faces/materials/textures, so the interactive view is a point-cloud
representation of the real model geometry, accompanied by top/front/side
orthographic views. For building categorisation this is normally enough to make
shape and scale obvious without needing an external OFP model viewer.

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
except ImportError as exc:  # pragma: no cover - platform packaging issue
    raise SystemExit(
        "Tkinter is required. On Windows, install Python from python.org with Tcl/Tk enabled."
    ) from exc

try:
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - dependency issue
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

# A preview does not need millions of points. Even 25k vertices gives a very
# readable silhouette while keeping Matplotlib responsive on older machines.
DEFAULT_MAX_PREVIEW_POINTS = 25_000


@dataclass(slots=True)
class PreviewModel:
    model_path: str
    source: str
    measurement: measure.ModelMeasurement
    points: np.ndarray
    original_vertex_count: int


@dataclass(slots=True)
class Classification:
    categories: list[str]
    reviewed: bool = False


def _extract_first_lod_points(
    data: bytes, *, model_path: str, source: str
) -> tuple[measure.ModelMeasurement, np.ndarray]:
    """Read the same first-LOD vertex data used by the measurement script."""
    if len(data) < 12:
        raise measure.ModelReadError("P3D is too small to contain a valid header")

    signature = data[:4]
    stream = io.BytesIO(data)

    if signature == b"ODOL":
        if measure._read_exact(stream, 4, "ODOL signature") != b"ODOL":
            raise measure.ModelReadError("not an ODOL P3D")
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
                f"ODOL first-LOD normal count {normal_count} does not match "
                f"{point_count} vertices"
            )
        measure._read_exact(
            stream, normal_count * measure._VEC3.size, "ODOL normals"
        )
        format_name = "ODOL"

    elif signature == b"MLOD":
        if measure._read_exact(stream, 4, "MLOD signature") != b"MLOD":
            raise measure.ModelReadError("not an MLOD P3D")
        version = measure._read_u32(stream, "MLOD version")
        lod_count = measure._read_u32(stream, "MLOD LOD count")
        if lod_count <= 0 or lod_count > 128:
            raise measure.ModelReadError(f"implausible MLOD LOD count {lod_count}")

        lod_signature = measure._read_exact(stream, 4, "MLOD first-LOD signature")
        if lod_signature != b"SP3X":
            try:
                readable = lod_signature.decode("ascii")
            except UnicodeDecodeError:
                readable = repr(lod_signature)
            raise measure.ModelReadError(
                f"unsupported MLOD LOD format {readable}; expected SP3X"
            )

        major = measure._read_u32(stream, "SP3X major version")
        _minor = measure._read_u32(stream, "SP3X minor version")
        point_count = measure._read_u32(stream, "SP3X point count")
        normal_count = measure._read_u32(stream, "SP3X normal count")
        _face_count = measure._read_u32(stream, "SP3X face count")
        _flags = measure._read_u32(stream, "SP3X flags")
        if point_count <= 0 or point_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible SP3X point count {point_count}")
        if major not in {27, 28}:
            raise measure.ModelReadError(f"unsupported SP3X major version {major}")

        # SP3X stores x/y/z plus a signed 32-bit point flag for each point.
        raw = measure._read_exact(stream, point_count * 16, "SP3X points")
        records = np.frombuffer(
            raw,
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("flags", "<i4"),
                ]
            ),
        )
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(
            np.float32, copy=False
        )
        measure._read_exact(
            stream, normal_count * measure._VEC3.size, "SP3X normals"
        )
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
    return measurement, points


def _decimate_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically sample points across the whole vertex array."""
    if len(points) <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _resolve_inputs(raw_inputs: Sequence[Path]) -> list[Path]:
    """Resolve CLI inputs and recover a single unquoted path containing spaces.

    Windows shells split an unquoted path such as ``ARMA Cold War Assault`` into
    several positional arguments. If the arguments do not exist individually
    but joining them with spaces produces a real path, treat that joined path as
    the intended single input. Genuine multi-input scans are left untouched.
    """
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
            measurement, points = _extract_first_lod_points(
                data, model_path=model_path, source=source
            )
            yield PreviewModel(
                model_path=model_path,
                source=source,
                measurement=measurement,
                points=_decimate_points(points, max_points),
                original_vertex_count=len(points),
            )
        except (
            measure.ModelReadError,
            OSError,
            struct.error,
            OverflowError,
            IndexError,
            ValueError,
        ) as exc:
            failure = measure.ModelFailure(
                model_path=model_path, source=source, error=str(exc)
            )
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
        categories = [str(value) for value in item.get("categories", [])]
        result[model_path] = Classification(
            categories=categories,
            reviewed=bool(item.get("reviewed", True)),
        )
    return result, saved_categories


def _set_2d_equal(ax, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xspan = max(xmax - xmin, 1.0e-5)
    yspan = max(ymax - ymin, 1.0e-5)
    span = max(xspan, yspan)
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    pad = span * 0.06
    half = span / 2.0 + pad
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


def _draw_bounds_3d(ax, m: measure.ModelMeasurement) -> None:
    # Matplotlib's vertical axis is Z. The P3D height axis is Y, so display
    # (x, z, y) to keep the model visually upright.
    xs = (m.min_x, m.max_x)
    ys = (m.min_z, m.max_z)
    zs = (m.min_y, m.max_y)
    corners = [(x, y, z) for x in xs for y in ys for z in zs]
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            diff = sum(1 for av, bv in zip(a, b) if av != bv)
            if diff == 1:
                ax.plot((a[0], b[0]), (a[1], b[1]), (a[2], b[2]), linewidth=0.55)


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
        self.figure.tight_layout(pad=2.0)

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
            ttk.Checkbutton(
                side,
                text=category,
                variable=var,
                command=self._category_changed,
            ).pack(anchor="w", fill="x", pady=2)

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
            text="Left/Right arrows navigate • Ctrl+S saves • checkbox changes autosave",
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
            categories=self._current_categories(),
            reviewed=reviewed or existing.reviewed,
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
                self._clear_busy()
                print(f"[scan error] {exc}", file=sys.stderr, flush=True)
                messagebox.showerror("Scan error", str(exc))
                return
            finally:
                self._clear_busy()
            if model is None:
                summary = (
                    f"End of scan: {len(self.models)} model(s), "
                    f"{len(self.failures)} failure(s)"
                )
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
            f"Previewed: {len(model.points):,}\n"
            f"Width: {m.width_m:g} m\n"
            f"Height: {m.height_m:g} m\n"
            f"Length: {m.length_m:g} m\n"
            f"Footprint: {m.footprint_area_m2:g} m²\n"
            f"Reviewed: {reviewed}"
        )
        self.prev_button.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL)
        self._draw_model(model)

    def _draw_model(self, model: PreviewModel) -> None:
        points = model.points
        x = points[:, 0]
        height = points[:, 1]
        depth = points[:, 2]

        for ax in (self.ax3d, self.ax_top, self.ax_front, self.ax_side):
            ax.clear()

        # 3D perspective: remap P3D Y (height) to Matplotlib Z (vertical).
        self.ax3d.scatter(x, depth, height, s=1.0, alpha=0.7, depthshade=False)
        _draw_bounds_3d(self.ax3d, model.measurement)
        self.ax3d.set_title("Perspective (drag to rotate)")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Z")
        self.ax3d.set_zlabel("Y height")
        self.ax3d.view_init(elev=24, azim=-55)

        m = model.measurement
        x_span = max(m.max_x - m.min_x, 1.0e-5)
        z_span = max(m.max_z - m.min_z, 1.0e-5)
        y_span = max(m.max_y - m.min_y, 1.0e-5)
        span = max(x_span, z_span, y_span)
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

        self.ax_top.scatter(x, depth, s=1.0, alpha=0.65)
        self.ax_top.set_title("Top: X / Z")
        self.ax_top.set_xlabel("X")
        self.ax_top.set_ylabel("Z")
        _set_2d_equal(self.ax_top, x, depth)

        self.ax_front.scatter(x, height, s=1.0, alpha=0.65)
        self.ax_front.set_title("Front: X / Y")
        self.ax_front.set_xlabel("X")
        self.ax_front.set_ylabel("Y height")
        _set_2d_equal(self.ax_front, x, height)

        self.ax_side.scatter(depth, height, s=1.0, alpha=0.65)
        self.ax_side.set_title("Side: Z / Y")
        self.ax_side.set_xlabel("Z")
        self.ax_side.set_ylabel("Y height")
        _set_2d_equal(self.ax_side, depth, height)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def save_state(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)

        model_items = []
        for model_path in sorted(self.state):
            classification = self.state[model_path]
            model_items.append(
                {
                    "model_path": model_path,
                    "categories": classification.categories,
                    "reviewed": classification.reviewed,
                }
            )

        report = {
            "schema": 1,
            "categories": self.categories,
            "reviewed_count": sum(1 for item in self.state.values() if item.reviewed),
            "classified_count": sum(1 for item in self.state.values() if item.categories),
            "models": model_items,
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
                "Could not save",
                f"Could not save classifications:\n{exc}\n\nClose anyway?",
            ):
                return
        self.root.destroy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview OFP/CWA P3D first-LOD geometry and assign model categories interactively."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="P3D, PBO, or directory to scan")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Only include canonical model paths matching GLOB. May be repeated; "
            "example: --include 'o\\hous\\*.p3d'."
        ),
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
        help=(
            "Use a custom category checkbox. Repeat for multiple categories. "
            "If omitted, built-in building categories are used."
        ),
    )
    parser.add_argument(
        "--max-preview-points",
        type=int,
        default=DEFAULT_MAX_PREVIEW_POINTS,
        metavar="N",
        help=(
            f"Maximum points drawn per model (default: {DEFAULT_MAX_PREVIEW_POINTS:,})."
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
    # Preserve user ordering but remove accidental duplicates.
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
