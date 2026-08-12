# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive OpenStreetMap area picker for the desktop GUI.

Only tiles currently visible in the picker are requested. They are kept in a
local cache so reopening the same view does not repeatedly burden the public
OpenStreetMap tile service.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import queue
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable
from urllib import request

from PIL import Image, ImageTk

from ._version import __version__
from .location_example import square_bbox
from .osm import EARTH_RADIUS_METRES

TILE_SIZE = 256
MIN_ZOOM = 2
MAX_ZOOM = 18
CLASSIC_CELLS = 256
COMMON_LARGE_CELLS = 1024
MAX_GENERATOR_CELLS = 2048
OFP_CELL_SIZE_METRES = 25.0
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = f"CWR-Worldgen/{__version__} interactive-area-picker"


@dataclass(frozen=True, slots=True)
class AreaSelectionPlan:
    """A square, engine-scaled source selection derived from a map drag."""

    bbox: tuple[float, float, float, float]
    requested_width_metres: float
    requested_height_metres: float
    cells: int
    cell_size_metres: float
    world_size_metres: float
    severity: str
    warning: str
    supported: bool = True

    @property
    def requires_warning(self) -> bool:
        return self.severity != "safe"


def _haversine_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    delta_lat = lat2_r - lat1_r
    delta_lon = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METRES * math.asin(min(1.0, math.sqrt(value)))


def bbox_dimensions_metres(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    south, west, north, east = bbox
    if not (-85.05112878 < south < north < 85.05112878 and -180.0 <= west < east <= 180.0):
        raise ValueError("selection is outside supported Web Mercator bounds")
    middle_lat = (south + north) / 2.0
    middle_lon = (west + east) / 2.0
    width = _haversine_metres(middle_lat, west, middle_lat, east)
    height = _haversine_metres(south, middle_lon, north, middle_lon)
    return width, height


def _next_power_of_two(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("selection size must be positive and finite")
    return 1 << max(0, math.ceil(math.log2(value)))


def _selection_severity(cells: int, cell_size_metres: float) -> tuple[str, str, bool]:
    supported = cells <= MAX_GENERATOR_CELLS
    if not supported:
        return (
            "unsupported",
            f"The selected area needs {cells}×{cells} cells at {cell_size_metres:g} m. "
            f"The generator maximum is {MAX_GENERATOR_CELLS}×{MAX_GENERATOR_CELLS} "
            f"({MAX_GENERATOR_CELLS * cell_size_metres / 1000:.1f} km).",
            False,
        )
    if cells <= CLASSIC_CELLS:
        return (
            "safe",
            f"Recommended OFP/CWA grid size (up to 256×256 cells / "
            f"{CLASSIC_CELLS * cell_size_metres / 1000:.1f} km at {cell_size_metres:g} m).",
            True,
        )
    if cells <= 512:
        return (
            "warning",
            "Larger than the recommended 256×256 OFP/CWA grid. It may load more slowly "
            "and should be tested on the target game version.",
            True,
        )
    if cells <= COMMON_LARGE_CELLS:
        return (
            "danger",
            "Very large for OFP/CWA. Memory use, loading time, AI navigation and object "
            "counts can become impractical.",
            True,
        )
    return (
        "extreme",
        "Experimental size beyond the normal generator range for the selected cell scale. "
        "Expect serious compatibility and performance risk.",
        True,
    )


def plan_center_selection(
    latitude: float,
    longitude: float,
    *,
    cells: int = CLASSIC_CELLS,
    cell_size_metres: float = OFP_CELL_SIZE_METRES,
) -> AreaSelectionPlan:
    """Create an exact square selection around one chosen map centre."""

    if not (-85.05112878 < latitude < 85.05112878 and -180.0 <= longitude <= 180.0):
        raise ValueError("selection centre is outside supported Web Mercator bounds")
    if not isinstance(cells, int) or cells < 16 or cells & (cells - 1):
        raise ValueError("selection cells must be a power of two of at least 16")
    if not math.isfinite(cell_size_metres) or cell_size_metres <= 0:
        raise ValueError("cell size must be positive and finite")
    world_size = cells * cell_size_metres
    bbox = square_bbox(latitude, longitude, world_size)
    severity, warning, supported = _selection_severity(cells, cell_size_metres)
    return AreaSelectionPlan(
        bbox=bbox,
        requested_width_metres=world_size,
        requested_height_metres=world_size,
        cells=min(cells, MAX_GENERATOR_CELLS),
        cell_size_metres=cell_size_metres,
        world_size_metres=min(cells, MAX_GENERATOR_CELLS) * cell_size_metres,
        severity=severity,
        warning=warning,
        supported=supported,
    )


def resize_area_selection(plan: AreaSelectionPlan, factor: int) -> AreaSelectionPlan:
    """Resize a square selection around its current centre while preserving cell size."""

    if factor not in {2}:
        raise ValueError("selection resize factor must be 2")
    south, west, north, east = plan.bbox
    centre_lat = (south + north) / 2.0
    centre_lon = (west + east) / 2.0
    return plan_center_selection(
        centre_lat,
        centre_lon,
        cells=plan.cells * factor,
        cell_size_metres=plan.cell_size_metres,
    )


def plan_area_selection(
    bbox: tuple[float, float, float, float],
    *,
    cell_size_metres: float = OFP_CELL_SIZE_METRES,
) -> AreaSelectionPlan:
    """Snap a map drag to the smallest power-of-two world containing it."""

    width, height = bbox_dimensions_metres(bbox)
    requested_side = max(width, height)
    # ``square_bbox`` and the Web-Mercator distance conversion can differ by a
    # few floating-point ulps. Without a tolerance, an exact 6.4 km / 25 m
    # selection can become 256.00000000004 cells and incorrectly jump to 512.
    required_cells = _next_power_of_two(max(1.0, requested_side / cell_size_metres - 1.0e-6))
    cells = max(16, required_cells)
    applied_cells = min(cells, MAX_GENERATOR_CELLS)
    south, west, north, east = bbox
    centre_lat = (south + north) / 2.0
    centre_lon = (west + east) / 2.0
    world_size = applied_cells * cell_size_metres
    snapped_bbox = square_bbox(centre_lat, centre_lon, world_size)
    severity, warning, supported = _selection_severity(cells, cell_size_metres)

    return AreaSelectionPlan(
        bbox=snapped_bbox,
        requested_width_metres=width,
        requested_height_metres=height,
        cells=applied_cells,
        cell_size_metres=cell_size_metres,
        world_size_metres=world_size,
        severity=severity,
        warning=warning,
        supported=supported,
    )


def plan_initial_selection(
    bbox: tuple[float, float, float, float],
    *,
    cells: int | None = None,
    cell_size_metres: float = OFP_CELL_SIZE_METRES,
) -> AreaSelectionPlan:
    """Restore a saved selection without promoting coordinate-rounding noise.

    Wizard coordinates are displayed and persisted to seven decimal places. A
    nominal 6.4 km box can consequently measure a few centimetres over its
    exact boundary and otherwise jump from 256 to 512 cells. Honour the saved
    grid when both measured sides remain within a small rounding tolerance;
    genuinely larger boxes still use the normal containing-grid calculation.
    """

    measured_plan = plan_area_selection(bbox, cell_size_metres=cell_size_metres)
    if cells is None:
        return measured_plan
    if not isinstance(cells, int) or cells < 16 or cells & (cells - 1):
        return measured_plan

    width, height = bbox_dimensions_metres(bbox)
    expected_side = cells * cell_size_metres
    rounding_tolerance = max(0.05, cell_size_metres * 0.002)
    if max(abs(width - expected_side), abs(height - expected_side)) > rounding_tolerance:
        return measured_plan

    south, west, north, east = bbox
    return plan_center_selection(
        (south + north) / 2.0,
        (west + east) / 2.0,
        cells=cells,
        cell_size_metres=cell_size_metres,
    )


def latlon_to_world_pixel(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    latitude = max(-85.05112878, min(85.05112878, latitude))
    scale = TILE_SIZE * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(latitude))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return x, y


def world_pixel_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    longitude = x / scale * 360.0 - 180.0
    mercator = math.pi * (1.0 - 2.0 * y / scale)
    latitude = math.degrees(math.atan(math.sinh(mercator)))
    return max(-85.05112878, min(85.05112878, latitude)), longitude


def zoom_for_bbox(
    bbox: tuple[float, float, float, float],
    width_pixels: int,
    height_pixels: int,
    *,
    padding: int = 80,
) -> int:
    available_width = max(64, width_pixels - padding)
    available_height = max(64, height_pixels - padding)
    south, west, north, east = bbox
    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        west_px, north_px = latlon_to_world_pixel(north, west, zoom)
        east_px, south_px = latlon_to_world_pixel(south, east, zoom)
        if east_px - west_px <= available_width and south_px - north_px <= available_height:
            return zoom
    return MIN_ZOOM


class OsmAreaPicker(tk.Toplevel):
    """Slippy-map window for choosing the OSM source area."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        initial_center: tuple[float, float],
        initial_zoom: int = 10,
        initial_bbox: tuple[float, float, float, float] | None = None,
        initial_cells: int | None = None,
        cell_size_metres: float = OFP_CELL_SIZE_METRES,
        on_apply: Callable[[AreaSelectionPlan, bool], None],
    ) -> None:
        super().__init__(master)
        self.title(f"Select OSM area - CWR Worldgen {__version__}")
        self.geometry("980x720")
        self.minsize(720, 520)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.center_lat, self.center_lon = initial_center
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, int(initial_zoom)))
        self.selection_plan: AreaSelectionPlan | None = None
        self.cell_size_metres = float(cell_size_metres)
        if not math.isfinite(self.cell_size_metres) or self.cell_size_metres <= 0:
            self.cell_size_metres = OFP_CELL_SIZE_METRES
        self.on_apply = on_apply
        self.selection_mode_var = tk.StringVar(value="center")
        self._drag_start: tuple[float, float] | None = None
        self._drag_end: tuple[float, float] | None = None
        self._pan_start: tuple[float, float] | None = None
        self._pan_center_pixel: tuple[float, float] | None = None
        self._tile_images: dict[tuple[int, int, int], ImageTk.PhotoImage] = {}
        self._pending_tiles: set[tuple[int, int, int]] = set()
        self._tile_results: queue.Queue[tuple[tuple[int, int, int], Path | None, str | None]] = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cwr-osm-tile")
        self._closed = False
        self.tile_cache = self._tile_cache_dir()

        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.pack(fill="x")
        ttk.Radiobutton(
            toolbar, text="Click center", variable=self.selection_mode_var, value="center",
            command=self._selection_mode_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            toolbar, text="Drag square", variable=self.selection_mode_var, value="drag",
            command=self._selection_mode_changed,
        ).pack(side="left", padx=(8, 0))
        self.mode_help_var = tk.StringVar(value=f"Click once to place a {CLASSIC_CELLS * self.cell_size_metres / 1000:.1f} km square around that centre.")
        ttk.Label(toolbar, textvariable=self.mode_help_var).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="−", width=3, command=lambda: self._zoom_by(-1)).pack(side="right")
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1)).pack(side="right", padx=(4, 2))
        ttk.Button(toolbar, text="Clear", command=self._clear_selection).pack(side="right", padx=(4, 8))
        self.double_button = ttk.Button(toolbar, text="Double size", command=self._double_selection, state="disabled")
        self.double_button.pack(side="right", padx=4)

        self.canvas = tk.Canvas(self, background="#d7d7d7", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._selection_start)
        self.canvas.bind("<B1-Motion>", self._selection_drag)
        self.canvas.bind("<ButtonRelease-1>", self._selection_finish)
        self.canvas.bind("<ButtonPress-3>", self._pan_begin)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._pan_finish)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, -1))

        info = ttk.Frame(self, padding=8)
        info.pack(fill="x")
        self.selection_var = tk.StringVar(value="Click the desired square centre on the map.")
        self.warning_var = tk.StringVar(value=self._default_warning_text())
        ttk.Label(info, textvariable=self.selection_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.warning_label = ttk.Label(info, textvariable=self.warning_var, wraplength=900)
        self.warning_label.pack(anchor="w", pady=(2, 0))
        ttk.Label(
            info,
            text="© OpenStreetMap contributors | Display tiles are for interactive selection only.",
            font=("Segoe UI", 8),
        ).pack(anchor="e", pady=(5, 0))

        buttons = ttk.Frame(self, padding=(8, 0, 8, 8))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self._close).pack(side="right")
        self.apply_button = ttk.Button(
            buttons,
            text="Use area",
            command=self._apply,
            state="disabled",
        )
        self.apply_button.pack(side="right", padx=6)

        if initial_bbox is not None:
            try:
                self.selection_plan = plan_initial_selection(
                    initial_bbox,
                    cells=initial_cells,
                    cell_size_metres=self.cell_size_metres,
                )
                self.center_lat = (initial_bbox[0] + initial_bbox[2]) / 2.0
                self.center_lon = (initial_bbox[1] + initial_bbox[3]) / 2.0
                self.zoom = zoom_for_bbox(initial_bbox, 900, 600)
                self._update_selection_text()
            except ValueError:
                self.selection_plan = None

        self.after(100, self._drain_tile_results)
        self.after_idle(self._redraw)
        self.grab_set()

    @staticmethod
    def _tile_cache_dir() -> Path:
        root = Path.home() / ".cache" / "cwr-worldgen" / "osm-tiles"
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "cwr-worldgen-osm-tiles"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _viewport_top_left(self) -> tuple[float, float]:
        center_x, center_y = latlon_to_world_pixel(self.center_lat, self.center_lon, self.zoom)
        return center_x - self.canvas.winfo_width() / 2.0, center_y - self.canvas.winfo_height() / 2.0

    def _canvas_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        left, top = self._viewport_top_left()
        return world_pixel_to_latlon(left + x, top + y, self.zoom)

    def _latlon_to_canvas(self, latitude: float, longitude: float) -> tuple[float, float]:
        left, top = self._viewport_top_left()
        x, y = latlon_to_world_pixel(latitude, longitude, self.zoom)
        return x - left, y - top

    def _redraw(self) -> None:
        if self._closed or not self.winfo_exists():
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        left, top = self._viewport_top_left()
        right = left + width
        bottom = top + height
        self.canvas.delete("all")
        self._tile_images.clear()
        tile_count = 2**self.zoom
        min_x = math.floor(left / TILE_SIZE)
        max_x = math.floor(right / TILE_SIZE)
        min_y = max(0, math.floor(top / TILE_SIZE))
        max_y = min(tile_count - 1, math.floor(bottom / TILE_SIZE))
        for raw_y in range(min_y, max_y + 1):
            for raw_x in range(min_x, max_x + 1):
                tile_x = raw_x % tile_count
                key = (self.zoom, tile_x, raw_y)
                draw_x = raw_x * TILE_SIZE - left
                draw_y = raw_y * TILE_SIZE - top
                tile_path = self.tile_cache / str(self.zoom) / str(tile_x) / f"{raw_y}.png"
                image = self._load_tile_image(key, tile_path)
                if image is not None:
                    self.canvas.create_image(draw_x, draw_y, image=image, anchor="nw", tags="map")
                else:
                    self.canvas.create_rectangle(
                        draw_x,
                        draw_y,
                        draw_x + TILE_SIZE,
                        draw_y + TILE_SIZE,
                        fill="#ececec",
                        outline="#d0d0d0",
                        tags="map",
                    )
                    self._request_tile(key, tile_path)

        if self.selection_plan is not None:
            south, west, north, east = self.selection_plan.bbox
            x1, y1 = self._latlon_to_canvas(north, west)
            x2, y2 = self._latlon_to_canvas(south, east)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#d02020", width=3, tags="selection")
            self.canvas.create_rectangle(x1 + 3, y1 + 3, x2 - 3, y2 - 3, outline="white", width=1, tags="selection")
            centre_x = (x1 + x2) / 2.0
            centre_y = (y1 + y2) / 2.0
            self.canvas.create_oval(centre_x - 5, centre_y - 5, centre_x + 5, centre_y + 5, fill="#d02020", outline="white", width=2, tags="selection")
            self.canvas.create_line(centre_x - 12, centre_y, centre_x + 12, centre_y, fill="white", width=1, tags="selection")
            self.canvas.create_line(centre_x, centre_y - 12, centre_x, centre_y + 12, fill="white", width=1, tags="selection")
        elif self._drag_start is not None and self._drag_end is not None:
            self.canvas.create_rectangle(
                self._drag_start[0],
                self._drag_start[1],
                self._drag_end[0],
                self._drag_end[1],
                outline="#d02020",
                width=3,
                tags="selection",
            )

    def _load_tile_image(self, key: tuple[int, int, int], path: Path) -> ImageTk.PhotoImage | None:
        if not path.is_file():
            return None
        try:
            with Image.open(path) as source:
                image = ImageTk.PhotoImage(source.convert("RGB"))
            self._tile_images[key] = image
            return image
        except (OSError, tk.TclError):
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def _request_tile(self, key: tuple[int, int, int], path: Path) -> None:
        if key in self._pending_tiles:
            return
        self._pending_tiles.add(key)
        self._executor.submit(self._download_tile, key, path)

    def _download_tile(self, key: tuple[int, int, int], path: Path) -> None:
        zoom, tile_x, tile_y = key
        try:
            req = request.Request(
                OSM_TILE_URL.format(z=zoom, x=tile_x, y=tile_y),
                headers={"User-Agent": OSM_USER_AGENT},
            )
            with request.urlopen(req, timeout=20) as response:
                data = response.read()
            with Image.open(BytesIO(data)) as image:
                image.verify()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
            self._tile_results.put((key, path, None))
        except Exception as exc:  # noqa: BLE001 - network errors belong in the UI status
            self._tile_results.put((key, None, str(exc)))

    def _drain_tile_results(self) -> None:
        if self._closed:
            return
        changed = False
        latest_error: str | None = None
        try:
            while True:
                key, path, error = self._tile_results.get_nowait()
                self._pending_tiles.discard(key)
                if path is not None:
                    changed = True
                elif error:
                    latest_error = error
        except queue.Empty:
            pass
        if changed:
            self._redraw()
        if latest_error and self.selection_plan is None:
            self.warning_var.set(f"Map tiles could not be loaded: {latest_error}")
        self.after(100, self._drain_tile_results)

    def _selection_start(self, event: tk.Event[tk.Misc]) -> None:
        if self.selection_mode_var.get() == "center":
            latitude, longitude = self._canvas_to_latlon(float(event.x), float(event.y))
            cells = self.selection_plan.cells if self.selection_plan is not None else CLASSIC_CELLS
            try:
                self.selection_plan = plan_center_selection(
                    latitude, longitude, cells=cells, cell_size_metres=self.cell_size_metres
                )
            except ValueError as exc:
                self.selection_plan = None
                messagebox.showerror("Invalid map centre", str(exc), parent=self)
            self._drag_start = None
            self._drag_end = None
            self._update_selection_text()
            self._redraw()
            return
        self._drag_start = (float(event.x), float(event.y))
        self._drag_end = self._drag_start
        self.selection_plan = None
        self._redraw()

    def _selection_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.selection_mode_var.get() != "drag" or self._drag_start is None:
            return
        start_x, start_y = self._drag_start
        delta_x = float(event.x) - start_x
        delta_y = float(event.y) - start_y
        side = max(abs(delta_x), abs(delta_y))
        end_x = start_x + math.copysign(side, delta_x if delta_x else 1.0)
        end_y = start_y + math.copysign(side, delta_y if delta_y else 1.0)
        self._drag_end = (
            max(0.0, min(float(self.canvas.winfo_width()), end_x)),
            max(0.0, min(float(self.canvas.winfo_height()), end_y)),
        )
        self._redraw()

    def _selection_finish(self, _event: tk.Event[tk.Misc]) -> None:
        if self.selection_mode_var.get() != "drag" or self._drag_start is None or self._drag_end is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = self._drag_end
        self._drag_start = None
        self._drag_end = None
        if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
            self.warning_var.set("Selection is too small. Drag a larger square.")
            self._redraw()
            return
        north_lat, west_lon = self._canvas_to_latlon(min(x1, x2), min(y1, y2))
        south_lat, east_lon = self._canvas_to_latlon(max(x1, x2), max(y1, y2))
        try:
            self.selection_plan = plan_area_selection((south_lat, west_lon, north_lat, east_lon), cell_size_metres=self.cell_size_metres)
        except ValueError as exc:
            self.selection_plan = None
            messagebox.showerror("Invalid map area", str(exc), parent=self)
        self._update_selection_text()
        self._redraw()

    def _update_selection_text(self) -> None:
        plan = self.selection_plan
        state = "normal" if plan is not None and plan.supported else "disabled"
        self.apply_button.configure(state=state)
        double_state = "normal" if plan is not None and plan.supported and plan.cells < MAX_GENERATOR_CELLS else "disabled"
        self.double_button.configure(state=double_state)
        if plan is None:
            self.selection_var.set(
                "Click the desired square centre on the map."
                if self.selection_mode_var.get() == "center"
                else "Drag a square around the area to import."
            )
            return
        self.selection_var.set(
            f"Selected {plan.requested_width_metres / 1000:.2f} × "
            f"{plan.requested_height_metres / 1000:.2f} km; snapped to "
            f"{plan.world_size_metres / 1000:.1f} km, {plan.cells}×{plan.cells} cells "
            f"at {plan.cell_size_metres:g} m."
        )
        self.warning_var.set(plan.warning)

    def _default_warning_text(self) -> str:
        return (
            f"Selections use a {self.cell_size_metres:g} m power-of-two OFP/CWA grid; "
            f"the default centred square is {CLASSIC_CELLS * self.cell_size_metres / 1000:.1f} km."
        )

    def _selection_mode_changed(self) -> None:
        if self.selection_mode_var.get() == "center":
            self.mode_help_var.set(
                f"Click once to place a {CLASSIC_CELLS * self.cell_size_metres / 1000:.1f} km square; existing size is preserved."
            )
        else:
            self.mode_help_var.set("Left-drag a square; Right-drag pans; mouse wheel zooms.")
        self._drag_start = None
        self._drag_end = None
        self._update_selection_text()
        self._redraw()

    def _double_selection(self) -> None:
        if self.selection_plan is None or self.selection_plan.cells >= MAX_GENERATOR_CELLS:
            return
        try:
            self.selection_plan = resize_area_selection(self.selection_plan, 2)
        except ValueError as exc:
            messagebox.showerror("Cannot resize map area", str(exc), parent=self)
            return
        south, west, north, east = self.selection_plan.bbox
        self.center_lat = (south + north) / 2.0
        self.center_lon = (west + east) / 2.0
        self.zoom = zoom_for_bbox(
            self.selection_plan.bbox,
            max(1, self.canvas.winfo_width()),
            max(1, self.canvas.winfo_height()),
        )
        self._update_selection_text()
        self._redraw()

    def _clear_selection(self) -> None:
        self.selection_plan = None
        self._drag_start = None
        self._drag_end = None
        self._update_selection_text()
        self.warning_var.set(self._default_warning_text())
        self._redraw()

    def _pan_begin(self, event: tk.Event[tk.Misc]) -> None:
        self._pan_start = (float(event.x), float(event.y))
        self._pan_center_pixel = latlon_to_world_pixel(self.center_lat, self.center_lon, self.zoom)
        self.canvas.configure(cursor="fleur")

    def _pan_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self._pan_start is None or self._pan_center_pixel is None:
            return
        delta_x = float(event.x) - self._pan_start[0]
        delta_y = float(event.y) - self._pan_start[1]
        center_x = self._pan_center_pixel[0] - delta_x
        center_y = self._pan_center_pixel[1] - delta_y
        self.center_lat, self.center_lon = world_pixel_to_latlon(center_x, center_y, self.zoom)
        self._redraw()

    def _pan_finish(self, _event: tk.Event[tk.Misc]) -> None:
        self._pan_start = None
        self._pan_center_pixel = None
        self.canvas.configure(cursor="crosshair")

    def _mouse_wheel(self, event: tk.Event[tk.Misc]) -> None:
        delta = 1 if getattr(event, "delta", 0) > 0 else -1
        self._zoom_at(float(event.x), float(event.y), delta)

    def _zoom_by(self, delta: int) -> None:
        self._zoom_at(self.canvas.winfo_width() / 2.0, self.canvas.winfo_height() / 2.0, delta)

    def _zoom_at(self, canvas_x: float, canvas_y: float, delta: int) -> None:
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom + delta))
        if new_zoom == self.zoom:
            return
        anchor_lat, anchor_lon = self._canvas_to_latlon(canvas_x, canvas_y)
        self.zoom = new_zoom
        anchor_world_x, anchor_world_y = latlon_to_world_pixel(anchor_lat, anchor_lon, self.zoom)
        center_world_x = anchor_world_x - (canvas_x - self.canvas.winfo_width() / 2.0)
        center_world_y = anchor_world_y - (canvas_y - self.canvas.winfo_height() / 2.0)
        self.center_lat, self.center_lon = world_pixel_to_latlon(center_world_x, center_world_y, self.zoom)
        self._redraw()

    def _apply(self) -> None:
        plan = self.selection_plan
        if plan is None or not plan.supported:
            return
        if plan.requires_warning:
            proceed = messagebox.askyesno(
                "Large OFP/CWA world",
                f"{plan.warning}\n\n"
                f"Selected output: {plan.world_size_metres / 1000:.1f} km, "
                f"{plan.cells}×{plan.cells} cells.\n\nContinue?",
                parent=self,
            )
            if not proceed:
                return
        self.on_apply(plan, False)
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
