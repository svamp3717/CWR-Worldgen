# SPDX-License-Identifier: GPL-3.0-or-later
"""Editable coordinate controls for the OpenStreetMap area picker."""
from __future__ import annotations

import math
from typing import Any


def parse_center_coordinates(latitude_text: str, longitude_text: str) -> tuple[float, float]:
    """Parse and validate a Web-Mercator-safe latitude/longitude pair."""
    try:
        latitude = float(latitude_text.strip())
        longitude = float(longitude_text.strip())
    except ValueError as exc:
        raise ValueError("center latitude and longitude must be numbers") from exc
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("center latitude and longitude must be finite")
    if not -85.05112878 < latitude < 85.05112878:
        raise ValueError("center latitude must be between -85.05112878 and 85.05112878")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("center longitude must be between -180 and 180")
    return latitude, longitude


def parse_bbox_coordinates(
    south_text: str,
    west_text: str,
    north_text: str,
    east_text: str,
) -> tuple[float, float, float, float]:
    """Parse and validate south/west/north/east coordinate fields."""
    try:
        south, west, north, east = (
            float(south_text.strip()),
            float(west_text.strip()),
            float(north_text.strip()),
            float(east_text.strip()),
        )
    except ValueError as exc:
        raise ValueError("south, west, north and east must all be numbers") from exc
    if not all(math.isfinite(value) for value in (south, west, north, east)):
        raise ValueError("bounding-box coordinates must be finite")
    if not -85.05112878 < south < north < 85.05112878:
        raise ValueError("south/north must be ordered inside Web Mercator latitude limits")
    if not -180.0 <= west < east <= 180.0:
        raise ValueError("west/east must be ordered inside -180..180")
    return south, west, north, east


def _format_coordinate(value: float) -> str:
    return f"{float(value):.7f}"


def install_osm_area_picker_coordinate_controls() -> None:
    """Replace the picker class with a coordinate-aware subclass."""
    from . import map_picker as picker

    original_class = picker.OsmAreaPicker
    if bool(getattr(original_class, "_cwr_coordinate_controls", False)):
        return

    class CoordinateOsmAreaPicker(original_class):
        _cwr_coordinate_controls = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            self.coord_center_lat_var = picker.tk.StringVar(self)
            self.coord_center_lon_var = picker.tk.StringVar(self)
            self.coord_south_var = picker.tk.StringVar(self)
            self.coord_west_var = picker.tk.StringVar(self)
            self.coord_north_var = picker.tk.StringVar(self)
            self.coord_east_var = picker.tk.StringVar(self)

            controls = picker.ttk.LabelFrame(self, text="Coordinates", padding=(8, 5))
            buttons = self.apply_button.master
            controls.pack(fill="x", padx=8, pady=(0, 6), before=buttons)

            center_row = picker.ttk.Frame(controls)
            center_row.pack(fill="x")
            picker.ttk.Label(center_row, text="Center Lat").pack(side="left")
            center_lat_entry = picker.ttk.Entry(
                center_row, textvariable=self.coord_center_lat_var, width=14
            )
            center_lat_entry.pack(side="left", padx=(4, 10))
            picker.ttk.Label(center_row, text="Lon").pack(side="left")
            center_lon_entry = picker.ttk.Entry(
                center_row, textvariable=self.coord_center_lon_var, width=14
            )
            center_lon_entry.pack(side="left", padx=(4, 8))
            picker.ttk.Button(
                center_row, text="Set center", command=self._apply_typed_center
            ).pack(side="left")

            bbox_row = picker.ttk.Frame(controls)
            bbox_row.pack(fill="x", pady=(5, 0))
            entries = []
            for label, variable in (
                ("South", self.coord_south_var),
                ("West", self.coord_west_var),
                ("North", self.coord_north_var),
                ("East", self.coord_east_var),
            ):
                picker.ttk.Label(bbox_row, text=label).pack(side="left")
                entry = picker.ttk.Entry(bbox_row, textvariable=variable, width=13)
                entry.pack(side="left", padx=(4, 8))
                entries.append(entry)
            picker.ttk.Button(
                bbox_row, text="Set bbox", command=self._apply_typed_bbox
            ).pack(side="left")

            for entry in (center_lat_entry, center_lon_entry):
                entry.bind("<Return>", lambda _event: self._apply_typed_center())
            for entry in entries:
                entry.bind("<Return>", lambda _event: self._apply_typed_bbox())

            self._sync_coordinate_fields()

        def _sync_coordinate_fields(self) -> None:
            if not hasattr(self, "coord_center_lat_var"):
                return
            plan = self.selection_plan
            if plan is None:
                center_lat = self.center_lat
                center_lon = self.center_lon
                bbox: tuple[float, float, float, float] | None = None
            else:
                south, west, north, east = plan.bbox
                center_lat = (south + north) / 2.0
                center_lon = (west + east) / 2.0
                bbox = plan.bbox

            self.coord_center_lat_var.set(_format_coordinate(center_lat))
            self.coord_center_lon_var.set(_format_coordinate(center_lon))
            if bbox is None:
                for variable in (
                    self.coord_south_var,
                    self.coord_west_var,
                    self.coord_north_var,
                    self.coord_east_var,
                ):
                    variable.set("")
            else:
                south, west, north, east = bbox
                self.coord_south_var.set(_format_coordinate(south))
                self.coord_west_var.set(_format_coordinate(west))
                self.coord_north_var.set(_format_coordinate(north))
                self.coord_east_var.set(_format_coordinate(east))

        def _fit_to_selection(self) -> None:
            plan = self.selection_plan
            if plan is None:
                return
            south, west, north, east = plan.bbox
            self.center_lat = (south + north) / 2.0
            self.center_lon = (west + east) / 2.0
            self.zoom = picker.zoom_for_bbox(
                plan.bbox,
                max(1, self.canvas.winfo_width()),
                max(1, self.canvas.winfo_height()),
            )

        def _apply_typed_center(self) -> None:
            try:
                latitude, longitude = parse_center_coordinates(
                    self.coord_center_lat_var.get(),
                    self.coord_center_lon_var.get(),
                )
                cells = (
                    self.selection_plan.cells
                    if self.selection_plan is not None
                    else picker.CLASSIC_CELLS
                )
                self.selection_plan = picker.plan_center_selection(
                    latitude,
                    longitude,
                    cells=cells,
                    cell_size_metres=self.cell_size_metres,
                )
            except ValueError as exc:
                picker.messagebox.showerror("Invalid coordinates", str(exc), parent=self)
                return
            self._fit_to_selection()
            self._update_selection_text()
            self._redraw()

        def _apply_typed_bbox(self) -> None:
            try:
                bbox = parse_bbox_coordinates(
                    self.coord_south_var.get(),
                    self.coord_west_var.get(),
                    self.coord_north_var.get(),
                    self.coord_east_var.get(),
                )
                self.selection_plan = picker.plan_area_selection(
                    bbox,
                    cell_size_metres=self.cell_size_metres,
                )
            except ValueError as exc:
                picker.messagebox.showerror("Invalid coordinates", str(exc), parent=self)
                return
            self._fit_to_selection()
            self._update_selection_text()
            self._redraw()

        def _update_selection_text(self) -> None:
            super()._update_selection_text()
            self._sync_coordinate_fields()

        def _pan_drag(self, event: Any) -> None:
            super()._pan_drag(event)
            if self.selection_plan is None:
                self._sync_coordinate_fields()

        def _zoom_at(self, canvas_x: float, canvas_y: float, delta: int) -> None:
            super()._zoom_at(canvas_x, canvas_y, delta)
            if self.selection_plan is None:
                self._sync_coordinate_fields()

    picker.OsmAreaPicker = CoordinateOsmAreaPicker
