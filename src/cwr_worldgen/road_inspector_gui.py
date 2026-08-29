# SPDX-License-Identifier: GPL-3.0-or-later
"""Tkinter front-end for the read-only Road Inspector."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import traceback
import webbrowser
from typing import Any, Mapping, Sequence


def discover_roads_geojson(input_path: str | Path) -> Path | None:
    """Return a nearby normalized roads file using the launcher search order."""
    raw = str(input_path).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = (
        path.parent / "normalized" / "roads.geojson",
        path.parent.parent / "normalized" / "roads.geojson",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def default_output_dir(input_path: str | Path) -> Path:
    """Return the report directory used by the Windows drag-and-drop launcher."""
    raw = str(input_path).strip()
    if not raw:
        return Path("road-inspector")
    path = Path(raw).expanduser()
    stem = path.stem or path.name or "road"
    return path.with_name(f"{stem}-road-inspector")


def positive_float(value: str, label: str) -> float:
    """Parse one finite positive diagnostic threshold."""
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _open_in_file_manager(path: Path) -> None:
    target = str(Path(path).expanduser().resolve())
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    command = ["open", target] if sys.platform == "darwin" else ["xdg-open", target]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _open_report(path: Path) -> None:
    report = Path(path).expanduser().resolve()
    if not webbrowser.open(report.as_uri()):
        _open_in_file_manager(report)


def _summary_text(result: Any, paths: Mapping[str, Path]) -> str:
    counts = Counter(issue.severity for issue in result.issues)
    categories = Counter(issue.category for issue in result.issues)
    lines = [
        "Inspection complete",
        "",
        f"Input: {result.input_path}",
        f"WRP entry: {result.wrp_entry}",
        f"Road objects: {result.road_object_count:,}",
        f"Source junctions: {result.source_junction_count:,}",
        f"Issues: {len(result.issues):,}",
        (
            "Severity: "
            f"critical {counts.get('critical', 0):,}, "
            f"high {counts.get('high', 0):,}, "
            f"medium {counts.get('medium', 0):,}, "
            f"low {counts.get('low', 0):,}"
        ),
    ]
    if categories:
        lines.extend(("", "Categories:"))
        lines.extend(f"  {name}: {count:,}" for name, count in sorted(categories.items()))
    lines.extend(("", f"Report: {paths['html']}"))
    return "\n".join(lines)


class RoadInspectorGui:
    """Small standalone UI that delegates all geometry work to Road Inspector."""

    def __init__(
        self,
        root: Any,
        *,
        tk: Any,
        ttk: Any,
        filedialog: Any,
        messagebox: Any,
        inspector: Any,
        defaults: Mapping[str, float],
        initial_input: str = "",
        initial_roads: str = "",
        initial_output: str = "",
    ) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.inspector = inspector
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.running = False
        self.report_path: Path | None = None
        self.output_path: Path | None = None

        self.root.title("CWR Road Inspector")
        self.root.geometry("860x700")
        self.root.minsize(720, 580)

        self.input_var = tk.StringVar(value=initial_input)
        self.roads_var = tk.StringVar(value=initial_roads)
        self.output_var = tk.StringVar(value=initial_output)
        self.endpoint_var = tk.StringVar(value=str(defaults["endpoint_tolerance"]))
        self.edge_gap_var = tk.StringVar(value=str(defaults["minimum_edge_gap"]))
        self.tangent_var = tk.StringVar(value=str(defaults["minimum_tangent_error"]))
        self.junction_var = tk.StringVar(value=str(defaults["junction_match_tolerance"]))
        self.open_when_done_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Choose a generated .wrp or .pbo to inspect.")

        self._build()
        if initial_input:
            self._apply_input_defaults(force=not bool(initial_output), discover=not bool(initial_roads))
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        outer = self.ttk.Frame(self.root, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        self.ttk.Label(outer, text="Road Inspector", font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(
            outer,
            text=(
                "Read-only audit of the stock-road geometry stored in a generated RVW4 world. "
                "The interactive HTML report remains the detailed inspection view."
            ),
            wraplength=800,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        files = self.ttk.LabelFrame(outer, text="Files", padding=10)
        files.grid(row=2, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._path_row(files, 0, "Generated world", self.input_var, self._browse_input)
        self._path_row(files, 1, "Normalized roads", self.roads_var, self._browse_roads)
        self._path_row(files, 2, "Report folder", self.output_var, self._browse_output)
        self.ttk.Button(files, text="Auto-detect roads", command=self._auto_detect_roads).grid(
            row=3, column=1, sticky="w", pady=(6, 0)
        )
        self.ttk.Button(files, text="Clear roads", command=lambda: self.roads_var.set("")).grid(
            row=3, column=2, sticky="e", padx=(8, 0), pady=(6, 0)
        )

        advanced = self.ttk.LabelFrame(outer, text="Diagnostic thresholds", padding=10)
        advanced.grid(row=3, column=0, sticky="ew", pady=(10, 10))
        for column in range(4):
            advanced.columnconfigure(column, weight=1 if column % 2 else 0)
        self._number_field(advanced, 0, 0, "Endpoint tolerance (m)", self.endpoint_var)
        self._number_field(advanced, 0, 2, "Minimum edge gap (m)", self.edge_gap_var)
        self._number_field(advanced, 1, 0, "Minimum tangent error (°)", self.tangent_var)
        self._number_field(advanced, 1, 2, "Junction match tolerance (m)", self.junction_var)

        results = self.ttk.LabelFrame(outer, text="Result", padding=10)
        results.grid(row=4, column=0, sticky="nsew")
        results.rowconfigure(0, weight=1)
        results.columnconfigure(0, weight=1)
        self.result_text = self.tk.Text(results, wrap="word", height=12, state="disabled")
        self.result_text.grid(row=0, column=0, sticky="nsew")
        scroll = self.ttk.Scrollbar(results, orient="vertical", command=self.result_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=scroll.set)

        footer = self.ttk.Frame(outer)
        footer.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(1, weight=1)
        self.run_button = self.ttk.Button(footer, text="Run inspection", command=self._start_inspection)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.progress = self.ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.grid(row=0, column=1, sticky="w", padx=(10, 8))
        self.ttk.Checkbutton(
            footer, text="Open report when finished", variable=self.open_when_done_var
        ).grid(row=0, column=2, sticky="e", padx=(8, 8))
        self.open_report_button = self.ttk.Button(
            footer, text="Open report", command=self._open_report_clicked, state="disabled"
        )
        self.open_report_button.grid(row=0, column=3, sticky="e", padx=(0, 8))
        self.open_folder_button = self.ttk.Button(
            footer, text="Open folder", command=self._open_folder_clicked, state="disabled"
        )
        self.open_folder_button.grid(row=0, column=4, sticky="e")
        self.ttk.Label(outer, textvariable=self.status_var).grid(row=6, column=0, sticky="ew", pady=(8, 0))

    def _path_row(self, parent: Any, row: int, label: str, variable: Any, command: Any) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        if variable is self.input_var:
            entry.bind("<FocusOut>", lambda _event: self._apply_input_defaults(force=False, discover=True))
        self.ttk.Button(parent, text="Browse…", command=command).grid(
            row=row, column=2, sticky="e", padx=(8, 0), pady=3
        )

    def _number_field(self, parent: Any, row: int, column: int, label: str, variable: Any) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=3)
        self.ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 16), pady=3
        )

    def _browse_input(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Choose generated world",
            filetypes=[("CWA world files", ("*.wrp", "*.pbo")), ("WRP files", "*.wrp"), ("PBO files", "*.pbo"), ("All files", "*.*")],
        )
        if selected:
            self.input_var.set(selected)
            self._apply_input_defaults(force=True, discover=True)

    def _browse_roads(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Choose normalized roads.geojson",
            filetypes=[("GeoJSON", "*.geojson *.json"), ("All files", "*.*")],
        )
        if selected:
            self.roads_var.set(selected)

    def _browse_output(self) -> None:
        selected = self.filedialog.askdirectory(title="Choose Road Inspector report folder")
        if selected:
            self.output_var.set(selected)

    def _auto_detect_roads(self) -> None:
        detected = discover_roads_geojson(self.input_var.get())
        if detected is None:
            self.status_var.set("No nearby normalized/roads.geojson found. Seam checks can still run.")
            return
        self.roads_var.set(str(detected))
        self.status_var.set(f"Detected source roads: {detected}")

    def _apply_input_defaults(self, *, force: bool, discover: bool) -> None:
        raw = self.input_var.get().strip()
        if not raw:
            return
        if force or not self.output_var.get().strip():
            self.output_var.set(str(default_output_dir(raw)))
        if discover and (force or not self.roads_var.get().strip()):
            detected = discover_roads_geojson(raw)
            if detected is not None:
                self.roads_var.set(str(detected))

    def _validated_inputs(self) -> dict[str, Any]:
        input_path = Path(self.input_var.get().strip()).expanduser()
        if not self.input_var.get().strip():
            raise ValueError("Choose a generated .wrp or .pbo first")
        if not input_path.is_file():
            raise ValueError(f"Generated world does not exist: {input_path}")
        if input_path.suffix.casefold() not in {".wrp", ".pbo"}:
            raise ValueError("Generated world must be a .wrp or .pbo file")

        roads_raw = self.roads_var.get().strip()
        roads_path = Path(roads_raw).expanduser() if roads_raw else None
        if roads_path is not None and not roads_path.is_file():
            raise ValueError(f"Normalized roads file does not exist: {roads_path}")

        output_raw = self.output_var.get().strip()
        output_path = Path(output_raw).expanduser() if output_raw else default_output_dir(input_path)
        return {
            "input_path": input_path,
            "roads_geojson": roads_path,
            "output_path": output_path,
            "endpoint_tolerance": positive_float(self.endpoint_var.get(), "Endpoint tolerance"),
            "minimum_edge_gap": positive_float(self.edge_gap_var.get(), "Minimum edge gap"),
            "minimum_tangent_error": positive_float(self.tangent_var.get(), "Minimum tangent error"),
            "junction_match_tolerance": positive_float(self.junction_var.get(), "Junction match tolerance"),
        }

    def _start_inspection(self) -> None:
        if self.running:
            return
        try:
            values = self._validated_inputs()
        except ValueError as exc:
            self.messagebox.showerror("Road Inspector", str(exc), parent=self.root)
            return

        self.running = True
        self.report_path = None
        self.output_path = None
        self.run_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Inspecting emitted road geometry…")
        self._set_result_text("Road Inspector is running. The window will remain responsive while the audit works.\n")
        threading.Thread(target=self._worker, args=(values,), daemon=True).start()

    def _worker(self, values: Mapping[str, Any]) -> None:
        try:
            result = self.inspector.inspect_road_geometry(
                values["input_path"],
                roads_geojson=values["roads_geojson"],
                endpoint_tolerance=values["endpoint_tolerance"],
                minimum_edge_gap=values["minimum_edge_gap"],
                minimum_tangent_error=values["minimum_tangent_error"],
                junction_match_tolerance=values["junction_match_tolerance"],
            )
            paths = self.inspector.write_inspection_report(result, values["output_path"])
        except Exception as exc:
            self.events.put(("error", (str(exc) or exc.__class__.__name__, traceback.format_exc())))
            return
        self.events.put(("done", (result, paths)))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "done":
                    self._inspection_done(*payload)
                elif event == "error":
                    self._inspection_failed(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _inspection_done(self, result: Any, paths: Mapping[str, Path]) -> None:
        self._finish_running_state()
        self.report_path = Path(paths["html"])
        self.output_path = self.report_path.parent
        self.open_report_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")
        self.status_var.set(
            f"Complete: {result.road_object_count:,} road objects, {len(result.issues):,} issues."
        )
        self._set_result_text(_summary_text(result, paths))
        if self.open_when_done_var.get():
            try:
                _open_report(self.report_path)
            except OSError as exc:
                self.status_var.set(f"Report written, but opening it failed: {exc}")

    def _inspection_failed(self, message: str, details: str) -> None:
        self._finish_running_state()
        self.status_var.set("Inspection failed. No geometry was modified.")
        self._set_result_text(f"Inspection failed\n\n{message}\n\n{details}")
        self.messagebox.showerror("Road Inspector failed", message, parent=self.root)

    def _finish_running_state(self) -> None:
        self.running = False
        self.progress.stop()
        self.run_button.configure(state="normal")

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _open_report_clicked(self) -> None:
        if self.report_path is None:
            return
        try:
            _open_report(self.report_path)
        except OSError as exc:
            self.messagebox.showerror("Road Inspector", str(exc), parent=self.root)

    def _open_folder_clicked(self) -> None:
        if self.output_path is None:
            return
        try:
            _open_in_file_manager(self.output_path)
        except OSError as exc:
            self.messagebox.showerror("Road Inspector", str(exc), parent=self.root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cwr-road-inspector-gui", description="Open the Road Inspector GUI.")
    parser.add_argument("input", nargs="?", default="", help="optional generated .wrp or .pbo to preselect")
    parser.add_argument("--roads", default="", help="optional normalized roads.geojson to preselect")
    parser.add_argument("--output", default="", help="optional report directory to preselect")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise SystemExit("Road Inspector GUI requires Python's tkinter support") from exc

    from . import road_inspector as core
    from . import road_inspector_entry as inspector

    root = tk.Tk()
    RoadInspectorGui(
        root,
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        inspector=inspector,
        defaults={
            "endpoint_tolerance": core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
            "minimum_edge_gap": core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
            "minimum_tangent_error": core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
            "junction_match_tolerance": core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
        },
        initial_input=args.input,
        initial_roads=args.roads,
        initial_output=args.output,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
