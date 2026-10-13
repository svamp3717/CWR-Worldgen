"""Startup input/output picker for the multi-game P3D model categoriser.

The command-line interface remains scriptable. When launched without model
arguments, this small dialog lets users choose actual model PBO/IFA archives
and the JSON that controls both save location and resume state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from p3d_frozen_support import dismiss_splash
from cwr_worldgen.pbo import is_pbo_path


@dataclass(frozen=True)
class StartupSelection:
    inputs: tuple[Path, ...]
    output: Path
    texture_sources: tuple[Path, ...] = ()


def validate_startup_selection(
    inputs: Sequence[Path],
    output: Path,
    texture_sources: Sequence[Path] = (),
) -> str | None:
    """Check user-selected inputs before starting a potentially lengthy scan."""
    if not inputs:
        return "Select at least one model PBO/IFA archive or model folder."
    for path in inputs:
        if not path.exists():
            return f"Model source does not exist:\n{path}"
        if path.is_file() and not (
            is_pbo_path(path) or path.suffix.casefold() == ".p3d"
        ):
            return (
                f"Unsupported model source:\n{path}\n\n"
                "Select .pbo, .ifa, .pbo.zst, .ifa.zst, a loose .p3d, or a folder."
            )
        if not path.is_file() and not path.is_dir():
            return f"Model source is not a file or folder:\n{path}"
    for path in texture_sources:
        if not path.exists():
            return f"Texture archive does not exist:\n{path}"
        if not path.is_file() or not is_pbo_path(path):
            return (
                f"Unsupported texture archive:\n{path}\n\n"
                "Choose individual .pbo, .ifa, .pbo.zst or .ifa.zst files."
            )
    if output.exists() and not output.is_file():
        return f"The JSON output is not a file:\n{output}"
    if not str(output).strip() or output.suffix.casefold() != ".json":
        return "Choose a .json file for your completed model classifications."
    parent = output.parent
    if parent.exists() and not parent.is_dir():
        return f"The JSON output parent is not a directory:\n{parent}"
    return None


class StartupSelectionDialog:
    """Select model sources and JSON output before the model iterator opens."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        initial_inputs: Sequence[Path] = (),
        default_output: Path = Path("p3d-model-categories.json"),
    ) -> None:
        self.root = root
        self.inputs = list(dict.fromkeys(Path(p).expanduser() for p in initial_inputs))
        self.texture_sources: list[Path] = []
        self.selection: StartupSelection | None = None
        self.output_var = tk.StringVar(master=root, value=str(default_output))
        root.title("P3D Model Categoriser • Choose files")
        root.geometry("800x690")
        root.minsize(670, 570)
        root.protocol("WM_DELETE_WINDOW", self.cancel)

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(
            frame,
            text="P3D Model Categoriser",
            font=("TkDefaultFont", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Choose Arma / OFP / Iron Front PBO or IFA archives, "
                "or a folder of model files."
            ),
            wraplength=690,
        ).grid(row=1, column=0, sticky="w", pady=(6, 10))

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.source_list = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, height=9, exportselection=False,
        )
        self.source_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.source_list.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.source_list.configure(yscrollcommand=scrollbar.set)
        self._refresh_sources()

        source_buttons = ttk.Frame(frame)
        source_buttons.grid(row=3, column=0, sticky="ew", pady=(8, 10))
        ttk.Button(
            source_buttons, text="Select PBO / IFA…", command=self.select_archives
        ).pack(side=tk.LEFT)
        ttk.Button(
            source_buttons, text="Add model folder…", command=self.add_folder
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            source_buttons, text="Remove selected", command=self.remove_selected
        ).pack(side=tk.LEFT)

        ttk.Label(
            frame,
            text="Additional texture PBO / IFA archives (not scanned for models):",
        ).grid(row=4, column=0, sticky="w", pady=(2, 5))
        texture_frame = ttk.Frame(frame)
        texture_frame.grid(row=5, column=0, sticky="nsew")
        texture_frame.columnconfigure(0, weight=1)
        texture_frame.rowconfigure(0, weight=1)
        self.texture_source_list = tk.Listbox(
            texture_frame, selectmode=tk.EXTENDED, height=5, exportselection=False,
        )
        self.texture_source_list.grid(row=0, column=0, sticky="nsew")
        texture_scrollbar = ttk.Scrollbar(
            texture_frame, orient=tk.VERTICAL, command=self.texture_source_list.yview,
        )
        texture_scrollbar.grid(row=0, column=1, sticky="ns")
        self.texture_source_list.configure(yscrollcommand=texture_scrollbar.set)
        self._refresh_texture_sources()

        texture_buttons = ttk.Frame(frame)
        texture_buttons.grid(row=6, column=0, sticky="ew", pady=(8, 12))
        ttk.Button(
            texture_buttons,
            text="Add texture PBOs / IFAs…",
            command=self.select_texture_archives,
        ).pack(side=tk.LEFT)
        ttk.Button(
            texture_buttons,
            text="Remove selected textures",
            command=self.remove_selected_textures,
        ).pack(side=tk.LEFT, padx=(6, 0))

        output_frame = ttk.LabelFrame(frame, text="Classification JSON output", padding=9)
        output_frame.grid(row=7, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            output_frame, text="Existing JSON…", command=self.select_existing_json,
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            output_frame, text="New JSON…", command=self.select_new_json,
        ).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(
            output_frame,
            text=(
                "Existing JSON resumes your saved categories and last position. "
                "Incomplete reviews use a .unclassified.json companion file."
            ),
            wraplength=685,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, sticky="e", pady=(12, 0))
        ttk.Button(actions, text="Cancel", command=self.cancel).pack(side=tk.LEFT)
        ttk.Button(
            actions, text="Start categorising", command=self.confirm,
        ).pack(side=tk.LEFT, padx=(10, 0))
        root.bind("<Escape>", lambda _event: self.cancel())

    def _refresh_sources(self) -> None:
        self.source_list.delete(0, tk.END)
        for path in self.inputs:
            self.source_list.insert(tk.END, str(path))

    def _add(self, sources: Sequence[Path]) -> None:
        existing = set(self.inputs)
        for source in sources:
            path = Path(source).expanduser()
            if path not in existing:
                self.inputs.append(path)
                existing.add(path)
        self._refresh_sources()

    def _refresh_texture_sources(self) -> None:
        self.texture_source_list.delete(0, tk.END)
        for path in self.texture_sources:
            self.texture_source_list.insert(tk.END, str(path))

    def _add_texture_archives(self, paths: Sequence[Path]) -> None:
        known = {str(path.resolve()).casefold() for path in self.texture_sources}
        for source in paths:
            path = Path(source).expanduser()
            key = str(path.resolve()).casefold()
            if key not in known:
                self.texture_sources.append(path)
                known.add(key)
        self._refresh_texture_sources()

    def select_texture_archives(self) -> None:
        """Add any number of individual archives without adding model scan inputs."""
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Add texture PBO / IFA archives (select multiple)",
            filetypes=(
                ("PBO / IFA archives", "*.pbo *.ifa *.pbo.zst *.ifa.zst"),
                ("PBO archives", "*.pbo *.pbo.zst"),
                ("Iron Front IFA archives", "*.ifa *.ifa.zst"),
                ("All files", "*.*"),
            ),
        )
        if paths:
            self._add_texture_archives(tuple(Path(p) for p in paths))

    def remove_selected_textures(self) -> None:
        indices = set(self.texture_source_list.curselection())
        if indices:
            self.texture_sources = [
                path for index, path in enumerate(self.texture_sources)
                if index not in indices
            ]
            self._refresh_texture_sources()

    def select_archives(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Select model PBO / IFA archives",
            filetypes=(
                ("PBO / IFA archives", "*.pbo *.ifa *.pbo.zst *.ifa.zst"),
                ("PBO archives", "*.pbo *.pbo.zst"),
                ("Iron Front IFA archives", "*.ifa *.ifa.zst"),
                ("Loose P3D models", "*.p3d"),
                ("All files", "*.*"),
            ),
        )
        if paths:
            self._add(tuple(Path(p) for p in paths))

    def add_folder(self) -> None:
        path = filedialog.askdirectory(
            parent=self.root, title="Select model archive or P3D folder",
        )
        if path:
            self._add((Path(path),))

    def remove_selected(self) -> None:
        indices = set(self.source_list.curselection())
        if indices:
            self.inputs = [
                path for index, path in enumerate(self.inputs)
                if index not in indices
            ]
            self._refresh_sources()

    def select_existing_json(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open existing model classifications JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.output_var.set(path)

    def select_new_json(self) -> None:
        current = Path(self.output_var.get().strip() or "p3d-model-categories.json")
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose model classifications JSON output",
            defaultextension=".json",
            initialdir=str(current.parent),
            initialfile=current.name,
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.output_var.set(path)

    def confirm(self) -> None:
        text = self.output_var.get().strip()
        output = Path(text).expanduser() if text else Path("")
        problem = validate_startup_selection(
            self.inputs, output, self.texture_sources
        ) if text else (
            "Choose a JSON filename for model classifications."
        )
        if problem:
            messagebox.showerror("Invalid categoriser setup", problem, parent=self.root)
            return

        # Validate an existing save before destroying the setup window. A
        # broken/irrelevant JSON must not accidentally be overwritten.
        if output.exists():
            from p3d_categoriser_storage import load_state
            try:
                load_state(output)
            except ValueError as exc:
                messagebox.showerror(
                    "Cannot load existing classifications", str(exc), parent=self.root,
                )
                return
        self.selection = StartupSelection(
            tuple(self.inputs), output, tuple(self.texture_sources)
        )
        self.root.quit()

    def cancel(self) -> None:
        self.selection = None
        self.root.quit()


def prompt_for_startup_selection(
    *,
    initial_inputs: Sequence[Path] = (),
    default_output: Path = Path("p3d-model-categories.json"),
) -> StartupSelection | None:
    """Show the setup GUI; return None without writing anything on cancellation."""
    root = tk.Tk()
    try:
        dialog = StartupSelectionDialog(
            root, initial_inputs=initial_inputs, default_output=default_output,
        )
        dismiss_splash(root)
        root.mainloop()
        return dialog.selection
    finally:
        root.destroy()
