"""Session/resume wrapper for the P3D categoriser UI."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

import measure_p3d_models as measure
from p3d_categoriser_app import CategoriserApp, Classification, PLACEMENTS


def load_resume_path(path: Path) -> str:
    """Return a saved browsing cursor without changing classification loading semantics."""
    if not path.exists():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = str(raw.get("resume_model_path", "")).strip()
    return measure._canonical_model_path(value) if value else ""


class SessionCategoriserApp(CategoriserApp):
    """Categoriser with a startup loading screen and persistent list position."""

    def __init__(self, root: tk.Tk, *, resume_model_path: str = "", **kwargs) -> None:
        self._startup_active = True
        self._startup_resume_target = (
            measure._canonical_model_path(resume_model_path) if resume_model_path else ""
        )
        self._resume_model_path = self._startup_resume_target
        self._loading_window: tk.Toplevel | None = None
        self._loading_label_var = tk.StringVar(master=root, value="Preparing model browser...")
        self._loading_detail_var = tk.StringVar(master=root, value="")
        self._loading_bar: ttk.Progressbar | None = None

        root.withdraw()
        self._create_loading_screen(root)
        root.update()

        super().__init__(root, **kwargs)
        self._install_session_menu()
        self._startup_active = False
        self._finish_loading_screen()

    def _create_loading_screen(self, root: tk.Tk) -> None:
        window = tk.Toplevel(root)
        self._loading_window = window
        window.title("Loading CWR P3D Model Categoriser")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(window, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            frame,
            text="Loading model catalogue",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            textvariable=self._loading_label_var,
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 6))
        ttk.Label(
            frame,
            textvariable=self._loading_detail_var,
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 12))
        bar = ttk.Progressbar(frame, mode="indeterminate", length=480)
        self._loading_bar = bar
        bar.pack(fill=tk.X)
        bar.start(12)
        ttk.Label(
            frame,
            text="The browser will open when the first requested textured preview is ready.",
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

        window.update_idletasks()
        width = max(window.winfo_reqwidth(), 540)
        height = max(window.winfo_reqheight(), 180)
        sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
        x, y = max(0, (sw - width) // 2), max(0, (sh - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.lift()

    def _set_loading_status(self, text: str, detail: str = "") -> None:
        if not self._startup_active or self._loading_window is None:
            return
        self._loading_label_var.set(text)
        self._loading_detail_var.set(detail)
        try:
            self.root.update()
        except tk.TclError:
            pass

    def _finish_loading_screen(self) -> None:
        try:
            # draw_idle() is normally enough, but at startup we explicitly finish the
            # first canvas draw before removing the loading screen.
            if hasattr(self, "canvas"):
                self.canvas.draw()
        except Exception as exc:  # UI should still become usable if final draw complains.
            print(f"[startup draw warning] {exc}", file=sys.stderr, flush=True)
        if self._loading_bar is not None:
            self._loading_bar.stop()
        if self._loading_window is not None:
            try:
                self._loading_window.destroy()
            except tk.TclError:
                pass
            self._loading_window = None
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _install_session_menu(self) -> None:
        menu = tk.Menu(self.root)
        session_menu = tk.Menu(menu, tearoff=False)
        session_menu.add_command(
            label="Restart from beginning",
            command=self.restart_from_beginning,
            accelerator="Ctrl+Home",
        )
        menu.add_cascade(label="Session", menu=session_menu)
        self.root.configure(menu=menu)
        self.root.bind("<Control-Home>", lambda _event: self.restart_from_beginning())

    def _busy(self, text: str) -> None:
        if self._startup_active:
            self._set_loading_status(text, self._loading_detail_var.get())
        super()._busy(text)

    def next_model(self, *, mark_current: bool = True) -> None:
        # CategoriserApp.__init__ calls next_model(False). Intercept that first call so
        # startup can scan directly to the saved cursor without rendering every model.
        if self._startup_active and not mark_current:
            target = self._startup_resume_target
            if not target:
                self._set_loading_status("Loading first model...", "Scanning model data and textures.")
                super().next_model(mark_current=False)
                return

            self._set_loading_status(
                "Resuming previous session...",
                f"Looking for {target}",
            )
            found_index: int | None = None
            while not self.exhausted:
                model = self._load_next()
                if model is None:
                    break
                candidate_index = len(self.models) - 1
                self._set_loading_status(
                    "Resuming previous session...",
                    f"Scanned {len(self.models):,} model(s) • {model.model_path}",
                )
                if model.model_path == target:
                    found_index = candidate_index
                    break

            if found_index is None:
                # The input set may have changed. Prefer the first unreviewed model;
                # otherwise keep the user at the last available model.
                for index, model in enumerate(self.models):
                    classification = self.state.get(model.model_path)
                    if classification is None or not classification.reviewed:
                        found_index = index
                        break
                if found_index is None and self.models:
                    found_index = len(self.models) - 1
                print(
                    f"[resume] saved model not found: {target}; using fallback position",
                    file=sys.stderr,
                    flush=True,
                )

            if found_index is None:
                summary = f"End of scan: 0 model(s), {len(self.failures)} failure(s)"
                self.progress_var.set(summary)
                self.next_button.configure(state=tk.DISABLED)
                return

            self.index = found_index
            self._set_loading_status(
                "Rendering saved model...",
                self.models[self.index].model_path,
            )
            self._show_model(self.models[self.index])
            return

        super().next_model(mark_current=mark_current)

    def _show_model(self, model) -> None:
        self._resume_model_path = model.model_path
        if self._startup_active:
            self._set_loading_status("Rendering textured model...", model.model_path)
        super()._show_model(model)

    def restart_from_beginning(self) -> None:
        """Move navigation to the first model without clearing classifications."""
        if self.current is not None:
            self._commit(True)
        if not self.models:
            self.next_model(mark_current=False)
            return
        self.index = 0
        self._resume_model_path = self.models[0].model_path
        self._show_model(self.models[0])
        self.save_state()
        self.status_var.set("Restarted browsing from the beginning; classifications were kept.")

    def save_state(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        resume_path = self.current.model_path if self.current is not None else self._resume_model_path
        report = {
            "schema": 3,
            "categories": self.categories,
            "placements": list(PLACEMENTS),
            "resume_model_path": resume_path or "",
            "reviewed_count": sum(1 for x in self.state.values() if x.reviewed),
            "classified_count": sum(1 for x in self.state.values() if x.categories),
            "placement_count": sum(1 for x in self.state.values() if x.placement),
            "models": [
                {
                    "model_path": key,
                    "categories": self.state[key].categories,
                    "placement": self.state[key].placement,
                    "reviewed": self.state[key].reviewed,
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
        if hasattr(self, "status_var"):
            self.status_var.set(f"Saved: {self.output}")
