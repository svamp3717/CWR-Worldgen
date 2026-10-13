#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Categorise P3D models from OFP, Arma and Iron Front archives."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from p3d_frozen_support import configure_windowed_logging, dismiss_splash

# PyInstaller --windowed has no stderr/stdout. Preserve scan diagnostics.
configure_windowed_logging("categorise-p3d-models")

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError as exc:
    raise SystemExit("Tkinter is required. Install Python with Tcl/Tk enabled.") from exc

try:
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit("Install dependencies with: python -m pip install matplotlib numpy") from exc

from p3d_categoriser_session import load_resume_path
from p3d_categoriser_startup import prompt_for_startup_selection
from p3d_categoriser_split import SplitStateCategoriserApp
from p3d_categoriser_storage import load_state
import measure_p3d_models as measure
from p3d_preview_geometry import preview_models
from p3d_texture_sources import TextureResolver

DEFAULT_CATEGORIES = (
    "Residential", "Commercial", "Industrial", "Agricultural", "Military",
    "Civic / Public", "Religious", "Infrastructure", "Ruins", "Prop / Misc",
)
DEFAULT_MAX_PREVIEW_POINTS = 25_000


def _resolve_inputs(raw_inputs: Sequence[Path]) -> list[Path]:
    inputs = [p.expanduser() for p in raw_inputs]
    if not inputs or all(p.exists() for p in inputs):
        return inputs
    if len(inputs) > 1:
        joined = Path(" ".join(str(p) for p in inputs)).expanduser()
        if joined.exists():
            return [joined]
    return inputs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Categorise P3D models from OFP/CWA, Arma 1/2/OA/3 and Iron Front archives."
    )
    p.add_argument(
        "inputs", nargs="*", type=Path,
        help=(
            "P3D, PBO, IFA, Zstd-wrapped archive, or folder to scan. "
            "If omitted, a GUI setup window opens to select model sources and JSON output."
        ),
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only include canonical model paths matching GLOB. May be repeated.",
    )
    p.add_argument(
        "--texture-source",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "Additional PBO/IFA archive or directory used only for texture lookup. "
            "May be repeated; it does not add models to the scan. "
            "Sibling Data.pbo, Structures.pbo, LIB_*.ifa and other "
            "namespace archives are discovered automatically."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("p3d-model-categories.json"),
        help=(
            "Completed classification JSON; reviewed incomplete models are stored "
            "automatically in a .unclassified.json companion file."
        ),
    )
    p.add_argument(
        "--category",
        action="append",
        dest="categories",
        metavar="NAME",
        help="Custom category checkbox; repeat for several.",
    )
    p.add_argument(
        "--max-preview-points",
        type=int,
        default=DEFAULT_MAX_PREVIEW_POINTS,
        metavar="N",
        help="Point limit used only when face topology is unavailable.",
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="Start browsing at the first model instead of resuming the saved position. Classifications are kept.",
    )
    p.add_argument(
        "--start-model",
        type=int,
        metavar="N",
        help=(
            "Start at 1-based model number N instead of resuming the saved position. "
            "The same jump is available in the GUI with Ctrl+G."
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_preview_points < 100:
        print("error: --max-preview-points must be at least 100", file=sys.stderr)
        return 2
    if args.start_model is not None and args.start_model < 1:
        print("error: --start-model must be at least 1", file=sys.stderr)
        return 2
    if args.restart and args.start_model is not None:
        print("error: --restart and --start-model cannot be used together", file=sys.stderr)
        return 2

    inputs = _resolve_inputs(args.inputs)
    selected_texture_sources: tuple[Path, ...] = ()
    if not inputs:
        # A double-click or "python categorise_p3d_models.py" should be a
        # usable GUI workflow, rather than argparse exiting for a missing PBO.
        try:
            selected = prompt_for_startup_selection(default_output=args.output)
        except tk.TclError as exc:
            print(f"error: could not open model/JSON selection GUI: {exc}", file=sys.stderr)
            return 2
        if selected is None:
            return 0  # Cancel: do not scan or touch the JSON output.
        inputs = list(selected.inputs)
        selected_texture_sources = selected.texture_sources
        args.output = selected.output
    missing = [p for p in inputs if not p.exists()]
    if missing:
        text = (
            "Input path does not exist:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nQuote paths containing spaces."
        )
        print(f"error: {text}", file=sys.stderr)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Scan path not found", text, parent=root)
            root.destroy()
        except tk.TclError:
            pass
        return 2

    # Only model inputs are counted/scanned. Extra archives are sent to the
    # texture resolver, not the model iterator (also when picked at startup).
    texture_sources = list(dict.fromkeys(
        path.expanduser()
        for path in (*args.texture_source, *selected_texture_sources)
    ))
    missing_texture_sources = [path for path in texture_sources if not path.exists()]
    if missing_texture_sources:
        text = (
            "Texture source path does not exist:\n"
            + "\n".join(f"  {path}" for path in missing_texture_sources)
        )
        print(f"error: {text}", file=sys.stderr)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Texture source not found", text, parent=root)
            root.destroy()
        except tk.TclError:
            pass
        return 2

    try:
        state, saved = load_state(args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    categories = args.categories or saved or list(DEFAULT_CATEGORIES)
    categories = list(dict.fromkeys(c.strip() for c in categories if c.strip()))
    if not categories:
        print("error: at least one category is required", file=sys.stderr)
        return 2

    resume_model_path = (
        ""
        if args.restart or args.start_model is not None
        else load_resume_path(args.output)
    )
    if resume_model_path:
        print(f"[resume] saved position: {resume_model_path}", file=sys.stderr, flush=True)
    elif args.restart:
        print("[resume] restart requested; starting from first model", file=sys.stderr, flush=True)

    texture_inputs = list(dict.fromkeys([*inputs, *texture_sources]))
    if texture_sources:
        print(
            "[textures] additional source(s): " + ", ".join(str(path) for path in texture_sources),
            file=sys.stderr,
            flush=True,
        )

    try:
        total_models = measure.count_models(inputs, args.include)
    except (OSError, ValueError) as exc:
        # Counting is only a progress-bar convenience. Do not block browsing
        # archives whose header variant the lightweight counter cannot understand
        # if the streaming scanner can still read models from them.
        total_models = None
        print(
            f"[count warning] could not count models: {exc}; continuing without a total",
            file=sys.stderr,
            flush=True,
        )

    if (
        args.start_model is not None
        and total_models is not None
        and args.start_model > total_models
    ):
        print(
            f"error: --start-model {args.start_model:,} exceeds "
            f"the {total_models:,} scanned model(s)",
            file=sys.stderr,
        )
        return 2

    iterator = preview_models(inputs, args.include, args.max_preview_points)
    root = tk.Tk()
    app = SplitStateCategoriserApp(
        root,
        model_iter=iterator,
        texture_resolver=TextureResolver(texture_inputs),
        output=args.output,
        categories=categories,
        state=state,
        resume_model_path=resume_model_path,
        total_models=total_models,
    )
    if args.start_model is not None and args.start_model != 1:
        app.jump_to_model_number(args.start_model)
    dismiss_splash(root)
    root.mainloop()
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
