#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Browse textured OFP/CWA P3D models and assign catalogue categories."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

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
from p3d_categoriser_split import SplitStateCategoriserApp
from p3d_categoriser_storage import load_state
from p3d_preview_geometry import preview_models
from p3d_texture_io import TextureResolver

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
        description="Preview textured OFP/CWA P3D first-LOD meshes and assign categories."
    )
    p.add_argument("inputs", nargs="+", type=Path, help="P3D, PBO, or directory to scan")
    p.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only include canonical model paths matching GLOB. May be repeated.",
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
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_preview_points < 100:
        print("error: --max-preview-points must be at least 100", file=sys.stderr)
        return 2

    inputs = _resolve_inputs(args.inputs)
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

    resume_model_path = "" if args.restart else load_resume_path(args.output)
    if resume_model_path:
        print(f"[resume] saved position: {resume_model_path}", file=sys.stderr, flush=True)
    elif args.restart:
        print("[resume] restart requested; starting from first model", file=sys.stderr, flush=True)

    iterator = preview_models(inputs, args.include, args.max_preview_points)
    root = tk.Tk()
    app = SplitStateCategoriserApp(
        root,
        model_iter=iterator,
        texture_resolver=TextureResolver(inputs),
        output=args.output,
        categories=categories,
        state=state,
        resume_model_path=resume_model_path,
    )
    root.mainloop()
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
