#!/usr/bin/env pythonw
"""Open the CWR P3D model and texture browser."""
from __future__ import annotations

try:
    import tkinter  # noqa: F401
except ImportError as exc:
    raise SystemExit("Tkinter is required for the asset browser.") from exc

from p3d_asset_browser_app import launch


if __name__ == "__main__":
    raise SystemExit(launch())
