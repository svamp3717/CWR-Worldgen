#!/usr/bin/env python3
"""Launch the CWR WRP mod dependency scanner from a source checkout."""
from __future__ import annotations

from pathlib import Path
import sys

if not getattr(sys, "frozen", False):
    # The frozen executable already contains cwr_worldgen; it has no src tree.
    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from cwr_worldgen.wrp_mod_dependency_scanner import main


if __name__ == "__main__":
    raise SystemExit(main())
