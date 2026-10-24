#!/usr/bin/env python3
"""Compare original CWA and known-working SFP road P3Ds from asset roots.

Usage:
    python tools/compare_paved_road_models.py --asset-root "C:/OFP/Addons" \
        --asset-root "C:/OFP/Res/Addons" --asset-root "C:/SFP/Addons" \
        --output paved-road-comparison.json

Roots can be directories containing .p3d/.pbo files or individual .pbo files.
No game or mod files are copied into the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cwr_worldgen.road_surface_comparison import (
    STOCK_PAVED, SFP_STRAIGHT, SFP_CURVE, compare_paved_models,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, action="append",
                        type=Path, help="Folder or PBO holding stock/SFP assets")
    parser.add_argument("--stock", default=STOCK_PAVED,
                        help="Original paved straight model path")
    parser.add_argument("--sfp-straight", default=SFP_STRAIGHT,
                        help="Working SFP paved straight model path")
    parser.add_argument("--sfp-curve", default=SFP_CURVE,
                        help="Working SFP paved curve model path")
    parser.add_argument("--output", type=Path, help="Write JSON comparison here")
    args = parser.parse_args(argv)
    report = compare_paved_models(
        args.asset_root, (args.stock, args.sfp_straight, args.sfp_curve),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    if not report["all_models_readable"]:
        print("WARNING: Some donors missing or unreadable; no geometry fix inferred.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
