#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cwr_worldgen.source_pipeline import (  # noqa: E402
    Milestone5Spec,
    SourceFetchSpec,
    build_milestone5,
    fetch_sources,
    validate_source_bundle,
)

MAP_URL = "https://opentopomap.org/#map=13/59.42766/16.89457"


def _zip_tree(source: Path, destination: Path, archive_root: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            info = ZipInfo(f"{archive_root}/{relative}", date_time=(2001, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the real 59.42766N, 16.89457E CWR-CE example")
    parser.add_argument("--output", type=Path, default=Path("build/malaren-example"))
    parser.add_argument("--source-dir", type=Path, default=Path("build/malaren-source-data"))
    parser.add_argument("--asset-root", type=Path, action="append", default=[])
    parser.add_argument("--strict-assets", action="store_true")
    parser.add_argument("--include-minor-roads", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="explicitly replace the frozen OSM and DEM snapshot")
    parser.add_argument("--reference-map", action="store_true")
    parser.add_argument("--dem-provider", choices=("dem-stitcher", "hgt"), default="dem-stitcher")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.fetch_only and args.build_only:
        parser.error("--fetch-only and --build-only are mutually exclusive")

    if not args.build_only:
        bundle = fetch_sources(
            SourceFetchSpec(
                source_dir=args.source_dir,
                map_url=MAP_URL,
                cells=256,
                cell_size=25.0,
                refresh=args.refresh,
                reference_map=args.reference_map,
                dem_provider=args.dem_provider,
            )
        )
    else:
        bundle = validate_source_bundle(args.source_dir).bundle

    print(f"Source bundle: {bundle.root}")
    print("BBox:          " + " ".join(f"{value:.12f}" for value in bundle.bbox))
    print(f"Fingerprint:   {bundle.fingerprint}")
    if args.fetch_only:
        return 0

    result = build_milestone5(
        args.output,
        Milestone5Spec(
            source_dir=bundle.root,
            name="cwr_malaren",
            display_name="Malaren Topographic Example",
            asset_roots=tuple(args.asset_root),
            strict_assets=args.strict_assets,
            include_minor_roads=args.include_minor_roads,
        ),
    )
    mod_root = result.pbo_path.parent.parent
    mission_root = result.mission_path.parent
    runtime_zip = result.output_dir / "cwr-malaren-runtime.zip"
    mission_zip = result.output_dir / "cwr-malaren-test-mission.zip"
    _zip_tree(mod_root, runtime_zip, mod_root.name)
    _zip_tree(mission_root, mission_zip, mission_root.name)
    print(f"Mod folder:    {mod_root}")
    print(f"WRP:           {result.wrp_path}")
    print(f"PBO:           {result.pbo_path}")
    print(f"Runtime ZIP:   {runtime_zip}")
    print(f"Mission ZIP:   {mission_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
