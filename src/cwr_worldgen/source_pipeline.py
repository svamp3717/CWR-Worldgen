# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

from .generator import BuildResult, build_milestone4
from .images import _image_values
from .location_example import (
    HgtTile,
    ProgressCallback,
    MapLink,
    _download,
    _fetch_overpass,
    _reference_map,
    _tile_origin,
    _write_heightmap,
    parse_opentopomap_link,
    required_hgt_tiles,
    reference_zoom_for_bbox,
    square_bbox,
)
from .model import DEFAULT_MAX_BUILDINGS, DEFAULT_MAX_FOREST_OBJECTS, DEFAULT_MAX_ROAD_OBJECTS, PlayabilitySpec
from .overture import fetch_overture_buildings_geojson
from .progress import report_progress
from .overpass_endpoints import GLOBAL_OVERPASS_URLS

SOURCE_SCHEMA = "cwr-worldgen-source-bundle"
SOURCE_SCHEMA_VERSION = 2
DEFAULT_OVERPASS_URLS = GLOBAL_OVERPASS_URLS
DEFAULT_HGT_URL_TEMPLATES = (
    "https://download.mapsforge.org/maps/dem/dem3/{latitude_band}/{tile}.hgt.zip",
    "https://bailu.ch/dem3/{latitude_band}/{tile}.hgt.zip",
)


def _mapped_progress(start: int, end: int) -> ProgressCallback:
    """Map a helper's 0..100 progress into a fetch-pipeline range."""

    span = end - start

    def callback(percent: int, stage: str) -> None:
        report_progress(start + round(span * max(0, min(100, int(percent))) / 100), stage)

    return callback


def _run_with_progress_heartbeat(
    operation: Callable[[], Any],
    *,
    progress_callback: ProgressCallback | None,
    percent: int,
    stage: str,
    interval_seconds: float = 10.0,
) -> Any:
    """Run a blocking provider call while emitting honest same-percent heartbeat updates."""

    if progress_callback is None:
        return operation()
    stopped = threading.Event()
    started = time.perf_counter()

    def heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            elapsed = int(time.perf_counter() - started)
            progress_callback(percent, f"{stage}; provider still working ({elapsed}s)")

    thread = threading.Thread(target=heartbeat, name="cwr-source-fetch-heartbeat", daemon=True)
    thread.start()
    try:
        return operation()
    finally:
        stopped.set()
        thread.join(timeout=max(0.1, interval_seconds))


@dataclass(frozen=True, slots=True)
class SourceFetchSpec:
    source_dir: Path
    map_url: str | None = None
    center: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    cells: int = 256
    cell_size: float = 25.0
    refresh: bool = False
    reference_map: bool = False
    dem_provider: str = "dem-stitcher"
    dem_name: str = "glo_30"
    overpass_urls: tuple[str, ...] = DEFAULT_OVERPASS_URLS
    overpass_timeout_seconds: int = 240
    hgt_url_templates: tuple[str, ...] = DEFAULT_HGT_URL_TEMPLATES

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size

    def validate(self) -> None:
        selected = sum(value is not None for value in (self.map_url, self.center, self.bbox))
        if selected != 1:
            raise ValueError("specify exactly one of map_url, center, or bbox")
        if self.cells < 16 or self.cells > 2048 or self.cells & (self.cells - 1):
            raise ValueError("cells must be a power of two between 16 and 2048")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError("cell size must be positive and finite")
        if self.dem_provider not in {"dem-stitcher", "hgt"}:
            raise ValueError("DEM provider must be 'dem-stitcher' or 'hgt'")
        if self.overpass_timeout_seconds < 30 or self.overpass_timeout_seconds > 600:
            raise ValueError("Overpass timeout must be within 30..600 seconds")
        if not self.overpass_urls:
            raise ValueError("at least one Overpass endpoint is required")
        if self.dem_provider == "hgt" and not self.hgt_url_templates:
            raise ValueError("at least one HGT mirror template is required")


@dataclass(frozen=True, slots=True)
class SourceRegridSpec:
    source_dir: Path
    output_source_dir: Path
    cells: int | None = None
    cell_size: float | None = 25.0
    replace_output: bool = False

    def resolved_grid(self, source: "FrozenSourceBundle") -> tuple[int, float]:
        world_size = source.cells * source.cell_size
        cells = self.cells
        cell_size = self.cell_size
        if cells is None and cell_size is None:
            cell_size = 25.0
        if cells is None:
            assert cell_size is not None
            if not math.isfinite(cell_size) or cell_size <= 0:
                raise ValueError("target cell size must be positive and finite")
            inferred = world_size / cell_size
            cells = int(round(inferred))
            if not math.isclose(cells, inferred, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError(
                    f"source world size {world_size:g}m is not evenly divisible by target cell size {cell_size:g}m"
                )
        if not isinstance(cells, int) or cells < 16 or cells > 2048 or cells & (cells - 1):
            raise ValueError("target cells must be a power of two between 16 and 2048")
        if cell_size is None:
            cell_size = world_size / cells
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("target cell size must be positive and finite")
        target_world_size = cells * cell_size
        if not math.isclose(target_world_size, world_size, rel_tol=0.0, abs_tol=0.01):
            raise ValueError(
                f"target grid is {target_world_size:g}m wide but the frozen bbox represents {world_size:g}m; "
                "choose cells and cell size that preserve the source world size"
            )
        return cells, float(cell_size)


@dataclass(frozen=True, slots=True)
class FrozenSourceBundle:
    root: Path
    manifest_path: Path
    checksum_path: Path
    heightmap_path: Path
    osm_json_path: Path
    overpass_query_path: Path
    osm_attribution_path: Path
    dem_attribution_path: Path
    reference_map_path: Path | None
    overture_buildings_geojson_path: Path | None
    bbox: tuple[float, float, float, float]
    cells: int
    cell_size: float
    heightmap_grid: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class SourceValidationReport:
    bundle: FrozenSourceBundle
    checks: tuple[tuple[str, bool, str], ...]
    report_path: Path

    @property
    def valid(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    _write_atomic(path, text.encode(encoding))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"source manifest path escapes bundle root: {relative}") from exc
    return candidate


def _validate_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain south, west, north, east")
    south, west, north, east = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (south, west, north, east)):
        raise ValueError("bbox coordinates must be finite")
    if not (-90 <= south < north <= 90):
        raise ValueError("bbox latitude order or range is invalid")
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitude order or range is invalid")
    return south, west, north, east


def _selection(spec: SourceFetchSpec) -> tuple[dict[str, Any], tuple[float, float, float, float], MapLink | None]:
    if spec.map_url is not None:
        link = parse_opentopomap_link(spec.map_url)
        bbox = square_bbox(link.latitude, link.longitude, spec.world_size)
        return (
            {
                "kind": "opentopomap-url",
                "value": link.url,
                "zoom": link.zoom,
                "center_latitude_longitude": [link.latitude, link.longitude],
            },
            bbox,
            link,
        )
    if spec.center is not None:
        latitude, longitude = spec.center
        if not (-85.05112878 < latitude < 85.05112878):
            raise ValueError("center latitude is outside supported Web Mercator coverage")
        if not (-180 <= longitude <= 180):
            raise ValueError("center longitude is outside -180..180")
        return (
            {
                "kind": "center",
                "center_latitude_longitude": [float(latitude), float(longitude)],
            },
            square_bbox(float(latitude), float(longitude), spec.world_size),
            None,
        )
    assert spec.bbox is not None
    bbox = _validate_bbox(spec.bbox)
    center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    return ({"kind": "bbox", "center_latitude_longitude": center}, bbox, None)


def _manifest_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("source manifest must be a JSON object")
    return document


def _bundle_from_manifest(root: Path, document: Mapping[str, Any]) -> FrozenSourceBundle:
    selection = document.get("selection")
    osm = document.get("osm")
    elevation = document.get("elevation")
    if not isinstance(selection, Mapping) or not isinstance(osm, Mapping) or not isinstance(elevation, Mapping):
        raise ValueError("source manifest is missing selection, osm, or elevation sections")
    bbox = _validate_bbox(selection.get("bbox_south_west_north_east", ()))
    cells = int(selection.get("cells", 0))
    cell_size = float(selection.get("cell_size_metres", 0.0))
    resampling = elevation.get("resampling")
    heightmap_grid = (
        str(resampling.get("target_grid", "game-cell-centres"))
        if isinstance(resampling, Mapping)
        else "game-cell-centres"
    )
    reference = document.get("reference_map")
    reference_path = None
    if isinstance(reference, Mapping) and isinstance(reference.get("path"), str):
        reference_path = _safe_path(root, reference["path"])
    overture = document.get("overture")
    overture_buildings_path = None
    if isinstance(overture, Mapping) and isinstance(overture.get("buildings_geojson"), str):
        overture_buildings_path = _safe_path(root, overture["buildings_geojson"])
    manifest_path = root / "source.json"
    return FrozenSourceBundle(
        root=root,
        manifest_path=manifest_path,
        checksum_path=root / "SHA256SUMS.txt",
        heightmap_path=_safe_path(root, str(elevation.get("heightmap", ""))),
        osm_json_path=_safe_path(root, str(osm.get("raw_json", ""))),
        overpass_query_path=_safe_path(root, str(osm.get("query", ""))),
        osm_attribution_path=root / "attribution" / "OSM-ATTRIBUTION.txt",
        dem_attribution_path=root / "attribution" / "DEM-ATTRIBUTION.txt",
        reference_map_path=reference_path,
        overture_buildings_geojson_path=overture_buildings_path,
        bbox=bbox,
        cells=cells,
        cell_size=cell_size,
        heightmap_grid=heightmap_grid,
        fingerprint=_sha256(manifest_path),
    )


def load_source_bundle(source_dir: Path) -> FrozenSourceBundle:
    root = source_dir.resolve()
    manifest_path = root / "source.json"
    if not manifest_path.is_file():
        raise ValueError(f"source bundle manifest does not exist: {manifest_path}")
    document = _manifest_json(manifest_path)
    return _bundle_from_manifest(root, document)


def _parse_checksum_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}: {line!r}") from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 digest on line {line_number}")
        entries[relative] = digest
    return entries


def validate_source_bundle(
    source_dir: Path,
    *,
    write_report: bool = True,
    progress_callback: Callable[[int, str], None] | None = None,
) -> SourceValidationReport:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, int(percent))), stage)

    root = source_dir.resolve()
    progress(0, "Reading source bundle manifest")
    checks: list[tuple[str, bool, str]] = []
    hash_cache: dict[Path, str] = {}

    def file_hash(path: Path) -> str:
        resolved = path.resolve()
        cached = hash_cache.get(resolved)
        if cached is None:
            cached = _sha256(resolved)
            hash_cache[resolved] = cached
        return cached

    manifest_path = root / "source.json"
    try:
        document = _manifest_json(manifest_path)
        manifest_ok = True
        manifest_detail = manifest_path.name
    except ValueError as exc:
        manifest_ok = False
        manifest_detail = str(exc)
        document = {}
    checks.append(("Source manifest readable", manifest_ok, manifest_detail))
    progress(8, "Source manifest loaded")
    if not manifest_ok:
        report_path = root / "source-validation-report.txt"
        lines = ["CWR World Generator - Milestone 5 source validation", ""]
        lines.extend(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}" for label, ok, detail in checks)
        if write_report:
            root.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(report_path, "\n".join(lines) + "\n")
        raise ValueError(manifest_detail)

    checks.append(("Source schema", document.get("schema") == SOURCE_SCHEMA, str(document.get("schema"))))
    checks.append(("Source schema version", document.get("schema_version") == SOURCE_SCHEMA_VERSION, str(document.get("schema_version"))))
    try:
        bundle = _bundle_from_manifest(root, document)
        bundle_fields_ok = bundle.cells >= 16 and bundle.cell_size > 0
        bundle_fields_detail = f"{bundle.cells}x{bundle.cells} @ {bundle.cell_size:g}m"
    except (ValueError, TypeError, OSError) as exc:
        bundle_fields_ok = False
        bundle_fields_detail = str(exc)
        bundle = FrozenSourceBundle(
            root=root,
            manifest_path=manifest_path,
            checksum_path=root / "SHA256SUMS.txt",
            heightmap_path=root / "missing-heightmap",
            osm_json_path=root / "missing-osm",
            overpass_query_path=root / "missing-query",
            osm_attribution_path=root / "attribution" / "OSM-ATTRIBUTION.txt",
            dem_attribution_path=root / "attribution" / "DEM-ATTRIBUTION.txt",
            reference_map_path=None,
            overture_buildings_geojson_path=None,
            bbox=(0.0, 0.0, 1.0, 1.0),
            cells=0,
            cell_size=0.0,
            heightmap_grid="game-cell-centres",
            fingerprint="",
        )
    checks.append(("Selection and path fields", bundle_fields_ok, bundle_fields_detail))

    files = document.get("files")
    file_entries_ok = isinstance(files, Mapping) and bool(files)
    progress(15, "Verifying frozen source file hashes")
    checks.append(("Manifest contains frozen file hashes", file_entries_ok, str(len(files) if isinstance(files, Mapping) else 0)))
    if isinstance(files, Mapping):
        sorted_files = sorted(files.items())
        file_total = max(1, len(sorted_files))
        for file_index, (relative, wanted) in enumerate(sorted_files, start=1):
            progress(15 + int(file_index * 35 / file_total), f"Verifying source file {file_index}/{file_total}: {relative}")
            try:
                path = _safe_path(root, str(relative))
                actual = file_hash(path) if path.is_file() else "missing"
                ok = isinstance(wanted, str) and actual == wanted
                detail = actual if ok else f"wanted={wanted}, actual={actual}"
            except (OSError, ValueError) as exc:
                ok = False
                detail = str(exc)
            checks.append((f"Frozen file {relative}", ok, detail))

    progress(55, f"Parsing frozen OpenStreetMap JSON: {bundle.osm_json_path.name}")
    try:
        osm_document = json.loads(bundle.osm_json_path.read_text(encoding="utf-8"))
        element_count = len(osm_document.get("elements", [])) if isinstance(osm_document, dict) else -1
        osm_ok = isinstance(osm_document, dict) and isinstance(osm_document.get("elements"), list)
        osm_detail = f"{element_count} elements"
    except (OSError, json.JSONDecodeError) as exc:
        osm_ok = False
        osm_detail = str(exc)
    checks.append(("Frozen Overpass JSON", osm_ok, osm_detail))

    progress(70, f"Checking frozen heightmap: {bundle.heightmap_path.name}")
    try:
        with Image.open(bundle.heightmap_path) as image:
            image.load()
            heightmap_size = image.size
            values = [float(value) for value in _image_values(image)]
        height_ok = heightmap_size == (bundle.cells, bundle.cells) and values and all(math.isfinite(v) for v in values)
        height_detail = f"{heightmap_size[0]}x{heightmap_size[1]}, {min(values):.2f}..{max(values):.2f}m" if values else "empty"
    except (OSError, ValueError, TypeError) as exc:
        height_ok = False
        height_detail = str(exc)
    checks.append(("Frozen metre heightmap", height_ok, height_detail))

    elevation_document = document.get("elevation")
    resampling_document = elevation_document.get("resampling") if isinstance(elevation_document, Mapping) else None
    method = resampling_document.get("method") if isinstance(resampling_document, Mapping) else None
    target_grid = resampling_document.get("target_grid") if isinstance(resampling_document, Mapping) else None
    resampling_ok = (
        method in {"georeferenced-bilinear-sampling", "direct-hgt-bilinear"}
        and target_grid in {"game-cell-centres", "game-terrain-vertices"}
    )
    checks.append((
        "Georeferenced DEM resampling",
        resampling_ok,
        f"method={method}, grid={target_grid}",
    ))

    checks.append(("OSM attribution present", bundle.osm_attribution_path.is_file(), bundle.osm_attribution_path.name))
    checks.append(("DEM attribution present", bundle.dem_attribution_path.is_file(), bundle.dem_attribution_path.name))

    progress(82, "Verifying SHA256SUMS using cached file hashes")
    checksum_ok = bundle.checksum_path.is_file()
    checksum_detail = bundle.checksum_path.name
    if checksum_ok:
        try:
            checksum_entries = _parse_checksum_file(bundle.checksum_path)
            for relative, wanted in checksum_entries.items():
                path = _safe_path(root, relative)
                if not path.is_file() or file_hash(path) != wanted:
                    checksum_ok = False
                    checksum_detail = f"checksum mismatch: {relative}"
                    break
        except (OSError, ValueError) as exc:
            checksum_ok = False
            checksum_detail = str(exc)
    checks.append(("SHA256SUMS verifies", checksum_ok, checksum_detail))

    progress(96, "Writing source validation report")
    report_path = root / "source-validation-report.txt"
    lines = ["CWR World Generator - Milestone 5 source validation", ""]
    lines.extend(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}" for label, ok, detail in checks)
    failures = [label for label, ok, _ in checks if not ok]
    lines.extend(["", f"Failures: {len(failures)}"])
    if write_report:
        _write_text_atomic(report_path, "\n".join(lines) + "\n")
    report = SourceValidationReport(bundle=bundle, checks=tuple(checks), report_path=report_path)
    if failures:
        raise ValueError("source bundle validation failed: " + "; ".join(failures))
    progress(100, "Source bundle validation complete")
    return report


def _source_selection_matches(document: Mapping[str, Any], bbox: tuple[float, float, float, float], spec: SourceFetchSpec) -> bool:
    selection = document.get("selection")
    elevation = document.get("elevation")
    if not isinstance(selection, Mapping) or not isinstance(elevation, Mapping):
        return False
    try:
        existing_bbox = _validate_bbox(selection.get("bbox_south_west_north_east", ()))
        existing_cells = int(selection.get("cells", 0))
        existing_cell_size = float(selection.get("cell_size_metres", 0.0))
    except (ValueError, TypeError):
        return False
    provider_matches = elevation.get("provider") == spec.dem_provider
    dem_name_matches = spec.dem_provider != "dem-stitcher" or elevation.get("dem_name") == spec.dem_name
    reference_matches = bool(document.get("reference_map")) == spec.reference_map
    return (
        existing_cells == spec.cells
        and math.isclose(existing_cell_size, spec.cell_size, rel_tol=0.0, abs_tol=1e-9)
        and all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(existing_bbox, bbox))
        and provider_matches
        and dem_name_matches
        and reference_matches
    )


def _fetch_hgt_elevation(
    spec: SourceFetchSpec,
    bbox: tuple[float, float, float, float],
    root: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, list[dict[str, str]], float, float, str, dict[str, Any]]:
    raw_dir = root / "elevation" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles: dict[str, HgtTile] = {}
    records: list[dict[str, str]] = []
    tile_names = required_hgt_tiles(bbox)
    total = max(1, len(tile_names))
    if progress_callback is not None:
        progress_callback(0, f"Resolved {len(tile_names)} HGT elevation tiles")
    for tile_index, tile_name in enumerate(tile_names, start=1):
        latitude_origin, _ = _tile_origin(tile_name)
        latitude_band = f"{'N' if latitude_origin >= 0 else 'S'}{abs(latitude_origin):02d}"
        path = raw_dir / f"{tile_name}.hgt.zip"
        urls = [template.format(latitude_band=latitude_band, tile=tile_name) for template in spec.hgt_url_templates]
        tile_start = 2 + round((tile_index - 1) * 58 / total)
        tile_end = 2 + round(tile_index * 58 / total)

        def tile_progress(percent: int, stage: str, *, _start: int = tile_start, _end: int = tile_end) -> None:
            if progress_callback is not None:
                progress_callback(
                    _start + round((_end - _start) * percent / 100),
                    f"HGT tile {tile_index}/{len(tile_names)} {tile_name}: {stage}",
                )

        source = _download(
            urls,
            path,
            refresh=spec.refresh,
            timeout=180,
            progress_callback=tile_progress,
        )
        if progress_callback is not None:
            progress_callback(tile_end, f"Reading HGT tile {tile_index}/{len(tile_names)}: {tile_name}")
        tiles[tile_name] = HgtTile.from_zip(path, tile_name)
        records.append({"path": _relative(root, path), "source": source, "sha256": _sha256(path)})
    heightmap_path = root / "elevation" / "heightmap-meters.tif"

    def heightmap_progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(62 + round(38 * percent / 100), stage)

    minimum, maximum = _write_heightmap(
        heightmap_path,
        bbox,
        spec.cells,
        tiles,
        progress_callback=heightmap_progress,
    )
    resampling: dict[str, Any] = {
        "method": "direct-hgt-bilinear",
        "target_grid": "game-terrain-vertices",
        "target_width": spec.cells,
        "target_height": spec.cells,
        "target_bounds_west_south_east_north": [bbox[1], bbox[0], bbox[3], bbox[2]],
        "output_minimum_metres": minimum,
        "output_maximum_metres": maximum,
    }
    return heightmap_path, records, minimum, maximum, "SRTM3 HGT", resampling


def _heightmap_from_raster(
    path: Path,
    output: Path,
    cells: int,
    bbox: tuple[float, float, float, float],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Resample a georeferenced DEM onto the exact 4WVR terrain vertices.

    The old Milestone 5 implementation converted the complete raster to a Pillow
    image and resized it. ``dem-stitcher`` aligns its output to the source DEM
    pixels, so the raster bounds can extend beyond the requested bbox. Treating
    that array as an unreferenced image shifts and stretches the terrain and also
    performs a second, avoidable image-space resample.

    The target samples are the west/south-inclusive WRP vertices. The final
    east/north bbox edges are outside the stored ``cells × cells`` vertex array.
    """
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, stage)

    progress(0, f"Opening raw DEM raster: {path.name}")
    try:
        import numpy as np
        import rasterio
        from rasterio.fill import fillnodata
        from rasterio.transform import from_origin
        from rasterio.warp import transform as warp_coordinates
    except ImportError as exc:
        raise RuntimeError(
            "dem-stitcher source support requires the 'sources' extra: "
            "python -m pip install -e '.[sources]'"
        ) from exc

    south, west, north, east = bbox
    longitude_step = (east - west) / cells
    latitude_step = (north - south) / cells
    destination_transform = from_origin(
        west - longitude_step * 0.5,
        north - latitude_step * 0.5,
        longitude_step,
        latitude_step,
    )
    progress(5, f"Reading raw DEM samples for {cells}x{cells} target grid")

    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError("downloaded DEM has no coordinate reference system")
        source = dataset.read(1, masked=True).filled(float("nan")).astype("float32")
        source_finite = np.isfinite(source)
        if not bool(source_finite.any()):
            raise ValueError("downloaded DEM contains no finite elevation samples")

        progress(20, "Preparing target WRP terrain vertex coordinates")
        # North-up row zero becomes the highest stored runtime row after the
        # build flips the image. Runtime row zero is exactly the south edge.
        target_longitudes = west + np.arange(cells, dtype="float64") * longitude_step
        target_latitudes = north - (np.arange(cells, dtype="float64") + 1.0) * latitude_step
        target_x, target_y = np.meshgrid(target_longitudes, target_latitudes)
        if dataset.crs.to_string().upper() not in {"EPSG:4326", "OGC:CRS84"}:
            progress(28, f"Transforming target coordinates into {dataset.crs}")
            transformed_x, transformed_y = warp_coordinates(
                "EPSG:4326",
                dataset.crs,
                target_x.ravel().tolist(),
                target_y.ravel().tolist(),
            )
            sample_x = np.asarray(transformed_x, dtype="float64").reshape((cells, cells))
            sample_y = np.asarray(transformed_y, dtype="float64").reshape((cells, cells))
        else:
            sample_x = target_x
            sample_y = target_y

        progress(36, "Calculating bilinear DEM sample indices")
        inverse = ~dataset.transform
        # Affine coordinates address pixel corners. Subtract half a pixel so
        # integer positions address sample centres for bilinear interpolation.
        sample_columns = inverse.a * sample_x + inverse.b * sample_y + inverse.c - 0.5
        sample_rows = inverse.d * sample_x + inverse.e * sample_y + inverse.f - 0.5
        outside = (
            (sample_columns < -0.5)
            | (sample_columns > dataset.width - 0.5)
            | (sample_rows < -0.5)
            | (sample_rows > dataset.height - 0.5)
        )

        column0 = np.floor(sample_columns).astype("int64")
        row0 = np.floor(sample_rows).astype("int64")
        column1 = column0 + 1
        row1 = row0 + 1
        fraction_x = sample_columns - column0
        fraction_y = sample_rows - row0
        column0 = np.clip(column0, 0, dataset.width - 1)
        column1 = np.clip(column1, 0, dataset.width - 1)
        row0 = np.clip(row0, 0, dataset.height - 1)
        row1 = np.clip(row1, 0, dataset.height - 1)

        corner_values = (
            source[row0, column0],
            source[row0, column1],
            source[row1, column0],
            source[row1, column1],
        )
        corner_weights = (
            (1.0 - fraction_x) * (1.0 - fraction_y),
            fraction_x * (1.0 - fraction_y),
            (1.0 - fraction_x) * fraction_y,
            fraction_x * fraction_y,
        )
        weighted_sum = np.zeros((cells, cells), dtype="float64")
        weight_sum = np.zeros((cells, cells), dtype="float64")
        progress(48, "Interpolating DEM onto the game terrain grid")
        for corner_index, (corner, weight) in enumerate(zip(corner_values, corner_weights), start=1):
            usable = np.isfinite(corner)
            weighted_sum += np.where(usable, corner * weight, 0.0)
            weight_sum += np.where(usable, weight, 0.0)
            progress(48 + corner_index * 5, f"Interpolated DEM corner set {corner_index}/4")
        destination = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full((cells, cells), np.nan, dtype="float64"),
            where=weight_sum > 0.0,
        ).astype("float32")
        destination[outside] = np.nan

        raw_bounds = dataset.bounds
        raw_resolution = dataset.res
        raw_width = dataset.width
        raw_height = dataset.height
        raw_crs = dataset.crs.to_string()
        raw_finite_fraction = float(source_finite.mean())
        raw_minimum = float(np.nanmin(source))
        raw_maximum = float(np.nanmax(source))
        outside_target_samples = int(outside.sum())

    progress(70, "Checking resampled DEM for missing elevation cells")
    valid = np.isfinite(destination)
    missing_before_fill = int(destination.size - int(valid.sum()))
    if not bool(valid.any()):
        raise ValueError("requested bbox does not overlap finite DEM samples")
    if not bool(valid.all()):
        # Fill small edge holes from nearby terrain. The previous global-median
        # replacement could create conspicuous flat shelves along DEM gaps.
        working = np.where(valid, destination, 0.0).astype("float32")
        destination = fillnodata(
            working,
            mask=valid.astype("uint8"),
            max_search_distance=max(8, cells),
            smoothing_iterations=0,
        ).astype("float32")
        remaining = ~np.isfinite(destination)
        if bool(remaining.any()):
            destination[remaining] = float(np.nanmedian(destination))

    progress(80, "Calculating heightmap statistics")
    values = destination.astype("float64")
    minimum = float(values.min())
    maximum = float(values.max())
    percentiles = np.percentile(values, (1, 5, 50, 95, 99))
    horizontal = np.abs(np.diff(values, axis=1))
    vertical = np.abs(np.diff(values, axis=0))
    mean_neighbour_delta = float(
        (horizontal.sum() + vertical.sum()) / max(1, horizontal.size + vertical.size)
    )

    progress(90, f"Writing {cells}x{cells} georeferenced heightmap TIFF")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.tif")
    profile = {
        "driver": "GTiff",
        "width": cells,
        "height": cells,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": destination_transform,
        "compress": "deflate",
        "predictor": 3,
    }
    with rasterio.open(temporary, "w", **profile) as dataset:
        dataset.write(destination, 1)
        dataset.update_tags(
            AREA_OR_POINT="Point",
            CWR_GRID="game-terrain-vertices",
            CWR_RESAMPLING="georeferenced-bilinear",
        )
    os.replace(temporary, output)
    progress(100, f"Heightmap written: {minimum:.2f}..{maximum:.2f} m")

    metadata: dict[str, Any] = {
        "method": "georeferenced-bilinear-sampling",
        "target_grid": "game-terrain-vertices",
        "target_width": cells,
        "target_height": cells,
        "target_bounds_west_south_east_north": [west, south, east, north],
        "raw_width": raw_width,
        "raw_height": raw_height,
        "raw_crs": raw_crs,
        "raw_bounds_west_south_east_north": [
            float(raw_bounds.left),
            float(raw_bounds.bottom),
            float(raw_bounds.right),
            float(raw_bounds.top),
        ],
        "raw_resolution_degrees": [float(raw_resolution[0]), float(raw_resolution[1])],
        "raw_finite_fraction": raw_finite_fraction,
        "raw_minimum_metres": raw_minimum,
        "raw_maximum_metres": raw_maximum,
        "target_samples_outside_raw_raster": outside_target_samples,
        "missing_target_samples_filled": missing_before_fill,
        "output_minimum_metres": minimum,
        "output_maximum_metres": maximum,
        "output_mean_metres": float(values.mean()),
        "output_standard_deviation_metres": float(values.std()),
        "output_mean_neighbour_delta_metres": mean_neighbour_delta,
        "output_percentiles_metres": {
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p95": float(percentiles[3]),
            "p99": float(percentiles[4]),
        },
    }
    return minimum, maximum, metadata


def _fetch_dem_stitcher_elevation(
    spec: SourceFetchSpec,
    bbox: tuple[float, float, float, float],
    root: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, list[dict[str, str]], float, float, str, dict[str, Any]]:
    def progress(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, stage)

    progress(0, f"Loading DEM provider support for {spec.dem_name}")
    try:
        import rasterio
        from dem_stitcher import stitch_dem
    except ImportError as exc:
        raise RuntimeError(
            "dem-stitcher source support requires the 'sources' extra: "
            "python -m pip install -e '.[sources]'"
        ) from exc
    raw_path = root / "elevation" / "raw" / f"{spec.dem_name}.tif"
    if spec.refresh or not raw_path.is_file():
        south, west, north, east = bbox
        progress(5, f"Requesting and mosaicking {spec.dem_name} DEM coverage")

        def acquire_dem() -> tuple[Any, Any]:
            return stitch_dem(
                [west, south, east, north],
                dem_name=spec.dem_name,
                dst_ellipsoidal_height=False,
                dst_area_or_point="Point",
            )

        elevation, profile = _run_with_progress_heartbeat(
            acquire_dem,
            progress_callback=progress_callback,
            percent=5,
            stage=f"Requesting and mosaicking {spec.dem_name} DEM coverage",
        )
        progress(38, f"DEM provider returned {elevation.shape[1]}x{elevation.shape[0]} samples")
        profile = dict(profile)
        profile["count"] = 1
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_suffix(".tmp.tif")
        progress(42, f"Writing raw {spec.dem_name} DEM cache")
        with rasterio.open(temporary, "w", **profile) as dataset:
            dataset.write(elevation, 1)
            dataset.update_tags(AREA_OR_POINT="Point")
        os.replace(temporary, raw_path)
        source = f"dem-stitcher:{spec.dem_name}"
    else:
        source = "cache"
        progress(45, f"Using cached raw DEM raster: {raw_path.name}")
    heightmap_path = root / "elevation" / "heightmap-meters.tif"

    def resample_progress(percent: int, stage: str) -> None:
        progress(48 + round(48 * percent / 100), stage)

    minimum, maximum, resampling = _heightmap_from_raster(
        raw_path,
        heightmap_path,
        spec.cells,
        bbox,
        progress_callback=resample_progress,
    )
    progress(98, f"Hashing raw DEM raster: {raw_path.name}")
    records = [{"path": _relative(root, raw_path), "source": source, "sha256": _sha256(raw_path)}]
    product = "Copernicus GLO-30" if spec.dem_name == "glo_30" else spec.dem_name
    progress(100, f"DEM and heightmap preparation complete: {product}")
    return heightmap_path, records, minimum, maximum, product, resampling


def _attribution_texts(dem_product: str) -> tuple[str, str]:
    osm = (
        "OpenStreetMap attribution\n\n"
        "Map data © OpenStreetMap contributors.\n"
        "OpenStreetMap data is available under the Open Database License (ODbL) 1.0.\n"
        "https://www.openstreetmap.org/copyright\n"
    )
    if dem_product == "Copernicus GLO-30":
        dem = (
            "Elevation attribution\n\n"
            "Elevation product: Copernicus DEM GLO-30.\n"
            "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
            "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
            "by the European Union and ESA; all rights reserved.\n"
            "The exact fetched raster and its SHA-256 are recorded in source.json.\n"
        )
    else:
        dem = (
            "Elevation attribution\n\n"
            f"Elevation product: {dem_product}.\n"
            "The HGT mirror files are repackaged terrain data from Viewfinder Panoramas; "
            "most tiles originate from the 2000 Shuttle Radar Topography Mission.\n"
            "The exact mirror URL and archive SHA-256 are recorded in source.json.\n"
        )
    return osm, dem


def _write_checksums(root: Path, paths: Iterable[Path]) -> Path:
    entries = sorted((_relative(root, path), _sha256(path)) for path in paths if path.is_file())
    checksum_path = root / "SHA256SUMS.txt"
    text = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    _write_text_atomic(checksum_path, text, encoding="ascii")
    return checksum_path


def fetch_sources(spec: SourceFetchSpec) -> FrozenSourceBundle:
    report_progress(0, "Validating source-fetch settings")
    spec.validate()
    report_progress(1, "Resolving selected geographic area and terrain grid")
    selection_input, bbox, map_link = _selection(spec)
    root = spec.source_dir.resolve()
    manifest_path = root / "source.json"

    if manifest_path.is_file() and spec.refresh:
        report_progress(2, "Preparing atomic refresh of existing frozen source bundle")
        root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.refresh-", dir=root.parent))
        backup = root.with_name(f".{root.name}.backup")
        try:
            fetch_sources(replace(spec, source_dir=stage, refresh=True))
            report_progress(96, "Swapping refreshed source bundle into place")
            # Validation completed inside the staged fetch. Swap only after the new
            # snapshot is complete, preserving the previous bundle on download failure.
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(root, backup)
            try:
                os.replace(stage, root)
            except Exception:
                os.replace(backup, root)
                raise
            shutil.rmtree(backup)
            report_progress(97, "Validating refreshed source bundle after atomic swap")
            result = validate_source_bundle(
                root,
                progress_callback=_mapped_progress(97, 100),
            ).bundle
            report_progress(100, "Frozen source refresh complete")
            return result
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    if manifest_path.is_file() and not spec.refresh:
        report_progress(2, "Existing frozen source bundle found; checking selection")
        document = _manifest_json(manifest_path)
        if not _source_selection_matches(document, bbox, spec):
            raise ValueError("source directory already contains a different selection; use another directory or --refresh")
        report_progress(4, "Selection matches existing bundle; validating local files")
        result = validate_source_bundle(
            root,
            progress_callback=_mapped_progress(4, 100),
        ).bundle
        report_progress(100, "Existing frozen source bundle is ready")
        return result

    report_progress(2, f"Creating frozen source bundle at {root}")
    root.mkdir(parents=True, exist_ok=True)
    osm_dir = root / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    overture_dir = root / "overture"
    overture_dir.mkdir(parents=True, exist_ok=True)
    osm_json_path = osm_dir / "raw-overpass.json"
    query_path = osm_dir / "overpass-query.txt"
    osm_source = _fetch_overpass(
        osm_json_path,
        query_path,
        bbox,
        spec.overpass_urls,
        refresh=spec.refresh,
        timeout=spec.overpass_timeout_seconds,
        progress_callback=_mapped_progress(3, 34),
    )
    report_progress(35, "Loading frozen Overpass JSON and counting elements")
    osm_document = json.loads(osm_json_path.read_text(encoding="utf-8"))
    element_count = len(osm_document["elements"])
    report_progress(37, f"Frozen OpenStreetMap snapshot contains {element_count:,} elements")

    report_progress(38, "Fetching optional Overture building fallback data")
    overture_buildings_path = fetch_overture_buildings_geojson(
        bbox,
        overture_dir / "buildings.geojson",
        refresh=spec.refresh,
    )
    overture_document: dict[str, Any] | None = None
    if overture_buildings_path is not None:
        overture_document = {
            "provider": "Overture Maps",
            "buildings_geojson": _relative(root, overture_buildings_path),
            "buildings_geojson_sha256": _sha256(overture_buildings_path),
        }
        report_progress(38, "Cached optional Overture building fallback data")
    else:
        report_progress(38, "Optional Overture building fallback data unavailable; continuing with OSM only")

    if spec.dem_provider == "dem-stitcher":
        heightmap_path, raw_elevation, minimum, maximum, dem_product, resampling = _fetch_dem_stitcher_elevation(
            spec, bbox, root, progress_callback=_mapped_progress(39, 74)
        )
    else:
        heightmap_path, raw_elevation, minimum, maximum, dem_product, resampling = _fetch_hgt_elevation(
            spec, bbox, root, progress_callback=_mapped_progress(39, 74)
        )

    reference_map_path: Path | None = None
    reference_document: dict[str, Any] | None = None
    if spec.reference_map:
        reference_zoom = (
            int(round(map_link.zoom))
            if map_link is not None
            else reference_zoom_for_bbox(bbox)
        )
        reference_map_path = root / "reference" / "opentopomap.png"
        _reference_map(
            reference_map_path,
            bbox,
            reference_zoom,
            root / "reference" / "tiles",
            refresh=spec.refresh,
            progress_callback=_mapped_progress(75, 87),
        )
        report_progress(88, "Hashing completed reference map")
        reference_document = {
            "provider": "OpenTopoMap",
            "path": _relative(root, reference_map_path),
            "zoom": reference_zoom,
            "sha256": _sha256(reference_map_path),
        }
    else:
        report_progress(87, "Reference-map download disabled")

    report_progress(89, "Writing OpenStreetMap and elevation attribution files")
    osm_attribution_path = root / "attribution" / "OSM-ATTRIBUTION.txt"
    dem_attribution_path = root / "attribution" / "DEM-ATTRIBUTION.txt"
    osm_attribution, dem_attribution = _attribution_texts(dem_product)
    _write_text_atomic(osm_attribution_path, osm_attribution)
    _write_text_atomic(dem_attribution_path, dem_attribution)

    frozen_paths = [
        osm_json_path,
        query_path,
        heightmap_path,
        osm_attribution_path,
        dem_attribution_path,
        *(Path(root / record["path"]) for record in raw_elevation),
    ]
    if reference_map_path is not None:
        frozen_paths.append(reference_map_path)
    if overture_buildings_path is not None:
        frozen_paths.append(overture_buildings_path)

    hash_cache: dict[Path, str] = {}
    for record in raw_elevation:
        record_path = (root / record["path"]).resolve()
        hash_cache[record_path] = record["sha256"]

    def file_hash(path: Path) -> str:
        resolved = path.resolve()
        digest = hash_cache.get(resolved)
        if digest is None:
            digest = _sha256(resolved)
            hash_cache[resolved] = digest
        return digest

    files: dict[str, str] = {}
    sorted_paths = sorted(frozen_paths)
    for file_index, path in enumerate(sorted_paths, start=1):
        report_progress(90 + round(3 * file_index / max(1, len(sorted_paths))), f"Hashing frozen source file {file_index}/{len(sorted_paths)}: {_relative(root, path)}")
        files[_relative(root, path)] = file_hash(path)

    report_progress(94, "Writing frozen-source manifest")
    manifest = {
        "schema": SOURCE_SCHEMA,
        "schema_version": SOURCE_SCHEMA_VERSION,
        "fetched_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection": {
            **selection_input,
            "bbox_south_west_north_east": list(bbox),
            "world_size_metres": spec.world_size,
            "cells": spec.cells,
            "cell_size_metres": spec.cell_size,
            "projection": "local equirectangular; latitude and longitude spans independently mapped to square world",
        },
        "osm": {
            "provider": "OpenStreetMap Overpass API",
            "source": osm_source,
            "raw_json": _relative(root, osm_json_path),
            "query": _relative(root, query_path),
            "element_count": element_count,
            "raw_json_sha256": file_hash(osm_json_path),
            "query_sha256": file_hash(query_path),
        },
        "elevation": {
            "provider": spec.dem_provider,
            "product": dem_product,
            "dem_name": spec.dem_name if spec.dem_provider == "dem-stitcher" else None,
            "raw_files": raw_elevation,
            "heightmap": _relative(root, heightmap_path),
            "heightmap_sha256": file_hash(heightmap_path),
            "minimum_metres": minimum,
            "maximum_metres": maximum,
            "orientation": "north-up; build flips rows for south-to-north WRP storage",
            "resampling": resampling,
        },
        "overture": overture_document,
        "reference_map": reference_document,
        "attribution": [
            _relative(root, osm_attribution_path),
            _relative(root, dem_attribution_path),
        ],
        "files": files,
    }
    _write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_progress(95, "Writing SHA256SUMS for frozen source bundle")
    checksum_paths = [manifest_path, *frozen_paths]
    checksum_entries = sorted((_relative(root, path), file_hash(path)) for path in checksum_paths if path.is_file())
    checksum_path = root / "SHA256SUMS.txt"
    _write_text_atomic(
        checksum_path,
        "".join(f"{digest}  {relative}\n" for relative, digest in checksum_entries),
        encoding="ascii",
    )
    report_progress(96, "Validating completed frozen source bundle")
    result = validate_source_bundle(
        root,
        progress_callback=_mapped_progress(96, 100),
    ).bundle
    report_progress(100, "OSM and heightmap source fetch complete")
    return result


def _copy_frozen_file(source_root: Path, target_root: Path, relative: str) -> Path:
    source = _safe_path(source_root, relative)
    if not source.is_file():
        raise ValueError(f"frozen source file is missing: {relative}")
    target = _safe_path(target_root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _regrid_hgt_elevation(
    manifest: Mapping[str, Any],
    source_root: Path,
    target_root: Path,
    bbox: tuple[float, float, float, float],
    cells: int,
) -> tuple[Path, list[dict[str, str]], float, float, dict[str, Any]]:
    elevation = manifest.get("elevation")
    if not isinstance(elevation, Mapping):
        raise ValueError("source manifest elevation section is missing")
    raw_entries = elevation.get("raw_files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("source bundle contains no reusable raw HGT files")
    copied_records: list[dict[str, str]] = []
    tiles: dict[str, HgtTile] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("source manifest contains an invalid raw HGT record")
        relative = str(entry["path"])
        copied = _copy_frozen_file(source_root, target_root, relative)
        name = copied.name
        if not name.casefold().endswith(".hgt.zip"):
            raise ValueError(f"raw HGT record does not reference a .hgt.zip file: {relative}")
        tile_name = name[:-8]
        tiles[tile_name] = HgtTile.from_zip(copied, tile_name)
        copied_records.append({
            "path": _relative(target_root, copied),
            "source": "reused-local-frozen-bundle",
            "sha256": _sha256(copied),
        })
    required = set(required_hgt_tiles(bbox))
    missing = sorted(required.difference(tiles))
    if missing:
        raise ValueError(f"source bundle is missing HGT tiles needed for its own bbox: {missing}")
    heightmap_path = target_root / "elevation" / "heightmap-meters.tif"
    minimum, maximum = _write_heightmap(heightmap_path, bbox, cells, tiles)
    resampling: dict[str, Any] = {
        "method": "direct-hgt-bilinear",
        "target_grid": "game-terrain-vertices",
        "target_width": cells,
        "target_height": cells,
        "target_bounds_west_south_east_north": [bbox[1], bbox[0], bbox[3], bbox[2]],
        "output_minimum_metres": minimum,
        "output_maximum_metres": maximum,
        "reused_local_raw_dem": True,
    }
    return heightmap_path, copied_records, minimum, maximum, resampling


def _regrid_raster_elevation(
    manifest: Mapping[str, Any],
    source_root: Path,
    target_root: Path,
    bbox: tuple[float, float, float, float],
    cells: int,
) -> tuple[Path, list[dict[str, str]], float, float, dict[str, Any]]:
    elevation = manifest.get("elevation")
    if not isinstance(elevation, Mapping):
        raise ValueError("source manifest elevation section is missing")
    raw_entries = elevation.get("raw_files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("source bundle contains no reusable raw DEM raster")
    copied_records: list[dict[str, str]] = []
    raster_path: Path | None = None
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("source manifest contains an invalid raw DEM record")
        relative = str(entry["path"])
        copied = _copy_frozen_file(source_root, target_root, relative)
        copied_records.append({
            "path": _relative(target_root, copied),
            "source": "reused-local-frozen-bundle",
            "sha256": _sha256(copied),
        })
        if raster_path is None and copied.suffix.casefold() in {".tif", ".tiff"}:
            raster_path = copied
    if raster_path is None:
        raise ValueError("source bundle has no reusable georeferenced raw DEM TIFF")
    heightmap_path = target_root / "elevation" / "heightmap-meters.tif"
    minimum, maximum, resampling = _heightmap_from_raster(raster_path, heightmap_path, cells, bbox)
    resampling = dict(resampling)
    resampling["reused_local_raw_dem"] = True
    return heightmap_path, copied_records, minimum, maximum, resampling


def regrid_sources(spec: SourceRegridSpec) -> FrozenSourceBundle:
    """Create a new frozen bundle at another grid resolution without networking."""

    source_root = spec.source_dir.resolve()
    target_root = spec.output_source_dir.resolve()
    if source_root == target_root:
        raise ValueError("output source directory must differ from the input source directory")

    report_progress(0, "Validating frozen source bundle for local regridding")
    source_report = validate_source_bundle(
        source_root,
        progress_callback=lambda percent, stage: report_progress(round(percent * 0.15), stage),
    )
    source = source_report.bundle
    cells, cell_size = spec.resolved_grid(source)
    world_size = cells * cell_size
    report_progress(16, f"Regridding {source.cells}x{source.cells} @ {source.cell_size:g}m to {cells}x{cells} @ {cell_size:g}m")

    if target_root.exists() and not spec.replace_output:
        raise ValueError(f"output source directory already exists: {target_root}; use --replace-output to replace it")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target_root.name}.regrid-", dir=target_root.parent))
    backup = target_root.with_name(f".{target_root.name}.backup")
    try:
        source_manifest = _manifest_json(source.manifest_path)
        osm = source_manifest.get("osm")
        elevation = source_manifest.get("elevation")
        selection = source_manifest.get("selection")
        if not isinstance(osm, Mapping) or not isinstance(elevation, Mapping) or not isinstance(selection, Mapping):
            raise ValueError("source manifest is missing selection, osm, or elevation sections")

        report_progress(22, "Copying frozen OpenStreetMap data and query")
        osm_json_path = _copy_frozen_file(source_root, stage, str(osm.get("raw_json", "")))
        query_path = _copy_frozen_file(source_root, stage, str(osm.get("query", "")))
        attribution_paths: list[Path] = []
        raw_attribution = source_manifest.get("attribution")
        if not isinstance(raw_attribution, list):
            raise ValueError("source manifest attribution list is missing")
        for relative in raw_attribution:
            if not isinstance(relative, str):
                raise ValueError("source manifest contains an invalid attribution path")
            attribution_paths.append(_copy_frozen_file(source_root, stage, relative))

        report_progress(35, "Copying reusable raw DEM coverage")
        provider = str(elevation.get("provider", ""))
        if provider == "hgt":
            report_progress(46, f"Resampling local HGT tiles onto {cells}x{cells} terrain grid")
            heightmap_path, raw_elevation, minimum, maximum, resampling = _regrid_hgt_elevation(
                source_manifest, source_root, stage, source.bbox, cells
            )
        elif provider == "dem-stitcher":
            report_progress(46, f"Resampling local georeferenced DEM onto {cells}x{cells} terrain grid")
            heightmap_path, raw_elevation, minimum, maximum, resampling = _regrid_raster_elevation(
                source_manifest, source_root, stage, source.bbox, cells
            )
        else:
            raise ValueError(f"unsupported frozen DEM provider for local regridding: {provider!r}")

        report_progress(78, "Copying optional reference map")
        reference_document: dict[str, Any] | None = None
        source_reference = source_manifest.get("reference_map")
        reference_map_path: Path | None = None
        if isinstance(source_reference, Mapping) and isinstance(source_reference.get("path"), str):
            reference_map_path = _copy_frozen_file(source_root, stage, str(source_reference["path"]))
            reference_document = dict(source_reference)
            reference_document["path"] = _relative(stage, reference_map_path)
            reference_document["sha256"] = _sha256(reference_map_path)

        report_progress(82, "Copying optional Overture building fallback data")
        overture_document: dict[str, Any] | None = None
        overture_buildings_path: Path | None = None
        source_overture = source_manifest.get("overture")
        if isinstance(source_overture, Mapping) and isinstance(source_overture.get("buildings_geojson"), str):
            overture_buildings_path = _copy_frozen_file(source_root, stage, str(source_overture["buildings_geojson"]))
            overture_document = dict(source_overture)
            overture_document["buildings_geojson"] = _relative(stage, overture_buildings_path)
            overture_document["buildings_geojson_sha256"] = _sha256(overture_buildings_path)

        report_progress(84, "Writing regridded frozen-source manifest")
        osm_document = json.loads(osm_json_path.read_text(encoding="utf-8"))
        if not isinstance(osm_document, Mapping) or not isinstance(osm_document.get("elements"), list):
            raise ValueError("frozen OpenStreetMap JSON is malformed")
        selection_document = dict(selection)
        selection_document.update({
            "bbox_south_west_north_east": list(source.bbox),
            "world_size_metres": world_size,
            "cells": cells,
            "cell_size_metres": cell_size,
        })
        frozen_paths = [
            osm_json_path,
            query_path,
            heightmap_path,
            *attribution_paths,
            *(stage / str(record["path"]) for record in raw_elevation),
        ]
        if reference_map_path is not None:
            frozen_paths.append(reference_map_path)
        if overture_buildings_path is not None:
            frozen_paths.append(overture_buildings_path)
        files = {_relative(stage, path): _sha256(path) for path in sorted(frozen_paths)}
        manifest_path = stage / "source.json"
        manifest = {
            "schema": SOURCE_SCHEMA,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "fetched_at_utc": source_manifest.get("fetched_at_utc"),
            "regridded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "regridded_from": {
                "manifest_sha256": source.fingerprint,
                "cells": source.cells,
                "cell_size_metres": source.cell_size,
                "world_size_metres": source.cells * source.cell_size,
            },
            "selection": selection_document,
            "osm": {
                **dict(osm),
                "source": "reused-local-frozen-bundle",
                "raw_json": _relative(stage, osm_json_path),
                "query": _relative(stage, query_path),
                "element_count": len(osm_document["elements"]),
                "raw_json_sha256": _sha256(osm_json_path),
                "query_sha256": _sha256(query_path),
            },
            "elevation": {
                "provider": provider,
                "product": elevation.get("product"),
                "dem_name": elevation.get("dem_name"),
                "raw_files": raw_elevation,
                "heightmap": _relative(stage, heightmap_path),
                "heightmap_sha256": _sha256(heightmap_path),
                "minimum_metres": minimum,
                "maximum_metres": maximum,
                "orientation": "north-up; build flips rows for south-to-north WRP storage",
                "resampling": resampling,
            },
            "overture": overture_document,
            "reference_map": reference_document,
            "attribution": [_relative(stage, path) for path in attribution_paths],
            "files": files,
        }
        _write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report_progress(90, "Writing SHA256SUMS for regridded source bundle")
        _write_checksums(stage, [manifest_path, *frozen_paths])
        report_progress(94, "Validating regridded frozen source bundle")
        validate_source_bundle(
            stage,
            progress_callback=lambda percent, stage_name: report_progress(94 + round(percent * 0.05), stage_name),
        )

        if backup.exists():
            shutil.rmtree(backup)
        if target_root.exists():
            os.replace(target_root, backup)
        try:
            os.replace(stage, target_root)
        except Exception:
            if backup.exists():
                os.replace(backup, target_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        report_progress(100, "Local source regridding complete")
        return load_source_bundle(target_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class Milestone5Spec:
    source_dir: Path
    name: str = "cwr_milestone5"
    display_name: str = "CWR Milestone 5"
    profile: str = "cwr-ce"
    include_minor_roads: bool = False
    forest_road_clearance: float = 0.0
    building_ground_clearance: float = 0.10
    forest_ground_clearance: float = 0.15
    point_building_footprint: float = 12.0
    water_depth: float = 5.0
    coastline_blend_cells: int = 2
    road_segment_length: float = 24.5
    max_road_objects: int = DEFAULT_MAX_ROAD_OBJECTS
    max_buildings: int = DEFAULT_MAX_BUILDINGS
    building_minimum_area: float = 20.0
    forest_tree_spacing: float = 50.0
    max_forest_objects: int = DEFAULT_MAX_FOREST_OBJECTS
    road_connection_tolerance: float = 5.0
    maximum_road_grade_percent: float = 12.0
    road_grade_radius: float = 100.0
    building_grade_radius: float = 25.0
    maximum_grade_adjustment: float = 12.0
    transition_cells: int = 2
    asset_roots: tuple[Path, ...] = ()
    strict_assets: bool = False
    osm_asset_mapping_path: Path | None = None
    cache_dir: Path | None = None
    cache_enabled: bool = True
    cache_refresh: bool = False
    town_name_limit: int = 64
    verify_regeneration: bool = False


def _copy_provenance(bundle: FrozenSourceBundle, result: BuildResult) -> tuple[Path, Path]:
    provenance_path = result.output_dir / "source.json"
    source_validation_path = result.output_dir / "source-validation-report.txt"
    shutil.copyfile(bundle.manifest_path, provenance_path)
    source_report = bundle.root / "source-validation-report.txt"
    if source_report.is_file():
        shutil.copyfile(source_report, source_validation_path)
    else:
        validate_source_bundle(bundle.root)
        shutil.copyfile(source_report, source_validation_path)

    runtime_root = result.pbo_path.parent.parent
    shutil.copyfile(bundle.manifest_path, runtime_root / "SOURCE-PROVENANCE.json")
    shutil.copyfile(bundle.osm_attribution_path, runtime_root / "OSM-ATTRIBUTION.txt")
    shutil.copyfile(bundle.dem_attribution_path, runtime_root / "DEM-ATTRIBUTION.txt")
    shutil.copyfile(bundle.osm_attribution_path, result.output_dir / "OSM-ATTRIBUTION.txt")
    shutil.copyfile(bundle.dem_attribution_path, result.output_dir / "DEM-ATTRIBUTION.txt")
    return provenance_path, source_validation_path


def build_milestone5(output_dir: Path, spec: Milestone5Spec, *, clean: bool = True) -> BuildResult:
    validation = validate_source_bundle(spec.source_dir)
    bundle = validation.bundle
    playability = PlayabilitySpec(
        heightmap_path=bundle.heightmap_path,
        name=spec.name,
        display_name=spec.display_name,
        profile=spec.profile,
        cells=bundle.cells,
        cell_size=bundle.cell_size,
        heightmap_grid=bundle.heightmap_grid,
        input_mode="meters",
        flip_y=True,
        sea_level=0.0,
        beach_height=3.0,
        rock_height=110.0,
        rock_slope_degrees=30.0,
        bbox=bundle.bbox,
        osm_json_path=bundle.osm_json_path,
        water_depth=spec.water_depth,
        coastline_blend_cells=spec.coastline_blend_cells,
        road_segment_length=spec.road_segment_length,
        max_road_objects=spec.max_road_objects,
        max_buildings=spec.max_buildings,
        building_minimum_area=spec.building_minimum_area,
        forest_tree_spacing=spec.forest_tree_spacing,
        forest_road_clearance=spec.forest_road_clearance,
        building_ground_clearance=spec.building_ground_clearance,
        forest_ground_clearance=spec.forest_ground_clearance,
        point_building_footprint=spec.point_building_footprint,
        max_forest_objects=spec.max_forest_objects,
        include_minor_roads=spec.include_minor_roads,
        road_connection_tolerance=spec.road_connection_tolerance,
        maximum_road_grade_percent=spec.maximum_road_grade_percent,
        road_grade_radius=spec.road_grade_radius,
        building_grade_radius=spec.building_grade_radius,
        maximum_grade_adjustment=spec.maximum_grade_adjustment,
        transition_cells=spec.transition_cells,
        asset_roots=spec.asset_roots,
        strict_assets=spec.strict_assets,
        osm_asset_mapping_path=spec.osm_asset_mapping_path,
        town_name_limit=spec.town_name_limit,
        deterministic_seed=f"milestone5:{bundle.fingerprint}",
        verify_regeneration=spec.verify_regeneration,
    )
    result = build_milestone4(
        output_dir,
        playability,
        clean=clean,
        mod_directory_name="@CWR-Milestone5",
        milestone_number=5,
    )
    provenance_path, source_validation_path = _copy_provenance(bundle, result)

    try:
        build_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build_manifest = {}
    build_manifest["schema"] = 5
    build_manifest["milestone"] = 5
    build_manifest["source_bundle"] = {
        "manifest_sha256": bundle.fingerprint,
        "bbox_south_west_north_east": list(bundle.bbox),
        "cells": bundle.cells,
        "cell_size_metres": bundle.cell_size,
        "heightmap_grid": bundle.heightmap_grid,
        "manifest": provenance_path.name,
        "validation": source_validation_path.name,
    }
    result.manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with result.report_path.open("a", encoding="utf-8", newline="\n") as report:
        report.write("\nMilestone 5 frozen-source checks\n\n")
        report.write(f"[PASS] Source bundle validates: {bundle.fingerprint}\n")
        report.write(f"[PASS] Build uses frozen OSM JSON: {bundle.osm_json_path.name}\n")
        report.write(f"[PASS] Build uses frozen metre heightmap: {bundle.heightmap_path.name}\n")
        report.write("[PASS] Runtime mod carries OSM and DEM attribution\n")

    return BuildResult(
        output_dir=result.output_dir,
        source_dir=result.source_dir,
        wrp_path=result.wrp_path,
        texture_paths=result.texture_paths,
        pbo_path=result.pbo_path,
        mission_path=result.mission_path,
        intro_mission_path=result.intro_mission_path,
        intro_script_path=result.intro_script_path,
        preview_path=result.preview_path,
        height_preview_path=result.height_preview_path,
        material_preview_path=result.material_preview_path,
        manifest_path=result.manifest_path,
        report_path=result.report_path,
        osm_preview_path=result.osm_preview_path,
        osm_source_path=result.osm_source_path,
        osm_query_path=result.osm_query_path,
        attribution_path=result.attribution_path,
        asset_catalogue_path=result.asset_catalogue_path,
        road_report_path=result.road_report_path,
        grading_report_path=result.grading_report_path,
        reproducibility_path=result.reproducibility_path,
        source_manifest_path=provenance_path,
        source_validation_path=source_validation_path,
    )
