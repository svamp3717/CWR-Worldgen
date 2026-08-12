# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import math
import re
import time
from typing import Callable, Iterable, Mapping, Sequence
from urllib import error, parse, request
from zipfile import ZipFile

from PIL import Image

from ._version import __version__
from .generator import BuildResult, build_milestone4
from .model import PlayabilitySpec
from .osm import EARTH_RADIUS_METRES, build_overpass_query
from .overpass_endpoints import GLOBAL_OVERPASS_URLS, overpass_urls_for_bbox

_MAP_LINK = re.compile(
    r"^https?://(?:www\.)?opentopomap\.org/(?:[^#]*)#map=(?P<zoom>\d+(?:\.\d+)?)/"
    r"(?P<lat>-?\d+(?:\.\d+)?)/(?P<lon>-?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_VOID = -32768


ProgressCallback = Callable[[int, str], None]


def _emit_progress(callback: ProgressCallback | None, percent: int, stage: str) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), stage)


def _read_response_payload(
    response,
    *,
    progress_callback: ProgressCallback | None,
    start_percent: int,
    end_percent: int,
    label: str,
) -> bytes:
    """Read a network response in chunks and report byte-accurate progress when possible."""

    headers = getattr(response, "headers", None)
    content_length_text = headers.get("Content-Length") if headers is not None else None
    try:
        content_length = int(content_length_text) if content_length_text else 0
    except (TypeError, ValueError):
        content_length = 0
    chunks: list[bytes] = []
    downloaded = 0
    last_reported_percent = -1
    last_reported_megabyte = -1
    while True:
        try:
            block = response.read(1024 * 1024)
        except TypeError:
            block = response.read()
        if not block:
            break
        chunks.append(block)
        downloaded += len(block)
        if content_length > 0:
            fraction = min(1.0, downloaded / content_length)
            percent = start_percent + round((end_percent - start_percent) * fraction)
            if percent != last_reported_percent:
                _emit_progress(
                    progress_callback,
                    percent,
                    f"{label}: {downloaded / (1024 * 1024):.1f}/{content_length / (1024 * 1024):.1f} MiB",
                )
                last_reported_percent = percent
        else:
            megabyte = downloaded // (1024 * 1024)
            if megabyte != last_reported_megabyte:
                _emit_progress(
                    progress_callback,
                    start_percent,
                    f"{label}: {downloaded / (1024 * 1024):.1f} MiB received",
                )
                last_reported_megabyte = megabyte
    _emit_progress(
        progress_callback,
        end_percent,
        f"{label}: {downloaded / (1024 * 1024):.1f} MiB complete",
    )
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class MapLink:
    url: str
    zoom: float
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class LocationExampleSpec:
    map_url: str
    output_dir: Path
    cache_dir: Path
    name: str = "cwr_malaren"
    display_name: str = "Malaren Topographic Example"
    cells: int = 256
    cell_size: float = 25.0
    profile: str = "cwr-ce"
    include_minor_roads: bool = False
    forest_road_clearance: float = 0.0
    asset_roots: tuple[Path, ...] = ()
    strict_assets: bool = False
    refresh: bool = False
    reference_map: bool = True
    overpass_urls: tuple[str, ...] = GLOBAL_OVERPASS_URLS
    hgt_url_templates: tuple[str, ...] = (
        "https://download.mapsforge.org/maps/dem/dem3/{latitude_band}/{tile}.hgt.zip",
        "https://bailu.ch/dem3/{latitude_band}/{tile}.hgt.zip",
    )

    @property
    def world_size(self) -> float:
        return self.cells * self.cell_size


@dataclass(frozen=True, slots=True)
class LocationInputs:
    link: MapLink
    bbox: tuple[float, float, float, float]
    heightmap_path: Path
    osm_json_path: Path
    overpass_query_path: Path
    source_manifest_path: Path
    reference_map_path: Path | None


def parse_opentopomap_link(url: str) -> MapLink:
    match = _MAP_LINK.fullmatch(url.strip())
    if match is None:
        raise ValueError("expected an OpenTopoMap URL ending in #map=ZOOM/LAT/LON")
    zoom = float(match.group("zoom"))
    latitude = float(match.group("lat"))
    longitude = float(match.group("lon"))
    if not (0 <= zoom <= 24):
        raise ValueError("map zoom must be within 0..24")
    if not (-85.05112878 < latitude < 85.05112878):
        raise ValueError("map latitude is outside Web Mercator coverage")
    if not (-180 <= longitude <= 180):
        raise ValueError("map longitude is outside -180..180")
    return MapLink(url=url, zoom=zoom, latitude=latitude, longitude=longitude)


def square_bbox(latitude: float, longitude: float, size_metres: float) -> tuple[float, float, float, float]:
    if not math.isfinite(size_metres) or size_metres <= 0:
        raise ValueError("world size must be positive and finite")
    half = size_metres / 2.0
    latitude_delta = math.degrees(half / EARTH_RADIUS_METRES)
    longitude_delta = math.degrees(
        half / (EARTH_RADIUS_METRES * math.cos(math.radians(latitude)))
    )
    south = latitude - latitude_delta
    north = latitude + latitude_delta
    west = longitude - longitude_delta
    east = longitude + longitude_delta
    if south < -90 or north > 90 or west < -180 or east > 180:
        raise ValueError("square selection crosses an unsupported pole or antimeridian")
    return south, west, north, east


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    urls: Iterable[str],
    path: Path,
    *,
    refresh: bool,
    timeout: int = 120,
    progress_callback: ProgressCallback | None = None,
) -> str:
    candidates = tuple(urls)
    if path.is_file() and not refresh:
        _emit_progress(progress_callback, 100, f"Using cached download: {path.name}")
        return "cache"
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    total = max(1, len(candidates))
    for attempt, url in enumerate(candidates, start=1):
        attempt_start = round((attempt - 1) * 100 / total)
        attempt_end = round(attempt * 100 / total)
        host = parse.urlparse(url).netloc or url
        try:
            _emit_progress(progress_callback, attempt_start, f"Connecting to download mirror {attempt}/{total}: {host}")
            req = request.Request(
                url,
                headers={
                    "User-Agent": f"cwr-worldgen-location-example/{__version__} (+https://github.com/ofpisnotdead-com/CWR-CE)",
                    "Accept": "*/*",
                },
            )
            with request.urlopen(req, timeout=timeout) as response:
                payload = _read_response_payload(
                    response,
                    progress_callback=progress_callback,
                    start_percent=min(attempt_end, attempt_start + max(1, round((attempt_end - attempt_start) * 0.15))),
                    end_percent=max(attempt_start, attempt_end - max(1, round((attempt_end - attempt_start) * 0.15))),
                    label=f"Downloading {path.name}",
                )
            if not payload:
                raise RuntimeError("empty response")
            _emit_progress(progress_callback, max(attempt_start, attempt_end - 1), f"Saving {path.name}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
            _emit_progress(progress_callback, attempt_end, f"Downloaded {path.name} from {host}")
            return url
        except Exception as exc:  # noqa: BLE001 - report each public mirror failure together.
            errors.append(f"{url}: {exc}")
            _emit_progress(progress_callback, attempt_end, f"Download mirror {attempt}/{total} failed: {host}")
            if attempt < total:
                time.sleep(1.0)
    raise RuntimeError("all download mirrors failed:\n" + "\n".join(errors))


def _tile_name(latitude: float, longitude: float) -> str:
    lat_floor = math.floor(latitude)
    lon_floor = math.floor(longitude)
    return f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}"


def required_hgt_tiles(bbox: tuple[float, float, float, float]) -> tuple[str, ...]:
    south, west, north, east = bbox
    north_probe = math.nextafter(north, south)
    east_probe = math.nextafter(east, west)
    names = {
        _tile_name(latitude, longitude)
        for latitude in range(math.floor(south), math.floor(north_probe) + 1)
        for longitude in range(math.floor(west), math.floor(east_probe) + 1)
    }
    return tuple(sorted(names))


def _tile_origin(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", name)
    if match is None:
        raise ValueError(f"invalid HGT tile name: {name}")
    latitude = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    longitude = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    return latitude, longitude


@dataclass(frozen=True, slots=True)
class HgtTile:
    name: str
    side: int
    samples: tuple[int, ...]
    latitude_origin: int
    longitude_origin: int

    @classmethod
    def from_zip(cls, path: Path, expected_name: str) -> "HgtTile":
        with ZipFile(path) as archive:
            candidates = [entry for entry in archive.namelist() if entry.casefold().endswith(".hgt")]
            if len(candidates) != 1:
                raise ValueError(f"expected one HGT file in {path.name}, found {len(candidates)}")
            payload = archive.read(candidates[0])
        if len(payload) % 2:
            raise ValueError(f"HGT payload has an odd byte count: {path}")
        sample_count = len(payload) // 2
        side = math.isqrt(sample_count)
        if side * side != sample_count or side < 2:
            raise ValueError(f"HGT payload is not a square sample grid: {path}")
        samples = tuple(
            int.from_bytes(payload[offset : offset + 2], "big", signed=True)
            for offset in range(0, len(payload), 2)
        )
        latitude_origin, longitude_origin = _tile_origin(expected_name)
        return cls(expected_name, side, samples, latitude_origin, longitude_origin)

    def _value(self, row: int, column: int) -> int:
        row = max(0, min(self.side - 1, row))
        column = max(0, min(self.side - 1, column))
        return self.samples[row * self.side + column]

    def sample(self, latitude: float, longitude: float) -> float:
        # HGT rows start at the north edge and columns at the west edge.
        x = (longitude - self.longitude_origin) * (self.side - 1)
        y = (self.latitude_origin + 1.0 - latitude) * (self.side - 1)
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = min(self.side - 1, x0 + 1)
        y1 = min(self.side - 1, y0 + 1)
        fx = x - x0
        fy = y - y0
        corners = (
            (self._value(y0, x0), (1.0 - fx) * (1.0 - fy)),
            (self._value(y0, x1), fx * (1.0 - fy)),
            (self._value(y1, x0), (1.0 - fx) * fy),
            (self._value(y1, x1), fx * fy),
        )
        usable = [(value, weight) for value, weight in corners if value != _VOID and weight > 0]
        if not usable:
            raise ValueError(f"DEM void at {latitude:.7f},{longitude:.7f}")
        total = sum(weight for _, weight in usable)
        return sum(value * weight for value, weight in usable) / total


def _write_heightmap(
    path: Path,
    bbox: tuple[float, float, float, float],
    cells: int,
    tiles: Mapping[str, HgtTile],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, float]:
    south, west, north, east = bbox
    values: list[float] = []
    _emit_progress(progress_callback, 0, f"Preparing {cells}x{cells} HGT heightmap grid")
    report_every = max(1, cells // 50)
    # Conventional north-up image. The worldgen call sets flip_y=True because WRP z rows run south to north.
    for row in range(cells):
        # Store the west/south-inclusive WRP terrain vertices. The east and
        # north outer edges are one implicit interval beyond the final sample.
        latitude = north - (north - south) * (row + 1.0) / cells
        for column in range(cells):
            longitude = west + (east - west) * column / cells
            tile = tiles[_tile_name(latitude, longitude)]
            values.append(tile.sample(latitude, longitude))
        if row == cells - 1 or (row + 1) % report_every == 0:
            _emit_progress(
                progress_callback,
                5 + round(82 * (row + 1) / cells),
                f"Sampling HGT heightmap rows {row + 1}/{cells}",
            )
    _emit_progress(progress_callback, 90, "Encoding floating-point heightmap TIFF")
    image = Image.new("F", (cells, cells))
    image.putdata(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="TIFF", compression="tiff_deflate")
    minimum, maximum = min(values), max(values)
    _emit_progress(progress_callback, 100, f"Heightmap written: {minimum:.2f}..{maximum:.2f} m")
    return minimum, maximum


def _fetch_overpass(
    path: Path,
    query_path: Path,
    bbox: tuple[float, float, float, float],
    urls: Sequence[str],
    *,
    refresh: bool,
    timeout: int = 180,
    progress_callback: ProgressCallback | None = None,
) -> str:
    _emit_progress(progress_callback, 0, "Building Overpass query")
    query_text = build_overpass_query(bbox, timeout_seconds=min(600, timeout))
    query_path.write_text(query_text, encoding="ascii", newline="\n")
    if path.is_file() and not refresh:
        _emit_progress(progress_callback, 100, f"Using cached Overpass JSON: {path.name}")
        return "cache"
    errors: list[str] = []
    payload = parse.urlencode({"data": query_text}).encode("ascii")
    candidates = overpass_urls_for_bbox(urls, bbox)
    total = max(1, len(candidates))
    _emit_progress(progress_callback, 3, f"Selected {len(candidates)} Overpass endpoint candidates")
    for attempt, url in enumerate(candidates, start=1):
        attempt_start = 5 + round((attempt - 1) * 88 / total)
        attempt_end = 5 + round(attempt * 88 / total)
        host = parse.urlparse(url).netloc or url
        try:
            _emit_progress(progress_callback, attempt_start, f"Submitting Overpass request {attempt}/{total}: {host}")
            req = request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded; charset=ascii",
                    "User-Agent": f"cwr-worldgen-location-example/{__version__} (+https://github.com/ofpisnotdead-com/CWR-CE)",
                },
            )
            with request.urlopen(req, timeout=timeout + 30) as response:
                data = _read_response_payload(
                    response,
                    progress_callback=progress_callback,
                    start_percent=min(attempt_end, attempt_start + 2),
                    end_percent=max(attempt_start + 2, attempt_end - 4),
                    label=f"Receiving Overpass JSON from {host}",
                )
            if not data:
                raise ValueError("empty response")
            _emit_progress(progress_callback, max(attempt_start, attempt_end - 3), "Parsing downloaded Overpass JSON")
            document = json.loads(data)
            if not isinstance(document, dict) or not isinstance(document.get("elements"), list):
                raise ValueError("response is not Overpass JSON with an elements array")
            _emit_progress(
                progress_callback,
                max(attempt_start, attempt_end - 2),
                f"Overpass returned {len(document['elements']):,} elements",
            )
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
            _emit_progress(progress_callback, 100, f"Saved Overpass snapshot from {host}")
            return url
        except error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
            detail = f"HTTP {exc.code}"
            if retry_after:
                detail += f" (Retry-After {retry_after})"
            errors.append(f"{url}: {detail}")
            _emit_progress(progress_callback, attempt_end, f"Overpass endpoint {attempt}/{total} failed: {detail}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
            _emit_progress(progress_callback, attempt_end, f"Overpass endpoint {attempt}/{total} failed: {exc}")
        if attempt < total:
            time.sleep(0.5)
    raise RuntimeError("all Overpass endpoints failed:\n" + "\n".join(errors))


def _global_pixel(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    scale = 256.0 * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(latitude))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def reference_zoom_for_bbox(
    bbox: tuple[float, float, float, float],
    *,
    target_pixels: int = 1024,
    minimum_zoom: int = 2,
    maximum_zoom: int = 16,
) -> int:
    """Choose a reference-map zoom that fits the bbox without fetching a tile continent."""
    if target_pixels < 128:
        raise ValueError("reference-map target size must be at least 128 pixels")
    south, west, north, east = bbox
    if not (-85.05112878 < south < north < 85.05112878 and -180 <= west < east <= 180):
        raise ValueError("reference-map bbox is outside Web Mercator coverage")
    for zoom in range(maximum_zoom, minimum_zoom - 1, -1):
        west_px, north_px = _global_pixel(north, west, zoom)
        east_px, south_px = _global_pixel(south, east, zoom)
        if east_px - west_px <= target_pixels and south_px - north_px <= target_pixels:
            return zoom
    return minimum_zoom


def _reference_map(
    path: Path,
    bbox: tuple[float, float, float, float],
    zoom: int,
    cache_dir: Path,
    *,
    refresh: bool,
    progress_callback: ProgressCallback | None = None,
) -> None:
    south, west, north, east = bbox
    west_px, north_px = _global_pixel(north, west, zoom)
    east_px, south_px = _global_pixel(south, east, zoom)
    min_tile_x = math.floor(west_px / 256)
    max_tile_x = math.floor(math.nextafter(east_px, west_px) / 256)
    min_tile_y = math.floor(north_px / 256)
    max_tile_y = math.floor(math.nextafter(south_px, north_px) / 256)
    tile_total = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1)
    _emit_progress(progress_callback, 0, f"Preparing OpenTopoMap mosaic from {tile_total} tiles at zoom {zoom}")
    mosaic = Image.new("RGB", ((max_tile_x - min_tile_x + 1) * 256, (max_tile_y - min_tile_y + 1) * 256))
    tile_index = 0
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile_index += 1
            tile_path = cache_dir / "opentopomap" / str(zoom) / str(tile_x) / f"{tile_y}.png"
            tile_start = 3 + round((tile_index - 1) * 82 / max(1, tile_total))
            tile_end = 3 + round(tile_index * 82 / max(1, tile_total))

            def tile_progress(percent: int, stage: str, *, _start: int = tile_start, _end: int = tile_end) -> None:
                mapped = _start + round((_end - _start) * percent / 100)
                _emit_progress(progress_callback, mapped, f"Reference tile {tile_index}/{tile_total}: {stage}")

            _download(
                [f"https://a.tile.opentopomap.org/{zoom}/{tile_x}/{tile_y}.png"],
                tile_path,
                refresh=refresh,
                progress_callback=tile_progress,
            )
            with Image.open(tile_path) as tile:
                mosaic.paste(tile.convert("RGB"), ((tile_x - min_tile_x) * 256, (tile_y - min_tile_y) * 256))
            _emit_progress(progress_callback, tile_end, f"Assembled reference tile {tile_index}/{tile_total}")
            time.sleep(0.1)
    _emit_progress(progress_callback, 90, "Cropping reference map to selected bounding box")
    left = int(round(west_px - min_tile_x * 256))
    top = int(round(north_px - min_tile_y * 256))
    right = int(round(east_px - min_tile_x * 256))
    bottom = int(round(south_px - min_tile_y * 256))
    cropped = mosaic.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    _emit_progress(progress_callback, 96, f"Writing reference map {cropped.width}x{cropped.height}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(path)
    _emit_progress(progress_callback, 100, "Reference map complete")


def prepare_location_inputs(spec: LocationExampleSpec) -> LocationInputs:
    link = parse_opentopomap_link(spec.map_url)
    bbox = square_bbox(link.latitude, link.longitude, spec.world_size)
    raw_dir = spec.cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tiles: dict[str, HgtTile] = {}
    tile_sources: dict[str, str] = {}
    for tile_name in required_hgt_tiles(bbox):
        latitude_origin, _ = _tile_origin(tile_name)
        latitude_band = f"{'N' if latitude_origin >= 0 else 'S'}{abs(latitude_origin):02d}"
        zip_path = raw_dir / f"{tile_name}.hgt.zip"
        urls = [template.format(latitude_band=latitude_band, tile=tile_name) for template in spec.hgt_url_templates]
        tile_sources[tile_name] = _download(urls, zip_path, refresh=spec.refresh)
        tiles[tile_name] = HgtTile.from_zip(zip_path, tile_name)

    heightmap_path = spec.cache_dir / "heightmap-meters.tif"
    minimum, maximum = _write_heightmap(heightmap_path, bbox, spec.cells, tiles)

    osm_json_path = raw_dir / "osm-source.json"
    overpass_query_path = raw_dir / "overpass-query.txt"
    osm_source = _fetch_overpass(
        osm_json_path,
        overpass_query_path,
        bbox,
        spec.overpass_urls,
        refresh=spec.refresh,
    )

    reference_map_path: Path | None = None
    if spec.reference_map:
        reference_map_path = spec.cache_dir / "opentopomap-reference.png"
        _reference_map(
            reference_map_path,
            bbox,
            int(round(link.zoom)),
            spec.cache_dir,
            refresh=spec.refresh,
        )

    source_manifest_path = spec.cache_dir / "source-manifest.json"
    document = {
        "source_link": asdict(link),
        "selection": {
            "bbox_south_west_north_east": list(bbox),
            "world_size_metres": spec.world_size,
            "cells": spec.cells,
            "cell_size_metres": spec.cell_size,
        },
        "elevation": {
            "dataset": "SRTM3 HGT",
            "tiles": list(required_hgt_tiles(bbox)),
            "download_sources": tile_sources,
            "heightmap": str(heightmap_path),
            "heightmap_sha256": _sha256(heightmap_path),
            "minimum_metres": minimum,
            "maximum_metres": maximum,
        },
        "osm": {
            "source": osm_source,
            "json": str(osm_json_path),
            "json_sha256": _sha256(osm_json_path),
            "query": str(overpass_query_path),
        },
        "reference_map": None if reference_map_path is None else {
            "provider": "OpenTopoMap",
            "path": str(reference_map_path),
            "sha256": _sha256(reference_map_path),
            "zoom": int(round(link.zoom)),
        },
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "attribution": [
            "Map data: OpenStreetMap contributors, ODbL 1.0",
            "Reference tiles: OpenTopoMap, CC BY-SA 3.0; map data OpenStreetMap contributors",
            "Elevation: SRTM3 HGT mirror",
        ],
    }
    source_manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return LocationInputs(
        link=link,
        bbox=bbox,
        heightmap_path=heightmap_path,
        osm_json_path=osm_json_path,
        overpass_query_path=overpass_query_path,
        source_manifest_path=source_manifest_path,
        reference_map_path=reference_map_path,
    )


def build_location_example(spec: LocationExampleSpec) -> tuple[LocationInputs, BuildResult]:
    inputs = prepare_location_inputs(spec)
    world_spec = PlayabilitySpec(
        heightmap_path=inputs.heightmap_path,
        name=spec.name,
        display_name=spec.display_name,
        profile=spec.profile,
        cells=spec.cells,
        cell_size=spec.cell_size,
        input_mode="meters",
        flip_y=True,
        sea_level=0.0,
        beach_height=3.0,
        rock_height=110.0,
        rock_slope_degrees=30.0,
        bbox=inputs.bbox,
        osm_json_path=inputs.osm_json_path,
        include_minor_roads=spec.include_minor_roads,
        forest_road_clearance=spec.forest_road_clearance,
        asset_roots=spec.asset_roots,
        strict_assets=spec.strict_assets,
        deterministic_seed=f"opentopomap:{spec.map_url}:{spec.world_size:.3f}",
    )
    result = build_milestone4(spec.output_dir, world_spec, clean=True)
    # Keep location source metadata next to generated validation material.
    target = result.output_dir / "location-source-manifest.json"
    target.write_bytes(inputs.source_manifest_path.read_bytes())
    if inputs.reference_map_path is not None:
        (result.output_dir / "opentopomap-reference.png").write_bytes(inputs.reference_map_path.read_bytes())
    return inputs, result
