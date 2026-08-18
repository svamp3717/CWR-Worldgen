# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from urllib.request import Request, urlopen

from ._version import __version__
from .network import UNVERIFIED_SSL_CONTEXT


OVERTURE_CLI_MARKER = "--cwr-overture"
# Last-known release used only if the live STAC catalog cannot be reached.
# Normally the worker resolves Overture's current release from STAC on each run.
DEFAULT_OVERTURE_RELEASE = "2026-07-22.0"
_OVERTURE_RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")
_STAC_ROOT_CATALOG = "https://stac.overturemaps.org/catalog.json"
_STAC_RELEASE_INDEX = "https://stac.overturemaps.org/{release}/collections.parquet"
_S3_BUCKET_PREFIX = "s3://overturemaps-us-west-2/"
_AZURE_DATA_PREFIX = "az://overturemapswestus2.blob.core.windows.net/"


def overture_buildings_cache_path(cache_dir: Path, bbox: tuple[float, float, float, float]) -> Path:
    digest = sha256(json.dumps(tuple(round(float(value), 7) for value in bbox)).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overture-buildings-{digest}.geojson"


def _latest_overture_release_from_stac(*, timeout: int = 15) -> str:
    """Resolve Overture's live current release from its root STAC catalog."""
    req = Request(
        _STAC_ROOT_CATALOG,
        headers={
            "User-Agent": f"CWR-Worldgen/{__version__} (+https://github.com/svamp3717/CWR-Worldgen)",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout, context=UNVERIFIED_SSL_CONTEXT) as response:
        document = json.loads(response.read().decode("utf-8"))
    release = str(document.get("latest") or "").strip() if isinstance(document, dict) else ""
    if not _OVERTURE_RELEASE_RE.fullmatch(release):
        raise RuntimeError(f"Overture STAC catalog returned invalid latest release: {release!r}")
    return release


def selected_overture_release() -> str:
    override = os.environ.get("CWR_OVERTURE_RELEASE", "").strip()
    if override:
        if not _OVERTURE_RELEASE_RE.fullmatch(override):
            raise ValueError(f"invalid CWR_OVERTURE_RELEASE value: {override!r}")
        return override

    try:
        release = _latest_overture_release_from_stac()
        print(
            f"CWR Overture worker: live STAC latest release={release}",
            file=sys.stderr,
            flush=True,
        )
        return release
    except Exception as exc:  # noqa: BLE001 - retain a last-known fallback for offline/broken STAC.
        print(
            "CWR Overture worker: could not resolve live STAC latest release; "
            f"falling back to {DEFAULT_OVERTURE_RELEASE}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return DEFAULT_OVERTURE_RELEASE


def _json_value(value):
    """Convert DuckDB nested values into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return str(value)


def _sql_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite Overture bbox coordinate: {value!r}")
    return format(number, ".10f")


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _azure_href_from_stac_asset(asset: object) -> str | None:
    """Resolve the Azure mirror URL from Overture's STAC asset metadata."""
    if not isinstance(asset, dict):
        return None

    aws = asset.get("aws")
    if isinstance(aws, dict):
        alternate = aws.get("alternate")
        if isinstance(alternate, dict):
            s3 = alternate.get("s3")
            if isinstance(s3, dict):
                href = str(s3.get("href") or "")
                if href.startswith(_S3_BUCKET_PREFIX):
                    return _AZURE_DATA_PREFIX + href[len(_S3_BUCKET_PREFIX) :]

    for value in asset.values():
        if not isinstance(value, dict):
            continue
        href = str(value.get("href") or "")
        if "overturemapswestus2.blob.core.windows.net" in href:
            if href.startswith("https://"):
                relative = href.split("overturemapswestus2.blob.core.windows.net/", 1)[1]
                return _AZURE_DATA_PREFIX + relative
            return href
        alternate = value.get("alternate")
        if isinstance(alternate, dict):
            for candidate in alternate.values():
                if not isinstance(candidate, dict):
                    continue
                href = str(candidate.get("href") or "")
                if href.startswith(_S3_BUCKET_PREFIX):
                    return _AZURE_DATA_PREFIX + href[len(_S3_BUCKET_PREFIX) :]
                if "overturemapswestus2.blob.core.windows.net" in href:
                    if href.startswith("https://"):
                        relative = href.split("overturemapswestus2.blob.core.windows.net/", 1)[1]
                        return _AZURE_DATA_PREFIX + relative
                    return href
    return None


def _intersecting_building_files_from_release_index(
    bbox: tuple[float, float, float, float],
    release: str,
    *,
    timeout: int = 30,
    cache_path: Path | None = None,
) -> list[str]:
    """Use Overture's per-release spatial index to select exact Parquet files.

    A release-scoped local copy can be shared by every bbox tile so a large
    world does not download and parse the same STAC index over and over.
    """
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read Overture's release spatial index") from exc

    south, west, north, east = (float(value) for value in bbox)
    index_url = _STAC_RELEASE_INDEX.format(release=release)
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.is_file():
        data = cache_path.read_bytes()
        print(
            f"CWR Overture worker: using cached release spatial index {cache_path} "
            f"({len(data):,} bytes)",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"CWR Overture worker: fetching release spatial index {index_url}",
            file=sys.stderr,
            flush=True,
        )
        request = Request(
            index_url,
            headers={
                "User-Agent": f"CWR-Worldgen/{__version__} (+https://github.com/svamp3717/CWR-Worldgen)"
            },
        )
        started = time.monotonic()
        with urlopen(request, timeout=timeout, context=UNVERIFIED_SSL_CONTEXT) as response:
            data = response.read()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = cache_path.with_name(cache_path.name + ".tmp")
            temporary_cache.write_bytes(data)
            os.replace(temporary_cache, cache_path)
        print(
            f"CWR Overture worker: spatial index fetched in {time.monotonic() - started:.1f}s "
            f"({len(data):,} bytes)",
            file=sys.stderr,
            flush=True,
        )

    table = pq.read_table(io.BytesIO(data))
    feature_filter = (pc.field("collection") == "building") & (pc.field("type") == "Feature")
    bbox_filter = (
        (pc.field("bbox", "xmin") < east)
        & (pc.field("bbox", "xmax") > west)
        & (pc.field("bbox", "ymin") < north)
        & (pc.field("bbox", "ymax") > south)
    )
    selected = table.filter(feature_filter & bbox_filter)
    if selected.num_rows == 0:
        print("CWR Overture worker: spatial index reports no building files for bbox", file=sys.stderr, flush=True)
        return []

    files: list[str] = []
    for asset in selected.column("assets").to_pylist():
        href = _azure_href_from_stac_asset(asset)
        if href and href not in files:
            files.append(href)
    if not files:
        raise RuntimeError(
            "Overture spatial index matched the bbox but no usable Azure/S3 asset hrefs were found"
        )
    print(
        f"CWR Overture worker: spatial index selected {len(files)} exact building file(s)",
        file=sys.stderr,
        flush=True,
    )
    return files


def download_overture_buildings_direct(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    release: str | None = None,
    release_index_cache: Path | None = None,
) -> str:
    """Fetch Overture buildings using the release index + exact Azure files."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is required for Overture building fallback support. Install the sources extra."
        ) from exc

    try:
        from shapely import from_wkb
        from shapely.geometry import mapping
    except ImportError as exc:
        raise RuntimeError("Shapely is required to convert Overture building geometry") from exc

    south, west, north, east = (float(value) for value in bbox)
    release = release or selected_overture_release()
    if not _OVERTURE_RELEASE_RE.fullmatch(release):
        raise ValueError(f"invalid Overture release value: {release!r}")
    west_sql = _sql_float(west)
    south_sql = _sql_float(south)
    east_sql = _sql_float(east)
    north_sql = _sql_float(north)

    print(
        f"CWR Overture worker: backend=STAC-index/DuckDB/Azure release={release}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "CWR Overture worker: bbox="
        f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
        file=sys.stderr,
        flush=True,
    )

    exact_files = _intersecting_building_files_from_release_index(
        bbox, release, cache_path=release_index_cache
    )
    output = Path(output)
    if not exact_files:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return release

    file_list_sql = "[" + ",".join(_sql_string(path) for path in exact_files) + "]"
    extract = output.with_name(output.name + ".extract.parquet")
    extract.unlink(missing_ok=True)
    connection = duckdb.connect(database=":memory:")
    try:
        for extension in ("azure", "spatial"):
            try:
                connection.execute(f"LOAD {extension}")
            except Exception:
                try:
                    connection.execute(f"INSTALL {extension}")
                    connection.execute(f"LOAD {extension}")
                except Exception as exc:
                    raise RuntimeError(
                        f"DuckDB {extension} extension could not be installed or loaded"
                    ) from exc

        connection.execute(
            "CREATE SECRET cwr_overture_azure ("
            "TYPE azure, PROVIDER config, ACCOUNT_NAME 'overturemapswestus2'"
            ")"
        )
        connection.execute("SET threads = 4")
        connection.execute("SET enable_http_metadata_cache = true")

        extract_query = f"""
            COPY (
                SELECT
                    id,
                    \"class\" AS building_class,
                    subtype,
                    sources,
                    has_parts,
                    height,
                    num_floors,
                    min_height,
                    min_floor,
                    facade_color,
                    facade_material,
                    roof_material,
                    roof_shape,
                    roof_direction,
                    roof_orientation,
                    roof_color,
                    roof_height,
                    geometry
                FROM read_parquet(
                    {file_list_sql},
                    hive_partitioning = true,
                    union_by_name = true
                )
                WHERE
                    bbox.xmin < {east_sql} AND bbox.xmax > {west_sql} AND
                    bbox.ymin < {north_sql} AND bbox.ymax > {south_sql}
            ) TO {_sql_string(str(extract))} (FORMAT PARQUET)
        """

        print(
            "CWR Overture worker: querying only spatial-index-selected Parquet files",
            file=sys.stderr,
            flush=True,
        )
        remote_started = time.monotonic()
        connection.execute(extract_query)
        if not extract.is_file():
            raise RuntimeError("Overture bbox extraction completed without creating local GeoParquet")
        print(
            f"CWR Overture worker: remote extract ready in {time.monotonic() - remote_started:.1f}s "
            f"({extract.stat().st_size:,} bytes); converting locally",
            file=sys.stderr,
            flush=True,
        )

        cursor = connection.execute(
            f"""
                SELECT
                    id,
                    building_class,
                    subtype,
                    sources,
                    has_parts,
                    height,
                    num_floors,
                    min_height,
                    min_floor,
                    facade_color,
                    facade_material,
                    roof_material,
                    roof_shape,
                    roof_direction,
                    roof_orientation,
                    roof_color,
                    roof_height,
                    ST_AsWKB(geometry)::BLOB AS geometry_wkb
                FROM read_parquet({_sql_string(str(extract))})
            """
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        feature_count = 0
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type":"FeatureCollection","features":[')
            first = True
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    (
                        source_id, building_class, subtype, sources, has_parts, height,
                        num_floors, min_height, min_floor, facade_color, facade_material,
                        roof_material, roof_shape, roof_direction, roof_orientation,
                        roof_color, roof_height, geometry_wkb,
                    ) = row
                    if geometry_wkb is None:
                        continue
                    try:
                        geometry = from_wkb(bytes(geometry_wkb))
                    except Exception:
                        continue
                    if geometry is None or geometry.is_empty:
                        continue
                    feature = {
                        "type": "Feature",
                        "id": str(source_id),
                        "properties": {
                            "id": str(source_id),
                            "class": _json_value(building_class),
                            "subtype": _json_value(subtype),
                            "sources": _json_value(sources),
                            "has_parts": _json_value(has_parts),
                            "height": _json_value(height),
                            "num_floors": _json_value(num_floors),
                            "min_height": _json_value(min_height),
                            "min_floor": _json_value(min_floor),
                            "facade_color": _json_value(facade_color),
                            "facade_material": _json_value(facade_material),
                            "roof_material": _json_value(roof_material),
                            "roof_shape": _json_value(roof_shape),
                            "roof_direction": _json_value(roof_direction),
                            "roof_orientation": _json_value(roof_orientation),
                            "roof_color": _json_value(roof_color),
                            "roof_height": _json_value(roof_height),
                        },
                        "geometry": mapping(geometry),
                    }
                    if not first:
                        stream.write(",")
                    json.dump(feature, stream, ensure_ascii=False, separators=(",", ":"))
                    first = False
                    feature_count += 1
                stream.flush()
            stream.write("]}")

        print(
            f"CWR Overture worker: completed with {feature_count:,} buildings; "
            f"wrote {output.stat().st_size:,} bytes",
            file=sys.stderr,
            flush=True,
        )
        return release
    finally:
        try:
            connection.close()
        except Exception:
            pass
        extract.unlink(missing_ok=True)


def _worker_error_path(output: Path) -> Path:
    return output.with_name(output.name + ".error.txt")


def run_overture_worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cwr-worldgen-overture-worker", add_help=False)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release")
    parser.add_argument("--release-index-cache")
    options = parser.parse_args(argv)

    output = Path(options.output)
    error_path = _worker_error_path(output)
    error_path.unlink(missing_ok=True)
    try:
        parts = [part.strip() for part in str(options.bbox).split(",")]
        if len(parts) != 4:
            raise ValueError("Overture worker bbox must be west,south,east,north")
        west, south, east, north = (float(part) for part in parts)
        worker_options: dict[str, object] = {}
        if options.release:
            worker_options["release"] = str(options.release).strip()
        if options.release_index_cache:
            worker_options["release_index_cache"] = Path(options.release_index_cache)
        download_overture_buildings_direct(
            (south, west, north, east),
            output,
            **worker_options,
        )
        return 0
    except Exception:
        diagnostic = traceback.format_exc()
        try:
            error_path.write_text(diagnostic, encoding="utf-8")
        except OSError:
            pass
        try:
            print(diagnostic, file=sys.stderr, flush=True)
        except Exception:
            pass
        return 1


def overture_command_prefix() -> list[str]:
    executable = str(Path(sys.executable).resolve())
    if bool(getattr(sys, "frozen", False)):
        return [executable, OVERTURE_CLI_MARKER]
    return [executable, "-m", "cwr_worldgen.gui_entry", OVERTURE_CLI_MARKER]


def _subprocess_window_kwargs() -> dict[str, object]:
    if os.name != "nt":
        return {}
    kwargs: dict[str, object] = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_type is not None:
        startupinfo = startupinfo_type()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        process.kill()
    except OSError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _bbox_size_km(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return approximate north/south and east/west bbox spans in kilometres."""
    south, west, north, east = (float(value) for value in bbox)
    latitude_km = abs(north - south) * 111.32
    mid_latitude = math.radians((south + north) * 0.5)
    longitude_scale = max(0.01, abs(math.cos(mid_latitude)))
    longitude_km = abs(east - west) * 111.32 * longitude_scale
    return latitude_km, longitude_km


def _overture_bbox_tiles(
    bbox: tuple[float, float, float, float],
    *,
    maximum_edge_km: float = 10.0,
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into a deterministic grid whose tile edges are about <= 10 km."""
    south, west, north, east = (float(value) for value in bbox)
    latitude_km, longitude_km = _bbox_size_km(bbox)
    edge = max(1.0, float(maximum_edge_km))
    # Avoid an extra row/column when a nominal 50 km span is represented as
    # 50.00000000000001 by floating-point arithmetic.
    rows = max(1, math.ceil(max(0.0, latitude_km - 1.0e-6) / edge))
    columns = max(1, math.ceil(max(0.0, longitude_km - 1.0e-6) / edge))
    # A malformed/global bbox should not accidentally create thousands of workers.
    if rows * columns > 400:
        scale = math.sqrt((rows * columns) / 400.0)
        rows = max(1, math.ceil(rows / scale))
        columns = max(1, math.ceil(columns / scale))
    latitude_step = (north - south) / rows
    longitude_step = (east - west) / columns
    tiles: list[tuple[float, float, float, float]] = []
    for row in range(rows):
        tile_south = south + latitude_step * row
        tile_north = north if row == rows - 1 else south + latitude_step * (row + 1)
        for column in range(columns):
            tile_west = west + longitude_step * column
            tile_east = east if column == columns - 1 else west + longitude_step * (column + 1)
            tiles.append((tile_south, tile_west, tile_north, tile_east))
    return tiles


def _overture_tile_timeout_seconds(
    bbox: tuple[float, float, float, float],
) -> int:
    """Choose a generous per-tile timeout instead of one fixed timeout for a whole world."""
    latitude_km, longitude_km = _bbox_size_km(bbox)
    largest_edge = max(latitude_km, longitude_km)
    if largest_edge <= 5.0:
        return 180
    if largest_edge <= 10.5:
        return 300
    if largest_edge <= 25.0:
        return 450
    return 600


def _bbox_digest(bbox: tuple[float, float, float, float]) -> str:
    values = tuple(round(float(value), 7) for value in bbox)
    return sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def _tile_cache_path(root: Path, bbox: tuple[float, float, float, float]) -> Path:
    return root / f"building-{_bbox_digest(bbox)}.geojson"


def _load_tile_manifest(path: Path) -> dict[str, object] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return document if isinstance(document, dict) else None


def _write_tile_manifest(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _feature_identity(feature: object) -> str:
    if not isinstance(feature, dict):
        return ""
    source_id = str(feature.get("id") or "").strip()
    properties = feature.get("properties")
    if not source_id and isinstance(properties, dict):
        source_id = str(properties.get("id") or "").strip()
    if source_id:
        return "id:" + source_id
    stable = json.dumps(feature, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + sha256(stable.encode("utf-8")).hexdigest()


def _merge_overture_tile_geojson(tile_paths: list[Path], output: Path) -> tuple[int, int]:
    """Merge cached tile GeoJSONs while removing cross-tile duplicate buildings."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".merge.tmp")
    seen: set[str] = set()
    written = 0
    duplicates = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"type":"FeatureCollection","features":[')
        first = True
        for tile_path in tile_paths:
            document = json.loads(tile_path.read_text(encoding="utf-8"))
            features = document.get("features", []) if isinstance(document, dict) else []
            if not isinstance(features, list):
                continue
            for feature in features:
                identity = _feature_identity(feature)
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                if not first:
                    stream.write(",")
                json.dump(feature, stream, ensure_ascii=False, separators=(",", ":"))
                first = False
                written += 1
        stream.write("]}")
    os.replace(temporary, output)
    return written, duplicates


def _run_overture_tile_worker(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    release: str,
    release_index_cache: Path,
    timeout: int,
    tile_number: int,
    tile_count: int,
) -> bool:
    temporary = output.with_name(output.name + ".tmp")
    diagnostic = _worker_error_path(temporary)
    temporary.unlink(missing_ok=True)
    diagnostic.unlink(missing_ok=True)
    south, west, north, east = bbox
    command = overture_command_prefix() + [
        "--bbox",
        f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
        "--output",
        str(temporary),
        "--release",
        release,
        "--release-index-cache",
        str(release_index_cache),
    ]

    started = time.monotonic()
    next_heartbeat = 10.0
    try:
        process = subprocess.Popen(command, **_subprocess_window_kwargs())
    except OSError as exc:
        print(
            f"CWR Overture tile {tile_number}/{tile_count} failed to start: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False

    while True:
        return_code = process.poll()
        if return_code is not None:
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            print(
                f"CWR Overture tile {tile_number}/{tile_count} timed out after {int(elapsed)}s",
                file=sys.stderr,
                flush=True,
            )
            _terminate_process(process)
            temporary.unlink(missing_ok=True)
            # Keep the diagnostic if the worker managed to create one; it is useful on retry/failure.
            return False
        if elapsed >= next_heartbeat:
            print(
                f"CWR Overture tile {tile_number}/{tile_count} still working "
                f"({int(elapsed)}s, timeout {timeout}s)",
                file=sys.stderr,
                flush=True,
            )
            next_heartbeat += 10.0
        time.sleep(0.25)

    if return_code != 0:
        if diagnostic.is_file():
            try:
                detail = diagnostic.read_text(encoding="utf-8", errors="replace")
                print(
                    f"CWR Overture tile {tile_number}/{tile_count} failed:\n{detail}",
                    file=sys.stderr,
                    flush=True,
                )
            except OSError:
                pass
        else:
            print(
                f"CWR Overture tile {tile_number}/{tile_count} worker exited with code {return_code}",
                file=sys.stderr,
                flush=True,
            )
        temporary.unlink(missing_ok=True)
        return False

    diagnostic.unlink(missing_ok=True)
    if not temporary.is_file():
        print(
            f"CWR Overture tile {tile_number}/{tile_count} failed: worker created no GeoJSON file",
            file=sys.stderr,
            flush=True,
        )
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(
        f"CWR Overture tile {tile_number}/{tile_count} ready after "
        f"{time.monotonic() - started:.1f}s ({output.stat().st_size:,} bytes)",
        file=sys.stderr,
        flush=True,
    )
    return True


def fetch_overture_buildings_geojson(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    refresh: bool = False,
    timeout: int | None = None,
    tile_edge_km: float = 10.0,
    max_attempts: int = 2,
    tile_cache_dir: Path | None = None,
) -> Path | None:
    """Fetch Overture buildings as resumable bbox tiles and merge them locally.

    Large worlds no longer share one global wall-clock timeout. Each roughly
    10 km tile gets its own adaptive timeout, successful tiles are cached, and
    an interrupted run resumes from those completed tile files.
    """
    output = Path(output)
    if output.is_file() and not refresh:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    release = selected_overture_release()
    tiles = _overture_bbox_tiles(bbox, maximum_edge_km=tile_edge_km)
    latitude_km, longitude_km = _bbox_size_km(bbox)
    world_digest = _bbox_digest(bbox)
    cache_base = (
        Path(tile_cache_dir)
        if tile_cache_dir is not None
        else output.parent / "overture-tiles"
    )
    cache_root = cache_base / release
    cache_root.mkdir(parents=True, exist_ok=True)
    release_index_cache = cache_root / "collections.parquet"
    manifest_path = cache_root / f"world-{world_digest}.tiles.json"
    previous_manifest = _load_tile_manifest(manifest_path)
    completed: set[str] = set()
    resume_incomplete_refresh = False
    if previous_manifest:
        previous_release = str(previous_manifest.get("release") or "")
        previous_world = str(previous_manifest.get("world_bbox_digest") or "")
        previous_complete = bool(previous_manifest.get("complete", False))
        if previous_release == release and previous_world == world_digest and not previous_complete:
            raw_completed = previous_manifest.get("completed", [])
            if isinstance(raw_completed, list):
                completed = {str(value) for value in raw_completed}
            resume_incomplete_refresh = refresh

    print(
        f"CWR Overture: world bbox approximately {longitude_km:.1f} x {latitude_km:.1f} km; "
        f"using {len(tiles)} tile(s) at <= {float(tile_edge_km):.1f} km, release={release}",
        file=sys.stderr,
        flush=True,
    )
    if len(tiles) > 1:
        print(
            "CWR Overture: completed tiles are cached and an interrupted run will resume them",
            file=sys.stderr,
            flush=True,
        )

    tile_paths: list[Path] = []
    max_attempts = max(1, int(max_attempts))
    for index, tile_bbox in enumerate(tiles, start=1):
        tile_key = _bbox_digest(tile_bbox)
        tile_path = _tile_cache_path(cache_root, tile_bbox)
        tile_paths.append(tile_path)
        can_reuse = tile_path.is_file() and (
            not refresh or (resume_incomplete_refresh and tile_key in completed)
        )
        if can_reuse:
            print(
                f"CWR Overture tile {index}/{len(tiles)}: cached, skipping download",
                file=sys.stderr,
                flush=True,
            )
            completed.add(tile_key)
            continue

        tile_timeout = int(timeout) if timeout is not None else _overture_tile_timeout_seconds(tile_bbox)
        success = False
        for attempt in range(1, max_attempts + 1):
            print(
                f"CWR Overture tile {index}/{len(tiles)}: downloading "
                f"(attempt {attempt}/{max_attempts}, timeout {tile_timeout}s)",
                file=sys.stderr,
                flush=True,
            )
            success = _run_overture_tile_worker(
                tile_bbox,
                tile_path,
                release=release,
                release_index_cache=release_index_cache,
                timeout=tile_timeout,
                tile_number=index,
                tile_count=len(tiles),
            )
            if success:
                break
            if attempt < max_attempts:
                print(
                    f"CWR Overture tile {index}/{len(tiles)}: retrying after 5s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5.0)
        if not success:
            manifest = {
                "schema": 1,
                "release": release,
                "world_bbox_digest": world_digest,
                "bbox": [float(value) for value in bbox],
                "tile_edge_km": float(tile_edge_km),
                "tile_count": len(tiles),
                "completed": sorted(completed),
                "complete": False,
            }
            _write_tile_manifest(manifest_path, manifest)
            print(
                f"CWR Overture: stopped after tile {index}/{len(tiles)} failed; "
                f"{len(completed)} completed tile(s) remain cached for resume; continuing with OSM only",
                file=sys.stderr,
                flush=True,
            )
            return None

        completed.add(tile_key)
        manifest = {
            "schema": 1,
            "release": release,
            "world_bbox_digest": world_digest,
            "bbox": [float(value) for value in bbox],
            "tile_edge_km": float(tile_edge_km),
            "tile_count": len(tiles),
            "completed": sorted(completed),
            "complete": False,
        }
        _write_tile_manifest(manifest_path, manifest)

    started_merge = time.monotonic()
    try:
        feature_count, duplicates = _merge_overture_tile_geojson(tile_paths, output)
    except Exception as exc:  # noqa: BLE001 - a corrupt tile should not abort the whole world build.
        print(f"CWR Overture tile merge failed: {exc}; continuing with OSM only", file=sys.stderr, flush=True)
        return None

    _write_tile_manifest(
        manifest_path,
        {
            "schema": 1,
            "release": release,
            "world_bbox_digest": world_digest,
            "bbox": [float(value) for value in bbox],
            "tile_edge_km": float(tile_edge_km),
            "tile_count": len(tiles),
            "completed": sorted(completed),
            "complete": True,
            "features": feature_count,
            "duplicates_removed": duplicates,
        },
    )
    print(
        f"CWR Overture ready: {feature_count:,} buildings from {len(tiles)} tile(s); "
        f"removed {duplicates:,} cross-tile duplicate(s); merge took "
        f"{time.monotonic() - started_merge:.1f}s: {output}",
        file=sys.stderr,
        flush=True,
    )
    return output
