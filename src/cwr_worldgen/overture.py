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
from urllib.request import Request, urlopen


OVERTURE_CLI_MARKER = "--cwr-overture"
# Known-good Overture release used when no explicit override is supplied.
# CWR_OVERTURE_RELEASE can select a newer retained release without a code update.
DEFAULT_OVERTURE_RELEASE = "2026-06-17.0"
_OVERTURE_RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")
_STAC_RELEASE_INDEX = "https://stac.overturemaps.org/{release}/collections.parquet"
_S3_BUCKET_PREFIX = "s3://overturemaps-us-west-2/"
_AZURE_DATA_PREFIX = "az://overturemapswestus2.blob.core.windows.net/"


def overture_buildings_cache_path(cache_dir: Path, bbox: tuple[float, float, float, float]) -> Path:
    digest = sha256(json.dumps(tuple(round(float(value), 7) for value in bbox)).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overture-buildings-{digest}.geojson"


def selected_overture_release() -> str:
    release = os.environ.get("CWR_OVERTURE_RELEASE", "").strip() or DEFAULT_OVERTURE_RELEASE
    if not _OVERTURE_RELEASE_RE.fullmatch(release):
        raise ValueError(f"invalid CWR_OVERTURE_RELEASE value: {release!r}")
    return release


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
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
    timeout: int = 15,
) -> list[str]:
    """Use Overture's per-release spatial index to select exact Parquet files."""
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read Overture's release spatial index") from exc

    south, west, north, east = (float(value) for value in bbox)
    index_url = _STAC_RELEASE_INDEX.format(release=release)
    print(
        f"CWR Overture worker: fetching release spatial index {index_url}",
        file=sys.stderr,
        flush=True,
    )
    request = Request(index_url, headers={"User-Agent": "CWR-Worldgen/0.9.240"})
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
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
    release = selected_overture_release()
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

    exact_files = _intersecting_building_files_from_release_index(bbox, release)
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
                for source_id, building_class, subtype, geometry_wkb in rows:
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
                            "class": _json_value(building_class),
                            "subtype": _json_value(subtype),
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
    options = parser.parse_args(argv)

    output = Path(options.output)
    error_path = _worker_error_path(output)
    error_path.unlink(missing_ok=True)
    try:
        parts = [part.strip() for part in str(options.bbox).split(",")]
        if len(parts) != 4:
            raise ValueError("Overture worker bbox must be west,south,east,north")
        west, south, east, north = (float(part) for part in parts)
        download_overture_buildings_direct((south, west, north, east), output)
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


def fetch_overture_buildings_geojson(
    bbox: tuple[float, float, float, float],
    output: Path,
    *,
    refresh: bool = False,
    timeout: int = 120,
) -> Path | None:
    """Fetch Overture data with heartbeat logs and a hard outer timeout."""
    if output.is_file() and not refresh:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.geojson")
    diagnostic = _worker_error_path(temporary)
    temporary.unlink(missing_ok=True)
    diagnostic.unlink(missing_ok=True)
    south, west, north, east = bbox
    command = overture_command_prefix() + [
        "--bbox",
        f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}",
        "--output",
        str(temporary),
    ]

    started = time.monotonic()
    next_heartbeat = 10.0
    try:
        process = subprocess.Popen(command, **_subprocess_window_kwargs())
    except OSError as exc:
        print(f"CWR Overture fallback failed to start: {exc}", file=sys.stderr, flush=True)
        return None

    while True:
        return_code = process.poll()
        if return_code is not None:
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            print(
                f"CWR Overture fallback timed out after {int(elapsed)}s; continuing with OSM only",
                file=sys.stderr,
                flush=True,
            )
            _terminate_process(process)
            temporary.unlink(missing_ok=True)
            diagnostic.unlink(missing_ok=True)
            return None
        if elapsed >= next_heartbeat:
            print(
                f"CWR Overture fallback still working ({int(elapsed)}s)",
                file=sys.stderr,
                flush=True,
            )
            next_heartbeat += 10.0
        time.sleep(0.25)

    if return_code != 0:
        if diagnostic.is_file():
            try:
                detail = diagnostic.read_text(encoding="utf-8", errors="replace")
                print("CWR Overture fallback failed:\n" + detail, file=sys.stderr, flush=True)
            except OSError:
                pass
        else:
            print(
                f"CWR Overture fallback worker exited with code {return_code}",
                file=sys.stderr,
                flush=True,
            )
        temporary.unlink(missing_ok=True)
        diagnostic.unlink(missing_ok=True)
        return None

    diagnostic.unlink(missing_ok=True)
    if not temporary.is_file():
        print("CWR Overture fallback failed: worker created no GeoJSON file", file=sys.stderr, flush=True)
        return None
    os.replace(temporary, output)
    print(
        f"CWR Overture fallback ready after {time.monotonic() - started:.1f}s: {output}",
        file=sys.stderr,
        flush=True,
    )
    return output
