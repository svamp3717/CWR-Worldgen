# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep Overture geometry conversion stable across DuckDB spatial versions.

DuckDB writes the GEOMETRY column from the remote Overture GeoParquet extract as
ordinary WKB/BLOB bytes in the temporary Parquet file.  Reopening that file and
calling ``ST_AsWKB`` a second time therefore fails on versions that correctly bind
``ST_AsWKB(GEOMETRY)`` only.  Freeze the remote geometry explicitly as WKB and read
those bytes directly during the local GeoJSON conversion.
"""
from __future__ import annotations

from typing import Any


_REMOTE_GEOMETRY_LINE = "                    geometry\n                FROM read_parquet("
_REMOTE_WKB_LINE = "                    ST_AsWKB(geometry)::BLOB AS geometry_wkb\n                FROM read_parquet("
_LOCAL_WKB_EXPRESSION = "ST_AsWKB(geometry)::BLOB AS geometry_wkb"


def rewrite_overture_geometry_sql(query: str) -> str:
    """Normalize the two-stage Overture extract to an explicit WKB contract."""

    text = str(query)
    # The remote COPY still reads Overture's spatial GEOMETRY value, so convert
    # it exactly once before Parquet serialization.  The resulting local column
    # is deliberately named geometry_wkb to make its byte representation explicit.
    if "COPY (" in text and _REMOTE_GEOMETRY_LINE in text:
        return text.replace(_REMOTE_GEOMETRY_LINE, _REMOTE_WKB_LINE, 1)

    # The local extract already contains WKB bytes.  Calling ST_AsWKB again is the
    # BinderException seen with DuckDB releases where read_parquet exposes it as BLOB.
    if _LOCAL_WKB_EXPRESSION in text:
        return text.replace(_LOCAL_WKB_EXPRESSION, "geometry_wkb", 1)
    return text


class _GeometryCompatibleConnection:
    """Thin DuckDB connection proxy that rewrites only the Overture geometry SQL."""

    def __init__(self, connection: Any):
        self._connection = connection

    def execute(self, query: str, *args: Any, **kwargs: Any):
        result = self._connection.execute(
            rewrite_overture_geometry_sql(query), *args, **kwargs
        )
        # DuckDBPyConnection.execute returns the connection itself and callers use
        # it as a cursor. Keep that fluent behavior while still proxying fetchmany.
        return self if result is self._connection else result

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def install_overture_geometry_policy() -> None:
    """Install the WKB compatibility layer around direct Overture downloads."""

    from . import overture

    current = overture.download_overture_buildings_direct
    if bool(getattr(current, "_cwr_geometry_wkb_compatibility", False)):
        return

    def download_overture_buildings_direct(*args: Any, **kwargs: Any):
        # Each bbox tile runs in its own worker process, so temporarily replacing
        # duckdb.connect here cannot interfere with another in-process tile. Keep
        # the patch scoped to this call and restore it even when a download fails.
        import duckdb

        original_connect = duckdb.connect

        def compatible_connect(*connect_args: Any, **connect_kwargs: Any):
            return _GeometryCompatibleConnection(
                original_connect(*connect_args, **connect_kwargs)
            )

        duckdb.connect = compatible_connect
        try:
            return current(*args, **kwargs)
        finally:
            duckdb.connect = original_connect

    download_overture_buildings_direct._cwr_geometry_wkb_compatibility = True  # type: ignore[attr-defined]
    overture.download_overture_buildings_direct = download_overture_buildings_direct
