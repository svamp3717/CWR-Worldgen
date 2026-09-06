from __future__ import annotations

from cwr_worldgen import overture
from cwr_worldgen.overture_geometry_policy import (
    _GeometryCompatibleConnection,
    rewrite_overture_geometry_sql,
)


def test_remote_extract_serializes_geometry_as_named_wkb_blob() -> None:
    query = """
            COPY (
                SELECT
                    id,
                    geometry
                FROM read_parquet(['az://example/building.parquet'])
            ) TO 'tile.extract.parquet' (FORMAT PARQUET)
    """

    rewritten = rewrite_overture_geometry_sql(query)

    assert "ST_AsWKB(geometry)::BLOB AS geometry_wkb" in rewritten
    assert "                    geometry\n                FROM read_parquet(" not in rewritten


def test_local_extract_reads_wkb_blob_without_second_spatial_conversion() -> None:
    query = """
        SELECT
            id,
            ST_AsWKB(geometry)::BLOB AS geometry_wkb
        FROM read_parquet('tile.extract.parquet')
    """

    rewritten = rewrite_overture_geometry_sql(query)

    assert "ST_AsWKB(geometry)" not in rewritten
    assert "geometry_wkb" in rewritten


def test_unrelated_duckdb_sql_is_untouched() -> None:
    query = "SELECT id, height FROM read_parquet('tile.extract.parquet')"
    assert rewrite_overture_geometry_sql(query) == query


def test_connection_proxy_rewrites_execute_and_keeps_cursor_behavior() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            return self

        def fetchmany(self, count: int):
            return [(count,)]

    raw = FakeConnection()
    connection = _GeometryCompatibleConnection(raw)
    cursor = connection.execute(
        "SELECT ST_AsWKB(geometry)::BLOB AS geometry_wkb FROM read_parquet('x')"
    )

    assert cursor is connection
    assert "ST_AsWKB(geometry)" not in raw.queries[0]
    assert cursor.fetchmany(1000) == [(1000,)]


def test_package_bootstrap_installs_overture_geometry_compatibility() -> None:
    assert bool(
        getattr(
            overture.download_overture_buildings_direct,
            "_cwr_geometry_wkb_compatibility",
            False,
        )
    )
