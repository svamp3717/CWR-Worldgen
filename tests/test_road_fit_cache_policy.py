from types import SimpleNamespace

import cwr_worldgen.road_fit_cache_policy as road_cache
from cwr_worldgen.playability import RoadFitReport


def _report(chain_count: int = 7) -> RoadFitReport:
    return RoadFitReport(
        objects=(),
        chain_count=chain_count,
        connection_count=0,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )


def _spec(tmp_path, **changes):
    values = {
        "cache_dir": tmp_path,
        "cache_enabled": True,
        "cache_refresh": False,
        "road_segment_length": 25.0,
        "road_connection_tolerance": 0.35,
        "verify_regeneration": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_road_fit_cache_reuses_identical_core_result(tmp_path, monkeypatch) -> None:
    calls = []
    progress = []

    def fit(*_args, **_kwargs):
        calls.append(1)
        return _report()

    monkeypatch.setattr(road_cache, "_ORIGINAL_FIT", fit)
    monkeypatch.setattr(road_cache, "_runtime_code_fingerprint", lambda: "code-v1")
    dataset = SimpleNamespace(normalized_fingerprint="dataset-v1")
    projection = (0.0, 0.0, 50000.0, 50000.0)
    elevations = (1.0, 2.0, 3.0, 4.0)
    spec = _spec(tmp_path)

    first = road_cache._fit(
        dataset,
        projection,
        elevations,
        spec,
        progress_callback=lambda value, message: progress.append((value, message)),
    )
    second = road_cache._fit(
        dataset,
        projection,
        elevations,
        spec,
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert first == second == _report()
    assert len(calls) == 1
    assert any("Road-fit cache stored" in message for _value, message in progress)
    assert any("Road fitting ready from cache" in message for _value, message in progress)
    assert len(tuple((tmp_path / "roads").glob("road-fit-*.pickle"))) == 1


def test_road_fit_cache_invalidates_when_final_terrain_changes(tmp_path, monkeypatch) -> None:
    calls = []

    def fit(*_args, **_kwargs):
        calls.append(1)
        return _report(chain_count=len(calls))

    monkeypatch.setattr(road_cache, "_ORIGINAL_FIT", fit)
    monkeypatch.setattr(road_cache, "_runtime_code_fingerprint", lambda: "code-v1")
    dataset = SimpleNamespace(normalized_fingerprint="dataset-v1")
    projection = (0.0, 0.0, 50000.0, 50000.0)
    spec = _spec(tmp_path)

    first = road_cache._fit(
        dataset,
        projection,
        (1.0, 2.0, 3.0, 4.0),
        spec,
        progress_callback=lambda *_args: None,
    )
    second = road_cache._fit(
        dataset,
        projection,
        (1.0, 2.0, 3.0, 4.5),
        spec,
        progress_callback=lambda *_args: None,
    )

    assert first.chain_count == 1
    assert second.chain_count == 2
    assert len(calls) == 2
    assert len(tuple((tmp_path / "roads").glob("road-fit-*.pickle"))) == 2


def test_road_fit_cache_invalidates_when_road_spec_changes(tmp_path, monkeypatch) -> None:
    calls = []

    def fit(*_args, **_kwargs):
        calls.append(1)
        return _report(chain_count=len(calls))

    monkeypatch.setattr(road_cache, "_ORIGINAL_FIT", fit)
    monkeypatch.setattr(road_cache, "_runtime_code_fingerprint", lambda: "code-v1")
    dataset = SimpleNamespace(normalized_fingerprint="dataset-v1")
    projection = (0.0, 0.0, 50000.0, 50000.0)
    elevations = (1.0, 2.0, 3.0, 4.0)

    first = road_cache._fit(
        dataset,
        projection,
        elevations,
        _spec(tmp_path, road_connection_tolerance=0.35),
        progress_callback=lambda *_args: None,
    )
    second = road_cache._fit(
        dataset,
        projection,
        elevations,
        _spec(tmp_path, road_connection_tolerance=0.40),
        progress_callback=lambda *_args: None,
    )

    assert first.chain_count == 1
    assert second.chain_count == 2
    assert len(calls) == 2


def test_regeneration_without_progress_bypasses_road_fit_cache(tmp_path, monkeypatch) -> None:
    calls = []

    def fit(*_args, **_kwargs):
        calls.append(1)
        return _report(chain_count=len(calls))

    monkeypatch.setattr(road_cache, "_ORIGINAL_FIT", fit)
    dataset = SimpleNamespace(normalized_fingerprint="dataset-v1")
    projection = (0.0, 0.0, 50000.0, 50000.0)
    elevations = (1.0, 2.0)
    spec = _spec(tmp_path)

    first = road_cache._fit(
        dataset, projection, elevations, spec, progress_callback=None
    )
    second = road_cache._fit(
        dataset, projection, elevations, spec, progress_callback=None
    )

    assert first.chain_count == 1
    assert second.chain_count == 2
    assert len(calls) == 2
    assert not (tmp_path / "roads").exists()


def test_cache_report_records_road_fit_hit(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.setattr(
        road_cache,
        "_ORIGINAL_WRITE_JSON",
        lambda path, value: captured.update(path=path, value=value),
    )
    token = road_cache._LAST_CACHE_INFO.set(
        {
            "hit": True,
            "key": "abc",
            "path": str(tmp_path / "roads" / "road-fit-abc.pickle"),
            "objects": 123,
            "chains": 45,
            "stage_schema": 1,
            "code_fingerprint": "code-v1",
        }
    )
    try:
        road_cache._write_json(tmp_path / "cache-report.json", {"schema": 2})
    finally:
        road_cache._LAST_CACHE_INFO.reset(token)

    assert captured["value"]["road_fit"]["hit"] is True
    assert captured["value"]["road_fit"]["objects"] == 123
    assert captured["value"]["road_fit"]["chains"] == 45
