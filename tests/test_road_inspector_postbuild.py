from __future__ import annotations

from pathlib import Path

from cwr_worldgen import road_inspector_postbuild as _postbuild


def test_generated_world_pbo_prefers_runtime_addons(tmp_path: Path) -> None:
    other = tmp_path / "Other" / "wg_demo.pbo"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other")
    expected = tmp_path / "CWR-Worldgen" / "Addons" / "wg_demo.pbo"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"world")

    assert _postbuild.generated_world_pbo(tmp_path, "wg_demo") == expected.resolve()


def test_normalized_roads_prefers_source_bundle(tmp_path: Path) -> None:
    build = tmp_path / "build"
    source = tmp_path / "source"
    roads = source / "normalized" / "roads.geojson"
    roads.parent.mkdir(parents=True)
    roads.write_text("{}", encoding="utf-8")

    assert _postbuild.normalized_roads_geojson(build, source) == roads.resolve()


def test_postbuild_report_is_written_inside_world_build_folder(
    tmp_path: Path, monkeypatch
) -> None:
    pbo = tmp_path / "CWR-Worldgen" / "Addons" / "wg_demo.pbo"
    pbo.parent.mkdir(parents=True)
    pbo.write_bytes(b"pbo")
    roads = tmp_path / "source" / "normalized" / "roads.geojson"
    roads.parent.mkdir(parents=True)
    roads.write_text("{}", encoding="utf-8")

    from cwr_worldgen import road_inspector_entry

    captured: list[str] = []

    def fake_main(argv) -> int:
        captured.extend(str(value) for value in argv)
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.html").write_text("report", encoding="utf-8")
        return 0

    monkeypatch.setattr(road_inspector_entry, "main", fake_main)

    assert (
        _postbuild.run_postbuild_road_inspector(
            tmp_path,
            "wg_demo",
            source_dir=tmp_path / "source",
        )
        == 0
    )
    report = tmp_path / _postbuild.REPORT_DIRNAME / "report.html"
    assert report.read_text(encoding="utf-8") == "report"
    assert str(pbo.resolve()) in captured
    assert str(roads.resolve()) in captured
    assert str(report.parent.resolve()) in captured


def test_postbuild_inspector_failure_is_nonfatal(tmp_path: Path, monkeypatch) -> None:
    pbo = tmp_path / "CWR-Worldgen" / "Addons" / "wg_demo.pbo"
    pbo.parent.mkdir(parents=True)
    pbo.write_bytes(b"pbo")

    from cwr_worldgen import road_inspector_entry

    def fail(_argv) -> int:
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(road_inspector_entry, "main", fail)

    assert _postbuild.run_postbuild_road_inspector(tmp_path, "wg_demo") == 0
    error = tmp_path / _postbuild.REPORT_DIRNAME / _postbuild.ERROR_FILENAME
    assert "diagnostic failed" in error.read_text(encoding="utf-8")


def test_postbuild_refresh_removes_stale_report(tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / _postbuild.REPORT_DIRNAME / "old.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    pbo = tmp_path / "wg_demo.pbo"
    pbo.write_bytes(b"pbo")

    from cwr_worldgen import road_inspector_entry

    def fake_main(argv) -> int:
        output = Path(argv[argv.index("--output") + 1])
        (output / "report.html").write_text("new", encoding="utf-8")
        return 0

    monkeypatch.setattr(road_inspector_entry, "main", fake_main)

    assert _postbuild.run_postbuild_road_inspector(tmp_path, "wg_demo") == 0
    assert not stale.exists()
    assert (stale.parent / "report.html").read_text(encoding="utf-8") == "new"
