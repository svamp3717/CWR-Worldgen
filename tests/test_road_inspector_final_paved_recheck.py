# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_final_paved_recheck as _final


def test_existing_grass_wedge_is_removed_when_final_surface_audit_covers_it(monkeypatch) -> None:
    issue = _core.RoadIssue(
        issue_id="RI-00001",
        severity="high",
        score=70.0,
        category="grass_wedge",
        x=10.0,
        z=20.0,
        object_ids=(1, 2),
        models=(r"o\road\sil6.p3d", r"o\road\sil6.p3d"),
        message="Early paved wedge classification.",
        candidate_fix="Refit the seam.",
        metrics={},
    )
    result = _core.InspectionResult(
        input_path="dummy.pbo",
        wrp_entry="dummy.wrp",
        road_object_count=2,
        source_junction_count=0,
        issues=(issue,),
        road_objects=(),
    )
    first = SimpleNamespace(object_id=1)
    second = SimpleNamespace(object_id=2)

    monkeypatch.setattr(_final._audit, "_candidate_pairs", lambda _roads: ((first, second),))
    monkeypatch.setattr(_final._grass, "_grass_wedge_geometry", lambda *_args: object())
    monkeypatch.setattr(
        _final._audit,
        "_strictly_covered_by_other_paved_surface",
        lambda *_args: True,
    )
    monkeypatch.setattr(_final._coverage, "_terrain_context", lambda _path: None)
    monkeypatch.setattr(
        _final,
        "_actual_wedge_contains_factory",
        lambda _path: _core._paved_wedge_contains,
    )

    fixed = _final._recheck_existing_grass_wedges(result, Path("dummy.pbo"))

    assert fixed.issues == ()


def test_existing_grass_wedge_stays_when_final_surface_audit_cannot_cover_it(monkeypatch) -> None:
    issue = _core.RoadIssue(
        issue_id="RI-00001",
        severity="high",
        score=70.0,
        category="grass_wedge",
        x=10.0,
        z=20.0,
        object_ids=(1, 2),
        models=(r"o\road\sil6.p3d", r"o\road\sil6.p3d"),
        message="Visible paved wedge.",
        candidate_fix="Refit the seam.",
        metrics={},
    )
    result = _core.InspectionResult(
        input_path="dummy.pbo",
        wrp_entry="dummy.wrp",
        road_object_count=2,
        source_junction_count=0,
        issues=(issue,),
        road_objects=(),
    )
    first = SimpleNamespace(object_id=1)
    second = SimpleNamespace(object_id=2)

    monkeypatch.setattr(_final._audit, "_candidate_pairs", lambda _roads: ((first, second),))
    monkeypatch.setattr(_final._grass, "_grass_wedge_geometry", lambda *_args: object())
    monkeypatch.setattr(
        _final._audit,
        "_strictly_covered_by_other_paved_surface",
        lambda *_args: False,
    )
    monkeypatch.setattr(_final._coverage, "_terrain_context", lambda _path: None)
    monkeypatch.setattr(
        _final,
        "_actual_wedge_contains_factory",
        lambda _path: _core._paved_wedge_contains,
    )

    fixed = _final._recheck_existing_grass_wedges(result, Path("dummy.pbo"))

    assert fixed.issues == (issue,)
