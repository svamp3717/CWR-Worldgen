"""Compare stock and SFP model data before changing junction rendering."""
from __future__ import annotations

import io
from pathlib import Path

from cwr_worldgen.procedural_buildings import (
    _Face, _Lod, _MLOD_HEADER, _write_lod,
)
from cwr_worldgen.road_surface_comparison import (
    STOCK_PAVED, SFP_STRAIGHT, SFP_CURVE,
    compare_paved_models, summarize_road_model,
)


def _model(
    *,
    point_flag: int = 0x13F,
    face_flag: int = 0x2C102,
    y: float = 0.025,
    width: float = 9.1,
) -> bytes:
    half = width / 2
    visual = _Lod(
        (
            (-half, y, -12.5), (-half, y, 12.5),
            (half, y, 12.5), (half, y, -12.5),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                r"textures\paved.paa",
                ((0, 0, 0., 0.), (1, 0, 0., 1.),
                 (2, 0, 1., 1.), (3, 0, 1., 0.)),
                face_flag,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road")),
        point_flags=(point_flag,) * 4,
    )
    output = io.BytesIO()
    output.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(output, visual)
    return output.getvalue()


def _write_model(root: Path, relative: str, payload: bytes) -> None:
    output = root / relative.replace("\\", "/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def test_model_summary_reports_visual_height_flags_and_winding() -> None:
    summary = summarize_road_model(_model(point_flag=0x13F, face_flag=0x2C102))
    assert summary["format"] == "MLOD"
    assert summary["visual_points"] == 4
    assert summary["visual_faces"] == 1
    assert summary["visual_point_flags"] == {"0x0000013f": 4}
    assert summary["visual_face_flags"] == {"0x0002c102": 1}
    assert summary["visual_height_min_metres"] == 0.025
    assert summary["visual_height_max_metres"] == 0.025
    assert summary["visual_land_flagged_points"] == 0
    assert summary["dominant_visual_style"]["point_flag"] == "0x0000013f"
    assert summary["bounds"]["width_metres"] == 9.1
    assert "roadway_error" in summary


def test_reference_comparison_does_not_assume_sfp_behaves_like_stock(
    tmp_path: Path,
) -> None:
    _write_model(tmp_path, STOCK_PAVED, _model(point_flag=0x13F))
    _write_model(
        tmp_path, SFP_STRAIGHT,
        _model(point_flag=0x140, face_flag=0x52, y=0.040),
    )
    _write_model(
        tmp_path, SFP_CURVE,
        _model(point_flag=0x140, face_flag=0x52, y=0.040, width=10.0),
    )
    result = compare_paved_models((tmp_path,))
    assert result["all_models_readable"] is True
    difference = result["comparisons"][SFP_STRAIGHT]["differences"]
    assert difference["visual_point_flags"]["baseline"] == {
        "0x0000013f": 4,
    }
    assert difference["visual_point_flags"]["candidate"] == {
        "0x00000140": 4,
    }
    assert difference["visual_face_flags"]["candidate"] == {
        "0x00000052": 1,
    }
    assert difference["visual_height_min_metres"]["candidate"] == 0.04
    assert "bounds" in result["comparisons"][SFP_CURVE]["differences"]


def test_missing_sfp_reference_is_explicit_not_faked(tmp_path: Path) -> None:
    _write_model(tmp_path, STOCK_PAVED, _model())
    result = compare_paved_models((tmp_path,))
    assert result["all_models_readable"] is False
    assert result["models"][STOCK_PAVED]["status"] == "ok"
    assert result["models"][SFP_STRAIGHT]["status"] == "missing"
    assert result["models"][SFP_CURVE]["status"] == "missing"
    assert not result["comparisons"]
