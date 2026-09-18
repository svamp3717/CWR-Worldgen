from __future__ import annotations

import json
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from p3d_categoriser_app import Classification
from p3d_categoriser_storage import incomplete_state_path, load_state, save_split_state


def test_complete_and_incomplete_reviews_are_saved_separately(tmp_path: Path) -> None:
    output = tmp_path / "catalogue.json"
    state = {
        r"o\hous\complete.p3d": Classification(
            categories=["Residential"],
            placement="Urban",
            reviewed=True,
            width_m=10.25,
            length_m=7.5,
            height_m=8.75,
            aspect_ratio=1.3667,
            origin_to_bottom_m=4.125,
        ),
        r"o\hous\missing-placement.p3d": Classification(
            categories=["Residential"], placement="", reviewed=True
        ),
        r"o\hous\untouched.p3d": Classification(
            categories=[], placement="", reviewed=False
        ),
    }

    complete_count, incomplete_count = save_split_state(
        output,
        categories=["Residential"],
        state=state,
        resume_model_path=r"o\hous\missing-placement.p3d",
    )

    assert complete_count == 1
    assert incomplete_count == 1

    primary = json.loads(output.read_text(encoding="utf-8"))
    companion_path = incomplete_state_path(output)
    companion = json.loads(companion_path.read_text(encoding="utf-8"))

    assert [item["model_path"] for item in primary["models"]] == [r"o\hous\complete.p3d"]
    saved = primary["models"][0]
    assert saved["width_m"] == 10.25
    assert saved["length_m"] == 7.5
    assert saved["height_m"] == 8.75
    assert saved["aspect_ratio"] == 1.3667
    assert saved["origin_to_bottom_m"] == 4.125
    assert [item["model_path"] for item in companion["models"]] == [
        r"o\hous\missing-placement.p3d"
    ]
    assert r"o\hous\untouched.p3d" not in json.dumps(primary)
    assert r"o\hous\untouched.p3d" not in json.dumps(companion)
    assert "failures" not in primary
    assert "failures" not in companion

    loaded, categories = load_state(output)
    assert categories == ["Residential"]
    assert loaded[r"o\hous\complete.p3d"].placement == "Urban"
    assert loaded[r"o\hous\complete.p3d"].width_m == 10.25
    assert loaded[r"o\hous\complete.p3d"].length_m == 7.5
    assert loaded[r"o\hous\complete.p3d"].height_m == 8.75
    assert loaded[r"o\hous\complete.p3d"].aspect_ratio == 1.3667
    assert loaded[r"o\hous\complete.p3d"].origin_to_bottom_m == 4.125
    assert loaded[r"o\hous\missing-placement.p3d"].reviewed is True
    assert r"o\hous\untouched.p3d" not in loaded


def test_completing_a_review_removes_stale_companion_entry(tmp_path: Path) -> None:
    output = tmp_path / "catalogue.json"
    key = r"o\hous\moving.p3d"

    save_split_state(
        output,
        categories=["Residential"],
        state={key: Classification(["Residential"], "", True)},
    )
    assert incomplete_state_path(output).exists()

    save_split_state(
        output,
        categories=["Residential"],
        state={key: Classification(["Residential"], "Both", True)},
    )
    assert not incomplete_state_path(output).exists()
    primary = json.loads(output.read_text(encoding="utf-8"))
    assert [item["model_path"] for item in primary["models"]] == [key]
    assert "failures" not in primary
