from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_p3d_models as measure


def test_measurement_accepts_numpy_vertex_array() -> None:
    points = np.array(
        [
            [-2.0, -1.5, -4.0],
            [3.0, 2.5, 6.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    result = measure._measurement(
        model_path=r"o\hous\example.p3d",
        source="fixture",
        format_name="ODOL",
        version=7,
        lod="first",
        points=points,
    )

    assert result.vertex_count == 3
    assert result.width_m == pytest.approx(5.0)
    assert result.height_m == pytest.approx(4.0)
    assert result.length_m == pytest.approx(10.0)
    assert result.origin_to_bottom_m == pytest.approx(1.5)
