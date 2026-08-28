# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable command entry for the read-only post-build Road Inspector."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import road_inspector as _core
from . import road_inspector_runtime as _runtime
from . import road_inspector_source_context as _source_context
from . import road_inspector_edge_view as _edge_view
from . import road_inspector_report_ui as _report_ui


# Keep every correction confined to the inspector process. Importing the normal
# world generator does not install any of these diagnostics or change road output.
_runtime.install()
_source_context.install()
_edge_view.install()
_report_ui.install()

RoadIssue = _core.RoadIssue
InspectionResult = _core.InspectionResult
write_inspection_report = _core.write_inspection_report


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    return _core.inspect_road_geometry(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return _core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
