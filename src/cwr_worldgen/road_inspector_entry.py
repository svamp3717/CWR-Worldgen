# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable command entry for the read-only post-build Road Inspector."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import road_inspector as _core
from . import road_inspector_runtime as _runtime
from . import road_inspector_intersection_context as _intersection_context
from . import road_inspector_source_context as _source_context
from . import road_inspector_edge_view as _edge_view
from . import road_inspector_report_ui as _report_ui
from . import road_inspector_coordinate_log as _coordinate_log
from . import road_inspector_surface_coverage as _surface_coverage
from . import road_inspector_surface_height as _surface_height
from . import road_inspector_overlap_filter as _overlap_filter
from . import road_inspector_grass_wedge as _grass_wedge
from . import road_inspector_paved_wedge_audit as _paved_wedge_audit
from . import road_inspector_kodiak_overlap as _kodiak_overlap
from . import road_inspector_embedded_paved_geometry as _embedded_paved_geometry
from . import road_inspector_wrptool_catalogue as _wrptool_catalogue
from . import road_inspector_native_junction_overlap as _native_junction_overlap


# Keep every correction confined to the inspector process. Importing the normal
# world generator does not install any of these diagnostics or change road output.
_runtime.install()
_intersection_context.install()
_source_context.install()
_edge_view.install()
_report_ui.install()
_coordinate_log.install()
# First prove genuinely covered paved seams invisible, then classify surviving
# physical outside wedges. Only after that may the generic overlap filter remove
# ordinary connector diagnostics; it deliberately does not suppress grass_wedge.
_surface_coverage.install()
_surface_height.install()
_grass_wedge.install()
_overlap_filter.install()
# Scan paved physical endpoints directly so shallow visible outside triangles
# are reported even when the ordinary seam thresholds emitted no base issue.
_paved_wedge_audit.install()
# Kodiak demonstrates that the intended stock pieces may legitimately overlap by
# about half a metre. Inspect that wider range and count the two real stock road
# surfaces as cover when they physically hide the outside triangle.
_kodiak_overlap.install()
# Existing PBOs may still contain the retired generated paved helpers. Their
# filename is not a geometry version, so use the actual embedded Visual LOD when
# deciding whether such an old object really covers a wedge. Fresh worlds no
# longer serialize these helpers at all.
_embedded_paved_geometry.install()
# Finish with the same Resistance road catalogue used by generation. This layer
# recommends the exact WrpTool-listed T/X P3D when a generic or wrong stock cap
# survives at a source junction, and it never guesses unmeasured connector data.
_wrptool_catalogue.install()
# A correct native T/X must also be the only road surface crossing its logical
# centre. Report any ordinary stock road that still penetrates the measured
# connector footprint, which is the stacked-road failure visible in game.
_native_junction_overlap.install()

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
