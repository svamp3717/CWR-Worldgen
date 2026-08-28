# SPDX-License-Identifier: GPL-3.0-or-later
"""Add road-strip edges to the Road Inspector HTML overview.

The numeric seam detector remains authoritative. This view makes its findings
easier to inspect by drawing the two road edges around each emitted centreline.
Straight P3Ds are exact in plan view. Native ten-degree curves are sampled along
their measured circular centreline before edge offsets are drawn, so the browser
view no longer disguises a real curve as one straight chord.
"""
from __future__ import annotations

import math

from . import road_inspector as _core
from . import road_inspector_runtime as _runtime
from . import stock_road_model_geometry as _geometry


_ORIGINAL_ROAD_PAYLOAD = None
_ORIGINAL_HTML_DOCUMENT = None
_INSTALLED = False
_CURVE_VIEW_STEPS = 8

_EDGE_CSS = r"""
.road.edge { stroke:#555; stroke-width:.8; opacity:.42; pointer-events:none; }
.road.edge.selected { stroke:#63d6ff; stroke-width:2; opacity:.9; }
"""

_EDGE_SCRIPT = r"""
<script>
(function(){
  if(typeof roads==='undefined'||typeof svg==='undefined'||typeof ns==='undefined') return;
  for(const r of roads){
    const width=Number(r.half_width||0);
    if(!(width>0)) continue;
    for(const s of r.segments){
      const dx=Number(s[2])-Number(s[0]), dz=Number(s[3])-Number(s[1]);
      const length=Math.hypot(dx,dz);
      if(!(length>1e-6)) continue;
      const nx=dz/length, nz=-dx/length;
      for(const side of [-1,1]){
        const line=document.createElementNS(ns,'line');
        line.setAttribute('x1',Number(s[0])+nx*width*side);
        line.setAttribute('y1',-(Number(s[1])+nz*width*side));
        line.setAttribute('x2',Number(s[2])+nx*width*side);
        line.setAttribute('y2',-(Number(s[3])+nz*width*side));
        line.classList.add('road','edge');
        line.dataset.object=r.object_id;
        svg.insertBefore(line,svg.firstChild);
        if(typeof roadEls!=='undefined'){
          if(!roadEls.has(r.object_id)) roadEls.set(r.object_id,[]);
          roadEls.get(r.object_id).push(line);
        }
      }
    }
  }
})();
</script>
"""


def _curve_segments(road):
    curve = _geometry.stock_curve_connectors(str(road.model_path))
    if curve is None:
        return None
    radius = float(curve.radius_metres)
    center = (float(curve.begin[0]) + radius, float(curve.begin[1]))
    points = []
    for step in range(_CURVE_VIEW_STEPS + 1):
        fraction = step / _CURVE_VIEW_STEPS
        angle = math.radians(_geometry.STOCK_CURVE_ANGLE_DEGREES) * fraction
        local = (
            center[0] - radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        points.append(
            _runtime._world_point(
                local,
                (float(road.x), float(road.z)),
                float(road.heading_degrees),
                float(road.pitch_degrees),
            )
        )
    return [
        [
            round(start[0], 4),
            round(start[1], 4),
            round(end[0], 4),
            round(end[1], 4),
        ]
        for start, end in zip(points, points[1:])
    ]


def _road_payload(road):
    if _ORIGINAL_ROAD_PAYLOAD is None:
        raise RuntimeError("Road Inspector edge view is not installed")
    payload = _ORIGINAL_ROAD_PAYLOAD(road)
    payload["half_width"] = float(
        _geometry.STOCK_HALF_WIDTHS_METRES.get(road.family, 0.0)
    )
    if road.kind == "curve":
        segments = _curve_segments(road)
        if segments:
            payload["segments"] = segments
    return payload


def _html_document(result) -> str:
    if _ORIGINAL_HTML_DOCUMENT is None:
        raise RuntimeError("Road Inspector edge view is not installed")
    document = _ORIGINAL_HTML_DOCUMENT(result)
    if "</style>" in document:
        document = document.replace("</style>", _EDGE_CSS + "\n</style>", 1)
    if "</body>" in document:
        document = document.replace("</body>", _EDGE_SCRIPT + "\n</body>", 1)
    return document


def install() -> None:
    global _ORIGINAL_ROAD_PAYLOAD, _ORIGINAL_HTML_DOCUMENT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_ROAD_PAYLOAD = _core._road_payload
    _ORIGINAL_HTML_DOCUMENT = _core._html_document
    _core._road_payload = _road_payload
    _core._html_document = _html_document
    _INSTALLED = True
