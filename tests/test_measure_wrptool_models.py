from __future__ import annotations

from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_wrptool_models as wrptool


def test_house_sections_include_houses_and_old_houses_only() -> None:
    text = """;
; houses
;
data3d\\dum01.p3d=houses
data3d\\kostelik.p3d=houses
;
; old houses
;
Data3D\\AFbarabizna.p3d=houses
Data3D\\kostel_trosky.p3d=houses
;
; bushes
;
data3d\\krovi.p3d=bushes
"""

    refs = wrptool.references_from_ini_sections(
        text,
        (),
        wrptool._HOUSE_SECTIONS,
    )

    assert refs == (
        r"data3d\afbarabizna.p3d",
        r"data3d\dum01.p3d",
        r"data3d\kostel_trosky.p3d",
        r"data3d\kostelik.p3d",
    )


def test_house_sections_still_honor_include_globs() -> None:
    text = """;
; houses
;
data3d\\dum01.p3d=houses
data3d\\kostelik.p3d=houses
;
; old houses
;
Data3D\\AFbarabizna.p3d=houses
"""

    refs = wrptool.references_from_ini_sections(
        text,
        (r"data3d\kost*.p3d",),
        wrptool._HOUSE_SECTIONS,
    )

    assert refs == (r"data3d\kostelik.p3d",)
