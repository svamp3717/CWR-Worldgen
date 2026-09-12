from __future__ import annotations

from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_wrptool_models as wrptool


def test_house_category_includes_assignments_anywhere_in_ini() -> None:
    text = """;
; houses
;
data3d\\dum01.p3d=houses
data3d\\kostelik.p3d=houses
;
; old houses
;
Data3D\\AFbarabizna.p3d=houses
;
; bushes
;
data3d\\krovi.p3d=bushes
;
; resistance objects
;
O\\Hous\\domek01.p3d=houses
O\\Hous\\domek02.p3d=houses
O\\Hous\\plot.p3d=fences
o\\misc\\leseni2x.p3d=houses
"""

    refs = wrptool.references_from_ini_categories(text, (), ("houses",))

    assert refs == (
        r"data3d\afbarabizna.p3d",
        r"data3d\dum01.p3d",
        r"data3d\kostelik.p3d",
        r"o\hous\domek01.p3d",
        r"o\hous\domek02.p3d",
        r"o\misc\leseni2x.p3d",
    )


def test_house_category_still_honors_include_globs() -> None:
    text = """data3d\\dum01.p3d=houses
data3d\\kostelik.p3d=houses
O\\Hous\\domek01.p3d=houses
O\\Hous\\domek02.p3d=houses
"""

    refs = wrptool.references_from_ini_categories(
        text,
        (r"o\hous\domek*.p3d",),
        ("houses",),
    )

    assert refs == (
        r"o\hous\domek01.p3d",
        r"o\hous\domek02.p3d",
    )


def test_comment_section_selector_is_still_available() -> None:
    text = """;
; houses
;
data3d\\dum01.p3d=houses
;
; old houses
;
Data3D\\AFbarabizna.p3d=houses
;
; bushes
;
data3d\\krovi.p3d=bushes
"""

    refs = wrptool.references_from_ini_sections(text, (), ("old houses",))

    assert refs == (r"data3d\afbarabizna.p3d",)
