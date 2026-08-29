from cwr_worldgen import road_inspector_report_ui as report_ui


def test_report_ui_defaults_to_paved_intersection_focus() -> None:
    script = report_ui._UI_SCRIPT

    assert "Show dirt intersections" in script
    assert "dirtIntersectionIds" in script
    assert "pavedFamilies=new Set(['sil','asf','kos'])" in script
    assert "source_surfaces" in script
    assert "!showDirt&&dirtIntersectionIds.has" in script


def test_report_ui_keeps_dirt_findings_available_on_request() -> None:
    script = report_ui._UI_SCRIPT

    assert "dirtCheckbox.addEventListener('change',applySearch)" in script
    assert "Dirt-only intersection diagnostic" in script
