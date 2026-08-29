from cwr_worldgen import road_inspector_report_ui as report_ui


def test_report_ui_defaults_to_paved_only_intersection_focus() -> None:
    script = report_ui._UI_SCRIPT

    assert "Show dirt/mixed intersections" in script
    assert "nonPavedIntersectionIds" in script
    assert "source_surfaces" in script
    assert "dirtSurfaceWords" in script
    assert "if(surfaces.some(value=>matchesSurfaceWord(value,dirtSurfaceWords))) return true;" in script
    assert "if(involved.some(road=>String(road.family||'').toLowerCase()==='ces')) return true;" in script
    assert "!showDirt&&nonPavedIntersectionIds.has" in script


def test_report_ui_keeps_dirt_and_mixed_findings_available_on_request() -> None:
    script = report_ui._UI_SCRIPT

    assert "dirtCheckbox.addEventListener('change',applySearch)" in script
    assert "Dirt or mixed paved/dirt intersection diagnostic" in script
