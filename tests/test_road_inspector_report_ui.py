from cwr_worldgen import road_inspector_report_ui as report_ui


def test_report_ui_defaults_to_paved_only_finding_focus() -> None:
    script = report_ui._UI_SCRIPT

    assert "Show dirt/mixed findings" in script
    assert "nonPavedFindingIds" in script
    assert "source_surfaces" in script
    assert "dirtSurfaceWords" in script
    assert "isMixedStockFamilyFinding" in script
    assert "families.has('ces')" in script
    assert "Array.from(pavedFamilies).some" in script
    assert "if(surfaces.some(value=>matchesSurfaceWord(value,dirtSurfaceWords))) return true;" in script
    assert "!showDirt&&nonPavedFindingIds.has" in script


def test_report_ui_hides_paved_seams_beside_native_mixed_t_by_default() -> None:
    script = report_ui._UI_SCRIPT

    assert "mixedJunctionRadiusMetres=7.0" in script
    assert "kr_new_(?:sil|asf|kos)_ces_t" in script
    assert "seamCategories.has(issue.category)" in script
    assert "if(nearNativeMixedJunction(issue)) return true;" in script


def test_report_ui_keeps_dirt_and_mixed_findings_available_on_request() -> None:
    script = report_ui._UI_SCRIPT

    assert "dirtCheckbox.addEventListener('change',applySearch)" in script
    assert "Dirt or mixed paved/dirt diagnostic" in script
