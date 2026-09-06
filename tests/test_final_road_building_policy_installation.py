from __future__ import annotations

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import final_road_building_audit_policy as audit
from cwr_worldgen import physical_road_overlap_policy as physical
from cwr_worldgen import road_building_priority_policy as priority


def test_physical_overlap_policy_is_the_live_final_conflict_resolver() -> None:
    assert clearance.resolve_final_building_road_conflicts is physical._physical_step4_resolve
    assert priority.resolve_road_building_priorities is physical._physical_step4_resolve
    assert audit.audit_final_road_building_conflicts is physical._physical_step4_resolve
    assert priority._ORIGINAL_RESOLVE is physical._preferred_step2_resolve
    assert audit._ORIGINAL_RESOLVE is physical._physical_step3_resolve
    assert physical._STEP2_RESOLVE is not None
    assert physical._STEP3_RESOLVE is not None
    assert physical._STEP4_RESOLVE is not None
