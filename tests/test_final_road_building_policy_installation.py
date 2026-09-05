from __future__ import annotations

from cwr_worldgen import final_building_road_clearance_policy as clearance
from cwr_worldgen import final_road_building_audit_policy as audit
from cwr_worldgen import road_building_priority_policy as priority


def test_step_four_audit_is_the_live_final_conflict_resolver() -> None:
    assert clearance.resolve_final_building_road_conflicts is audit.audit_final_road_building_conflicts
    assert priority.resolve_road_building_priorities is audit.audit_final_road_building_conflicts
    assert audit._ORIGINAL_RESOLVE is not None
    assert priority._ORIGINAL_RESOLVE is not None
