from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "cwr_worldgen" / "foundation_visual_policy.py"

needle = '''    from .interior_performance_policy import install_interior_performance_policy\n\n    install_interior_performance_policy()\n\n    # Utility material selection is deliberately data-only: all barn/shed/garage/\n'''
replacement = '''    from .interior_performance_policy import install_interior_performance_policy\n\n    install_interior_performance_policy()\n\n    # Bound exact polygon P3Ds after the final geometry/performance wrappers are\n    # installed. This keeps complex footprint fidelity without letting a hidden\n    # 2048-model side budget dominate the asset-generation stage.\n    from .building_asset_budget_policy import install_building_asset_budget_policy\n\n    install_building_asset_budget_policy()\n\n    # Utility material selection is deliberately data-only: all barn/shed/garage/\n'''

text = PATH.read_text(encoding="utf-8")
if replacement in text:
    raise SystemExit(0)
if needle not in text:
    raise RuntimeError("foundation performance-policy anchor not found")
PATH.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
