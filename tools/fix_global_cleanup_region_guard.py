from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "cwr_worldgen" / "house_style_catalogue.py"
text = path.read_text(encoding="utf-8")
old = '''    if len(profiles) != 24 or numbers != set(range(1, 25)):
        raise RuntimeError("house-style catalogue must contain exactly map regions 1 through 24")
'''
new = '''    # Sweden is represented exclusively by country_styles/SE_Sweden.json and
    # inherits the Northern Europe baseline, so the regional catalogue ends at 23.
    if len(profiles) != 23 or numbers != set(range(1, 24)):
        raise RuntimeError("house-style catalogue must contain exactly map regions 1 through 23")
'''
if text.count(old) != 1:
    raise RuntimeError("regional catalogue guard changed")
path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
print("Updated regional house-style catalogue guard to regions 1 through 23.")
