from pathlib import Path

path = Path(__file__).resolve().parent / "finish_osm_house_modeler_fidelity.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "    if key.roof_storey:\n        try:\n",
    "    if key.roof_storey and not key.footprint_vertices:\n        try:\n",
    1,
)
text = text.replace(
    "    if interior_storeys >= 2 and not key.roof_storey:\\n        minimum_eave = (\\n",
    "    if interior_storeys >= 2 and not (key.roof_storey and not key.footprint_vertices):\\n        minimum_eave = (\\n",
    1,
)
# Depending on whether this helper is applied before or after the triple-quoted
# replacement was materialized, the second pattern may exist with real newlines.
text = text.replace(
    "    if interior_storeys >= 2 and not key.roof_storey:\n        minimum_eave = (\n",
    "    if interior_storeys >= 2 and not (key.roof_storey and not key.footprint_vertices):\n        minimum_eave = (\n",
    1,
)
if "if key.roof_storey and not key.footprint_vertices:" not in text:
    raise RuntimeError("rectangular roof-storey guard was not applied")
path.write_text(text, encoding="utf-8", newline="\n")
