# Mixed paved/gravel road rendering: clean baseline and SFP references

## Verified status

- Restart branch: `feature/stock-road-surface-layering` was force-reset to
  `main` at `7613eafef7db15bca6faaeed429e234f3caa76bc`.
- Earlier experiments were archived as
  `backup/stock-road-surface-layering-before-reset-20261024`. Do not copy
  previous final clipping, miter, banking or height-offset code into this work.
- The in-game report says generated gravel and dirt appear above stock paved
  `o\road\sil*` intersections, but *not* above these SFP paved donors:
  `sfp_objects\roads\sil25.p3d` and
  `sfp_objects\roads\sil10 25.p3d`.
- CWA's actual P3D material/rendering and intersection behavior is the
  unresolved variable. Successful unit tests on guessed WRP heights did *not*
  predict what the game drew.

## First reproducible investigation

Compare each SFP model with its corresponding stock **P3D bytes**. The model
files are third-party assets and are **not** checked into the repository.

From a Python environment with the project installed, run:

```sh
python tools/compare_paved_road_models.py \
  --asset-root "/path/to/stock/addons" \
  --asset-root "/path/to/SFP/addons" \
  --output paved-road-comparison.json
```

The roots may be extracted directories or individual PBO/IFA archives.
The diagnostic looks for these pairs:

| Stock | Known-working SFP reference |
| --- | --- |
| `o\road\sil25.p3d` | `sfp_objects\roads\sil25.p3d` |
| `o\road\sil10 25.p3d` | `sfp_objects\roads\sil10 25.p3d` |

Model path overrides (`--stock`, `--stock-curve`, `--sfp-straight`,
`--sfp-curve`) are available if the installed mod uses another prefix.

The JSON records visual mesh height bounds, visual face winding, source
textures, raw face and point flags, land-clipping flags, model dimensions,
and Roadway texture and flags. Missing/unreadable donors are explicit.
This diagnostic **does not** infer CWA draw precedence or mutate any P3Ds.

## Acceptance test for any subsequent geometry/material fix

1. Preserve gravel/dirt artwork through the paved boundary without a
   visible empty strip before asphalt.
2. Asphalt must be the visible top surface throughout its entire footprint;
   no gravel/dirt paint over it.
3. Test the same crossing with the stock and SFP paved donor, holding terrain
   and gravel objects constant. Previous reproduction coordinate:
   `[3302.524, 3167.709]`.
4. Compare generated WRP object heights and P3D visual vertices separately
   from Roadway contact geometry.
5. Keep checks for straight and curved paved junctions. A passing geometry-only
   unit test is necessary but **not** sufficient proof of CWA rendering.
