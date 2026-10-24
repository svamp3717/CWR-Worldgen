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


## Measured from user-supplied `SFP_objects(2).pbo` (2026-10-24)

Extracted directly from this specific 29 MiB PBO, without copying or
committing its proprietary P3D or PAA payloads:

| SFP model | First visual LOD | Visual point flag | Visual face flag | Local visual Y (m) | Visual faces |
| --- | ---: | --- | --- | ---: | ---: |
| `roads/sil25.p3d` | 0.5 | `0x00000001` | `0x00000000` | 0.000 | 2 triangles |
| `roads/sil10 25.p3d` | 0.5 | `0x00000001` | `0x00000000` | 0.000 | 4 triangles |
| `roads/sil12.p3d` | 0.5 | `0x00000001` | `0x00000000` | 0.008 | 2 triangles |
| `roads/sil6.p3d` | 0.5 | `0x00000001` | `0x00000000` | 0.000 | 2 triangles |

Each model uses `SFP_objects\\roads\\sil.paa` for visual triangles.
Their Roadway LODs instead use `Data\\asfaltka.paa`, with zero face and
point flags. The straight `sil25` road spans 9.1 m width and 25 m length.
The short `sil12` model has an 8 mm visual plane lift, even though the
other three visual planes are at local origin height zero.

**Important distinction:** The *current generator's built-in stock-style
visual defaults* are `0x0000013F` point flags and `0x0002C102` face
flags. These are **not independently measured from the original stock
`o\\road\\sil25.p3d`**; that P3D was not included in the user upload.
The defaults and the working SFP geometry differ, but the causal property
is not proven. Stock texture opacity/transparency and the game's render
pipeline remain possible confounding variables.

The uploaded PBO was named `SFP_objects(2).pbo`, a common renamed-download
form. The comparison tool now recognizes that filename's suffix as the
`SFP_objects` asset prefix only when matching is unambiguous.

**Next experimental step:** compare real `o\\road\\sil25.p3d` and
`o\\road\\sil10 25.p3d` binaries against these measured SFP models,
then run controlled stock/SFP A/B renders with identical gravel geometry.
Do not infer that reducing the visual flag values alone is sufficient.


## Real stock-road ODOL models from user-supplied `Road(1).zip` (2026-10-24)

The supplied original stock road archive has SHA-256
`2eff7e0d8899f0805609c30fa5d49d3e9083fe382ff16beee5525529521b4eea`.
Its P3Ds are ODOL v7, not synthetic test stand-ins. Direct parsing of
each first drawable LOD produced:

| Model | Visual point flags | Visual face flags | Visual Y | Geometry |
| --- | --- | --- | --- | --- |
| Stock `o\\road\\sil25.p3d` | `0x13F` | `0x24102` | 0 m | 9.10 m x 25 m, 2 triangles |
| SFP `sfp_objects\\roads\\sil25.p3d` | `0x01` | `0x00000` | 0 m | 9.10 m x 25 m, 2 triangles |
| Stock `o\\road\\sil10 25.p3d` | `0x13F` | `0x2C102` | 0 m | 4 triangles |
| SFP `sfp_objects\\roads\\sil10 25.p3d` | `0x01` | `0x00000` | 0 m | 4 triangles |
| Stock `o\\road\\sil12.p3d` | `0x13F` | `0x24102` | 0 m | 2 triangles |
| SFP `sfp_objects\\roads\\sil12.p3d` | `0x01` | `0x00000` | +0.008 m | 2 triangles |
| Stock `o\\road\\sil6.p3d` | `0x13F` | `0x2C102` | 0 m | 2 triangles |
| SFP `sfp_objects\\roads\\sil6.p3d` | `0x01` | `0x00000` | 0 m | 2 triangles |

The first ODOL LOD also has downward normals. The original stock straight
model therefore does **not** sit below the SFP reference, nor is it narrower
or shorter. Those earlier explanations are ruled out for the *model-local*
straight paved geometry. Different world-object elevations are a separate
question, to be controlled in an in-game A/B experiment.

Other observations:

- Stock `asf25.p3d` and `ces25.p3d` also have `0x13F` point flags
  and `0x24102` face flags, and their first visual surface is at 0 m.
- Stock `kr_new_sil_ces_t.p3d` itself contains some dirt-surface vertices
  approximately **6.63 cm higher** than its asphalt-surface vertices. Native
  mixed-junction artwork and the independent paved/gravel issue must not be
  conflated.
- The original stock asphalt texture `o\\road\\sil_new.paa` begins with
  format word `0x8080`; the uploaded SFP `SFP_objects\\roads\\sil.paa`
  starts with `0xE000`. These are raw header values, not a verified claim
  about engine blending or opacity. Both P3D surface flags and texture
  format are independent variables.
- Importantly, the default `_ROAD_SURFACE_FACE_FLAG=0x2C102` in
  `procedural_infrastructure.py` reflects stock `sil6` and stock curved
  models, **not** stock `sil25` (which uses `0x24102`). A single uniform
  "stock face flag" for every variant would be technically inaccurate.

The comparator now also supports the source archive `Road(1).zip` directly:
`Road/sil25.p3d` is recognized as the canonical stock path
`o\\road\\sil25.p3d`, and the existing `SFP_objects(2).pbo` is accepted
as the SFP model namespace. Ambiguous duplicates are explicitly rejected.
No game files are added to git.

**Most useful controlled next step:** hold the same asphalt texture,
same actual P3D vertices and same WRP Y constant, and vary only the paved
P3D first-visual-Lod point/face flags in a separate local test asset.
Then hold the flags fixed and swap only the asphalt texture. Do this
in-game, for *both* straight and curved pieces. Only after one of these
is measured should the production road generator change its defaults.
