# Generated roads: donor groundtype and contact verification

The visible asphalt, gravel, and dirt textures are **not** a reliable proxy for
the surface a vehicle or pedestrian contacts. Generated P3Ds have a separate
**Roadway LOD**. Its face texture is used for surface/environment selection in
OFP/CWA, and its contact geometry must actually be above the terrain where
vehicles travel. The donor model's Roadway texture, point flags, and face flags
are independent of its visible texture and rendering flags.

## What the generator guarantees

- When the configured donor P3D exposes a readable Roadway LOD, Worldgen
  preserves the selected Roadway face texture (including an intentionally
  blank texture), point flags, and face flags on generated road variants and
  junctions. Failure to parse the donor's visual LOD does not discard its
  independently readable Roadway LOD.
- After writing or restoring a generated road P3D, the asset task **reads the
  serialized Roadway LOD back** and checks **every contact polygon's texture,
  face flag, and referenced point flags**, not merely the most common face.
  A mismatch fails the build rather than silently shipping a partial off-road
  patch. The infrastructure manifest records `roadway_contact_verified`,
  `roadway_faces_verified`, and the written values.
- If no donor Roadway LOD could be inspected, existing visual-texture fallback
  behavior is retained for compatibility, but the donor report identifies it
  explicitly as `visual-fallback` rather than calling it verified.
- When no gravel donor is configured, generated gravel intentionally reuses
  the dirt donor's contact style; `road-donors.json` records
  `roadway_source_format: borrowed-dirt-donor`. That is not native gravel
  groundtype inheritance.

## Check a *packed* world PBO

Run from a source checkout (with Worldgen dependencies installed):

```sh
python tools/audit_generated_road_groundtypes.py path/to/wg_mymap.pbo
```

The report counts generated paved, dirt and gravel P3Ds and checks **every
usable Roadway polygon** in the packed models against the embedded
`i/road-donors.json`, including its referenced point flags. A nonzero exit
indicates mismatches; provenance warnings
identify surfaces whose style was borrowed or inferred rather than inspected
from a readable donor. Checking the packed PBO also catches stale cached
models or any difference between model generation and packaging.

**Important:** a successful match verifies the P3D's *encoded contact
metadata*. It does **not** prove that the OFP/CWA engine uses that Roadway LOD
while a vehicle crosses it. If the contact plane is buried under the terrain,
the vehicle can interact with the terrain underneath instead. Verify
in-game vehicle handling and sounds using the same car on a stock donor road
and a generated road. For a definitive byte-level donor comparison, supply
the original donor P3D (e.g. from `O.pbo`) as well as the generated PBO.

## Lundby evidence

The supplied `wg_lundby5.pbo` already contains generated paved contact faces
with `o\\road\\sil_new.paa`, and generated dirt faces with
`o\\road\\ces_hned.paa`, matching the **styles recorded by its generator**.
A separate scan of that PBO found 448 generated-road P3D variants matching
those reported per-surface textures and flags. However, a geometry/terrain
audit found that **501 of 1,373 placed paved pieces** have sampled Roadway
points or cross-section midpoints over 5 mm below the bilinearly interpolated
WRP terrain. That is a plausible *runtime ground-contact* issue distinct from
metadata inheritance. The actual game triangulation may differ from bilinear
interpolation, so this number is a diagnostic estimate, not proof of a specific
wheel-contact failure. Correcting it requires terrain/road-fit or per-object
grounding work, not further copying of the same donor flags.

Neither these donor contact checks nor the PBO auditor alter generated road
positions, curve directions, terrain heights, visuals, or post-build cleanup.


## Runtime caveat confirmed on Lundby

The packed contact faces already match their embedded donor styles; this
integrity audit does not reclassify how the CWA engine drives on them. **In
particular, an apparently paved P3D is not proof that the vehicle's wheels
are making contact with its Roadway LOD.** Where the model's contact polygons
are under the terrain, the terrain can be the contacted surface regardless
of the texture on those buried faces. Correcting those locations requires
a separately tested terrain/placement/contact-height change. This work does
not alter roadway height or the road placement/seam fix from merged PR #63.
