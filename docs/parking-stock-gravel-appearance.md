# Compacted dirt-and-gravel parking (natural ground)

Worldgen's stock gravel donor is `o\\road\\asf25.p3d`, but **unpaved parking
does not reuse that road's texture**. This update responds to the in-game
report that the previous high-contrast, monochrome, hash-based gravel looked
like camouflage rather than worn ground.

## Appearance

Unpaved parking now resembles a levelled roadside turnout or a compacted
dirt/gravel mix:

- **Base colour:** muted, dusty earth grey-brown near RGB **(78, 76, 70)**,
  informed by the neighbouring dark stock gravel, while keeping a natural
  earth tint and lower contrast than the previous version.
- **Surface variation:** two scales of interpolated, world-coordinate ground
  noise. Smoothstep blending across hash lattice boundaries eliminates hard
  rectangular camouflage-like patches.
- **Aggregate:** faint, sparse fine-gravel specks. No road-photo imagery or
  repeated source bitmap is used.
- **Wear:** shallow tyre ruts and faint parking circulation compaction with
  reduced contrast. Paved parking appearance and painted lines do not change.
- **Continuity:** the procedural material is generated from world X/Z,
  preserving coherence across adjacent WRP terrain texture cells.

No game road model, `asf_new.paa`, donor texture image, or external photo is
read when generating the parking material.

## Selection and preservation

Explicit OSM `surface=gravel`, `dirt`, `unpaved`, `earth`, `ground`,
`sand`, `compacted` and similar tags remain non-paved. Untagged parking
continues to infer a surface from the nearest supported road. Paved parking,
runway and sports-field priority, and non-parking terrain are unchanged.

## Rebuilding and validation

`PARKING_TEXTURE_CACHE_SCHEMA` is bumped to 8, with cache namespace
`parking-cell-v8`, and the recipe is now
`procedural-asf25-colour-compacted-dirt-gravel-v3`. Rebuild the world PBO
to replace cached lighter parking PAAs with the darker hardpack version.

The existing `parking-lot-textures.json` report describes the procedural
material and notes that **no stock road texture was used**. Its
`gravel_reference_rgb` field still records the actual stock road's
(74, 74, 74) colour reference; the new parking **base** is warmer as above.

Tests validate the muted channel palette, small intensity range, smooth
interpolation, deterministic world-space texture, lack of stock image
dependencies, preserved paved lots and existing parking-cell caching.
CWA still needs in-game visual confirmation after regeneration.
