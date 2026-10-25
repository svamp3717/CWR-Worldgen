# Large-world Overture building coverage and retries

The Overture downloader processes a large world as independent ~10 km tiles.
One or more failed tiles no longer discard the buildings obtained from all
other tiles. Every remaining tile is attempted even after a failure. The
downloader merges every usable tile into a valid GeoJSON and the build uses
that **partial** Overture layer alongside OpenStreetMap.

## Status and diagnostics

A fetch writes `buildings.geojson.status.json` beside the output (even when
the GeoJSON cannot be produced). The persistent per-release tile cache also
contains `world-<hash>.tiles.json`. Both contain:

- `status`: `complete` (all tiles), `partial` (some tiles), or
  `unavailable` (no usable output, e.g. all downloads or the merge failed)
- `bbox_south_west_north_east`: the overall requested geographic bounds
- `tile_count` and `completed_tile_count`
- `failed_tiles`: each failed tile's 1-based `tile_number`, exact
  `bbox_south_west_north_east`, attempt count, and error text
- `release`, successful tile digests, feature count, and duplicate count

The **frozen source manifest** (`source.json`, often called the source
manifest in the GUI) copies this status and failed-tile coverage. It records
the building GeoJSON path only if a usable output exists. Disabled Overture
appears explicitly as `status: "unavailable"`, `reason: "disabled"`.
Older cached GeoJSONs without a status file are still accepted; they are
marked `complete` with `reason: "legacy cached GeoJSON; full coverage
unverified"` rather than falsely implying a verified tile audit.

The log reports `CWR Overture tile N/M FAILED` with geographic bounds,
followed by `CWR Overture partial: ...` if other tiles were usable.

## Resume

Within the same Overture release, completed tiles live under
`.cwr-worldgen-cache/overture/<release>/`. A retry reuses those files and
fetches only missing/failed tile locations; corrupt cached tile outputs are
quarantined for subsequent retry. This works with `refresh=False` and when
resuming an incomplete `refresh=True` run.

**Frozen source bundles are intentionally immutable:** invoking the normal
build again uses its recorded partial GeoJSON, rather than silently changing
the source data. To retry missing coverage and create a new frozen snapshot,
use the explicit **Refresh sources** option (or `fetch-sources --refresh`).
The source refresh itself is atomic; previously completed Overture tiles
survive outside the staging directory and are reused. If the Overture release
changes between runs, the new release is treated as a different data source
and fetched separately.

The direct Milestone 9 fetch path uses the same Overture downloader and can
also return partial GeoJSON.

## Memory use

Tile merging incrementally decodes **one feature at a time** and writes the
merged GeoJSON to a temporary file before atomically replacing the result.
It retains only the current feature, a bounded input buffer, and a set of
previous building IDs (needed to remove cross-tile duplicates). It does
**not** call `json.load` on complete tile GeoJSONs.

Individual buildings with unusually complex polygons, or enormous ID sets,
still consume memory proportional to that feature or number of unique IDs.
The remote DuckDB extraction still has its own external data / disk / network
constraints. These changes remove the previous Python whole-tile merge
memory spike; they do not guarantee that a remote Overture tile succeeds.

## Validation

`.github/workflows/overture-partial-buildings.yml` validates:

- a 50 km world split into 25 tiles
- one tile failing, while 24 others are downloaded and used
- geographic failure records in both status files and source manifest
- a second run fetching only the missing tile
- a fully unavailable fetch
- a corrupt tile being quarantined without discarding other buildings
- incremental feature reading, cross-tile ID deduplication and atomic output
- frozen source manifest and WKB / DuckDB compatibility

Tests mock network workers. A real 50 km download is still recommended
before relying on the feature for production island snapshots.
