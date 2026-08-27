**CWR Worldgen**

A tool that generates playable Operation Flashpoint / Cold War Assault islands from real-world OpenStreetMap + elevation data, it also uses Overture as fallback for missing buildings in OSM.

It can generate terrain, roads, gravel roads, bridges, buildings, forests, beaches, parks, water and vegetation, then package everything into a CWA/OFP pbo file into a modfolder for you.

The goal is to turn real places into recognizable, playable OFP maps with as little manual work as possible.

## Build a frozen source bundle

Install the checkout you want to test, then run the production pipeline directly against a frozen source bundle:

```text
python -m pip install -e .
worldgen build path/to/source --output build/wg_test --name wg_test --display-name "WG Test"
```

`worldgen build` always targets the current production generation pipeline, so branch fixes can be tested on real source data without relying on RoadLab. `--source path/to/source` and the existing `--source-dir path/to/source` spelling are also accepted.

For all production options, including deployment and surface/forest controls:

```text
worldgen build --help
```

The legacy/internal spelling remains available as `cwr-worldgen milestone9 ...`.
