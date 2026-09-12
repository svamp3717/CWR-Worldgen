**CWR Worldgen**

A tool that generates playable Operation Flashpoint / Cold War Assault islands from real-world OpenStreetMap + elevation data, it also uses Overture as fallback for missing buildings in OSM.

It can generate terrain, roads, gravel roads, bridges, buildings, forests, beaches, parks, water and vegetation, then package everything into a CWA/OFP pbo file into a modfolder for you.

The goal is to turn real places into recognizable, playable OFP maps with as little manual work as possible.

## Modding JSON in packaged builds

PyInstaller distributions include an editable `config` directory beside the executable. Runtime JSON is grouped exactly like the source package:

- `config/data` contains catalogues such as road types, stock buildings and worship styles.
- `config/house_styles` contains regional house-style profiles.
- `config/country_styles` contains country profiles and their index.

The external files are loaded at application startup and take precedence over the copies bundled inside PyInstaller. Restart CWR Worldgen after editing them. Files may also be added or removed from these directories; each external directory is treated as the authoritative JSON set when it contains JSON files.

For custom launchers or portable setups, set `CWR_WORLDGEN_CONFIG_DIR` to an alternate directory containing the same `data`, `house_styles` and `country_styles` layout.
