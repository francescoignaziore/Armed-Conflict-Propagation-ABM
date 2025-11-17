# Geo-Sim

Geo-Sim contains a small Python package plus command-line interface for aligning geospatial rasters, rasterizing OpenStreetMap-derived vector features, and sampling from the resulting feature stack for simulation studies over the Democratic Republic of Congo.

## Installation

```bash
cd geo-sim
python -m venv .venv
source .venv/bin/activate
pip install -e .
# install extra spatial deps that are not declared in pyproject
pip install geopandas shapely pyproj affine xarray xrspatial
```

## Configuration

- Edit `geo_sim/config/paths.py` when you want to:
  - Change the location of the repository (`ROOT`) and data folder (`DATA_DIR`).
- Edit `geo_sim/config/consts.py` when you need to:
  - Control which rasters the sampling distribution should account for via `GEO_FEATURES_DISTRIBUTION` (default is just 'viirs', that is night lights.)

## CLI overview

The CLI is powered by [Typer](https://typer.tiangolo.com/); general usage is:

```bash
geo-sim [COMMAND] [OPTIONS]
geo-sim --help           # top-level help
geo-sim COMMAND --help   # per-command help (where applicable)
```

Quick reference of the available commands, you can run them in the order:

| Command | Purpose | Key inputs | Outputs |
| --- | --- | --- | --- |
| `tiff-alignment` | Reproject and snap a list of rasters to a common equal-area grid and create quicklook PNGs. | `TIFF_PATHS`, `OUTPUT_CRS_NIGHT_LIGHTS`, `OUTPUT_AREA_CELL` | `_aligned_XXXm.tif` files + PNGs under `data/output/tiffs/pngs/`. |
| `roads-to-tiff` | Rasterize the configured road network into per-cell length estimates. | `SHP_ROAD_PATH`, `ALIGNED_TIFF_NIGHT_LIFE_OUT` | `roads_length_m.tif`, vector/raster previews. |
| `natural-to-tiff` | Rasterize natural features (forests, reserves, etc.). | `SHP_NATURAL_PATH`, aligned reference raster. | `natural_feats.tif` and PNG previews. |
| `waters-to-tiff` | Rasterize water bodies to the shared grid. | `SHP_WATER_PATH`, aligned reference raster. | `waters_length_m.tif` and PNG previews. |
| `landuse-to-tiff` | Rasterize land-use polygons using `xrspatial` and convert coverage to pseudo-length. | `SHP_LANDUSE_PATH`, aligned reference raster. | `landuses_length_m.tif` and PNG previews. |
| `run-simulation` | Stack the rasters selected by `GEO_FEATURES_DISTRIBUTION`, normalize them, and sample cells. | All rasters in `TIFF_OUT_DIR` whose stem matches the distribution list. | Plots in `data/output/tiffs/simulation_plots/` and printed sample statistics. |

### Typical workflow

1. Align the base rasters: `geo-sim tiff-alignment`.
2. Convert each vector layer to raster form (`roads-to-tiff`, `natural-to-tiff`, etc.). 'building-to-tiff' stalls as of now, so do not run it. 
3. Once the directory contains the rasters you care about, update the GEO_FEATURES_DISTRIBUTION with the names of the raster, e.g. pop, viirs, roads ... and call `geo-sim run-simulation --n-samples 2000` (or any other count) to build sampling distributions that are proportional to the raster values.
