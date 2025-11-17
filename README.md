# Geo-Sim

Geo-Sim contains a small Python package plus command-line interface for aligning geospatial rasters, rasterizing OpenStreetMap-derived vector features, and sampling from the resulting feature stack for simulation studies (the current configuration targets the Democratic Republic of Congo, but the tooling is generic after re-pointing the paths).

## Repository layout

| Path | Description |
| --- | --- |
| `geo-sim/` | Installable Python package (`typer` CLI lives under `geo_sim/cli`). |
| `data/input/` | Raw inputs (night lights, population rasters, OSM shapefiles, etc.). |
| `data/output/` | Products written by the CLI (`tiffs/` plus `pngs/` quicklooks & simulation plots). |
| `old/`, `download_hrsl_cod.py`, `try_script.py` | Scratch scripts that helped assemble the current pipeline. |

## Requirements

- Python 3.9+ with a working C toolchain.
- GDAL/PROJ/GEOS available on the system (they are required by `rasterio` and `geopandas`).
- Python dependencies from `geo-sim/pyproject.toml` (`typer`, `rich`, `numpy`, `matplotlib`, `rasterio`).
- Additional spatial packages that the CLI imports but are not pinned in `pyproject.toml`: `geopandas`, `shapely`, `pyproj`, `affine`, `xarray`, and `xrspatial`.
- Large raster/shape inputs under `data/input/` (see the configuration section below for the expected paths).

## Installation

```bash
cd geo-sim
python -m venv .venv
source .venv/bin/activate
pip install -e .
# install extra spatial deps that are not declared in pyproject
pip install geopandas shapely pyproj affine xarray xrspatial
```

Once the package is installed, the `geo-sim` console script becomes available everywhere in the environment.

## Configuration

All file-system and CRS choices are centralized so that the CLI commands stay argument-free in the happy path.

- Edit `geo_sim/config/paths.py` when you want to:
  - Change the location of the repository (`ROOT`), data folder (`DATA_DIR`), and where aligned rasters get written (`TIFF_OUT_DIR`).
  - Point `NIGHT_LIGHTS_TIF`, `POPULATION_TIF`, and the `TIFF_PATHS` list to the rasters that should be aligned.
  - Point each `SHP_*_PATH` constant at the desired OpenStreetMap layers (roads, buildings, natural, water, landuse).
  - Update `ALIGNED_TIFF_NIGHT_LIFE_OUT` if you change the naming convention of the raster alignment outputs; most vector-to-raster commands use this file as their reference grid.
- Edit `geo_sim/config/consts.py` when you need to:
  - Choose the target CRS (`OUTPUT_CRS_NIGHT_LIGHTS`, default EPSG:3857) and desired per-cell area (`OUTPUT_AREA_CELL`, default 1 km²) for the alignment step.
  - Tune `_SUPER_K` (supersampling factor), `_GRID_STEP` (grid overlay spacing for PNG quicklooks), and `_ALL_TOUCHED` (forwarded to `rasterize`).
  - Control which rasters the simulator consumes via `GEO_FEATURES_DISTRIBUTION` (only files whose stem contains any of these tokens are stacked).

Update those two files before running the CLI the first time; most commands will abort early with a clear error if the configured inputs are missing.

## CLI overview

The CLI is powered by [Typer](https://typer.tiangolo.com/); general usage is:

```bash
geo-sim [COMMAND] [OPTIONS]
geo-sim --help           # top-level help
geo-sim COMMAND --help   # per-command help (where applicable)
```

Quick reference of the available commands:

| Command | Purpose | Key inputs | Outputs |
| --- | --- | --- | --- |
| `tiff-alignment` | Reproject and snap a list of rasters to a common equal-area grid and create quicklook PNGs. | `TIFF_PATHS`, `OUTPUT_CRS_NIGHT_LIGHTS`, `OUTPUT_AREA_CELL` | `_aligned_XXXm.tif` files + PNGs under `data/output/tiffs/pngs/`. |
| `roads-to-tiff` | Rasterize the configured road network into per-cell length estimates. | `SHP_ROAD_PATH`, `ALIGNED_TIFF_NIGHT_LIFE_OUT` | `roads_length_m.tif`, vector/raster previews. |
| `buildings-to-tiff` | Buffer/rasterize building footprints to highlight density. | `SHP_BUILDINGS_PATH`, aligned reference raster. | `buildings.tif` and PNG previews. |
| `natural-to-tiff` | Rasterize natural features (forests, reserves, etc.). | `SHP_NATURAL_PATH`, aligned reference raster. | `natural_feats.tif` and PNG previews. |
| `waters-to-tiff` | Rasterize water bodies to the shared grid. | `SHP_WATER_PATH`, aligned reference raster. | `waters_length_m.tif` and PNG previews. |
| `landuse-to-tiff` | Rasterize land-use polygons using `xrspatial` and convert coverage to pseudo-length. | `SHP_LANDUSE_PATH`, aligned reference raster. | `landuses_length_m.tif` and PNG previews. |
| `run-simulation` | Stack the rasters selected by `GEO_FEATURES_DISTRIBUTION`, normalize them, and sample cells. | All rasters in `TIFF_OUT_DIR` whose stem matches the distribution list. | Plots in `data/output/tiffs/simulation_plots/` and printed sample statistics. |

### Typical workflow

1. Align the base rasters: `geo-sim tiff-alignment`.
2. Convert each vector layer to raster form (`roads-to-tiff`, `buildings-to-tiff`, etc.). Each command will write a GeoTIFF that already matches the aligned grid as well as vector/raster PNG previews for QC.
3. Once the directory contains the rasters you care about, call `geo-sim run-simulation --n-samples 2000` (or any other count) to build sampling distributions.

## Command details

### `geo-sim tiff-alignment`

Aligns every raster in `TIFF_PATHS` to a single projected grid:

- **Inputs:** List of source rasters (`TIFF_PATHS`), target CRS (`OUTPUT_CRS_NIGHT_LIGHTS`), and target cell area (`OUTPUT_AREA_CELL`).
- **Behavior:** Validates that every raster shares the same source CRS, computes a union extent in the target CRS, snaps the grid via `rasterio.aligned_target`, reprojects every layer with `Resampling.average`, applies any provided scale/offset metadata, and logs descriptive stats.
- **Outputs:** For each input raster a new GeoTIFF named `*_aligned_<resolution>m.tif` in `TIFF_OUT_DIR` plus a PNG quicklook with the configured `_GRID_STEP`. The command also prints summary stats per input and per output grid.
- **Example:** `geo-sim tiff-alignment`

### `geo-sim roads-to-tiff`

Creates a raster of estimated road length per coarse cell:

- **Inputs:** `SHP_ROAD_PATH` (roads shapefile) and `ALIGNED_TIFF_NIGHT_LIFE_OUT` (reference raster). Only features with `fclass` in `primary/secondary/tertiary/service` are kept. `_SUPER_K` controls the supersampling factor when buffering.
- **Behavior:** Clips roads to the reference extent, reprojects them to the aligned CRS, buffers individual segments so they retain coverage after rasterization, and collapses the supersampled raster into average coverage which is finally converted to length per cell.
- **Outputs:** `roads_length_m.tif` plus `pngs/roads_vector_preview.png` and `pngs/roads_length_m.png`.
- **Example:** `geo-sim roads-to-tiff`

### `geo-sim buildings-to-tiff`

- Processes `SHP_BUILDINGS_PATH` through the same supersampled rasterization pipeline (buffer ➜ rasterize ➜ aggregate) to produce a per-cell proxy for building density.
- Outputs `buildings.tif` and matching PNG previews under `data/output/tiffs/`.
- You can tweak `_SUPER_K` and `_GRID_STEP` in `consts.py` to balance runtime vs. fidelity.

### `geo-sim natural-to-tiff`

- Clips and rasterizes the natural-features shapefile (`SHP_NATURAL_PATH`) into `natural_feats.tif`.
- Generates PNG previews named `natural_vector_preview.png` and `natural_feats.png`.
- **Note:** The function currently contains a `pdb.set_trace()` call for debugging; expect the command to pause in the debugger unless you remove/comment that line.

### `geo-sim waters-to-tiff`

- Rasterizes the configured water-body layer into `waters_length_m.tif`, using the same buffering/supersampling approach as the roads command.
- Produces `waters_vector_preview.png` and `waters_length_m.png` quicklooks.
- Intended for river/lake density derivatives that can be stacked with the other features.

### `geo-sim landuse-to-tiff`

- Rasterizes polygons from `SHP_LANDUSE_PATH` using `xrspatial.rasterize`, accumulates the covered area per pixel, and converts it to a pseudo-length metric (`area / stroke_width`) so that it shares units with the other per-cell features.
- Writes `landuses_length_m.tif` and PNG quicklooks (`landuses_vector_preview.png`, `landuses_length_m.png`).
- Requires `xarray` and `xrspatial` in the active environment.

### `geo-sim run-simulation`

- **Options:** `--n-samples / -n` (Typer exposes the `N_samples` parameter) controls how many cells are drawn; default is 1000.
- **Inputs:** Every GeoTIFF inside `TIFF_OUT_DIR` whose filename stem contains any token from `GEO_FEATURES_DISTRIBUTION`. Before sampling, each raster is normalized to `[0, 1]` with NaNs preserved as nodata.
- **Behavior:** Stacks the rasters into an `(M, H, W)` tensor, builds two sampling schemes (exponential weighting vs. uniform over valid cells), and saves overlays/heatmaps into `data/output/tiffs/simulation_plots/`.
- **Outputs:** `samples_overlay_exp.png`, `samples_overlay_uniform.png`, `samples_overlay_exp_heatmap.png`, plus per-layer debug PNGs saved by `_plot_array`.
- **Example:** `geo-sim run-simulation --n-samples 5000`

## Troubleshooting & tips

- If any command fails with “path not found”, confirm that `geo_sim/config/paths.py` points to the correct absolute paths; the repo currently ships with machine-specific defaults.
- `rasterio`, `geopandas`, and `xrspatial` depend on GDAL/PROJ; make sure the system libraries are reachable before installing Python wheels.
- Commands emit PNG quicklooks in `data/output/tiffs/pngs/` — they are invaluable for spot-checking alignment issues without opening a GIS.
- `natural-to-tiff` will intentionally halt through `pdb.set_trace()`; type `c` to continue, or remove the breakpoint when you no longer need it.

With the README in place you should be able to install the package, wire up datasets, and run any of the CLI commands end-to-end.
