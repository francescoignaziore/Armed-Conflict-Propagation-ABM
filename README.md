# Geo-Sim
Geo-Sim leverages **geo**graphic features to **sim**ulate the establishment and diffusion of conflicts among armed groups in the Democratic Republic of Congo.

We model groups as **agents** whose objective is to **maximize control** over resources.
Drawing only on this assumption, we develop a family of simulation environments with configurable dynamics and demonstrate that complex diffusion dynamics arise across them. 

# Setup
Geo-Sim contains a small Python package plus a command-line interface for 
1. aligning geospatial rasters, 
2. rasterizing OpenStreetMap-derived vector features, and 
3. executing simulations based on these features.


## Installation
```bash
python -m venv .venv
source .venv/bin/activate
cd geo-sim
pip install -e .
# install extra spatial deps that are not declared in pyproject
pip install -r requirements.txt
```

## Downloading geo data
1. Download the data.zip from [polybox](https://polybox.ethz.ch/index.php/s/PP9CFEXdJ3Q5LG9)
2. In Ubuntu:
```bash
unzip data.zip -d ~/css/Armed-Conflict-Propagation-ABM/data/
```
## Repository overview
See [here](#repository-overview-1).

## Command-line overview
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
| `landuse-to-tiff` | Rasterize land-use polygons to the shared grid. | `SHP_LANDUSE_PATH`, aligned reference raster. | `landuses_length_m.tif` and PNG previews. |
| `sample` | Stack the rasters selected by `GEO_FEATURES_DISTRIBUTION`, normalize them, and sample cells. | All rasters in `TIFF_OUT_DIR` whose stem matches the distribution list. | Plots in `data/output/tiffs/simulation_plots/` and printed sample statistics. |
| `sim-resource-conflict` | Execute the purely resource-driven simulation. | - | Images in `data/output/resource_conflict_state/` visualize the environment state for each time step. |

### Typical workflow

1. Align the base rasters: `geo-sim tiff-alignment`.
2. Convert each vector layer to raster form (`roads-to-tiff`, `natural-to-tiff`, etc.). 
3. Once the directory contains the rasters you care about, update the GEO_FEATURES_DISTRIBUTION with the names of the raster, e.g. pop, viirs, roads ... and call `geo-sim sample --n-samples 2000` to sample coordinates from a distribution that is proportional to the raster values.
4. Execute the simulations
    - **Baseline simulation:** Execute `run_many.py` in the top-level folder.
    - **Purely resource-driven simulation:** Run `geo-sim sim-resource-conflict` to execute an extended simulation that integrates resource maximization into more aspects of the world dynamics.

# Extending the codebase


## Repository overview
```
geo_sim
├── cli
│   ├── __init__.py
│   ├── app.py
│   ├── buildings.py
│   ├── env.py                              # Resource-driven simulation: Agent decision-making
│   ├── landuse.py
│   ├── natural.py
│   ├── roads.py
│   ├── sim.py                              # Sampling from geofeatures
│   ├── sim_env.py                          # Resource-driven simulation: World dynamics
│   ├── spatial_accumulation.py
│   ├── tiff_alignment.py
│   └── water.py
└── config
    ├── consts.py
    ├── env.py                              # Resource-driven simulation: Configurable hyperparameters for simulation
    ├── features.py                         # Resource-driven simulation: Configure resource features that the simulation is based upon
    └── paths.py                            
```

## Configuration

- Edit `geo_sim/config/paths.py` when you want to:
  - Change the location of the repository (`ROOT`) and data folder (`DATA_DIR`).
- Edit `geo_sim/config/consts.py` when you need to:
  - Control which rasters the sampling distribution should account for via `GEO_FEATURES_DISTRIBUTION` (default is just 'viirs', that is night lights.)
