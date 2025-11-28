from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent # geo-sim/geo_sim/config

ROOT_PACKAGE = CURRENT_DIR.parents[1] # geo-sim
ROOT_REPO = CURRENT_DIR.parents[2] # git repo

ROOT = ROOT_PACKAGE
DATA_DIR = ROOT_REPO / "data" 

INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
NIGHT_LIGHTS_OUT_DIR = OUTPUT_DATA_DIR / "night_lights"

NIGHT_LIGHTS_TIF = INPUT_DATA_DIR / "COD_viirs_annual_2012.tif"
POPULATION_TIF = INPUT_DATA_DIR / "cod_pop_2021_CN_1km_R2025A_UA_v1.tif"
DRC_SHAPEFILES = INPUT_DATA_DIR / "DRC_shapefiles"
TIFF_PATHS = [NIGHT_LIGHTS_TIF, POPULATION_TIF]
TIFF_OUT_DIR = OUTPUT_DATA_DIR / "tiffs"


SHP_ROAD_PATH = DRC_SHAPEFILES / "gis_osm_roads_free_1.shp"
SHP_BUILDINGS_PATH = DRC_SHAPEFILES / "gis_osm_buildings_a_free_1.shp"
SHP_NATURAL_PATH = DRC_SHAPEFILES / "gis_osm_natural_a_free_1.shp"
SHP_WATER_PATH = DRC_SHAPEFILES / "gis_osm_water_a_free_1.shp"
SHP_LANDUSE_PATH = DRC_SHAPEFILES / "gis_osm_landuse_a_free_1.shp"


ALIGNED_TIFF_NIGHT_LIFE_OUT = TIFF_OUT_DIR / "COD_viirs_annual_2012_aligned_1000m.tif"
