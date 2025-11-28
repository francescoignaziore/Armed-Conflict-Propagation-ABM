# geo_sim/config/consts.py
# Target projected CRS for night lights (meters)
OUTPUT_CRS_NIGHT_LIGHTS = "EPSG:3857"  # or your preferred equal-area CRS

# Desired per-cell area in m² (e.g., 1 km²)
OUTPUT_AREA_CELL = 1_000_000.0


GEO_FEATURES_DISTRIBUTION = ["viirs"]

# Plotting
# plotting constants
_DPI = 220  # higher DPI for crisp grids
_GRID_STEP = 0  # draw a line at every coarse cell (set 0 to disable)


# --- tuneables (shp files) ---
_SUPER_K = 1  # supersampling factor
_ALL_TOUCHED = True  # conservative coverage
