import numpy as np
SEED = 31919

# Initialization
INIT_MAX_VAL = 255
INIT_RESOURCE_GAMMA = np.log(INIT_MAX_VAL)/INIT_MAX_VAL
# That way P(0) will be INIT_MAX_VAL times more likely than INIT_MAX_VAL

# Observations
## Keys for observation dictionary
KEY_OBS_OCCUPANCY = "occupancy"
KEY_OBS_FEATURES_LOCATION  = "features"
KEY_OBS_FEATURES_GROUP  = "group"

# Action
ACTION_VALUE_DEFAULT = 1
MASK_RADIUS = 10

# World Dynamics
## Occupancy
MAX_OCCUPANCY_GAIN = 2.00