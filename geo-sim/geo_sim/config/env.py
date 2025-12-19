import numpy as np
from enum import IntEnum, Enum, auto
SEED = 31919


# Initialization
N_GROUPS = 10
INIT_MAX_VAL = 255
INIT_RESOURCE_GAMMA = np.log(INIT_MAX_VAL)/INIT_MAX_VAL
# That way P(0) will be INIT_MAX_VAL times more likely than INIT_MAX_VAL


# Observations
## Keys for observation dictionary
KEY_OBS_OCCUPANCY = "occupancy"
KEY_OBS_FEATURES_LOCATION  = "features"
KEY_OBS_FEATURES_GROUP  = "group"

# Action
class ActionIdx(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4
ACTION_VALUE_DEFAULT = 1
MASK_RADIUS = 10

# World Dynamics
## Occupancy
MAX_OCCUPANCY_GAIN = 1.50

## Conflict
class ConflictStrategy(Enum):
    TOURNAMENT = auto()
    UNIFORM = auto()