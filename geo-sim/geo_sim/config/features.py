from dataclasses import dataclass
from enum import IntEnum, Enum, auto

ABSORPTION_RATE_DEFAULT = 0.02
ABSORPTION_INIT         = 0.02 # What fraction of a cell's resources is already owned by the group at initialization?

class FeatureKey(Enum):
    RESOURCES = auto()
    PEOPLE = auto()
    IS_VALID = auto()
    IS_ALIVE = auto()

class GeoFeatureIdx(IntEnum):
    RESOURCES = 0
    PEOPLE = 1
    IS_VALID = 2

class GrpFeatureIdx(IntEnum):
    RESOURCES = 0
    PEOPLE = 1
    IS_ALIVE = 2

class GeoGrpFeatureIdx(IntEnum):
    RESOURCES = 0

@dataclass(frozen=True)
class GeoFeatureSpec:
    index: GeoFeatureIdx
    is_absorbable: bool = False
    absorption_rate: float = ABSORPTION_RATE_DEFAULT


@dataclass(frozen=True)
class GrpFeatureSpec:
    index: GrpFeatureIdx
    is_absolute: bool = False

@dataclass(frozen=True)
class FeatureSpec:
    geo: GeoFeatureSpec | None = None
    grp: GrpFeatureSpec | None = None
    
    def get_absorption_rate(self):
        return self.geo.absorption_rate if self.geo else None

FEATURES_SPEC: dict[FeatureKey, FeatureSpec] = {
    FeatureKey.RESOURCES: FeatureSpec(
        geo=GeoFeatureSpec(index=GeoFeatureIdx.RESOURCES, is_absorbable=True),
        grp=GeoFeatureSpec(index=GrpFeatureIdx.RESOURCES, is_absolute=True)
    ),
    FeatureKey.PEOPLE: FeatureSpec(
        geo=GeoFeatureSpec(index=GeoFeatureIdx.PEOPLE, is_absorbable=True),
        grp=GeoFeatureSpec(index=GrpFeatureIdx.PEOPLE, is_absolute=True)
    ),
    FeatureKey.IS_VALID: FeatureSpec(
        geo=GeoFeatureSpec(index=GeoFeatureIdx.IS_VALID, is_absorbable=False),
        grp=None
    ),
    FeatureKey.IS_ALIVE: FeatureSpec(
        geo=None,
        grp=GrpFeatureSpec(index=GrpFeatureIdx.IS_ALIVE, is_absolute=False),
    ),
}