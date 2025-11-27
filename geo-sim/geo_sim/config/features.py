from dataclasses import dataclass
from enum import IntEnum, Enum, auto

class FeatureKey(Enum):
    RESOURCES = auto()
    PEOPLE = auto()
    IS_VALID = auto()

class GeoFeatureIdx(IntEnum):
    RESOURCES     = 0
    WITHIN_BORDER = 1

class GrpFeatureIdx(IntEnum):
    RESOURCES = 0

@dataclass(frozen=True)
class GeoFeatureSpec:
    index: GeoFeatureIdx
    is_movable: bool

@dataclass(frozen=True)
class GrpFeatureSpec:
    index: GrpFeatureIdx
    is_absolute: bool

@dataclass(frozen=True)
class FeatureSpec:
    geo: GeoFeatureSpec | None = None
    grp: GrpFeatureSpec | None = None

FEATURES: dict[FeatureKey, FeatureSpec] = {
    FeatureKey.RESOURCES: FeatureSpec(
        geo=GeoFeatureSpec(index=GeoFeatureIdx.RESOURCES, is_movable=True),
        grp=GeoFeatureSpec(index=GrpFeatureIdx.RESOURCES, is_absolute=True)
    ),
    FeatureKey.IS_VALID: FeatureSpec(
        geo=GeoFeatureSpec(index=GeoFeatureIdx.IS_VALID, is_movable=False),
        grp=None
    ),
}