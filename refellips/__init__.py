from .structureSE import (
    RI,
    StructureSE,
    SlabSE,
    Cauchy,
    ComponentSE,
    materials,
    load_material,
)
from .dataSE import DataSE
from .reflect_modelSE import ReflectModelSE
from .objectiveSE import ObjectiveSE


__all__ = [s for s in dir() if not s.startswith("_")]

try:
    from refellips.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"
