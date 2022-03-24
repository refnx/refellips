import os
import re
import sys

from .structureSE import (
    RI,
    StructureSE,
    SlabSE,
    Cauchy,
    ComponentSE,
    materials,
    load_material,
    MixedSlabSE,
)
from .dataSE import DataSE
from .reflect_modelSE import ReflectModelSE
from .objectiveSE import ObjectiveSE


class _PytestTester:
    """
    Pytest test runner entry point.
    """

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(
        self,
        label="fast",
        verbose=1,
        extra_argv=None,
        doctests=False,
        coverage=False,
        tests=None,
    ):
        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        pytest_args = ["-l"]

        if doctests:
            raise ValueError("Doctests not supported")

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v" * (int(verbose) - 1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if tests is None:
            tests = [self.module_name]

        pytest_args += ["--pyargs"] + list(tests)

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0


test = _PytestTester(__name__)
del _PytestTester


__all__ = [s for s in dir() if not s.startswith("_")]

try:
    from refellips.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"
