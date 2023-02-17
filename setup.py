from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os
import subprocess
import sys


packages = find_packages()


# versioning
MAJOR = 0
MINOR = 0
MICRO = 5
ISRELEASED = True
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"


# are we on windows, darwin, etc?
platform = sys.platform
packages = find_packages()


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = "refellips"

    def run_tests(self):
        import shlex
        import pytest

        print("Running tests with pytest")
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


info = {
    "packages": packages,
    "include_package_data": True,
    "cmdclass": {"test": PyTest},
}


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env
        ).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of refellips.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists(".git"):
        GIT_REVISION = git_version()
    elif os.path.exists("refellips/version.py"):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load refellips/__init__.py
        import imp

        version = imp.load_source("refellips.version", "refellips/version.py")
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += ".dev0+" + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename="refellips/version.py"):
    cnt = """
# THIS FILE IS GENERATED FROM refellips SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        a.close()


def setup_package():
    # Rewrite the version file every time
    write_version_py()
    info["version"] = get_version_info()[0]
    info["zip_safe"] = False
    print(info["version"])

    setup(**info)


if __name__ == "__main__":
    setup_package()
