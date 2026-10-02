import sys
import importlib.abc
import importlib.machinery
from types import ModuleType


class _DummyLoader(importlib.abc.Loader):
    """A loader that creates an empty module."""

    def exec_module(self, module):
        # Create an empty module instead of executing __init__.py
        module.__path__ = []


class _InitFinder(importlib.abc.MetaPathFinder):
    """Custom import hook to handle root __init__.py imports by pytest."""

    def find_spec(self, fullname, path, target=None):
        # When pytest tries to import __init__, intercept it to prevent
        # it from trying to execute the file (which has relative imports that fail)
        if fullname == "__init__" or fullname.endswith(".__init__"):
            # Return a spec with our dummy loader
            spec = importlib.machinery.ModuleSpec(fullname, _DummyLoader())
            spec.submodule_search_locations = []
            return spec
        return None


# Add the custom finder to sys.meta_path at the beginning
sys.meta_path.insert(0, _InitFinder())
