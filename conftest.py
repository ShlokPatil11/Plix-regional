import sys
import importlib.util
import importlib.abc
import importlib.machinery
from pathlib import Path
from types import ModuleType

_app_root = Path(__file__).parent


class _DummyLoader(importlib.abc.Loader):
    """A loader that creates an empty module."""

    def exec_module(self, module):
        # Create an empty module instead of executing __init__.py
        module.__path__ = []


class _InitFinder(importlib.abc.MetaPathFinder):
    """Custom import hook to handle root __init__.py imports."""

    def find_spec(self, fullname, path, target=None):
        # When pytest tries to import __init__, intercept it
        if fullname == "__init__" or fullname.endswith(".__init__"):
            # Return a spec with our dummy loader
            spec = importlib.machinery.ModuleSpec(fullname, _DummyLoader())
            spec.submodule_search_locations = []
            return spec
        return None


# Add the custom finder to sys.meta_path at the beginning
sys.meta_path.insert(0, _InitFinder())

# Also preemptively register dir_language_pipeline
if "dir_language_pipeline" not in sys.modules:
    try:
        spec = importlib.util.spec_from_file_location(
            "dir_language_pipeline",
            _app_root / "__init__.py",
            submodule_search_locations=[str(_app_root)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["dir_language_pipeline"] = module
        spec.loader.exec_module(module)
    except (ImportError, AttributeError):
        module = ModuleType("dir_language_pipeline")
        module.__path__ = [str(_app_root)]
        module.__file__ = str(_app_root / "__init__.py")
        sys.modules["dir_language_pipeline"] = module
