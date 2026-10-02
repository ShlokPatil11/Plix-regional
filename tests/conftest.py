import importlib.util
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
SERVICE_B = APP_ROOT / "vendor" / "service_b"
TESTS_DIR = Path(__file__).parent


def _register_package():
    """The app is deployed as the package `dir_language_pipeline`; mirror that here."""
    if "dir_language_pipeline" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "dir_language_pipeline", APP_ROOT / "__init__.py",
        submodule_search_locations=[str(APP_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dir_language_pipeline"] = module
    spec.loader.exec_module(module)


_register_package()
if str(SERVICE_B) not in sys.path:
    sys.path.insert(0, str(SERVICE_B))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
