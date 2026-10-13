"""Local ASGI entry point for checkouts whose directory is not the package name."""

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "dir_language_pipeline"

# `/data` is the Docker volume path. A copied Docker-style `.env` should still
# work from a local checkout without requiring a root-level directory.
if os.getenv("PIPELINE_DATA_DIR") in (None, "", "/data"):
    os.environ["PIPELINE_DATA_DIR"] = str(PROJECT_ROOT / ".local-data")


def _load_project_package() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PROJECT_ROOT / "__init__.py",
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PACKAGE_NAME} from {PROJECT_ROOT}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)


_load_project_package()

from dir_language_pipeline.api import app  # noqa: E402


__all__ = ["app"]
