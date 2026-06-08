"""
Registry subpackage.

Responsibilities:
  - parse registry sources;
  - build and validate antigen_registry.csv;
  - expose the registry CLI.
"""

from .core import (
    build_antigen_registry,
    load_antigen_registry,
    load_like_registry,
    merge_registry_updates,
    validate_antigen_registry,
    write_antigen_registry,
)
from .workflow import (
    RegistryBuildResult,
    build_registry,
    build_registry_from_paths,
    write_registry_result,
)

__all__ = [
    "RegistryBuildResult",
    "build_antigen_registry",
    "build_registry",
    "build_registry_from_paths",
    "load_antigen_registry",
    "load_like_registry",
    "merge_registry_updates",
    "validate_antigen_registry",
    "write_antigen_registry",
    "write_registry_result",
]
