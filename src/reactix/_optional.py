"""
Utilities for checking optional dependencies.
"""

from __future__ import annotations

import importlib


_INSTALL_URL = 'https://astoeriko.github.io/reactix/installation.html'

OPTIONAL_DEPENDENCIES = {
    'pymc': ('pymc', 'arviz', 'nutpie'),
    'notebooks': ('matplotlib', 'ipympl', 'ipykernel'),
}


def _require_package(package: str) -> bool:
    """
    Return True if the package can be imported, False otherwise.
    """
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def _check_dependencies(feature: str) -> None:
    """
    Check whether all packages required for a feature are installed.
    """
    missing = [package for package in OPTIONAL_DEPENDENCIES[feature] if not _require_package(package)]

    if missing:
        missing_str = ', '.join(missing)

        raise ImportError(
            f'This notebook requires optional dependencies.\n'
            f'Missing packages: {missing_str}\n\n'
            f'Please see the installation instructions:\n'
            f'{_INSTALL_URL}'
        )


def require_pymc() -> None:
    """Check that PyMC-related dependencies are installed."""
    _check_dependencies('pymc')


def require_notebooks() -> None:
    """Check that notebook-related dependencies are installed."""
    _check_dependencies('notebooks')
