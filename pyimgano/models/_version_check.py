# -*- coding: utf-8 -*-
"""
Utility for PyOD version compatibility checks.
"""

from __future__ import annotations

import warnings
from typing import Optional


def check_pyod_version(min_version: str = "1.1.0") -> bool:
    """
    Check if PyOD version meets minimum requirements.

    Parameters
    ----------
    min_version : str, optional (default="1.1.0")
        Minimum required PyOD version.

    Returns
    -------
    bool
        True if version requirement is met, False otherwise.
    """
    try:
        import pyod
        from packaging import version

        current_version = version.parse(pyod.__version__)
        required_version = version.parse(min_version)

        if current_version < required_version:
            warnings.warn(
                f"PyOD version {pyod.__version__} is installed, "
                f"but version >={min_version} is recommended for optimal performance. "
                f"Some algorithms may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
            return False
        return True

    except ImportError:
        warnings.warn(
            "PyOD is not installed. Install it with: pip install pyod>=1.1.0",
            UserWarning,
            stacklevel=2,
        )
        return False
    except Exception as e:
        warnings.warn(
            f"Could not check PyOD version: {e}",
            UserWarning,
            stacklevel=2,
        )
        return True  # Don't block if we can't check


def get_pyod_version() -> Optional[str]:
    """
    Get the installed PyOD version.

    Returns
    -------
    str or None
        PyOD version string, or None if not installed.
    """
    try:
        import pyod
        return pyod.__version__
    except ImportError:
        return None


__all__ = ["check_pyod_version", "get_pyod_version"]
