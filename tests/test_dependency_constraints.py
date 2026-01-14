"""Tests that validate critical dependency constraints.

These tests catch incompatible dependency combinations early in CI,
before they cause cryptic import failures. See CONTRIBUTING.md for
the full dependency compatibility matrix.
"""

import importlib

import pytest


def test_numpy_version_below_2():
    """Verify NumPy is <2.0 for TensorFlow/OpenCV compatibility.

    TensorFlow and OpenCV distribute pre-compiled wheels built against
    NumPy 1.x ABI. NumPy 2.x has breaking ABI changes that cause
    `_ARRAY_API not found` errors at import time.

    See CONTRIBUTING.md for details.
    """
    import numpy as np

    major_version = int(np.__version__.split(".")[0])
    assert major_version < 2, (
        f"NumPy {np.__version__} detected. "
        "NumPy 2.x breaks TensorFlow and OpenCV imports. "
        "Pin to numpy<2.0.0 in pyproject.toml. "
        "See CONTRIBUTING.md for the dependency compatibility matrix."
    )


def test_ml_libraries_import_without_array_api_errors():
    """Verify all ML libraries can be imported without ABI mismatch errors.

    This test catches the specific failure mode where a library compiled
    against NumPy 1.x is loaded with NumPy 2.x, causing:
    `AttributeError: _ARRAY_API not found`
    """
    libraries = [
        ("tensorflow", "tensorflow"),
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("PIL", "Pillow"),
    ]

    for module_name, package_name in libraries:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            error_msg = str(e)
            if "_ARRAY_API" in error_msg or "numpy.core.multiarray" in error_msg:
                pytest.fail(
                    f"NumPy ABI mismatch importing {package_name}. "
                    "This usually means NumPy was upgraded to 2.x while "
                    f"{package_name} wheels are compiled against NumPy 1.x. "
                    "Revert to numpy<2.0.0 in pyproject.toml. "
                    "See CONTRIBUTING.md for details."
                )
            raise


def test_python_version_supported():
    """Verify Python version is within supported range."""
    import sys

    assert sys.version_info >= (3, 11), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
        "This project requires Python 3.11+."
    )
