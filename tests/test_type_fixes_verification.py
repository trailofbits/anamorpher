"""Regression tests verifying type annotation fixes don't change behavior.

These tests document assumptions made during the type annotation fixes
and will catch if those assumptions become invalid in future versions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


class TestPILConstantEquivalence:
    """Verify PIL resampling constants are equivalent across namespaces."""

    def test_bicubic_constant_equivalence(self):
        """Image.BICUBIC == Image.Resampling.BICUBIC"""
        from PIL import Image

        assert Image.BICUBIC == Image.Resampling.BICUBIC
        assert isinstance(Image.Resampling.BICUBIC, int)

    def test_nearest_constant_equivalence(self):
        """Image.NEAREST == Image.Resampling.NEAREST"""
        from PIL import Image

        assert Image.NEAREST == Image.Resampling.NEAREST
        assert isinstance(Image.Resampling.NEAREST, int)


class TestCV2ColorConversion:
    """Verify cv2.cvtColor dtype preservation behavior."""

    def test_cvtcolor_preserves_float32_dtype(self):
        """cv2.cvtColor should preserve float32 dtype on input."""
        import cv2

        bgr_f32 = np.random.rand(100, 100, 3).astype(np.float32)
        rgb_result = cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2RGB)

        assert rgb_result.dtype == np.float32, (
            f"cv2.cvtColor changed dtype from float32 to {rgb_result.dtype}"
        )

    def test_astype_float32_is_noop_on_float32(self):
        """Calling .astype(np.float32) on float32 array should be a no-op."""
        import cv2

        bgr_f32 = np.random.rand(100, 100, 3).astype(np.float32)
        rgb_result = cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2RGB)
        rgb_casted = rgb_result.astype(np.float32)

        assert np.array_equal(rgb_result, rgb_casted), (
            "astype(np.float32) changed data on a float32 array"
        )

    def test_cvtcolor_preserves_values(self):
        """cv2.cvtColor BGR->RGB should only swap channels, not change values."""
        import cv2

        # Create known values
        bgr = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)  # B=0.1, G=0.2, R=0.3
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # RGB should be [0.3, 0.2, 0.1]
        expected = np.array([[[0.3, 0.2, 0.1]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(rgb, expected)


class TestSanitizerOverloads:
    """Verify @overload decorators don't change sanitize_numeric runtime behavior."""

    def test_sanitize_numeric_returns_int(self):
        """sanitize_numeric with data_type=int should return int."""
        from sanitizer import sanitize_numeric

        result = sanitize_numeric("42", data_type=int)
        assert isinstance(result, int)
        assert result == 42

    def test_sanitize_numeric_returns_float(self):
        """sanitize_numeric with data_type=float should return float."""
        from sanitizer import sanitize_numeric

        result = sanitize_numeric("3.14", data_type=float)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_sanitize_numeric_bounds_checking(self):
        """sanitize_numeric should still enforce bounds."""
        from sanitizer import sanitize_numeric

        with pytest.raises(ValueError):
            sanitize_numeric("100", min_val=0, max_val=50, data_type=int)

    def test_sanitize_numeric_default_is_float(self):
        """sanitize_numeric without data_type should default to float."""
        from sanitizer import sanitize_numeric

        result = sanitize_numeric("42")
        assert isinstance(result, float)
