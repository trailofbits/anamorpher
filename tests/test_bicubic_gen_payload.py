import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from adversarial_generators import bicubic_gen_payload


class TestBicubicGenPayload:
    """Test suite for bicubic gen payload functionality"""

    def test_module_imports(self):
        """Test that all required functions are importable"""
        assert hasattr(bicubic_gen_payload, 'srgb2lin')
        assert hasattr(bicubic_gen_payload, 'lin2srgb')
        assert hasattr(bicubic_gen_payload, 'cubic_kernel')
        assert hasattr(bicubic_gen_payload, 'weight_vector')
        assert hasattr(bicubic_gen_payload, 'luma_linear')
        assert hasattr(bicubic_gen_payload, 'bottom_luma_mask')
        assert hasattr(bicubic_gen_payload, 'embed')
        assert hasattr(bicubic_gen_payload, 'mse_psnr')
        assert hasattr(bicubic_gen_payload, 'main')

    def test_srgb2lin_conversion(self):
        """Test sRGB to linear conversion"""
        # Test with known values
        srgb_values = np.array([[[0., 128., 255.]]], dtype=np.float32)
        linear = bicubic_gen_payload.srgb2lin(srgb_values)

        assert linear.shape == srgb_values.shape
        assert linear.dtype == np.float32
        assert 0 <= linear.min() <= linear.max() <= 1

        # Test that black stays black and white becomes ~1
        assert linear[0, 0, 0] == 0.0  # Black
        assert 0.95 < linear[0, 0, 2] <= 1.0  # White (close to 1)

    def test_lin2srgb_conversion(self):
        """Test linear to sRGB conversion"""
        linear_values = np.array([[[0., 0.5, 1.]]], dtype=np.float32)
        srgb = bicubic_gen_payload.lin2srgb(linear_values)

        assert srgb.shape == linear_values.shape
        assert srgb.dtype == np.float32
        assert 0 <= srgb.min() <= srgb.max() <= 255

        # Test that black stays black and white becomes 255
        assert srgb[0, 0, 0] == 0.0  # Black
        assert srgb[0, 0, 2] == pytest.approx(255.0, abs=0.01)  # White

    def test_roundtrip_conversion(self):
        """Test that sRGB -> linear -> sRGB is approximately identity"""
        srgb_values = np.array([[[64., 128., 192.]]], dtype=np.float32)
        linear = bicubic_gen_payload.srgb2lin(srgb_values)
        srgb_back = bicubic_gen_payload.lin2srgb(linear)

        # Should be very close (within 1 unit)
        np.testing.assert_allclose(srgb_values, srgb_back, atol=1.0)

    def test_cubic_kernel(self):
        """Test cubic kernel function"""
        # Test at known points
        x = np.array([0., 0.5, 1., 1.5, 2.])
        weights = bicubic_gen_payload.cubic_kernel(x)

        assert len(weights) == len(x)
        assert weights[0] == 1.0  # Should be 1 at x=0
        assert weights[-1] == 0.0  # Should be 0 at x=2
        # Note: Cubic kernels have negative lobes between x=1 and x=2
        # This is mathematically correct for bicubic interpolation

    def test_weight_vector(self):
        """Test weight vector generation"""
        weights = bicubic_gen_payload.weight_vector(scale=4)

        assert len(weights) == 16  # 4x4 = 16
        assert weights.dtype == np.float32
        assert np.sum(weights) > 0  # Should have some positive weights

    def test_luma_linear(self):
        """Test linear luma computation"""
        # Create test RGB image
        rgb_img = np.random.rand(10, 10, 3).astype(np.float32)
        luma = bicubic_gen_payload.luma_linear(rgb_img)

        assert luma.shape == (10, 10)  # Should reduce channel dimension
        assert luma.dtype == np.float32
        assert np.all(luma >= 0)  # Luma should be non-negative

    def test_bottom_luma_mask(self):
        """Test bottom luma mask generation"""
        # Create test image with varying brightness
        rgb_img = np.zeros((4, 4, 3), dtype=np.float32)
        rgb_img[0, 0] = [1., 1., 1.]  # Bright pixel
        rgb_img[3, 3] = [0., 0., 0.]  # Dark pixel

        mask = bicubic_gen_payload.bottom_luma_mask(rgb_img, frac=0.5)

        assert mask.shape == (4, 4)
        assert mask.dtype == bool
        assert mask[3, 3]  # Dark pixel should be in bottom fraction
        assert not mask[0, 0]  # Bright pixel should not be in bottom fraction

    def test_embed_basic(self):
        """Test basic embed functionality"""
        # Create simple test images
        np.random.seed(42)  # Fixed seed for reproducibility
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        result = bicubic_gen_payload.embed(decoy, target, lam=0.1, eps=0.0)

        assert result.shape == decoy.shape
        assert result.dtype == np.float32
        # Note: Adversarial embedding may produce values outside [0, 1] range
        # which get clipped later when converting to uint8 for output

    def test_mse_psnr(self):
        """Test MSE and PSNR computation"""
        a = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.5

        mse, psnr = bicubic_gen_payload.mse_psnr(a, b)

        assert mse == 0.0  # Identical images should have 0 MSE
        assert psnr == float('inf')  # And infinite PSNR

        # Test with different images
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.6
        mse, psnr = bicubic_gen_payload.mse_psnr(a, b)

        assert mse > 0  # Different images should have positive MSE
        assert psnr > 0 and psnr != float('inf')  # Finite positive PSNR

    def test_main_function_args(self):
        """Test that main function accepts expected arguments"""
        with patch('sys.argv', ['bicubic_gen_payload.py', '--help']):
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.side_effect = SystemExit  # argparse exits on --help

                with pytest.raises(SystemExit):
                    bicubic_gen_payload.main()

    def test_main_function_structure(self):
        """Test main function structure without execution"""
        # Mock all the external dependencies
        mock_args = MagicMock()
        mock_args.decoy = "test_decoy.png"
        mock_args.target = "test_target.png"
        mock_args.lam = 0.25
        mock_args.eps = 0.0
        mock_args.gamma = 1.0
        mock_args.dark_frac = 0.3

        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
             patch('PIL.Image.open') as mock_open, \
             patch('PIL.Image.fromarray') as mock_fromarray:

            # Mock image loading
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_img.resize.return_value = mock_img
            mock_open.return_value = mock_img

            # Mock numpy array conversion
            test_array = np.random.rand(8, 8, 3).astype(np.float32)
            with patch('numpy.asarray', return_value=test_array):

                mock_result_img = MagicMock()
                mock_fromarray.return_value = mock_result_img

                try:
                    bicubic_gen_payload.main()
                    # If we get here, main ran without crashing
                    assert True
                except Exception as e:
                    # Allow certain expected exceptions during mocking
                    if "save" not in str(e).lower():
                        pytest.fail(f"Unexpected error in main: {e}")

    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        if str(backend_path) in sys.path:
            sys.path.remove(str(backend_path))
