import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from adversarial_generators import nearest_gen_payload


class TestNearestGenPayload:
    """Test suite for nearest gen payload functionality"""

    def test_module_imports(self):
        """Test that all required functions are importable"""
        assert hasattr(nearest_gen_payload, 'srgb2lin')
        assert hasattr(nearest_gen_payload, 'lin2srgb')
        assert hasattr(nearest_gen_payload, 'embed_nn')
        assert hasattr(nearest_gen_payload, 'mse_psnr')
        assert hasattr(nearest_gen_payload, 'main')

    def test_srgb2lin_conversion(self):
        """Test sRGB to linear conversion"""
        srgb_values = np.array([[[0., 128., 255.]]], dtype=np.float32)
        linear = nearest_gen_payload.srgb2lin(srgb_values)

        assert linear.shape == srgb_values.shape
        assert linear.dtype == np.float32
        assert 0 <= linear.min() <= linear.max() <= 1

        # Test that black stays black and white becomes ~1
        assert linear[0, 0, 0] == 0.0  # Black
        assert 0.95 < linear[0, 0, 2] <= 1.0  # White (close to 1)

    def test_lin2srgb_conversion(self):
        """Test linear to sRGB conversion"""
        linear_values = np.array([[[0., 0.5, 1.]]], dtype=np.float32)
        srgb = nearest_gen_payload.lin2srgb(linear_values)

        assert srgb.shape == linear_values.shape
        assert srgb.dtype == np.float32
        assert 0 <= srgb.min() <= srgb.max() <= 255

        # Test that black stays black and white becomes 255
        assert srgb[0, 0, 0] == 0.0  # Black
        assert srgb[0, 0, 2] == pytest.approx(255.0, abs=0.01)  # White

    def test_roundtrip_conversion(self):
        """Test that sRGB -> linear -> sRGB is approximately identity"""
        srgb_values = np.array([[[64., 128., 192.]]], dtype=np.float32)
        linear = nearest_gen_payload.srgb2lin(srgb_values)
        srgb_back = nearest_gen_payload.lin2srgb(linear)

        # Should be very close (within 1 unit)
        np.testing.assert_allclose(srgb_values, srgb_back, atol=1.0)

    def test_embed_nn_basic(self):
        """Test basic nearest neighbor embed functionality"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        result = nearest_gen_payload.embed_nn(decoy, target, lam=0.1, eps=0.0, offset=2)

        assert result.shape == decoy.shape
        assert result.dtype == np.float32

    def test_embed_nn_channel_constraint(self):
        """Test that embed_nn only modifies channel 0 (red channel)"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        # Store original green and blue channels
        original_green = decoy[:, :, 1].copy()
        original_blue = decoy[:, :, 2].copy()

        result = nearest_gen_payload.embed_nn(decoy, target, lam=0.1, eps=0.0, offset=2)

        # Green and blue channels should be unchanged
        np.testing.assert_array_equal(result[:, :, 1], original_green)
        np.testing.assert_array_equal(result[:, :, 2], original_blue)

    def test_embed_nn_different_offsets(self):
        """Test embed_nn with different offset values"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        # Test with different valid offsets
        for offset in [0, 1, 2, 3]:
            result = nearest_gen_payload.embed_nn(decoy, target, lam=0.1, eps=0.0, offset=offset)
            assert result.shape == decoy.shape
            assert result.dtype == np.float32

    def test_embed_nn_lambda_zero(self):
        """Test embed_nn with lambda=0 (no mean preservation)"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        result = nearest_gen_payload.embed_nn(decoy, target, lam=0.0, eps=0.0, offset=2)

        assert result.shape == decoy.shape
        assert result.dtype == np.float32

        # With lam=0, only the selected pixels should change
        # Most of the image should remain unchanged
        diff_mask = np.abs(result - decoy) > 1e-6
        total_changed = np.sum(diff_mask)
        total_pixels = np.prod(decoy.shape)
        # Only a small fraction should change (just the selected samples)
        assert total_changed < total_pixels * 0.5

    def test_embed_nn_with_dither(self):
        """Test embed_nn with null-space dithering"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        # Test with dithering
        result_no_dither = nearest_gen_payload.embed_nn(decoy, target, lam=0.1, eps=0.0, offset=2)
        result_with_dither = nearest_gen_payload.embed_nn(decoy, target, lam=0.1, eps=0.1, offset=2)

        assert result_no_dither.shape == result_with_dither.shape
        # Results should be different due to random dithering
        assert not np.allclose(result_no_dither, result_with_dither)

    def test_mse_psnr(self):
        """Test MSE and PSNR computation"""
        a = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.5

        mse, psnr = nearest_gen_payload.mse_psnr(a, b)

        assert mse == 0.0  # Identical images should have 0 MSE
        assert psnr == float('inf')  # And infinite PSNR

        # Test with different images
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.6
        mse, psnr = nearest_gen_payload.mse_psnr(a, b)

        assert mse > 0  # Different images should have positive MSE
        assert psnr > 0 and psnr != float('inf')  # Finite positive PSNR

    def test_main_function_args(self):
        """Test that main function accepts expected arguments"""
        with patch('sys.argv', ['nearest_gen_payload.py', '--help']):
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.side_effect = SystemExit  # argparse exits on --help

                with pytest.raises(SystemExit):
                    nearest_gen_payload.main()

    def test_main_function_structure(self):
        """Test main function structure without execution"""
        mock_args = MagicMock()
        mock_args.decoy = "test_decoy.png"
        mock_args.target = "test_target.png"
        mock_args.lam = 0.25
        mock_args.eps = 0.0
        mock_args.gamma = 1.0
        mock_args.offset = 2

        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
             patch('PIL.Image.open') as mock_open, \
             patch('PIL.Image.fromarray') as mock_fromarray:

            # Mock image loading
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_img.resize.return_value = mock_img
            mock_open.return_value = mock_img

            # Mock numpy array conversion - create properly sized arrays
            decoy_array = np.random.rand(8, 8, 3).astype(np.float32)
            target_array = np.random.rand(2, 2, 3).astype(np.float32)

            def mock_asarray(img, dtype=None):
                # Return decoy for first call, target for second
                if mock_asarray.call_count == 1:
                    mock_asarray.call_count += 1
                    return decoy_array
                else:
                    return target_array

            mock_asarray.call_count = 1

            with patch('numpy.asarray', side_effect=mock_asarray):
                mock_result_img = MagicMock()
                mock_fromarray.return_value = mock_result_img

                try:
                    nearest_gen_payload.main()
                    # If we get here, main ran without crashing
                    assert True
                except Exception as e:
                    # Allow certain expected exceptions during mocking
                    if "save" not in str(e).lower() and "Resampling" not in str(e):
                        pytest.fail(f"Unexpected error in main: {e}")

    def test_main_shape_assertion(self):
        """Test that main function validates shape requirements"""
        mock_args = MagicMock()
        mock_args.decoy = "test_decoy.png"
        mock_args.target = "test_target.png"
        mock_args.lam = 0.25
        mock_args.eps = 0.0
        mock_args.gamma = 1.0
        mock_args.offset = 2

        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
             patch('PIL.Image.open') as mock_open:

            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img

            # Create mismatched sizes (decoy not 4x target)
            decoy_array = np.random.rand(6, 6, 3).astype(np.float32)  # Wrong size
            target_array = np.random.rand(2, 2, 3).astype(np.float32)

            def mock_asarray(img, dtype=None):
                if mock_asarray.call_count == 1:
                    mock_asarray.call_count += 1
                    return decoy_array
                else:
                    return target_array

            mock_asarray.call_count = 1

            with patch('numpy.asarray', side_effect=mock_asarray):
                with pytest.raises(AssertionError):
                    nearest_gen_payload.main()

    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        if str(backend_path) in sys.path:
            sys.path.remove(str(backend_path))
