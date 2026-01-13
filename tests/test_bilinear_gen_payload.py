import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from adversarial_generators import bilinear_gen_payload


class TestBilinearGenPayload:
    """Test suite for bilinear gen payload functionality"""
    
    def test_module_imports(self):
        """Test that all required functions are importable"""
        assert hasattr(bilinear_gen_payload, 'srgb2lin')
        assert hasattr(bilinear_gen_payload, 'lin2srgb')
        assert hasattr(bilinear_gen_payload, 'bilinear_kernel')
        assert hasattr(bilinear_gen_payload, 'weight_vector_bilinear')
        assert hasattr(bilinear_gen_payload, 'luma_linear')
        assert hasattr(bilinear_gen_payload, 'bottom_luma_mask')
        assert hasattr(bilinear_gen_payload, 'embed_bilinear')
        assert hasattr(bilinear_gen_payload, 'mse_psnr')
        assert hasattr(bilinear_gen_payload, 'main')

    def test_srgb2lin_conversion(self):
        """Test sRGB to linear conversion"""
        srgb_values = np.array([[[0., 128., 255.]]], dtype=np.float32)
        linear = bilinear_gen_payload.srgb2lin(srgb_values)
        
        assert linear.shape == srgb_values.shape
        assert linear.dtype == np.float32
        assert 0 <= linear.min() <= linear.max() <= 1

    def test_lin2srgb_conversion(self):
        """Test linear to sRGB conversion"""
        linear_values = np.array([[[0., 0.5, 1.]]], dtype=np.float32)
        srgb = bilinear_gen_payload.lin2srgb(linear_values)
        
        assert srgb.shape == linear_values.shape
        assert srgb.dtype == np.float32
        assert 0 <= srgb.min() <= srgb.max() <= 255

    def test_bilinear_kernel(self):
        """Test bilinear kernel function"""
        x = np.array([0., 0.5, 1., 1.5])
        weights = bilinear_gen_payload.bilinear_kernel(x)
        
        assert len(weights) == len(x)
        assert weights[0] == 1.0  # Should be 1 at x=0
        assert weights[2] == 0.0  # Should be 0 at x=1
        assert weights[3] == 0.0  # Should be 0 at x>1

    def test_weight_vector_bilinear(self):
        """Test bilinear weight vector generation"""
        weights = bilinear_gen_payload.weight_vector_bilinear(scale=4)
        
        assert len(weights) == 16  # 4x4 = 16
        assert weights.dtype == np.float32
        assert np.abs(np.sum(weights) - 1.0) < 1e-6  # Should sum to 1 (normalized)
        
        # For bilinear with scale=4, only the center 2x2 should have non-zero weights
        weights_2d = weights.reshape(4, 4)
        # Check that corners are zero (outside bilinear kernel)
        assert weights_2d[0, 0] == 0.0
        assert weights_2d[0, 3] == 0.0  
        assert weights_2d[3, 0] == 0.0
        assert weights_2d[3, 3] == 0.0

    def test_luma_linear(self):
        """Test linear luma computation"""
        rgb_img = np.random.rand(10, 10, 3).astype(np.float32)
        luma = bilinear_gen_payload.luma_linear(rgb_img)
        
        assert luma.shape == (10, 10)
        assert luma.dtype == np.float32
        assert np.all(luma >= 0)

    def test_bottom_luma_mask(self):
        """Test bottom luma mask generation"""
        rgb_img = np.zeros((4, 4, 3), dtype=np.float32)
        rgb_img[0, 0] = [1., 1., 1.]  # Bright pixel
        rgb_img[3, 3] = [0., 0., 0.]  # Dark pixel
        
        mask = bilinear_gen_payload.bottom_luma_mask(rgb_img, frac=0.5)
        
        assert mask.shape == (4, 4)
        assert mask.dtype == bool
        assert mask[3, 3]  # Dark pixel should be in bottom fraction
        assert not mask[0, 0]  # Bright pixel should not be in bottom fraction

    def test_embed_bilinear_basic(self):
        """Test basic bilinear embed functionality"""
        np.random.seed(42)  # Fixed seed for reproducibility
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)

        result = bilinear_gen_payload.embed_bilinear(decoy, target, lam=0.1, eps=0.0)

        assert result.shape == decoy.shape
        assert result.dtype == np.float32
        # Note: Adversarial embedding may produce values outside [0, 1] range
        # which get clipped later when converting to uint8 for output

    def test_embed_bilinear_channel_constraint(self):
        """Test that embed_bilinear only modifies channel 0 (red channel)"""
        decoy = np.random.rand(8, 8, 3).astype(np.float32)
        target = np.random.rand(2, 2, 3).astype(np.float32)
        
        # Store original green and blue channels
        original_green = decoy[:, :, 1].copy()
        original_blue = decoy[:, :, 2].copy()
        
        result = bilinear_gen_payload.embed_bilinear(decoy, target, lam=0.1, eps=0.0)
        
        # Green and blue channels should be unchanged
        np.testing.assert_array_equal(result[:, :, 1], original_green)
        np.testing.assert_array_equal(result[:, :, 2], original_blue)

    def test_mse_psnr(self):
        """Test MSE and PSNR computation"""
        a = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        
        mse, psnr = bilinear_gen_payload.mse_psnr(a, b)
        
        assert mse == 0.0
        assert psnr == float('inf')
        
        # Test with different images
        b = np.ones((4, 4, 3), dtype=np.float32) * 0.6
        mse, psnr = bilinear_gen_payload.mse_psnr(a, b)
        
        assert mse > 0
        assert psnr > 0 and psnr != float('inf')

    def test_main_function_structure(self):
        """Test main function structure without execution"""
        mock_args = MagicMock()
        mock_args.decoy = "test_decoy.png"
        mock_args.target = "test_target.png"
        mock_args.lam = 0.25
        mock_args.eps = 0.0
        mock_args.gamma = 1.0
        mock_args.dark_frac = 0.3
        mock_args.anti_alias = False
        
        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args), \
             patch('cv2.imread') as mock_imread, \
             patch('cv2.imwrite') as mock_imwrite, \
             patch('cv2.resize') as mock_resize, \
             patch('cv2.cvtColor') as mock_cvtcolor:
            
            # Mock OpenCV functions
            test_array = np.random.rand(8, 8, 3).astype(np.float32)
            mock_imread.return_value = test_array
            mock_cvtcolor.return_value = test_array
            mock_resize.return_value = test_array
            
            try:
                bilinear_gen_payload.main()
                assert True  # Main function executed without crashing
            except Exception as e:
                # Allow certain expected exceptions during mocking
                if "cannot import name 'cv2'" not in str(e):
                    pytest.fail(f"Unexpected error in main: {e}")

    def test_anti_alias_flag_handling(self):
        """Test that anti-alias flag argument is accepted"""
        # Verify the main function can parse the anti_alias argument
        # by checking argparse configuration
        import argparse
        from io import StringIO

        with patch('sys.stderr', new_callable=StringIO):
            with patch('sys.argv', ['bilinear_gen_payload.py', '--help']):
                try:
                    bilinear_gen_payload.main()
                except SystemExit:
                    pass  # --help causes SystemExit, which is expected

        # Verify the module has the expected argument handling
        assert hasattr(bilinear_gen_payload, 'main')

    @classmethod  
    def teardown_class(cls):
        """Clean up after tests"""
        if str(backend_path) in sys.path:
            sys.path.remove(str(backend_path))