import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseDownsampler

class PyTorchDownsampler(BaseDownsampler):
    """PyTorch-based downsampler"""
    
    def __init__(self):
        self._method_map = {
            'nearest': 'nearest',
            'bilinear': 'bilinear',
            'bicubic': 'bicubic'
        }
    
    @property 
    def name(self) -> str:
        return "PyTorch"
    
    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())
    
    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """
        Downsample using PyTorch interpolate
        
        Args:
            image: Input image (H, W, C)
            target_size: (width, height) 
            method: 'nearest', 'bilinear', or 'bicubic'
        """
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")
        
        # Convert to torch tensor (C, H, W) and add batch dimension
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        
        # PyTorch expects (height, width) for size
        torch_size = (target_size[1], target_size[0])
        
        # Interpolate
        torch_method = self._method_map[method]
        with torch.no_grad():
            resized = F.interpolate(tensor, size=torch_size, mode=torch_method, 
                                  align_corners=False if method != 'nearest' else None)
        
        # Convert back to numpy (H, W, C)
        return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)