from abc import ABC, abstractmethod

import numpy as np


class BaseDownsampler(ABC):
    """Base class for all downsamplers"""

    @abstractmethod
    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """
        Downsample an image to target size using specified method
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target size as (width, height)
            method: Interpolation method ('bilinear', 'bicubic', 'nearest')
            
        Returns:
            Downsampled image as numpy array
        """
        pass

    @abstractmethod
    def get_supported_methods(self) -> list:
        """Return list of supported interpolation methods"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this downsampler"""
        pass
