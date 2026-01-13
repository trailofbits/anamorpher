import numpy as np
from PIL import Image

from .base import BaseDownsampler


class PillowDownsampler(BaseDownsampler):
    """Pillow-based downsampler"""

    def __init__(self):
        self._method_map = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }

    @property
    def name(self) -> str:
        return "Pillow"

    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """
        Downsample using Pillow resize

        Args:
            image: Input image (H, W, C)
            target_size: (width, height)
            method: 'nearest', 'bilinear', or 'bicubic'
        """
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Resize using Pillow
        pil_method = self._method_map[method]
        resized_pil = pil_image.resize(target_size, pil_method)

        # Convert back to numpy array
        return np.array(resized_pil)
