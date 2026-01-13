import cv2
import numpy as np

from .base import BaseDownsampler


class OpenCVDownsampler(BaseDownsampler):
    """OpenCV-based downsampler"""

    def __init__(self):
        self._method_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }

    @property
    def name(self) -> str:
        return "OpenCV"

    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """
        Downsample using OpenCV resize

        Args:
            image: Input image (H, W, C)
            target_size: (width, height)
            method: 'nearest', 'bilinear', or 'bicubic'
        """
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        cv_method = self._method_map[method]
        return cv2.resize(image, target_size, interpolation=cv_method)
