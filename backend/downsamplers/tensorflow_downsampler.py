import numpy as np
import tensorflow as tf

from .base import BaseDownsampler


class TensorFlowDownsampler(BaseDownsampler):
    """TensorFlow-based downsampler"""

    def __init__(self):
        self._method_map = {
            "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "bicubic": tf.image.ResizeMethod.BICUBIC,
        }

    @property
    def name(self) -> str:
        return "TensorFlow"

    def get_supported_methods(self) -> list:
        return list(self._method_map.keys())

    def downsample(self, image: np.ndarray, target_size: tuple, method: str) -> np.ndarray:
        """
        Downsample using TensorFlow resize

        Args:
            image: Input image (H, W, C)
            target_size: (width, height)
            method: 'nearest', 'bilinear', or 'bicubic'
        """
        if method not in self._method_map:
            raise ValueError(f"Unsupported method: {method}")

        # Convert to TensorFlow tensor and add batch dimension
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)

        # TensorFlow expects (height, width) for new_size
        tf_size = (target_size[1], target_size[0])

        # Resize
        tf_method = self._method_map[method]
        resized = tf.image.resize(tensor, tf_size, method=tf_method)

        # Convert back to numpy and remove batch dimension
        result = tf.squeeze(resized, 0).numpy().astype(np.uint8)
        return result
