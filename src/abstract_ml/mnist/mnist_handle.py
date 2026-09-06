from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


class MnistDataloader:
    """Load IDX-formatted MNIST image and label files into NumPy arrays."""

    def __init__(
        self,
        training_images_filepath: str | Path,
        training_labels_filepath: str | Path,
        test_images_filepath: str | Path,
        test_labels_filepath: str | Path,
    ) -> None:
        self.training_images_filepath = Path(training_images_filepath)
        self.training_labels_filepath = Path(training_labels_filepath)
        self.test_images_filepath = Path(test_images_filepath)
        self.test_labels_filepath = Path(test_labels_filepath)

    @staticmethod
    def _read_images_labels(
        images_filepath: str | Path,
        labels_filepath: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels_path = Path(labels_filepath)
        with labels_path.open("rb") as file:
            label_header = file.read(8)
            if len(label_header) != 8:
                raise ValueError(f"MNIST label file is truncated: {labels_path}")
            magic, size = struct.unpack(">II", label_header)
            if magic != 2049:
                raise ValueError(
                    f"Magic number mismatch for labels: expected 2049, got {magic}"
                )
            labels = np.frombuffer(file.read(), dtype=np.uint8).copy()

        if labels.size != size:
            raise ValueError(
                f"Label count mismatch: header declares {size}, found {labels.size}"
            )

        images_path = Path(images_filepath)
        with images_path.open("rb") as file:
            image_header = file.read(16)
            if len(image_header) != 16:
                raise ValueError(f"MNIST image file is truncated: {images_path}")
            magic, image_count, rows, cols = struct.unpack(">IIII", image_header)
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch for images: expected 2051, got {magic}"
                )
            image_data = np.frombuffer(file.read(), dtype=np.uint8).copy()

        if image_count != size:
            raise ValueError(
                f"Image/label count mismatch: images={image_count}, labels={size}"
            )

        expected_values = image_count * rows * cols
        if image_data.size != expected_values:
            raise ValueError(
                f"Image data mismatch: expected {expected_values} bytes, "
                f"found {image_data.size}"
            )

        images = image_data.reshape(image_count, rows, cols)
        return images, labels

    def load_data(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Load and return ((x_train, y_train), (x_test, y_test))."""
        x_train, y_train = self._read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self._read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)
