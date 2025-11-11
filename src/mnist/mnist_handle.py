import struct
from array import array
import numpy as np


class MnistDataloader:
    """
    Class to load MNIST dataset from files.
    
    Attributes:
        training_images_filepath (str): Filepath for training images.
        training_labels_filepath (str): Filepath for training labels.
        test_images_filepath (str): Filepath for test images.
        test_labels_filepath (str): Filepath for test labels.
    """
    training_images_filepath: str
    training_labels_filepath: str
    test_images_filepath: str
    test_labels_filepath: str

    def __init__(self, training_images_filepath: str, training_labels_filepath: str,
                 test_images_filepath: str, test_labels_filepath: str):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath


    def _read_images_labels(self, images_filepath: str, labels_filepath: str):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels


    def load_data(self) -> tuple[tuple, tuple]:
        """
        Load the MNIST dataset from the files specified in the constructor.

        Returns:
            tuple[tuple, tuple]: Training and test datasets.
                                 Unpack format: ((x_train, y_train), (x_test, y_test))
        """
        x_train, y_train = self._read_images_labels(
            self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self._read_images_labels(
            self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
