import warnings
import numpy as np

rng = np.random.default_rng(1234)


class TrainingData:
    """
    A class to manage training and validation data for machine learning models.

    Attributes:
        supervised (bool): Indicates if the data is supervised (i.e., has labels).
        full_training_size (int): The number of samples used for training.
    """

    supervised: bool
    full_training_size: int
    original_training_data: np.ndarray
    original_training_labels: np.ndarray | None
    rand_gen: np.random.Generator
    x_train: np.ndarray
    y_train: np.ndarray | None
    x_val: np.ndarray
    y_val: np.ndarray | None

    def __init__(self,
                 training_data: np.ndarray,
                 training_labels: np.ndarray | None = None,
                 validation_ratio: float = 0,
                 rand_gen: np.random.Generator = rng,
                 ) -> None:
        if validation_ratio < 0 or validation_ratio >= 1:
            raise ValueError("validation_ratio must be in the range [0, 1).")
        if training_labels is not None:
            if training_data.shape[0] != training_labels.shape[0]:
                raise ValueError(
                    "The number of training samples and labels must match.")
            self.supervised = True
        else:
            self.supervised = False
        self.original_training_data = training_data
        self.original_training_labels = training_labels
        self.rand_gen = rand_gen

        self.full_training_size = int(
            training_data.shape[0] * (1 - validation_ratio))
        shuffle_ind = self.rand_gen.permutation(training_data.shape[0])
        training_ind = np.sort(shuffle_ind[:self.full_training_size])

        self.x_train = training_data[training_ind]
        self.y_train = training_labels[training_ind] if training_labels is not None else None
        if self.full_training_size == training_data.shape[0]:
            warnings.warn("All data used for both training and validation.")
            self.x_val = training_data.copy()
            self.y_val = training_labels.copy() if training_labels is not None else None
        else:
            validation_ind = np.sort(shuffle_ind[self.full_training_size:])
            self.x_val = training_data[validation_ind]
            self.y_val = training_labels[validation_ind] if training_labels is not None else None

    def shuffle_and_divide(self,
                           batch_size: int | None = None,
                           always_shuffle: bool = False
                           ) -> list[tuple[np.ndarray, ...]]:
        """
        Shuffle the training data and divide it into a list of mini-batches.
        The number of batches is
        N = floor(num_samples / batch_size). If the batch_size does not divide
        the number of samples, the last few samples are distributed to the previous batches.
        To get all samples in one batch, set batch_size to None or a value
        greater than or equal to the number of training samples.

        Args:
            batch_size (int | None): The size of each mini-batch. If None, use full training size.
            always_shuffle (bool): If True, always shuffle the data even if using full batch.

        Returns:
            list[tuple[np.ndarray, ...]]: A list of mini-batches, each as a tuple of arrays.

        Raises:
            ValueError: If batch_size is not a positive integer
        """
        if batch_size is None or batch_size >= self.full_training_size:
            batch_size = self.full_training_size
            if not always_shuffle:
                if self.y_train is not None:
                    return [(self.x_train, self.y_train)]
                return [(self.x_train,)]
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        batches_num = self.x_train.shape[0] // batch_size
        if self.y_train is not None:
            shuffle_index = self.rand_gen.permutation(self.x_train.shape[0])
            x_train_shuffled = self.x_train[shuffle_index]
            y_train_shuffled = self.y_train[shuffle_index]
            x_batches = np.array_split(x_train_shuffled, batches_num)
            y_batches = np.array_split(y_train_shuffled, batches_num)
            return list(zip(x_batches, y_batches))
        x_train_shuffled = self.rand_gen.permutation(self.x_train)
        x_batches = np.array_split(x_train_shuffled, batches_num)
        return list(zip(x_batches))


    def get_full_training_data(self) -> tuple[np.ndarray, ...]:
        """
        Get the full training data.

        Returns:
            tuple[np.ndarray, ...]: The full training data and labels (if supervised).
        """
        if self.supervised:
            return self.original_training_data, self.original_training_labels  # type: ignore
        return (self.original_training_data,)


    def get_validation_data(self) -> tuple[np.ndarray, ...]:
        """
        Get the validation data.

        Returns:
            tuple[np.ndarray, ...]: The validation data and labels (if supervised).
        """
        if self.y_val is not None:
            return self.x_val, self.y_val
        return (self.x_val,)
