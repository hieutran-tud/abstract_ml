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
        self._batch_base = 0
        self._batch_high_mask = np.zeros(self.full_training_size, dtype=bool)


    def get_next_batch(self, batch_size: int | None = None) -> tuple[np.ndarray, ...]:
        """
        Get the next mini-batch of training data using a balanced sampler.

        Args:
            batch_size (int | None): The size of the mini-batch.
                                     If None or >= the number of training samples,
                                     the full training data is returned.
                                     Defaults to None.

        Returns:
            tuple[np.ndarray, ...]: The next mini-batch of training data and labels (if supervised).

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        if batch_size is None or batch_size >= self.full_training_size:
            if self.y_train is not None:
                return self.x_train, self.y_train
            return (self.x_train,)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        base = self._batch_base
        high_mask = self._batch_high_mask
        low_mask = np.logical_not(high_mask)
        have_low = np.any(low_mask)
        minima_mask = low_mask if have_low else np.ones(self.full_training_size, dtype=bool)
        indices_min = np.nonzero(minima_mask)[0]

        if indices_min.size >= batch_size:
            chosen = self.rand_gen.choice(indices_min, size=batch_size, replace=False)
            if have_low:
                high_mask[chosen] = True
            else:
                base += 1
                high_mask[:] = False
                high_mask[chosen] = True
        else:
            chosen_min = indices_min
            remaining = batch_size - indices_min.size
            others = np.nonzero(np.logical_not(minima_mask))[0]
            extra = self.rand_gen.choice(others, size=remaining, replace=False)
            chosen = np.concatenate([chosen_min, extra])
            self.rand_gen.shuffle(chosen)
            base += 1
            high_mask[:] = False
            high_mask[extra] = True
        self._batch_base = base
        self._batch_high_mask = high_mask

        x_batch = self.x_train[chosen]
        if self.y_train is not None:
            y_batch = self.y_train[chosen]
            return x_batch, y_batch
        return (x_batch,)


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
