from __future__ import annotations

import warnings

import numpy as np

rng = np.random.default_rng(1234)


class TrainingData:
    """Manage supervised or unsupervised training and validation data."""

    def __init__(
        self,
        training_data: np.ndarray,
        training_labels: np.ndarray | None = None,
        validation_ratio: float = 0.0,
        rand_gen: np.random.Generator = rng,
    ) -> None:
        if training_data.ndim == 0 or training_data.shape[0] == 0:
            raise ValueError("training_data must contain at least one sample.")
        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in the range [0, 1).")

        if training_labels is not None:
            if training_data.shape[0] != training_labels.shape[0]:
                raise ValueError(
                    "The number of training samples and labels must match."
                )
            self.supervised = True
        else:
            self.supervised = False

        self.original_training_data = training_data
        self.original_training_labels = training_labels
        self.rand_gen = rand_gen

        self.full_training_size = int(training_data.shape[0] * (1.0 - validation_ratio))
        if self.full_training_size < 1:
            raise ValueError(
                "validation_ratio leaves no samples for training; reduce the ratio."
            )

        shuffle_indices = self.rand_gen.permutation(training_data.shape[0])
        training_indices = np.sort(shuffle_indices[: self.full_training_size])

        self.x_train = training_data[training_indices]
        self.y_train = (
            training_labels[training_indices]
            if training_labels is not None
            else None
        )

        if self.full_training_size == training_data.shape[0]:
            warnings.warn(
                "validation_ratio=0; validation data duplicates training data.",
                UserWarning,
                stacklevel=2,
            )
            self.x_val = training_data.copy()
            self.y_val = training_labels.copy() if training_labels is not None else None
        else:
            validation_indices = np.sort(shuffle_indices[self.full_training_size :])
            self.x_val = training_data[validation_indices]
            self.y_val = (
                training_labels[validation_indices]
                if training_labels is not None
                else None
            )

        self._batch_high_mask = np.zeros(self.full_training_size, dtype=bool)

    def get_next_batch(
        self, batch_size: int | None = None
    ) -> tuple[np.ndarray, ...]:
        """Return a balanced random batch from the training partition."""
        if batch_size is not None and (
            not isinstance(batch_size, (int, np.integer)) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        if batch_size is None or batch_size >= self.full_training_size:
            if self.y_train is not None:
                return self.x_train, self.y_train
            return (self.x_train,)

        low_mask = np.logical_not(self._batch_high_mask)
        have_low = bool(np.any(low_mask))
        minimum_mask = low_mask if have_low else np.ones(
            self.full_training_size, dtype=bool
        )
        minimum_indices = np.nonzero(minimum_mask)[0]

        if minimum_indices.size >= batch_size:
            chosen = self.rand_gen.choice(
                minimum_indices, size=batch_size, replace=False
            )
            if have_low:
                self._batch_high_mask[chosen] = True
            else:
                self._batch_high_mask[:] = False
                self._batch_high_mask[chosen] = True
        else:
            remaining = batch_size - minimum_indices.size
            other_indices = np.nonzero(np.logical_not(minimum_mask))[0]
            extra = self.rand_gen.choice(
                other_indices, size=remaining, replace=False
            )
            chosen = np.concatenate([minimum_indices, extra])
            self.rand_gen.shuffle(chosen)
            self._batch_high_mask[:] = False
            self._batch_high_mask[extra] = True

        x_batch = self.x_train[chosen]
        if self.y_train is not None:
            return x_batch, self.y_train[chosen]
        return (x_batch,)

    def get_full_training_data(self) -> tuple[np.ndarray, ...]:
        """Return the original, unsplit training data."""
        if self.supervised:
            return self.original_training_data, self.original_training_labels  # type: ignore[return-value]
        return (self.original_training_data,)

    def get_validation_data(self) -> tuple[np.ndarray, ...]:
        """Return the validation partition."""
        if self.y_val is not None:
            return self.x_val, self.y_val
        return (self.x_val,)
