import numpy as np
import pytest

from abstract_ml.utils.data_handler import TrainingData


def test_training_data_splits_and_handles_large_batches() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    handler = TrainingData(
        x,
        y,
        validation_ratio=0.2,
        rand_gen=np.random.default_rng(3),
    )

    assert handler.full_training_size == 8
    assert handler.x_val.shape[0] == 2
    x_batch, y_batch = handler.get_next_batch(100)
    assert x_batch.shape == (8, 2)
    assert y_batch.shape == (8,)


def test_training_data_rejects_empty_training_partitions() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        TrainingData(np.empty((0, 2)))

    with pytest.raises(ValueError, match="leaves no samples"):
        TrainingData(np.ones((1, 2)), validation_ratio=0.999)
