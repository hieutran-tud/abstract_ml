import numpy as np

from abstract_ml.utils.helpers import fid


def test_frechet_style_distance_is_zero_for_identical_inputs() -> None:
    samples = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False)

    assert np.isclose(fid(mean, mean, covariance, covariance), 0.0, atol=1e-10)
