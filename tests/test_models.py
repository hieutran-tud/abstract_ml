import numpy as np

from abstract_ml.regression.regression_model import LinearRegressionModel


def test_linear_regression_fit_and_predict() -> None:
    x = np.arange(5, dtype=float).reshape(-1, 1)
    y = 3.0 * x + 2.0
    model = LinearRegressionModel()

    model.fit(x, y)

    assert np.allclose(model.predict(x), y)
    assert np.isclose(model.r2_score(x, y), 1.0)
