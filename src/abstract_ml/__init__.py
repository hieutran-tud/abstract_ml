"""NumPy-first educational machine-learning toolkit."""

__version__ = "0.1.0"

from .classification.neural_classification import NeuralClassifier
from .gan_model.neural_gan import NeuralGAN, NeuralWGAN
from .general_model.optimizer import Adam, SGD
from .mlp_structure.multi_layer_perceptron import MultiLayerPerceptron
from .regression.neural_regression import NeuralRegressor
from .regression.regression_model import LinearRegressionModel

__all__ = [
    "Adam",
    "LinearRegressionModel",
    "MultiLayerPerceptron",
    "NeuralClassifier",
    "NeuralGAN",
    "NeuralRegressor",
    "NeuralWGAN",
    "SGD",
]
