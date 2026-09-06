import numpy as np
from .regression_model import NonLinearRegressionModel
from ..mlp_structure.multi_layer_perceptron import MultiLayerPerceptron
from ..utils.data_handler import TrainingData
from ..utils import function_collections as fc
from ..general_model.optimizer import Adam
from ..general_model.validation import EarlyStopper

rng = np.random.default_rng(2024)


class NeuralRegressor(NonLinearRegressionModel):
    """
    A neural network-based regressor using a multi-layer perceptron architecture.
    Inherits from NonLinearRegressionModel to provide regression capabilities.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: list[int],
                 activation: fc.RealDifferentialFunc = fc.leaky_relu,
                 rand_gen: np.random.Generator = rng
                 ) -> None:
        mlp_model = MultiLayerPerceptron(
            [input_dim] + hidden_layers + [output_dim],
            activation_func=activation,
            rand_gen=rand_gen
        )
        optimizer = Adam()
        super().__init__(mlp_model, optimizer)

    def train_model(self,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    epochs: int,
                    batch_size: int,
                    learning_rate: float = 0.001,
                    validation_ratio: float = 0,
                    rand_gen: np.random.Generator = rng
                    ) -> tuple[list[np.floating], list[float]]:
        """
        Train the model using input data and target values.
        Args:
            x_train (np.ndarray): Training input data of shape (n_samples, n_features), dtype float.
            y_train (np.ndarray): Target values of shape (n_samples, output_dim), dtype float.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float, optional): Learning rate for the optimizer.
                                             Defaults to 0.001.
            validation_ratio (float, optional): Ratio of training data to use for validation. 
                                                Defaults to 0.
        Returns:
            tuple[list, list[float]]: A tuple containing the training history (list of losses) 
                                      and validation loss history (list of losses).
        """
        training_data_handler = TrainingData(
            x_train, y_train, validation_ratio, rand_gen=rand_gen)
        early_stopper = EarlyStopper(
            max_epochs=epochs, patience=max(5, epochs//10))
        return super().train(training_data_handler, early_stopper,
                             batch_size, learning_rate=learning_rate)
