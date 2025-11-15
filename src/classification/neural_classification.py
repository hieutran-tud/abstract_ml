import numpy as np
from .classification_model import ProbabilisticClassificationModel
from ..mlp_structure.multi_layer_perceptron import MultiLayerPerceptron
from ..utils import function_collections as fc
from ..general_model.optimizer import Adam

rng = np.random.default_rng(2024)

class NeuralClassifier(ProbabilisticClassificationModel):
    """
    A neural network-based classifier using a multi-layer perceptron (MLP) architecture.
    Inherits from ProbabilisticClassificationModel 
    to provide probabilistic outputs for classification tasks.
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_layers: list[int],
                 activation: fc.RealDifferentialFunc = fc.leaky_relu,
                 rand_gen: np.random.Generator = rng
    ) -> None:
        mlp_model = MultiLayerPerceptron(
            [input_dim] + hidden_layers + [num_classes],
            activation_func=activation,
            rand_gen=rand_gen
        )
        optimizer = Adam()
        super().__init__(mlp_model, optimizer)


    def train_model_with_labels(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int,
              batch_size: int,
              learning_rate: float = 0.001,
              validation_ratio: float = 0,
              rand_gen: np.random.Generator = rng
              ) -> tuple[list[np.floating], list[float]]:
        """
        Train the model using input data and integer class labels.

        Args:
            x_train (np.ndarray): Training input data of shape (n_samples, n_features), dtype float.
            y_train (np.ndarray): Integer class labels of shape (n_samples,), dtype int.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float, optional): Learning rate for the optimizer.
                                             Defaults to 0.001.
            validation_ratio (float, optional): Ratio of training data to use for validation. 
                                                Defaults to 0.

        Returns:
            tuple[list[np.floating], list[float]]: A tuple containing
                                                - the training loss (list of losses)
                                                - the validation loss (list of accuracies).
        """
        return super().train_with_labels(x_train, y_train, epochs,
                                         batch_size, learning_rate,
                                         validation_ratio, rand_gen=rand_gen)
