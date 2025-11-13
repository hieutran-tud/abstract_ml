from abc import abstractmethod, ABC
from typing import override
import numpy as np

from ..general_model.parameterized_model import ParameterizedModel
from ..general_model.optimizer import Adam, GradientOptimizer
from ..general_model.validation import ValidatableTrainingModel
from ..utils.data_handler import TrainingData


class RegressionModel(ABC):
    """
    Abstract base class for the parameterized regression models
    """

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Compute the Mean Squared Error (MSE) loss between predicted and true values.

        Args:
            y_pred (np.ndarray): Predicted output values, 
                                 shape (num_samples, output_dim), dtype float
            y_true (np.ndarray): True output values, 
                                 shape (num_samples, output_dim), dtype float
        Returns:
            np.floating: The MSE loss.
        """
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for input data x.

        Args:
            x (np.ndarray): Input data with the shape of (n_samples, n_features), dtype float.
        Returns:
            np.ndarray: Predicted output with the shape of (n_samples, n_outputs), dtype float.
        """

    def r2_score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the R-squared score of the model on the test set.

        Args:
            x_test (np.ndarray): Test data, shape (n_samples, n_features), dtype float.
            y_test (np.ndarray): True target values, shape (n_samples, n_outputs), dtype float.
        Returns:
            float: The R-squared score of the model on the test set.
        """
        y_pred = self.predict(x_test)
        ss_total = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


class NonLinearRegressionModel(RegressionModel, ValidatableTrainingModel[np.floating]):
    """
    A regression model that uses a non linear parameterized model to make predictions:
    f: R^n -> R^m, where n is the input dimension and m is the output dimension.
    The model is trained using gradient descent optimization to minimize the (MSE) loss.

    Attributes:
        model (ParameterizedModel): The parameterized model used for regression.
        optimizer (GradientOptimizer): The gradient descent optimizer used for training.
    """

    model: ParameterizedModel

    @staticmethod
    def grad_given_output(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute the loss gradient given the true and predicted outputs:
            dL/dy_pred = (-1/N) * (y_true - y_pred)

        Args:
            y_true (np.ndarray): True output, shape (num_samples, output_dim), dtype float
            y_pred (np.ndarray): Predicted output, shape (num_samples, output_dim), dtype float
        Returns:
            np.ndarray: Gradient of the loss w.r.t. y_pred, shape (num_samples, output_dim)
        """
        grad_pred = (y_pred - y_true) / y_true.size
        return grad_pred

    def __init__(self, model: ParameterizedModel) -> None:
        self.model = model


    @override
    def get_parameters(self) -> np.ndarray:
        """
        Get the model (set of) parameters.

        Returns:
            np.ndarray: The model set parameters
        """
        return self.model.get_parameter()


    @override
    def set_parameters(self, params: np.ndarray) -> None:
        """
        Set the model (set of) parameters.

        Args:
            params (np.ndarray): The new model parameters
        """
        self.model.set_parameter(params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for input data x using the regression model.

        Args:
            x (np.ndarray): Input data, shape (num_samples, input_dim), dtype float
        Returns:
            np.ndarray: Predicted output, shape (num_samples, output_dim), dtype float
        """
        return self.model.forward(x)

    def train_epoch(self,
                    training_data_handler: TrainingData,
                    batch_size: int,
                    *,
                    optimizer: GradientOptimizer | None = None,
                    **_) -> np.floating:
        """
        Train the model for one epoch.
        The training data is shuffled and divided into batches. 
        For each batch, the model performs a forward pass,
        computes the loss and its gradient, and updates the model parameters using the optimizer.
        The average loss over the epoch is returned.

        Args:
            training_data (TrainingData): The training data handler object 
                                          containing training and validation data.
            batch_size (int): The size of each training batch.
        Returns:
            float: The average loss over the epoch.
        """
        total_loss: np.floating = np.float64(0)
        shuffled_batches = training_data_handler.shuffle_and_divide(batch_size)
        if optimizer is None:
            optimizer = Adam(0.001)
        for (x_data, y_true) in shuffled_batches:
            y_pred = self.model.forward(x_data)
            loss = self.compute_loss(y_pred, y_true)
            dloss_dout = NonLinearRegressionModel.grad_given_output(
                y_true, y_pred)
            grad_params = self.model.loss_gradient_by_param(x_data, dloss_dout)
            optimizer.stepwise_update(self.model, grad_params)
            total_loss += loss
        return total_loss / len(shuffled_batches)

    def validate(self, training_data_handler: TrainingData) -> np.floating:
        """
        Validate the model using the validation dataset.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.

        Returns:
            float: The validation loss.
        """
        validation_x, validation_y = training_data_handler.get_validation_data()
        validation_loss = self.compute_loss(
            validation_y, self.model.forward(validation_x))
        return validation_loss


class LinearRegressionModel(RegressionModel):
    """
    A simple linear regression model.

    Attributes:
        weights (np.ndarray): The weights of the linear model.
        bias (np.ndarray): The bias of the linear model.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the linear regression model to the training data.

        Args:
            x_train (np.ndarray): Training data, shape (n_samples, n_features), dtype float.
            y_train (np.ndarray): Training target, shape (n_samples, n_outputs), dtype float.
        """
        n_samples = x_train.shape[0]
        x_with_bias = np.hstack([x_train, np.ones((n_samples, 1))])
        params, _, _, _ = np.linalg.lstsq(x_with_bias, y_train, rcond=None)
        self.weights = params[:-1]
        self.bias = params[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for input data x using the linear regression model.

        Args:
            x (np.ndarray): Input data, shape (n_samples, n_features), dtype float.
        Returns:
            np.ndarray: Predicted output, shape (n_samples, n_outputs), dtype float.
        """
        if self.weights is None or self.bias is None:
            raise ValueError(
                "Model parameters not initialized. Call 'fit' first.")
        return x.dot(self.weights) + self.bias
