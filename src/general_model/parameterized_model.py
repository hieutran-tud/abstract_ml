from abc import ABC, abstractmethod
import numpy as np


class ParameterizedModel(ABC):
    """
    Abstract base class for parameterized models
    This represents a (multivariate input, multivariate output) function 
    parametrized by a set of parameters params.
    """

    params: np.ndarray

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the model on input x.

        Args:
            x (np.ndarray): Input array, shape (batch_size, ...), dtype Any

        Returns:
            np.ndarray: Output array, shape (batch_size, ...), dtype Any
        """

    def get_parameter(self) -> np.ndarray:
        """
        Get the model parameters.

        Returns:
            np.ndarray: The model parameters, AnyShape, dtype Any
        """
        return self.params.copy()

    def set_parameter(self, new_param: np.ndarray) -> None:
        """
        Set the model parameters.

        Args:
            new_param (np.ndarray): The new model parameters, AnyShape, dtype Any
        """
        self.params = new_param

    @abstractmethod
    def loss_gradient_by_param(self, x: np.ndarray, dloss_doutput: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss w.r.t. the model parameters.

        Args:
            x (np.ndarray): Input array of shape (batch_size, ...), dtype Any.
            dLoss_dOutput (np.ndarray): Gradient of the loss w.r.t. the model output, 
                                        shape (batch_size, ...), dtype Any.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the model parameters, dtype Any.
        """

    @abstractmethod
    def loss_gradient_by_input(self, x: np.ndarray, dloss_doutput: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss w.r.t. the model input.

        Args:
            x (np.ndarray): Input array of shape (batch_size, ...), dtype Any.
            dLoss_dOutput (np.ndarray): Gradient of the loss w.r.t. the model output,
                                        shape (batch_size, ...), dtype Any.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the model input, 
                        shape (batch_size, ...), dtype Any.
        """

    @abstractmethod
    def input_dim(self) -> int:
        """
        Get the input dimension of the model.

        Returns:
            int: The input dimension of the model.
        """

    @abstractmethod
    def output_dim(self) -> int:
        """
        Get the output dimension of the model.

        Returns:
            int: The output dimension of the model.
        """
