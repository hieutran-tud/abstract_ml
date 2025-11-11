from __future__ import annotations
from abc import ABC, abstractmethod
from typing import override
import numpy as np
from .parameterized_model import ParameterizedModel


class GradientOptimizer(ABC):
    """""
    Base class for gradient-based optimizers.
    """

    @abstractmethod
    def stepwise_update(
        self,
        model: ParameterizedModel,
        param_gradients: np.ndarray
    ) -> None:
        """
        Perform a stepwise update of the model's parameters using the provided gradients.

        Args:
            model (ParameterizedModel): The model to be updated.
            param_gradients (np.ndarray): The gradients of the model's parameters, dtype Any.
        """

    def reset(self) -> None:
        """
        Reset any internal state of the optimizer, if applicable.
        """

    @abstractmethod
    def copy(self) -> GradientOptimizer:
        """
        Create a copy of the optimizer instance.
        """


class SGD(GradientOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Performs parameter updates using the formula:
        new_params = old_params - learning_rate * param_gradients
    For deterministic full-batch gradient descent, 
    set the batch size at utils.data_handler.TrainingData to the entire dataset.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
    """

    learning_rate: float

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @override
    def stepwise_update(
        self,
        model: ParameterizedModel,
        param_gradients: np.ndarray
    ) -> None:
        new_params = model.get_parameter() - self.learning_rate * param_gradients
        model.set_parameter(new_params)

    @override
    def copy(self) -> SGD:
        return SGD(learning_rate=self.learning_rate)


class Adam(GradientOptimizer):
    """
    Adam optimizer.
    Combines the advantages of AdaGrad and RMSProp.
    Performs parameter updates using adaptive estimates of lower-order moments.
    Should not work with full-batch sampling.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant to prevent division by zero.
        m (np.ndarray | float): First moment vector.
        v (np.ndarray | float): Second moment vector.
        t (int): Time step counter.
    """

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    m: np.ndarray | float
    v: np.ndarray | float
    t: int


    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0


    @override
    def stepwise_update(
        self,
        model: ParameterizedModel,
        param_gradients: np.ndarray
    ) -> None:
        params = model.get_parameter()
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * param_gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * param_gradients**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        new_params = params - self.learning_rate * \
            m_hat / (v_hat**0.5 + self.epsilon)
        model.set_parameter(new_params)


    @override
    def reset(self) -> None:
        self.m = 0
        self.v = 0
        self.t = 0


    @override
    def copy(self) -> Adam:
        new_optimizer = Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon
        )
        new_optimizer.m = np.copy(self.m)
        new_optimizer.v = np.copy(self.v)
        new_optimizer.t = self.t
        return new_optimizer
