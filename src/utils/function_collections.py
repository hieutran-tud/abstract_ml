from collections.abc import Callable
import warnings
import numpy as np
from .helpers import minimize_scalar


class RealDifferentialFunc:
    """
    General model of a real-valued univariate differentiable function.
    """

    def __init__(self, function_def: Callable,
                 derivative: Callable | None = None,
                 lipschitz_constant: float | None = None):
        self.function_def = function_def
        if derivative is None:
            warnings.warn(
                "Numerical derivative approximation is used.", RuntimeWarning)

            def numerical_derivative(x):
                eps = 1e-7
                h = eps * np.maximum(1.0, np.abs(x))
                return (function_def(x+h) - function_def(x-h))/(2*h)
            derivative = numerical_derivative
        self.derivative = derivative
        if lipschitz_constant is None:
            warnings.warn(
                "Lipschitz constant is not provided.", RuntimeWarning)
            lipschitz_constant = self.numerical_lipschitz_constant()
        self.lipschitz_constant = lipschitz_constant

    def __call__(self, x):
        try:
            return self.function_def(x)
        except RuntimeWarning as e:
            print("Warning during function evaluation:", e)
            return self.function_def(x)

    def numerical_lipschitz_constant(self) -> float:
        """
        Estimate the Lipschitz constant numerically.

        Returns:
            float: Estimated Lipschitz constant
        """
        return minimize_scalar(
            lambda x: -np.abs(self.derivative(x)))

    def get_lipschitz_constant(self) -> float:
        """
        Return the function lipschitz constant, can be np.inf

        Returns:
            float: the function lipschitz constant
        """
        return self.lipschitz_constant


identity = RealDifferentialFunc(
    lambda x: x,
    np.ones_like,
    lipschitz_constant=1.0,
)

logistic = RealDifferentialFunc(
    lambda x: 0.5 + 0.5*np.tanh(0.5*x),
    lambda x: 0.25 - 0.25*np.tanh(0.5*x)**2,
    lipschitz_constant=0.25,
)

tanh = RealDifferentialFunc(
    np.tanh,
    lambda x: 1 - np.tanh(x)**2,
    lipschitz_constant=1.0,
)

relu = RealDifferentialFunc(
    lambda x: np.maximum(0, x),
    lambda x: (np.sign(x) + 1.0) / 2.0,
    lipschitz_constant=1.0,
)

swish = RealDifferentialFunc(
    lambda x: x * logistic(x),
    lambda x: logistic(x) + x * logistic(x) * (1 - logistic(x)),
    lipschitz_constant=1.09984,
)

softplus = RealDifferentialFunc(
    lambda x: np.logaddexp(0, x),
    logistic,
    lipschitz_constant=1.0,
)

leaky_relu = RealDifferentialFunc(
    lambda x: 0.1*x + relu(x),
    lambda x: 0.1 + relu.derivative(x),
    lipschitz_constant=1.1,
)

leaky_tanh = RealDifferentialFunc(
    lambda x: 0.1*x + tanh(x),
    lambda x: 0.1 + tanh.derivative(x),
    lipschitz_constant=1.1,
)

leaky_logistic = RealDifferentialFunc(
    lambda x: 0.05*x + logistic(x),
    lambda x: 0.05 + logistic.derivative(x),
    lipschitz_constant=0.30,
)

soft_leaky = RealDifferentialFunc(
    lambda x: 0.1*x + softplus(x),
    lambda x: 0.1 + logistic(x),
    lipschitz_constant=1.1,
)

linear_units: list[RealDifferentialFunc] = [
    relu, swish, leaky_relu, soft_leaky, softplus]

sigmoids: list[RealDifferentialFunc] = [
    logistic, tanh, leaky_tanh, leaky_logistic]
