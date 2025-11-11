from typing import override
import numpy as np

from ..general_model.parameterized_model import ParameterizedModel
from ..utils import function_collections as func

rng = np.random.default_rng(1905)


class MultiLayerPerceptron(ParameterizedModel):
    """
    A multi-layer perceptron (MLP) model.

    Attributes:
        layers (list[int]): A list specifying the number of neurons in each layer.
        activation_func (func.RealDifferentialFunc | list[func.RealDifferentialFunc]):
            The activation function(s) to be used in the hidden layers. 
            If a single function is provided, it will be used for all hidden layers. 
            If a list is provided, its length must be equal to the number of hidden layers.
        params (np.ndarray): The model parameters, stored as an array of weight matrices.
                             shape (num_layers - 1,), dtype object. Each element is a weight matrix
                             of shape (layer_size_in + 1, layer_size_out), dtype float.
    """

    def __init__(
        self,
        layers: list[int],
        activation_func: func.RealDifferentialFunc | list[func.RealDifferentialFunc],
        rand_gen: np.random.Generator = rng
    ) -> None:
        self.layers = layers
        self.activation_func = activation_func

        self.activation_func_list: list[func.RealDifferentialFunc] = []
        if isinstance(activation_func, list):
            if len(activation_func) != len(layers) - 2:
                raise ValueError(
                    "Length of activation_func list must be equal to number of layers - 2"
                    )
            self.activation_func_list = activation_func
        else:
            self.activation_func_list = [activation_func] * (len(layers) - 2)
        self._initialize_parameters(rand_gen)

    def _initialize_parameters(self, rand_gen: np.random.Generator = rng) -> None:
        param_shapes = [(self.layers[i] + 1, self.layers[i + 1])
                        for i in range(len(self.layers) - 1)]
        mean = 0.0
        stddev = 0.1
        initial_params = [rand_gen.normal(mean, stddev, size=shape)
                          for shape in param_shapes]
        self.params = np.empty(len(initial_params), dtype=object)
        self.params[:] = initial_params




    def _forward_internal_pass(
        self,
        x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations_aug: list[np.ndarray] = []
        pre_activations: list[np.ndarray] = []
        activation_funcs = self.activation_func_list + [func.identity]
        a_current = np.insert(x, x.shape[1], 1, axis=1)

        for _, (param, activation_func) in enumerate(zip(self.params, activation_funcs)):
            activations_aug.append(a_current)
            z = a_current @ param
            pre_activations.append(z)
            a_current = activation_func(z)
            a_current = np.insert(a_current, a_current.shape[1], 1, axis=1)

        return activations_aug, pre_activations

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        _, pre_activations = self._forward_internal_pass(x)
        return pre_activations[-1]

    @override
    def loss_gradient_by_param(
        self,
        x: np.ndarray,
        dloss_doutput: np.ndarray
    ) -> np.ndarray:
        activations_aug, pre_activations = self._forward_internal_pass(x)

        batch_size = x.shape[0]
        deltas: list[np.ndarray] = [np.empty(0)] * len(self.params)
        deltas[-1] = dloss_doutput

        for l in range(len(self.params) - 1, 0, -1):
            deltas[l - 1] = (
                deltas[l] @ self.params[l].T[:, :-1]
                * self.activation_func_list[l - 1].derivative(pre_activations[l - 1])
            )
        gradient_params = [(activations_aug[l].T @ deltas[l]) /
                           batch_size
                           for l in range(len(self.params))]
        gradient_array = np.empty(len(gradient_params), dtype=object)
        gradient_array[:] = gradient_params
        return gradient_array

    @override
    def loss_gradient_by_input(self, x: np.ndarray, dloss_doutput: np.ndarray) -> np.ndarray:
        _, pre_activations = self._forward_internal_pass(x)

        delta = dloss_doutput

        for l in range(len(self.params) - 1, 0, -1):
            delta = (
                delta @ self.params[l].T[:, :-1]
                * self.activation_func_list[l - 1].derivative(pre_activations[l - 1])
            )
        return delta @ self.params[0].T[:, :-1]

    @override
    def input_dim(self) -> int:
        return self.layers[0]

    @override
    def output_dim(self) -> int:
        return self.layers[-1]
