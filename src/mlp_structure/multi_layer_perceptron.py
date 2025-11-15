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
        self.rand_gen = rand_gen

        self.activation_func_list: list[func.RealDifferentialFunc] = []
        if isinstance(activation_func, list):
            if len(activation_func) != len(layers) - 2:
                raise ValueError(
                    "Length of activation_func list must be equal to number of layers - 2"
                )
            self.activation_func_list = activation_func
        else:
            self.activation_func_list = [activation_func] * (len(layers) - 2)

        self._lipschitz_activation: bool = True
        for a in self.activation_func_list:
            lips_cons = float(a.get_lipschitz_constant())
            if (not np.isfinite(lips_cons)) \
                    or np.isclose(lips_cons, 0.0) or abs(lips_cons - 1.0) > 0.0:
                self._lipschitz_activation = False

        self._initialize_parameters()

    @override
    def get_parameter(self) -> np.ndarray:
        copied_array = np.empty(len(self.params), dtype=object)
        copied_array[:] = [np.copy(param) for param in self.params]
        return copied_array

    def _initialize_parameters(self) -> None:
        param_shapes = [(self.layers[i] + 1, self.layers[i + 1])
                        for i in range(len(self.layers) - 1)]
        mean = 0.0
        stddev = 0.1
        initial_params = [self.rand_gen.normal(mean, stddev, size=shape)
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

        ones = np.ones((x.shape[0], 1), dtype=x.dtype)
        a_current = np.concatenate([x, ones], axis=1)

        for _, (param, activation_func) in enumerate(zip(self.params, activation_funcs)):
            activations_aug.append(a_current)
            z = a_current @ param
            pre_activations.append(z)
            a_current = activation_func(z)
            a_current = np.concatenate([a_current, ones], axis=1)

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

        deltas: list[np.ndarray] = [np.empty(0)] * len(self.params)
        deltas[-1] = dloss_doutput

        for l in range(len(self.params) - 1, 0, -1):
            deltas[l - 1] = (
                deltas[l] @ self.params[l].T[:, :-1]
                * self.activation_func_list[l - 1].derivative(pre_activations[l - 1])
            )
        gradient_params = [(activations_aug[l].T @ deltas[l])
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

    def _normalize_activation_lipschitz(self) -> None:
        normalized: list[func.RealDifferentialFunc] = []
        for a in self.activation_func_list:
            current_lips_cons = float(a.get_lipschitz_constant())
            if (not np.isfinite(current_lips_cons)) or np.isclose(current_lips_cons, 0.0):
                raise ValueError(
                    f"Activation has non-usable Lipschitz constant (got {current_lips_cons})."
                )
            if current_lips_cons > 1.0:
                # Use a factory function to properly capture loop variables
                def make_scaled_funcs(activation_fn, lipschitz_const):
                    def f_scaled(x: np.ndarray) -> np.ndarray:
                        return activation_fn(x) / lipschitz_const

                    def df_scaled(x: np.ndarray) -> np.ndarray:
                        return activation_fn.derivative(x) / lipschitz_const
                    return f_scaled, df_scaled
                f_scaled, df_scaled = make_scaled_funcs(
                    a, current_lips_cons)
                a = func.RealDifferentialFunc(
                    function_def=f_scaled,
                    derivative=df_scaled,
                    lipschitz_constant=1.0
                )
            normalized.append(a)
        self.activation_func_list = normalized
        self._lipschitz_activation = True

    def _normalize_weight_lipschitz(self) -> None:
        new_params = np.empty_like(self.params, dtype=object)
        for i, w in enumerate(self.params):
            weight_matrix = w[:-1, :]
            bias_vector = w[-1:, :]
            try:
                svals = np.linalg.svd(weight_matrix, compute_uv=False)
                sigma_max = float(svals[0]) if svals.size > 0 else 0.0
            except np.linalg.LinAlgError:
                u = self.rand_gen.normal(size=(weight_matrix.shape[0],))
                u /= np.linalg.norm(u)
                v = self.rand_gen.normal(size=(weight_matrix.shape[1],))
                v /= np.linalg.norm(v)
                for _ in range(10):
                    v = weight_matrix.T @ u
                    v /= np.linalg.norm(v)
                    u = weight_matrix @ v
                    u /= np.linalg.norm(u)
                sigma_max = float(u @ weight_matrix @ v)

            if not np.isfinite(sigma_max):
                raise RuntimeError(
                    "Failed to compute spectral norm during lipschitz_normalize()."
                )
            if sigma_max > 1.0:
                weight_matrix = weight_matrix / sigma_max
            new_params[i] = np.vstack([weight_matrix, bias_vector])

        self.params = new_params

    @override
    def lipschitz_normalize(self) -> None:
        if not self._lipschitz_activation:
            self._normalize_activation_lipschitz()
        self._normalize_weight_lipschitz()
