import numpy as np

from abstract_ml.mlp_structure.multi_layer_perceptron import MultiLayerPerceptron
from abstract_ml.utils import function_collections as fc


def test_mlp_parameter_gradient_matches_finite_difference() -> None:
    model = MultiLayerPerceptron(
        [2, 3, 1],
        activation_func=fc.tanh,
        rand_gen=np.random.default_rng(7),
    )
    x = np.array([[0.2, -0.1], [-0.4, 0.3]], dtype=float)
    upstream = np.array([[0.7], [-0.2]], dtype=float)
    analytical = model.loss_gradient_by_param(x, upstream)
    original = model.get_parameter()
    epsilon = 1e-6

    for layer_index, parameter_matrix in enumerate(original):
        coordinates = [(0, 0), tuple(np.array(parameter_matrix.shape) - 1)]
        for coordinate in coordinates:
            model.set_parameter(original)
            plus = model.get_parameter()
            minus = model.get_parameter()
            plus[layer_index][coordinate] += epsilon
            minus[layer_index][coordinate] -= epsilon
            model.set_parameter(plus)
            loss_plus = float(np.sum(model.forward(x) * upstream))
            model.set_parameter(minus)
            loss_minus = float(np.sum(model.forward(x) * upstream))
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            assert np.isclose(
                analytical[layer_index][coordinate],
                numerical,
                rtol=1e-4,
                atol=1e-6,
            )

    model.set_parameter(original)
