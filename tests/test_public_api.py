from abstract_ml import MultiLayerPerceptron, __version__


def test_public_package_imports() -> None:
    assert __version__ == "0.1.0"
    assert MultiLayerPerceptron.__name__ == "MultiLayerPerceptron"
