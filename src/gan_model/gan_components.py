import numpy as np
from ..general_model.parameterized_model import ParameterizedModel
from ..utils.function_collections import logistic


class Generator:
    """
    Generator model for generating data from latent variables.

    Attributes:
        model (ParameterizedModel): The underlying parameterized model used for generation.
    """

    model: ParameterizedModel

    def __init__(self, model: ParameterizedModel) -> None:
        self.model = model


    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate data from latent variables z using the generator model.
        
        Args:
            z (np.ndarray): Latent variables, shape (batch_size, latent_dim), dtype float.

        Returns:
            np.ndarray: Generated data, shape (batch_size, data_dim), dtype float.
        """
        return self.model.forward(z)




class Discriminator:
    """
    Discriminator model for distinguishing between real and generated data.

    Attributes:
        model (ParameterizedModel): The underlying parameterized model used for discrimination.
    """

    model: ParameterizedModel

    def __init__(self, model: ParameterizedModel) -> None:
        if model.output_dim() != 1:
            raise ValueError("Discriminator model must have output dimension of 1.")
        self.model = model


    def logits(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the logits for input data x using the discriminator model.
        
        Args:
            x (np.ndarray): Input data, shape (batch_size, data_dim), dtype float.
            
        Returns:
            np.ndarray: Logits, shape (batch_size, 1), dtype float.
        """
        return self.model.forward(x)


    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for input data x using the discriminator model.
        
        Args:
            x (np.ndarray): Input data, shape (batch_size, data_dim), dtype float.
            
        Returns:
            np.ndarray: Class probabilities, shape (batch_size, 1), dtype float.
        """
        return logistic(self.logits(x))


class Critic:
    """
    Critic model for WGAN: maps input x to a real-valued score C_phi(x) (no logistic).

    Attributes:
        model (ParameterizedModel): The underlying parameterized model used as the critic.
                                    Must output a single scalar per sample (output_dim == 1).
    """
    model: ParameterizedModel

    def __init__(self, model: ParameterizedModel) -> None:
        if model.output_dim() != 1:
            raise ValueError("Critic model must have output dimension of 1.")
        self.model = model
        self.model.lipschitz_normalize()

    def score(self, x: np.ndarray) -> np.ndarray:
        """
        Compute C_phi(x), the critic score(s).

        Args:
            x (np.ndarray): Input data, shape (batch_size, data_dim), dtype float.

        Returns:
            np.ndarray: Scores, shape (batch_size, 1), dtype float.
        """
        return self.model.forward(x)
