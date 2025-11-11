from abc import ABC, abstractmethod
import numpy as np

rng = np.random.default_rng(1111)


class NoiseSampler(ABC):
    """
    Abstract base class for noise samplers.
    """

    @abstractmethod
    def sample(self, batch_size: int) -> np.ndarray:
        """
        Sample noise vectors.

        Args:
            batch_size (int): Number of noise vectors to sample.

        Returns:
            np.ndarray: Sampled noise vectors of shape (batch_size, noise_dim), dtype float
        """


class GaussianNoiseSampler(NoiseSampler):
    """
    Gaussian noise sampler: samples noise vectors from a Gaussian distribution.

    Attributes:
        noise_dim (int): Dimension of the noise vectors.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        random_gen (np.random.Generator): Random number generator.
    """

    noise_dim: int
    mu: float
    sigma: float
    random_gen: np.random.Generator

    def __init__(self, noise_dim: int,
                 mu: float = 0.0, sigma: float = 1.0,
                 random_gen: np.random.Generator = rng) -> None:
        self.noise_dim = noise_dim
        self.mu = mu
        self.sigma = sigma
        self.random_gen = random_gen

    def sample(self, batch_size: int) -> np.ndarray:
        return self.random_gen.normal(
            loc=self.mu,
            scale=self.sigma,
            size=(batch_size, self.noise_dim)
        )