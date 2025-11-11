import numpy as np
from .gan_trainer import GANTrainer
from ..mlp_structure.multi_layer_perceptron import MultiLayerPerceptron
from ..utils import function_collections as fc
from ..utils.data_handler import TrainingData
from .noise_generator import GaussianNoiseSampler
from ..general_model.optimizer import Adam
from ..general_model.validation import EarlyStopper

rng = np.random.default_rng(123456)


class NeuralGAN(GANTrainer):
    """
    Generative Adversarial Network model with neural network-based generator and discriminator.
    This class implements a GAN model where both the generator and discriminator are modeled
    using multi-layer perceptrons (MLPs). It also includes a Gaussian noise sampler for generating
    latent variables.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        hidden_layer: list[int],
        activation: fc.RealDifferentialFunc = fc.leaky_relu,
        rand_gen: np.random.Generator = rng
    ) -> None:
        gen_mlp = MultiLayerPerceptron(
            [latent_dim] + hidden_layer + [data_dim],
            activation,
            rand_gen=rand_gen
        )
        disc_mlp = MultiLayerPerceptron(
            [data_dim] + hidden_layer + [1],
            activation,
            rand_gen=rand_gen
        )
        noise_sampler = GaussianNoiseSampler(latent_dim, 0, 1, rand_gen)
        super().__init__(
            gen_model=gen_mlp,
            disc_model=disc_mlp,
            noise_sampler=noise_sampler
        )

    def train_model(
        self,
        real_data: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float = 0.001,
        d_steps_per_one_g_step: int = 1,
        validation_ratio: float = 0,
        rand_gen: np.random.Generator = rng
    ) -> tuple[list, list[float]]:
        """
        Train the GAN model for a specified number of epochs using the provided real data
        using Adam optimizers for both the generator and discriminator.
        After each epoch, the model is validated using the provided training data handler.
        The method returns lists of training and validation losses for each epoch.

        Args:
            real_data (np.ndarray): The real data to train on
                                    shape (num_samples, data_dim), dtype float.
            epochs (int): Number of training epochs.    
            batch_size (int): Size of each training batch.
            learning_rate (float, optional): Learning rate for the optimizers. Defaults to 0.001.
            d_steps_per_one_g_step (int, optional): Number of discriminator steps per generator step
                                                    Defaults to 1.
            validation_ratio (float, optional): Ratio of data to use for validation. Defaults to 0.
            rand_gen (np.random.Generator, optional): Random number generator. Defaults to rng.

        Returns:
            tuple[list, list[float]]: Training and validation losses for each epoch.
        """
        gen_optimizer = Adam(learning_rate)
        disc_optimizer = Adam(learning_rate)
        training_data_handler = TrainingData(
            real_data, validation_ratio=validation_ratio, rand_gen=rand_gen)
        early_stopper = EarlyStopper(max_epochs=epochs, patience=max(5, epochs//10))
        return super().train(
            training_data_handler,
            early_stopper,
            batch_size,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            d_steps_per_one_g_step=d_steps_per_one_g_step
        )
